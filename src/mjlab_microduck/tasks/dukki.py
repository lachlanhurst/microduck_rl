"""Dukki task variants: any microduck env cfg on the J288 robot.

``make_dukki_variant(cfg)`` turns a microduck env cfg into its Dukki twin
(task ids ``Mjlab-<Task>-<Flat|Rough>[-Backlash]-Dukki``):

1. Robot -> the Dukki robot cfg on the same MJCF (robot/dukki_constants.py):
   J288 masses, 6S pack, head electronics, and the J288 BAM actuator with
   per-joint kp/kd. The backlash twin uses the plain J288 actuator and keeps
   the joint obs rotor-side (``make_backlash_variant(...,
   encoder_through_backlash=False)``): the J288's position loop and its
   reported position are before the gear play, the XL330's after it.
2. Actor joint_vel -> what the robot will actually observe, chosen by
   ``DUKKI_JOINT_VEL`` (default ``position_difference``):

   * ``position_difference``: (q_t - q_{t-1}) / step_dt, the difference of the
     rotor-side position over one 50 Hz policy tick, as the bridge would
     compute it from consecutive state frames. Recommended by dukki
     docs/j288-testing.md section 9.
   * ``reported_speed``: the servo's own speed field, which BAM models as the
     firmware's 15 ms first-order filter (the same signal its kd uses).

   The XL330's one-tick lag on joint_vel is removed (a Dynamixel moving
   average, not a J288 property); the noise is kept.

The critic keeps the true joint velocity (privileged).

3. An ``action_over_limit`` penalty (as the roller tasks have): commanding a
   joint target more than ``OVER_LIMIT_TORQUE / kp`` past its hard stop is
   penalised. The first Dukki velocity runs (02/10/2026) learned to hold one
   hip_yaw commanded about 2 rad past its stop, the servo at its 1.0 N.m clamp
   100% of the time (a brace that only works because the sim's stop is
   perfectly rigid, and that would overheat the real servo in seconds). The
   allowance lets a joint press its stop with at most OVER_LIMIT_TORQUE.

Gain sweep. ``DUKKI_KP`` [N.m/rad] and ``DUKKI_KD`` [N.m.s/rad] set the gains
of every joint (default dukki_constants.J288_KP / J288_KD). They are read when
the tasks are registered, so set them on the command line:

    DUKKI_KP=3.0 DUKKI_KD=0.08 uv run train Mjlab-Velocity-Flat-Dukki ...

``--hf-jobs`` forwards every ``DUKKI_*`` variable to the job.
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.actuator import J288DRActuator
from mjlab_microduck.robot.dukki_constants import (
    J288_KD,
    J288_KP,
    make_dukki_robot_cfg,
    make_j288_actuators,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backlash import make_backlash_variant

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#: Torque [N.m] a joint may hold against its hard stop before the over-limit
#: penalty starts (the target may go this / kp past the stop).
OVER_LIMIT_TORQUE = 0.2
OVER_LIMIT_WEIGHT = -1.0

JOINT_VEL_SOURCES = ("position_difference", "reported_speed")


def dukki_gains_from_env() -> tuple[float, float]:
    """(kp, kd) for every joint, from DUKKI_KP / DUKKI_KD or the defaults."""
    return (
        float(os.environ.get("DUKKI_KP", J288_KP)),
        float(os.environ.get("DUKKI_KD", J288_KD)),
    )


def dukki_joint_vel_source_from_env() -> str:
    source = os.environ.get("DUKKI_JOINT_VEL", "position_difference")
    if source not in JOINT_VEL_SOURCES:
        raise ValueError(f"DUKKI_JOINT_VEL must be one of {JOINT_VEL_SOURCES}, got {source!r}")
    return source


# ─────────────────────────────────────────────────────────────────────────────
# Joint velocity observations
# ─────────────────────────────────────────────────────────────────────────────


class joint_vel_position_difference(ManagerTermBase):
    """Joint velocity as the position difference over one policy step.

    (q_t - q_{t-1}) / step_dt on the selected joints, minus the default joint
    velocity. Positions are the servo joints' (rotor-side) ``joint_pos``; an
    encoder bias would cancel in the difference. Environments just reset
    report zero until they have two samples.

    A new sample is taken per environment when its physics time has moved
    since the last call, so repeated calls within one step return the same
    value, and loops that step the sim by hand (scripts/*sim2real.py) work
    without the env's step counter.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self._prev: torch.Tensor | None = None
        self._cur: torch.Tensor | None = None
        self._time: torch.Tensor | None = None
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        q = asset.data.joint_pos[:, asset_cfg.joint_ids]
        time = env.sim.data.time[:].clone()
        if self._cur is None:
            self._prev, self._cur, self._time = q.clone(), q.clone(), time
        moved = (time != self._time).unsqueeze(-1)
        self._prev = torch.where(moved, self._cur, self._prev)
        self._cur, self._time = q.clone(), time
        self._prev[self._fresh] = q[self._fresh]
        self._fresh[:] = False
        default_joint_vel = asset.data.default_joint_vel
        assert default_joint_vel is not None
        return (q - self._prev) / env.step_dt - default_joint_vel[:, asset_cfg.joint_ids]


def j288_reported_joint_vel(
    env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """The J288's reported speed (firmware-filtered) on the selected joints."""
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, "_dukki_reported_speed_cols", None)
    if cache is None:
        selected = torch.arange(asset.num_joints, device=env.device)[asset_cfg.joint_ids]
        cols = []
        for local_id in selected.tolist():
            for act in asset.actuators:
                if isinstance(act, J288DRActuator) and local_id in act._target_ids_list:
                    cols.append((act, act._target_ids_list.index(local_id)))
                    break
            else:
                raise ValueError(f"joint {asset.joint_names[local_id]} has no J288 actuator")
        if len({id(a) for a, _ in cols}) != 1:
            raise NotImplementedError("joints spread over several J288 actuators")
        cache = (cols[0][0], torch.tensor([c for _, c in cols], device=env.device))
        env._dukki_reported_speed_cols = cache
    act, col_ids = cache
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return act.reported_speed[:, col_ids] - default_joint_vel[:, asset_cfg.joint_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Variant
# ─────────────────────────────────────────────────────────────────────────────


def make_dukki_variant(
    cfg: ManagerBasedRlEnvCfg,
    model: str = "walk",
    backlash: bool = False,
    kp: float | None = None,
    kd: float | None = None,
    joint_vel_source: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Convert a microduck env cfg to Dukki (J288 robot and joint_vel obs).

    ``model`` picks the MJCF the base task uses ("walk", "standup",
    "allcollisions"); ``backlash`` its backlash twin. ``kp``/``kd``/
    ``joint_vel_source`` default to the DUKKI_* environment variables.
    """
    env_kp, env_kd = dukki_gains_from_env()
    kp = env_kp if kp is None else kp
    actuators = make_j288_actuators(kp_fw=kp, kd_fw=env_kd if kd is None else kd)
    source = joint_vel_source or dukki_joint_vel_source_from_env()
    if source not in JOINT_VEL_SOURCES:
        raise ValueError(f"joint_vel_source must be one of {JOINT_VEL_SOURCES}")

    if backlash:
        robot_cfg = make_dukki_robot_cfg(f"{model}_backlash", actuators)
        cfg = make_backlash_variant(cfg, robot_cfg, encoder_through_backlash=False)
    else:
        cfg.scene.entities = {
            **cfg.scene.entities,
            "robot": make_dukki_robot_cfg(model, actuators),
        }

    terms = cfg.observations["actor"].terms
    if "joint_vel" in terms:
        # Deepcopy: base templates share term objects across groups and make() calls.
        term = deepcopy(terms["joint_vel"])
        term.func = (
            joint_vel_position_difference
            if source == "position_difference"
            else j288_reported_joint_vel
        )
        term.params = {"asset_cfg": term.params.get("asset_cfg", _DEFAULT_ASSET_CFG)}
        term.delay_min_lag = 0
        term.delay_max_lag = 0
        term.delay_update_period = 0
        terms["joint_vel"] = term

    if "joint_pos" in cfg.actions:
        cfg.rewards["action_over_limit"] = RewardTermCfg(
            func=microduck_mdp.action_over_limit_penalty,
            weight=OVER_LIMIT_WEIGHT,
            params={"action_name": "joint_pos", "overshoot": OVER_LIMIT_TORQUE / kp},
        )
    return cfg
