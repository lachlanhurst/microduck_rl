"""Dukki task variants: any microduck env cfg on the J288 robot.

``make_dukki_variant(cfg)`` turns a microduck env cfg into its Dukki twin
(task ids ``Mjlab-<Task>-<Flat|Rough>[-Backlash]-Dukki``):

1. Robot -> the Dukki robot cfg on the same MJCF (robot/dukki_constants.py):
   Fusion inertials plus the cables and fasteners, and the J288 BAM actuator with
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
from mjlab.envs import mdp as mjlab_mdp
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

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


def check_cfg_matches_run(cfg: ManagerBasedRlEnvCfg, checkpoint: str) -> None:
    """Raise if ``cfg`` differs from the one ``checkpoint`` was trained with.

    Compares timestep, decimation and J288 gains against the run's
    ``params/env.yaml``. Gains come from DUKKI_KP / DUKKI_KD, so a script that
    loads a sweep checkpoint without them would silently run other gains.
    Checkpoints without a params folder (Hub downloads, .onnx) are not checked.
    """
    import re
    from pathlib import Path

    path = Path(checkpoint).parent / "params" / "env.yaml"
    if not path.exists():
        print(f"[dukki] {path} not found: not checking {checkpoint} against its training config")
        return
    text = path.read_text()

    def saved(key: str, indent: str = r"\s*") -> list[float]:
        return [float(v) for v in re.findall(rf"^{indent}{key}: ([-0-9.e]+)$", text, re.M)]

    actuators = cfg.scene.entities["robot"].articulation.actuators
    have = {
        "decimation": [float(cfg.decimation)],
        "timestep": [float(cfg.sim.mujoco.timestep)],
        "kp_fw": [float(a.kp_fw) for a in actuators if hasattr(a, "kp_fw")],
        "kd_fw": [float(a.kd_fw) for a in actuators if hasattr(a, "kd_fw")],
    }
    want = {
        "decimation": saved("decimation", ""),
        "timestep": saved("timestep"),
        "kp_fw": saved("kp_fw"),
        "kd_fw": saved("kd_fw"),
    }
    wrong = {k: (have[k], want[k]) for k in have if have[k] != want[k]}
    if wrong:
        detail = ", ".join(f"{k} {h} (trained with {w})" for k, (h, w) in wrong.items())
        raise ValueError(
            f"{checkpoint}: env differs from its training config: {detail}. "
            "Set DUKKI_KP / DUKKI_KD to the run's gains."
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


# ─────────────────────────────────────────────────────────────────────────────
# Sit-stand: practice from stillness
# ─────────────────────────────────────────────────────────────────────────────

#: Share of sit-stand robots that are "still" for an episode: head command held
#: at 0, no pushes, no head jerks. The first Dukki sit-stand run (23/10/2026)
#: sat from standing 88% of the time under training conditions but 63% with the
#: head still and no pushes: it leaned on outside motion to start the descent,
#: and on the robot the head is often still when it is told to sit.
STILL_PROB = 0.5


def _still(env) -> torch.Tensor:
    still = getattr(env, "_dukki_still", None)
    if still is None:
        still = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._dukki_still = still
    return still


def _all_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device, dtype=torch.long)


def sample_still_envs(env, env_ids, prob: float) -> None:
    """Reset event: marks each resetting robot still with probability ``prob``."""
    ids = _all_ids(env, env_ids)
    _still(env)[ids] = torch.rand(len(ids), device=env.device) < prob


def hold_head_still(env, env_ids, head_command_name: str = "head_pose") -> None:
    """Step event: still robots' head command stays at 0. Runs after the command manager's
    resample and before interval events, which skip still robots themselves."""
    still = _still(env)
    if head_command_name in env.command_manager.active_terms and still.any():
        env.command_manager.get_term(head_command_name)._command[still] = 0.0


def push_unless_still(env, env_ids, **params) -> None:
    """``push_by_setting_velocity`` for the robots that are not still."""
    ids = _all_ids(env, env_ids)
    ids = ids[~_still(env)[ids]]
    if len(ids):
        mjlab_mdp.push_by_setting_velocity(env, ids, **params)


def head_jerk_unless_still(env, env_ids, **params) -> None:
    """``seated_head_command_jerk`` for the robots that are not still."""
    ids = _all_ids(env, env_ids)
    ids = ids[~_still(env)[ids]]
    if len(ids):
        microduck_mdp.seated_head_command_jerk(env, ids, **params)


def make_still_practice(cfg: ManagerBasedRlEnvCfg, prob: float = STILL_PROB) -> ManagerBasedRlEnvCfg:
    """Gives a share of sit-stand robots episodes with no head motion and no pushes, so the
    policy learns to sit from a still stand. The push and head-jerk events keep their
    parameters, so their curricula still apply to the other robots."""
    cfg.events["sample_still"] = EventTermCfg(func=sample_still_envs, mode="reset", params={"prob": prob})
    cfg.events["hold_head_still"] = EventTermCfg(func=hold_head_still, mode="step", params={})
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].func = push_unless_still
    if "seated_head_jerk" in cfg.events:
        cfg.events["seated_head_jerk"].func = head_jerk_unless_still
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Sit-stand: head off the ground
# ─────────────────────────────────────────────────────────────────────────────

#: The sit-stand task lets the head touch the ground as a third support. On the
#: Fusion inertials (08/11/2026) the October policy sat by pitching forward onto
#: its head and rocking back off it into the seat. Dukki's head carries the
#: camera, the compute module and the speaker, so it is kept off the ground: a
#: cost for every step the neck or head touches the terrain, and a cost on the
#: contact force above HEAD_GROUND_FREE_N, so a hard smack costs more than a
#: graze.
#
#: Rewards are rates (scaled by dt), so -20 costs 20 per second of contact. One
#: head plant (about 0.3 s down, a 100 N impact over a few steps, ~10 N while
#: propped) then costs about 13: well below a clean sit, still above refusing
#: to sit (the sitting rewards are worth roughly 15 to 20 over a dwell). -3 and
#: -0.2 cost about 2 and changed nothing. Tripled after the soft-transition
#: fine-tune (model_5500) learned to fold the neck right forward (neck_pitch
#: -1.4 rad) at the bottom of the sit as a counterweight, grazing the head at
#: about 15 N in 2 to 5% of sits: at -20 such a graze cost only about 5. The
#: sit-stall cost now covers refusing to sit, so the margin below refusing is no
#: longer needed.
HEAD_GROUND_SENSOR = "head_ground_contact"
HEAD_GROUND_CONTACT_WEIGHT = -60.0  # per second touching
HEAD_GROUND_FORCE_WEIGHT = -3.0  # per second, per N above HEAD_GROUND_FREE_N
HEAD_GROUND_FREE_N = 1.0


def head_ground_contact(env, sensor_name: str = HEAD_GROUND_SENSOR) -> torch.Tensor:
    """1 where the neck or head touches the terrain, else 0."""
    found = microduck_mdp._sensor_any_contact(env, sensor_name)
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    return found.float()


def make_head_off_ground(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Adds the head-ground sensor (the neck's subtree against the terrain) and its two costs."""
    sensor = ContactSensorCfg(
        name=HEAD_GROUND_SENSOR,
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != HEAD_GROUND_SENSOR) + (sensor,)
    cfg.rewards["head_ground_contact"] = RewardTermCfg(
        func=head_ground_contact,
        weight=HEAD_GROUND_CONTACT_WEIGHT,
        params={"sensor_name": HEAD_GROUND_SENSOR},
    )
    cfg.rewards["head_ground_force"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=HEAD_GROUND_FORCE_WEIGHT,
        params={"sensor_name": HEAD_GROUND_SENSOR, "threshold": HEAD_GROUND_FREE_N},
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Sit-stand: no stalling on a sit command
# ─────────────────────────────────────────────────────────────────────────────

#: With the head kept off the ground, the fine-tune from 3250 (08/11/2026) sat
#: without the head by 3500 but then drifted into refusing: by 3750 two thirds
#: of sits from standing never left standing. Per second of a sit segment,
#: standing on collected 7.8 (1.5 of it upright_while_tall, which pays for
#: being upright and tall whatever the command), sitting 11.2, the transition
#: being the dip in between. Staying above the halfway height once the posture
#: target's 2 s ramp has run (plus 0.5 s) under a sit command now costs
#: SIT_STALL_WEIGHT per second, so refusing nets about 5 against 11.
SIT_STALL_WEIGHT = -5.0
SIT_STALL_GRACE_S = 2.5
SIT_STALL_HEIGHT = 0.0875  # [m] halfway between SIT_Z 0.060 and STAND_Z 0.115


class SitStall(ManagerTermBase):
    """1 while a posture command is older than ``grace_s`` and the root is still on the wrong
    side of ``height``: above it under a sit command (``posture="sit"``), below it under a
    stand command (``posture="stand"``)."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._sit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._age = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._age[ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str = "twist",
        grace_s: float = SIT_STALL_GRACE_S,
        height: float = SIT_STALL_HEIGHT,
        posture: str = "sit",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        sit = env.command_manager.get_term(command_name).vel_command_b[:, 0] > 0.5
        self._age = torch.where(sit == self._sit, self._age + env.step_dt, torch.zeros_like(self._age))
        self._sit = sit
        z = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
        if posture == "sit":
            stalled = sit & (z > height)
        else:
            stalled = ~sit & (z < height)
        return (stalled & (self._age > grace_s)).float()


def make_no_sit_stall(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Adds the sit-stall cost to a sit-stand cfg."""
    cfg.rewards["sit_stall"] = RewardTermCfg(func=SitStall, weight=SIT_STALL_WEIGHT, params={})
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Sit-stand: soft transitions
# ─────────────────────────────────────────────────────────────────────────────

#: The low-noise sit-stand (model_4749, 08/11/2026) stands up from seated in
#: 0.15 s, the trunk peaking at 0.5 m/s against the task's 0.08 m/s rise cap, and
#: sits by crouching slowly to about 100 mm and then dropping the last 40 mm onto
#: its seat in 0.1 to 0.2 s at up to 0.57 m/s (descent cap 0.05 m/s). Both caps
#: cost the excess speed linearly, so a whole pop or drop sums to about weight x
#: distance (0.4 for the rise), and the slewed posture target hardly charges for
#: arriving early. A cost on the SQUARE of the excess makes a pop or a drop
#: expensive and a steady transition nearly free: at -500, a 0.5 m/s pop costs
#: about 10, a rise over about 0.5 s about 1, one at the cap nothing. The
#: stand-stall twin of the sit-stall cost keeps it from refusing to stand instead.
SOFT_RISE_MAX_VZ = 0.08  # [m/s] up, the task's MAX_RISE_SPEED
SOFT_DESCENT_MAX_VZ = 0.05  # [m/s] down, the task's MAX_DESCENT_SPEED
SOFT_TRANSITION_WEIGHT = -500.0  # per second, per (m/s)^2 above the cap


def rise_speed_squared(env, max_vz: float = SOFT_RISE_MAX_VZ, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """(upward root speed above ``max_vz``)^2 [(m/s)^2]; 0 at or below it and going down."""
    vz = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2], nan=0.0)
    return torch.clamp(vz - max_vz, min=0.0) ** 2


def descent_speed_squared(env, max_vz: float = SOFT_DESCENT_MAX_VZ, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """(downward root speed above ``max_vz``)^2 [(m/s)^2]; 0 at or below it and going up."""
    vz = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2], nan=0.0)
    return torch.clamp(-vz - max_vz, min=0.0) ** 2


def make_soft_transitions(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Adds the squared rise- and descent-speed costs and the stand-stall cost to a sit-stand cfg."""
    cfg.rewards["rise_speed_squared"] = RewardTermCfg(
        func=rise_speed_squared, weight=SOFT_TRANSITION_WEIGHT, params={"max_vz": SOFT_RISE_MAX_VZ}
    )
    cfg.rewards["descent_speed_squared"] = RewardTermCfg(
        func=descent_speed_squared, weight=SOFT_TRANSITION_WEIGHT, params={"max_vz": SOFT_DESCENT_MAX_VZ}
    )
    cfg.rewards["stand_stall"] = RewardTermCfg(
        func=SitStall, weight=SIT_STALL_WEIGHT, params={"posture": "stand"}
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Stand-up taught by Pollen's stand expert
# ─────────────────────────────────────────────────────────────────────────────

#: Pollen's deployed stand-up policy (Hub ``pollen-robotics/microduck-policies``,
#: ``alpha_stand.onnx``, run 69u48n8l at 9750), kept out of the repository. On
#: the Dukki model at kp 1 it gets up from face down in 0.56 s and face up in
#: 0.86 s (23/10/2026), where the first Dukki stand-up run, unaided, settled
#: into a crouch propped on its head (trunk 113 mm, tilted 33 deg).
STAND_EXPERT_DEFAULT = "logs/hf/microduck-policies/alpha_stand.onnx"

#: Frames tilted past this are taught by the expert. Below the velstand recipe's
#: 35 deg, because the head prop sits at 33 deg; under 20 deg counts as upright
#: and is left to PPO (body pose control included).
STAND_EXPERT_GATE_DEG = 20.0


#: Soft stand-up (08/11/2026): the distilled stand-up (model_3500) gets up from
#: sitting in 0.14 s. Near upright the expert does not teach, so PPO alone sets the
#: rise there: the sit-stand's squared rise- and descent-speed costs, applied only
#: while the trunk is tilted less than STAND_EXPERT_GATE_DEG. The flips themselves
#: (taught by the expert above the gate) are left alone.
SOFT_STANDUP_TILT_DEG = STAND_EXPERT_GATE_DEG


def _tilt_deg(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.rad2deg(torch.arccos((-g[:, 2] / g.norm(dim=1)).clamp(-1.0, 1.0)))


def vertical_speed_squared_upright(
    env,
    max_up: float = SOFT_RISE_MAX_VZ,
    max_down: float = SOFT_DESCENT_MAX_VZ,
    tilt_deg: float = SOFT_STANDUP_TILT_DEG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Squared vertical speed beyond ``max_up`` rising or ``max_down`` falling, only while the
    trunk is tilted less than ``tilt_deg`` [(m/s)^2]."""
    vz = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2], nan=0.0)
    excess = torch.clamp(vz - max_up, min=0.0) + torch.clamp(-vz - max_down, min=0.0)
    return excess**2 * (_tilt_deg(env, asset_cfg) < tilt_deg).float()


def make_soft_standup(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Adds the near-upright squared vertical-speed cost to a stand-up cfg."""
    cfg.rewards["vertical_speed_squared_upright"] = RewardTermCfg(
        func=vertical_speed_squared_upright, weight=SOFT_TRANSITION_WEIGHT, params={}
    )
    return cfg


def make_dukki_standup_distill_rl_cfg(base):
    """``base`` (the stand-up runner cfg) with PPO plus behaviour cloning from the stand
    expert on tilted frames: no walk anchor. The expert path is ``DUKKI_STAND_EXPERT``."""
    from dataclasses import fields, replace

    from mjlab_microduck.tasks.distill import PpoWithExpertBcCfg, default_bc_cfg

    algo = {f.name: getattr(base.algorithm, f.name) for f in fields(base.algorithm) if f.name != "class_name"}
    bc_cfg = {
        **default_bc_cfg(),
        "wandb_run_path": None,
        "checkpoint_path": os.environ.get("DUKKI_STAND_EXPERT", STAND_EXPERT_DEFAULT),
        "gate_tilt_deg": STAND_EXPERT_GATE_DEG,
        "anchor_coef": 0.0,
        "anchor_wandb_run_path": None,
    }
    return replace(base, algorithm=PpoWithExpertBcCfg(**algo, bc_cfg=bc_cfg), run_name="dukki_stand_distill")
