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

4. Joint friction for the assembled robot: every Dukki task samples each joint's friction
   scale on BAM's bench fit independently per episode, in ``DUKKI_FRICTION_SCALE_RANGE``
   (env ``DUKKI_FRICTION``, "lo,hi"), where the base task samples one scale for all joints in
   0.9 to 1.1. Standing on the robot (21/11/2026) the ankles, hip rolls, a hip yaw and the knees
   broke free at 1.5 to 4.7 times the fit's torque, the hip pitches, neck and head at about 1;
   the policy trained at 0.9 to 1.1 stuck, jerked and over-corrected.

5. robotd's leg target low-pass, off by default (``DUKKI_LEGS_LOWPASS`` 1.0; env
   ``DUKKI_LEGS_LOWPASS``). On the robot robotd filters the legs at 0.7 and the head at 0.5,
   which steadied the walk policy's stand (21/11/2026). Trained with the leg filter, the walk
   policy learned to stand still for every slow command, sideways and backwards (run
   *_dukki-walk-lowpass-*), as Pollen found (microduck_rl 9cc8fc1a, reverted 0a865d93): train
   unfiltered, filter on the robot.

6. Calibration and servo spread: each joint's encoder offset in ±2 degrees, fixed per simulated
   robot for the run like a calibration error (``DUKKI_ENCODER_BIAS_RANGE``; the jig repeats to about 1 to 1.5 degrees, the base task uses
   ±0.86), and each joint's kp scaled per episode in ``DUKKI_KP_SCALE_RANGE`` (0.85 to 1.15; the
   base task's gain randomisation is off). ``DUKKI_CALIBRATION_SPREAD=0`` turns both off (the base
   task's ±0.86 degree offsets stay): the two policies trained with them (runs *_dukki-walk-robust-*
   and *_dukki-walk-lownoise-*) were calmer standing on the robot but fell under pushes the
   unspread 9250 recovered from (21/11/2026).

7. The joint stops: every joint target is clipped as the firmware's safety
   layer clips it on the robot (dukki_constants.target_clip_rad: 3 degrees
   inside the measured travel, head roll at its ±15 degree limit), so the
   policy meets in training the clamp it meets on the robot and a servo is
   never driven into a stop. The head pose command and its curriculum are
   capped at ``HEAD_ROLL_COMMAND_SHARE`` of what each head joint can reach from
   HOME inside that clip: the neck can pitch only about 12 degrees above HOME.

8. Slow walking under high friction, all off by default. 9250 fell on the robot at the end of a
   0.15 m/s walk, and in sim does so at 3 to 5 times the bench friction; fine-tuned straight
   onto 1 to 5 times, it stopped walking at every command (22/11/2026): the stock tracking term
   pays a slow walk no more than standing still. ``DUKKI_FRICTION_RAMP="lo,hi,iterations"``
   moves the friction range from ``DUKKI_FRICTION`` to lo..hi over that many iterations of the
   run; ``DUKKI_FINE_LIN_VEL="weight,std"`` adds a tight tracking term on the velocity error
   averaged over a stride (``track_lin_vel_smoothed``), beside the stock one;
   ``DUKKI_SLOW_WALK=share[,lo,hi]`` makes that share of command resamples slow straight
   walking, |vx| lo to hi (default 0.1 to 0.25 m/s). With those three alone it still stopped
   walking (run *_dukki-walk-slowwalk-*; its ramp did not ramp, a bug fixed since, so it trained at
   1 to 5 times friction throughout): ``DUKKI_ACTION_RATE=weight`` holds the
   action-rate penalty, the largest cost of stepping, at a lighter fixed weight (the recipe ends
   at -1.0).

9. Servo damping, off by default. ``DUKKI_KD_RANGE="lo,hi"`` draws each joint's kd in lo..hi
   (N.m.s/rad) per episode, in place of the fixed ``DUKKI_KD``. On the robot kd 0.07 stood still
   where the trained 0.05 left it restless, and caught pushes worse (22/11/2026).

10. Turning in place, off by default. ``DUKKI_FINE_ANG_VEL="weight,std"`` adds a tight term on the
    yaw-rate error averaged over a stride (``track_ang_vel_smoothed``) beside the stock one, and
    ``DUKKI_TURN_IN_PLACE="share,min"`` sets the turn-in-place command share and its smallest
    rate (recipe: 15 % from 0.4 rad/s). 9250 did not turn at all at 0.5 rad/s on the robot, and
    the gear-play fine-tune (run *_dukki-walk-backlash-*) stopped turning in place altogether.

11. A softer sit-stand, off by default. ``DUKKI_SOFT_TRUNK="weight,cap"`` adds a cost on the square
    of the trunk's rotation rate above ``cap`` rad/s (``trunk_rate_squared``) to the sit-stand
    tasks. The gear-play sit-stand 9248 rose every time on the robot, but its trunk peaked at 5 to
    8 rad/s doing it (22/11/2026); the soft-transition costs only see vertical speed. Charged
    everywhere at -10 (run *_dukki-sitstand-soft10-*), it stopped sitting rather than rise more
    gently: ``DUKKI_SOFT_TRUNK="weight,cap,rise"`` charges only the rise. The stand-up tasks
    take the same switch (``make_soft_standup``), there to keep the roll-over from face up
    inside the trunk gyro's +-500 dps.

12. Getting up after a fall with gear play: ``Mjlab-StandUp-Flat-Backlash-Dukki-Distill`` is the
    distilled, soft stand-up on the groundcontact model's backlash twin, for fine-tuning the
    stand-up with the walk's and sit-stand's recipe (friction ramp, kd range).

Gain sweep. ``DUKKI_KP`` [N.m/rad] and ``DUKKI_KD`` [N.m.s/rad] set the gains
of every joint (default dukki_constants.J288_KP / J288_KD). They are read when
the tasks are registered, so set them on the command line:

    DUKKI_KP=3.0 DUKKI_KD=0.08 uv run train Mjlab-Velocity-Flat-Dukki ...

``--hf-jobs`` forwards every ``DUKKI_*`` variable to the job.
"""

from __future__ import annotations

import math
import os
import re
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
    target_clip_rad,
)
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backlash import make_backlash_variant

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#: Torque [N.m] a joint may hold against its hard stop before the over-limit
#: penalty starts (the target may go this / kp past the stop).
OVER_LIMIT_TORQUE = 0.2
OVER_LIMIT_WEIGHT = -1.0

JOINT_VEL_SOURCES = ("position_difference", "reported_speed")

#: Each joint's friction scale on BAM's bench fit (m6), sampled per joint per episode. See (4) in
#: the module docstring. ``DUKKI_FRICTION="lo,hi"`` overrides it when the tasks are registered.
DUKKI_FRICTION_SCALE_RANGE = (1.0, 3.0)


def dukki_friction_range() -> tuple[float, float]:
    """``DUKKI_FRICTION`` ("lo,hi") or ``DUKKI_FRICTION_SCALE_RANGE``."""
    text = os.environ.get("DUKKI_FRICTION")
    if not text:
        return DUKKI_FRICTION_SCALE_RANGE
    lo, hi = (float(v) for v in text.split(","))
    if not 0.0 < lo <= hi:
        raise ValueError(f"DUKKI_FRICTION must be 0 < lo <= hi, got {text!r}")
    return lo, hi


def randomize_bam_friction_per_joint(
    env,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Like ``mdp.randomize_bam_friction``, but each joint gets its own scale.

    The friction-DR actuators hold ``friction_scale`` as (num_envs, 1), one scale for every joint;
    BAM multiplies its (num_envs, num_joints) friction budget by it, so a (num_envs, num_joints)
    scale works the same way, a joint at a time. Widened to that shape the first time.
    """
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRMixin

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)
    lo, hi = scale_range
    for actuator in env.scene[asset_cfg.name].actuators:
        if not isinstance(actuator, FrictionDRMixin):
            continue
        joints = len(actuator._dof_ids)
        if actuator.friction_scale.shape[1] != joints:
            actuator.friction_scale = actuator.friction_scale.expand(-1, joints).clone()
        actuator.friction_scale[env_ids] = torch.rand(len(env_ids), joints, device=env.device) * (hi - lo) + lo


def use_dukki_friction(cfg: ManagerBasedRlEnvCfg, scale_range: tuple[float, float] | None = None) -> ManagerBasedRlEnvCfg:
    """Swap every BAM friction randomisation in ``cfg`` for the per-joint Dukki one."""
    if scale_range is None:
        scale_range = dukki_friction_range()
    for term in cfg.events.values():
        if term is not None and term.func is microduck_mdp.randomize_bam_friction:
            term.func = randomize_bam_friction_per_joint
            term.params = {**term.params, "scale_range": tuple(scale_range)}
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Slow walking under high friction (8 in the module docstring)
# ─────────────────────────────────────────────────────────────────────────────

#: PPO steps per iteration (the runners' num_steps_per_env): curricula count env steps.
STEPS_PER_ITERATION = 24

#: A step-counter jump between friction-ramp calls bigger than this is a checkpoint load (resets
#: come at least once an episode, 1000 steps), not training: the ramp restarts there.
RAMP_RESTART_JUMP = 10 * 1000

#: Time constant of the smoothed velocity error the fine tracking term scores [s]. Long enough
#: to average over a stride (the body speed swings within each step), short enough to follow a
#: command change.
FINE_LIN_VEL_TAU_S = 1.0


def dukki_friction_ramp() -> tuple[float, float, int] | None:
    """``DUKKI_FRICTION_RAMP`` ("lo,hi,iterations"): the friction range to reach, and over how
    many iterations from the start of this run. Unset: no ramp."""
    text = os.environ.get("DUKKI_FRICTION_RAMP")
    if not text:
        return None
    lo, hi, iters = (v.strip() for v in text.split(","))
    lo, hi, iters = float(lo), float(hi), int(iters)
    if not 0.0 < lo <= hi or iters <= 0:
        raise ValueError(f"DUKKI_FRICTION_RAMP must be 0 < lo <= hi, iterations > 0; got {text!r}")
    return lo, hi, iters


def friction_ramp_curriculum(
    env,
    env_ids: torch.Tensor,
    event_name: str,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    ramp_steps: int,
) -> torch.Tensor:
    """Move ``event_name``'s ``scale_range`` linearly from ``start_range`` to ``end_range`` over
    ``ramp_steps`` env steps, counted from where this run's training starts: a plain resume
    continues the step counter, so absolute stages would already be over. The env resets (and
    calls this) before the runner loads the checkpoint and restores the counter, so a jump in the
    counter between calls, which training never makes, restarts the ramp from there. Returns the
    upper bound in use."""
    del env_ids
    now = env.common_step_counter
    start = getattr(env, "_dukki_friction_ramp_start", None)
    last = getattr(env, "_dukki_friction_ramp_last", None)
    if start is None or last is None or now - last > RAMP_RESTART_JUMP or now < last:
        start = now
        env._dukki_friction_ramp_start = start
    env._dukki_friction_ramp_last = now
    frac = min(1.0, max(0.0, (now - start) / max(ramp_steps, 1)))
    lo = start_range[0] + frac * (end_range[0] - start_range[0])
    hi = start_range[1] + frac * (end_range[1] - start_range[1])
    env.event_manager.get_term_cfg(event_name).params["scale_range"] = (lo, hi)
    return torch.tensor(hi)


def use_friction_ramp(cfg: ManagerBasedRlEnvCfg, ramp: tuple[float, float, int] | None = None) -> ManagerBasedRlEnvCfg:
    """Ramp the per-joint friction range from its ``DUKKI_FRICTION`` value to ``ramp`` (default
    ``DUKKI_FRICTION_RAMP``). Dropped straight onto 1 to 5 times the bench friction, the walk
    fine-tune stopped walking at every command (run *_dukki-walk-friction5-*, 22/11/2026)."""
    if ramp is None:
        ramp = dukki_friction_ramp()
    if ramp is None:
        return cfg
    from mjlab.managers import CurriculumTermCfg

    for name, term in cfg.events.items():
        if term is not None and term.func is randomize_bam_friction_per_joint:
            cfg.curriculum[f"{name}_ramp"] = CurriculumTermCfg(
                func=friction_ramp_curriculum,
                params={
                    "event_name": name,
                    "start_range": tuple(term.params["scale_range"]),
                    "end_range": (ramp[0], ramp[1]),
                    "ramp_steps": ramp[2] * STEPS_PER_ITERATION,
                },
            )
    return cfg


def track_lin_vel_smoothed(
    env,
    std: float,
    command_name: str = "twist",
    tau_s: float = FINE_LIN_VEL_TAU_S,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """exp(-|EMA(v_cmd - v)|² / std²) on the body's xy velocity, the error averaged over ``tau_s``.

    Added beside the stock ``track_linear_velocity`` (std 0.32), never instead of it. The stock
    term scores the instantaneous error, and a walking body's speed swings within each stride
    by about as much as a slow command: at 0.15 m/s, standing still keeps 80 % of it and 9250's
    walk (lin err 0.15) about the same, so it pays nothing for walking slowly. Averaging over a
    stride lets the swing cancel, so a tight std prices only how far the robot's mean speed is
    from the command (Pollen added a fine yaw term beside the stock one for the same dead zone,
    microduck_rl 4048a3ea; replacing it lost turning). Freshly reset envs start from zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    err = cmd - asset.data.root_link_lin_vel_b[:, :2]
    ema = getattr(env, "_dukki_lin_vel_err_ema", None)
    if ema is None or ema.shape != err.shape:
        ema = torch.zeros_like(err)
    ema = ema.clone()
    ema[env.episode_length_buf <= 1] = 0.0
    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    ema = (1.0 - alpha) * ema + alpha * err
    env._dukki_lin_vel_err_ema = ema
    return torch.exp(-torch.sum(torch.square(ema), dim=1) / std**2)


def track_ang_vel_smoothed(
    env,
    std: float,
    command_name: str = "twist",
    tau_s: float = FINE_LIN_VEL_TAU_S,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """exp(-EMA(wz_cmd - wz)^2 / std^2): the yaw-rate twin of ``track_lin_vel_smoothed``, beside the
    stock ``track_angular_velocity`` (std 0.71), whose loose tolerance pays a robot that stands
    still when asked to turn slowly almost as well as one that turns. Pollen added a fine yaw term
    (std 0.35) for the same dead zone (their microduck_rl 4048a3ea). Averaged over ``tau_s`` so the
    yaw wobble of each stride cancels; at zero command it also prices veering while walking."""
    asset: Entity = env.scene[asset_cfg.name]
    err = env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    ema = getattr(env, "_dukki_ang_vel_err_ema", None)
    if ema is None or ema.shape != err.shape:
        ema = torch.zeros_like(err)
    ema = ema.clone()
    ema[env.episode_length_buf <= 1] = 0.0
    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    ema = (1.0 - alpha) * ema + alpha * err
    env._dukki_ang_vel_err_ema = ema
    return torch.exp(-torch.square(ema) / std**2)


def _weight_std(name: str) -> tuple[float, float] | None:
    text = os.environ.get(name)
    if not text:
        return None
    weight, std = (float(v) for v in text.split(","))
    if weight <= 0.0 or std <= 0.0:
        raise ValueError(f"{name} must be weight,std > 0; got {text!r}")
    return weight, std


def dukki_fine_ang_vel() -> tuple[float, float] | None:
    """``DUKKI_FINE_ANG_VEL`` ("weight,std"): the smoothed fine yaw-rate term. Unset: off."""
    return _weight_std("DUKKI_FINE_ANG_VEL")


def dukki_turn_in_place() -> tuple[float, float] | None:
    """``DUKKI_TURN_IN_PLACE`` ("share,min"): the turn-in-place command share and its smallest
    |yaw rate|, rad/s, in place of the recipe's 15 % from 0.4 rad/s. Unset: the recipe's."""
    text = os.environ.get("DUKKI_TURN_IN_PLACE")
    if not text:
        return None
    share, low = (float(v) for v in text.split(","))
    if not 0.0 <= share < 1.0 or low < 0.0:
        raise ValueError(f"DUKKI_TURN_IN_PLACE must be share in [0, 1), min >= 0; got {text!r}")
    return share, low


def dukki_fine_lin_vel() -> tuple[float, float] | None:
    """``DUKKI_FINE_LIN_VEL`` ("weight,std"): the smoothed fine tracking term. Unset: off."""
    text = os.environ.get("DUKKI_FINE_LIN_VEL")
    if not text:
        return None
    weight, std = (float(v) for v in text.split(","))
    if weight <= 0.0 or std <= 0.0:
        raise ValueError(f"DUKKI_FINE_LIN_VEL must be weight,std > 0; got {text!r}")
    return weight, std


def dukki_slow_walk() -> tuple[float, tuple[float, float]]:
    """``DUKKI_SLOW_WALK`` ("share" or "share,lo,hi"): share of command resamples that are slow
    straight walking, and their |vx| range (default 0.1 to 0.25 m/s). Unset: share 0."""
    parts = [float(v) for v in (os.environ.get("DUKKI_SLOW_WALK", "") or "0").split(",")]
    share = parts[0]
    speed = (parts[1], parts[2]) if len(parts) == 3 else (0.1, 0.25)
    if len(parts) not in (1, 3) or not 0.0 <= share < 1.0 or not 0.0 < speed[0] <= speed[1]:
        raise ValueError(f"DUKKI_SLOW_WALK must be share[,lo,hi] with share in [0, 1), 0 < lo <= hi; got {parts}")
    return share, speed


def dukki_action_rate() -> float | None:
    """``DUKKI_ACTION_RATE``: a fixed ``action_rate_l2`` weight (negative) in place of the
    velocity recipe's ramp, whose final stage is -1.0. Unset: the recipe's."""
    text = os.environ.get("DUKKI_ACTION_RATE")
    if not text:
        return None
    weight = float(text)
    if weight > 0.0:
        raise ValueError(f"DUKKI_ACTION_RATE is a penalty weight, <= 0; got {weight}")
    return weight


def use_action_rate_weight(cfg: ManagerBasedRlEnvCfg, weight: float | None = None) -> ManagerBasedRlEnvCfg:
    """Hold ``action_rate_l2`` at ``weight`` (default ``DUKKI_ACTION_RATE``). It is the largest
    cost of stepping, and grows with joint friction: the slow-walk fine-tune (run
    *_dukki-walk-slowwalk-*, 22/11/2026) stopped walking as friction rose, the air-time reward
    halving while the action-rate penalty (-0.8 per episode) barely moved. Pollen lightened it
    "while the steps are discovered" (branch precision-mode)."""
    weight = dukki_action_rate() if weight is None else weight
    if weight is None or "action_rate_l2" not in cfg.rewards:
        return cfg
    cfg.rewards["action_rate_l2"].weight = weight
    ramp = cfg.curriculum.get("action_rate_weight")
    if ramp is not None and ramp.params.get("reward_name") == "action_rate_l2":
        ramp.params = {**ramp.params, "weight_stages": [{"step": 0, "weight": weight}]}
    return cfg


def use_slow_walk_training(
    cfg: ManagerBasedRlEnvCfg,
    fine: tuple[float, float] | None = None,
    slow_share: float | None = None,
    slow_speed: tuple[float, float] | None = None,
    fine_yaw: tuple[float, float] | None = None,
    turn: tuple[float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
    """The fine smoothed tracking terms (``DUKKI_FINE_LIN_VEL``, ``DUKKI_FINE_ANG_VEL``) and the
    slow-walk and turn-in-place command buckets (``DUKKI_SLOW_WALK``, ``DUKKI_TURN_IN_PLACE``), on
    tasks that track a twist command. All off (the recipe's) by default."""
    fine = dukki_fine_lin_vel() if fine is None else fine
    fine_yaw = dukki_fine_ang_vel() if fine_yaw is None else fine_yaw
    turn = dukki_turn_in_place() if turn is None else turn
    env_share, env_speed = dukki_slow_walk()
    slow_share = env_share if slow_share is None else slow_share
    slow_speed = env_speed if slow_speed is None else slow_speed
    if "track_linear_velocity" not in cfg.rewards:
        return cfg
    if fine is not None:
        cfg.rewards["track_lin_vel_fine"] = RewardTermCfg(
            func=track_lin_vel_smoothed,
            weight=fine[0],
            params={"std": fine[1], "command_name": "twist"},
        )
    if fine_yaw is not None:
        cfg.rewards["track_ang_vel_fine"] = RewardTermCfg(
            func=track_ang_vel_smoothed,
            weight=fine_yaw[0],
            params={"std": fine_yaw[1], "command_name": "twist"},
        )
    if slow_share > 0.0 or turn is not None:
        twist = cfg.commands.get("twist")
        if not isinstance(twist, microduck_mdp.VelocityCommandCommandOnlyCfg):
            raise ValueError("the command buckets need the twist command to be VelocityCommandCommandOnlyCfg")
        if turn is not None:
            twist.rel_turn_in_place_envs = turn[0]
            top = max(abs(v) for v in twist.ranges.ang_vel_z)
            twist.turn_in_place_min_frac = min(1.0, turn[1] / top)
        if twist.rel_turn_in_place_envs + slow_share >= 1.0:
            raise ValueError("turn-in-place and slow-walk shares must sum to under 1")
        if slow_share > 0.0:
            twist.rel_slow_walk_envs = slow_share
            twist.slow_walk_speed = tuple(slow_speed)
    return cfg


#: robotd's leg target low-pass in training. Off: see (5) in the module docstring.
DUKKI_LEGS_LOWPASS = 1.0
#: Each joint's encoder offset, rad, drawn once per simulated robot (the base task's startup event). See (6) in the module docstring.
DUKKI_ENCODER_BIAS_RANGE = (-0.035, 0.035)
#: Each joint's kp scale, per joint per episode. See (6) in the module docstring.
DUKKI_KP_SCALE_RANGE = (0.85, 1.15)


def randomize_j288_kp_per_joint(
    env,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Scale each joint's J288 kp by its own factor in ``scale_range`` for the episode.

    Runs after the scene reset has put every kp back to its default (J288DRActuator.reset), so
    the scale is applied to the default and does not accumulate.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)
    lo, hi = scale_range
    for actuator in env.scene[asset_cfg.name].actuators:
        if isinstance(actuator, J288DRActuator) and actuator.kp is not None:
            scale = torch.rand(len(env_ids), actuator.kp.shape[1], device=env.device) * (hi - lo) + lo
            actuator.kp[env_ids] = actuator.default_kp[env_ids] * scale


def dukki_kd_range() -> tuple[float, float] | None:
    """``DUKKI_KD_RANGE`` ("lo,hi", N.m.s/rad): each joint's kd drawn in it per episode. Unset: off,
    every joint at ``DUKKI_KD``."""
    text = os.environ.get("DUKKI_KD_RANGE")
    if not text:
        return None
    lo, hi = (float(v) for v in text.split(","))
    if not 0.0 <= lo <= hi:
        raise ValueError(f"DUKKI_KD_RANGE must be 0 <= lo <= hi, got {text!r}")
    return lo, hi


def randomize_j288_kd_per_joint(
    env,
    env_ids: torch.Tensor | None,
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Draw each joint's J288 kd in ``kd_range`` for the episode (absolute, not a scale on the default).

    Runs after the scene reset has restored every kd to its default (J288DRActuator.reset)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)
    lo, hi = kd_range
    for actuator in env.scene[asset_cfg.name].actuators:
        if isinstance(actuator, J288DRActuator) and actuator.kd is not None:
            actuator.kd[env_ids] = torch.rand(len(env_ids), actuator.kd.shape[1], device=env.device) * (hi - lo) + lo


def use_kd_range(cfg: ManagerBasedRlEnvCfg, kd_range: tuple[float, float] | None = None) -> ManagerBasedRlEnvCfg:
    """Randomise each joint's kd per episode in ``kd_range`` (default ``DUKKI_KD_RANGE``). The robot
    stood still at kd 0.07 where the trained 0.05 left it restless, but caught pushes worse
    (dukki docs/policy-testing.md section 5.7): a range lets one policy run at either."""
    if kd_range is None:
        kd_range = dukki_kd_range()
    if kd_range is None:
        return cfg
    cfg.events["randomize_j288_kd"] = EventTermCfg(
        func=randomize_j288_kd_per_joint,
        mode="reset",
        params={"kd_range": tuple(kd_range)},
    )
    return cfg


def dukki_calibration_spread() -> bool:
    """``DUKKI_CALIBRATION_SPREAD`` from the environment: on unless "0", "false" or "off"."""
    return os.environ.get("DUKKI_CALIBRATION_SPREAD", "1").strip().lower() not in ("0", "false", "off")


def use_dukki_calibration_spread(cfg: ManagerBasedRlEnvCfg, enabled: bool | None = None) -> ManagerBasedRlEnvCfg:
    """Widen the encoder offsets to the jig's repeatability and randomise each joint's kp, unless
    ``enabled`` (default ``DUKKI_CALIBRATION_SPREAD``) is off."""
    if enabled is None:
        enabled = dukki_calibration_spread()
    if not enabled:
        return cfg
    if "encoder_bias" in cfg.events and cfg.events["encoder_bias"] is not None:
        term = cfg.events["encoder_bias"]
        term.params = {**term.params, "bias_range": DUKKI_ENCODER_BIAS_RANGE}
    cfg.events["randomize_j288_kp"] = EventTermCfg(
        func=randomize_j288_kp_per_joint,
        mode="reset",
        params={"scale_range": DUKKI_KP_SCALE_RANGE},
    )
    return cfg
#: The joints it filters: everything in the legs, not the head.
LEG_JOINTS = r".*(hip|knee|ankle).*"


def dukki_legs_lowpass() -> float:
    """``DUKKI_LEGS_LOWPASS`` from the environment, or the default."""
    alpha = float(os.environ.get("DUKKI_LEGS_LOWPASS", DUKKI_LEGS_LOWPASS))
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"DUKKI_LEGS_LOWPASS must be in (0, 1], got {alpha}")
    return alpha


def use_legs_lowpass(cfg: ManagerBasedRlEnvCfg, alpha: float | None = None) -> ManagerBasedRlEnvCfg:
    """Make the joint_pos action low-pass the leg targets as robotd does; 1.0 leaves it plain."""
    from dataclasses import fields

    from mjlab_microduck.tasks.lowpass_action import LowPassJointPositionActionCfg

    if alpha is None:
        alpha = dukki_legs_lowpass()
    action = cfg.actions.get("joint_pos")
    if action is None or alpha >= 1.0:
        return cfg
    kept = {f.name: getattr(action, f.name) for f in fields(action) if f.init and f.name != "lowpass"}
    cfg.actions["joint_pos"] = LowPassJointPositionActionCfg(**kept, lowpass={LEG_JOINTS: alpha})
    return cfg


#: Head pose commands stay within this share of what each head joint can reach
#: from HOME inside its target clip: the margin the base curriculum keeps from
#: the joint limits.
HEAD_ROLL_COMMAND_SHARE = 0.9
#: The head pose command's joints, in its order. Each dim is a delta from HOME.
HEAD_POSE_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
#: head_roll's place in the head pose command.
HEAD_ROLL_COMMAND_INDEX = HEAD_POSE_JOINTS.index("head_roll")


def home_rad(joint: str) -> float:
    """``joint``'s HOME angle, from HOME_FRAME's name patterns."""
    for pattern, value in HOME_FRAME.joint_pos.items():
        if re.fullmatch(pattern, joint):
            return float(value)
    raise KeyError(f"no HOME angle for {joint}")


def head_pose_caps(clip: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    """(lo, hi) bounds on each head pose command dim: ``HEAD_ROLL_COMMAND_SHARE``
    of the delta from HOME that ``clip`` allows; unbounded for an unclipped joint."""
    caps = []
    for joint in HEAD_POSE_JOINTS:
        if joint not in clip:
            caps.append((-float("inf"), float("inf")))
            continue
        lo, hi = clip[joint]
        home = home_rad(joint)
        caps.append((HEAD_ROLL_COMMAND_SHARE * (lo - home), HEAD_ROLL_COMMAND_SHARE * (hi - home)))
    return caps


def _cap_ranges(ranges, caps):
    ranges = list(ranges)
    for i, (cap_lo, cap_hi) in enumerate(caps[: len(ranges)]):
        lo, hi = ranges[i]
        ranges[i] = (max(lo, cap_lo), min(hi, cap_hi))
    return tuple(ranges)


def apply_joint_stops(
    cfg: ManagerBasedRlEnvCfg, clip: dict[str, tuple[float, float]] | None = None
) -> ManagerBasedRlEnvCfg:
    """Clip joint targets as the robot's safety layer does, and cap head pose commands to match.

    ``clip`` defaults to dukki_constants.target_clip_rad(). The joint_pos
    action's targets (after scale and offset) are clipped to it. The head pose
    command and its range curriculum are capped at ``HEAD_ROLL_COMMAND_SHARE``
    of what each head joint can reach from HOME inside the clip.
    """
    if clip is None:
        clip = target_clip_rad()
    if not clip:
        return cfg
    action = cfg.actions.get("joint_pos")
    if action is not None:
        action.clip = {**(action.clip or {}), **clip}
    caps = head_pose_caps(clip)
    command = cfg.commands.get("head_pose")
    if command is not None:
        command.ranges = _cap_ranges(command.ranges, caps)
    curriculum = cfg.curriculum.get("head_pose_range")
    if curriculum is not None:
        stages = deepcopy(curriculum.params["range_stages"])
        for stage in stages:
            stage["ranges"] = _cap_ranges(stage["ranges"], caps)
        curriculum.params = {**curriculum.params, "range_stages": stages}
    return cfg


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
    cfg = use_kd_range(use_friction_ramp(use_dukki_friction(cfg)))
    cfg = use_action_rate_weight(use_slow_walk_training(cfg))
    return use_legs_lowpass(apply_joint_stops(use_dukki_calibration_spread(cfg)))


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


#: With ``rising_only``, the trunk-rate cost counts only under a stand command with the root below
#: this height [m], 10 mm short of the standing target: the rise, not sitting or standing pushed.
SOFT_TRUNK_RISE_BELOW = 0.105


def trunk_rate_squared(
    env,
    cap: float,
    rising_only: bool = False,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """(trunk rotation rate above ``cap``)^2 [(rad/s)^2]: the body-frame angular velocity's
    magnitude, what the trunk IMU's gyro reads; 0 at or below ``cap``. With ``rising_only``, 0
    except under a stand command below ``SOFT_TRUNK_RISE_BELOW``."""
    data = env.scene[asset_cfg.name].data
    w = torch.nan_to_num(data.root_link_ang_vel_b.norm(dim=1), nan=0.0)
    cost = torch.clamp(w - cap, min=0.0) ** 2
    if rising_only:
        stand = env.command_manager.get_term(command_name).vel_command_b[:, 0] <= 0.5
        cost = cost * (stand & (data.root_link_pos_w[:, 2] < SOFT_TRUNK_RISE_BELOW)).float()
    return cost


def dukki_soft_trunk() -> tuple[float, float, bool] | None:
    """``DUKKI_SOFT_TRUNK`` ("weight,cap[,rise]"): the trunk-rate cost's weight (<= 0, per second
    per (rad/s)^2), its free rate [rad/s], and with a third field "rise" only while rising. Unset: off."""
    text = os.environ.get("DUKKI_SOFT_TRUNK")
    if not text:
        return None
    parts = text.split(",")
    if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2].strip() != "rise"):
        raise ValueError(f"DUKKI_SOFT_TRUNK must be weight,cap[,rise]; got {text!r}")
    weight, cap = float(parts[0]), float(parts[1])
    if weight > 0.0 or cap < 0.0:
        raise ValueError(f"DUKKI_SOFT_TRUNK must be weight <= 0, cap >= 0; got {text!r}")
    return weight, cap, len(parts) == 3


def make_soft_transitions(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Adds the squared rise- and descent-speed costs and the stand-stall cost to a sit-stand cfg,
    and with ``DUKKI_SOFT_TRUNK`` the squared trunk-rate cost (module docstring, 11)."""
    cfg.rewards["rise_speed_squared"] = RewardTermCfg(
        func=rise_speed_squared, weight=SOFT_TRANSITION_WEIGHT, params={"max_vz": SOFT_RISE_MAX_VZ}
    )
    cfg.rewards["descent_speed_squared"] = RewardTermCfg(
        func=descent_speed_squared, weight=SOFT_TRANSITION_WEIGHT, params={"max_vz": SOFT_DESCENT_MAX_VZ}
    )
    cfg.rewards["stand_stall"] = RewardTermCfg(
        func=SitStall, weight=SIT_STALL_WEIGHT, params={"posture": "stand"}
    )
    soft = dukki_soft_trunk()
    if soft is not None:
        cfg.rewards["trunk_rate_squared"] = RewardTermCfg(
            func=trunk_rate_squared, weight=soft[0], params={"cap": soft[1], "rising_only": soft[2]}
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
#: At -500 the rise from sitting fell from 0.55 to about 0.35 m/s and the time to
#: upright from 0.14 to 0.26 s, recovery unchanged (soft stand-up run, model_4750,
#: the deployed one). A follow-on at -2000 from 4750 plateaued at 0.33 m/s with more
#: yaw spin, so -500 stays. The remaining peak is the lift-off from the seat; the
#: rise stays under 20 degrees of tilt, so it is PPO's, not the expert's.
SOFT_STANDUP_TILT_DEG = STAND_EXPERT_GATE_DEG
SOFT_STANDUP_WEIGHT = -500.0  # per second, per (m/s)^2 above the cap, near upright


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


#: The trunk IMU's gyro range on Dukki (LSM6DSV16X at +-500 dps, src/bridge imu.rs) [rad/s].
#: Past it the gyro clips and the chip's fused gravity goes wrong: sit-stand 7749 rising
#: unfiltered at about 10 rad/s read the robot as upside down (22/11/2026). The stand-up
#: checkpoints roll over from face up at 10 to 13 rad/s in simulation.
GYRO_RANGE_RAD_S = 500.0 * math.pi / 180.0


def make_soft_standup(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Adds the near-upright squared vertical-speed cost to a stand-up cfg, and with
    ``DUKKI_SOFT_TRUNK`` ("weight,cap[,rise]") the squared trunk-rate cost of the sit-stand
    (module docstring, 11). With "rise" it counts only below SOFT_TRUNK_RISE_BELOW, which in
    the stand-up is the whole way up from the floor and not standing pushed. Meant to keep the
    roll-over from face up inside the gyro's range (GYRO_RANGE_RAD_S)."""
    cfg.rewards["vertical_speed_squared_upright"] = RewardTermCfg(
        func=vertical_speed_squared_upright, weight=SOFT_STANDUP_WEIGHT, params={}
    )
    soft = dukki_soft_trunk()
    if soft is not None:
        cfg.rewards["trunk_rate_squared"] = RewardTermCfg(
            func=trunk_rate_squared, weight=soft[0], params={"cap": soft[1], "rising_only": soft[2]}
        )
    mix = dukki_standup_mix()
    if mix is not None:
        cfg.events["set_ground_state"].params.update(mix)
        curriculum = cfg.curriculum.get("ground_state_mix")
        if curriculum is not None:
            stages = deepcopy(curriculum.params["param_stages"])
            stages[-1]["params"] = {**stages[-1]["params"], **mix}
            curriculum.params = {**curriculum.params, "param_stages": stages}
    return cfg


STANDUP_STARTS = ("standing", "sitting", "face_down", "face_up")


def dukki_standup_mix() -> dict[str, float] | None:
    """``DUKKI_STANDUP_MIX`` ("standing,sitting,face_down,face_up", summing to 1): the stand-up's
    start mix once its curriculum has finished (its last stage is 0.15, 0.20, 0.30, 0.35). Unset: as
    the task has it. On the corrected ground contact (dukki_constants.add_body_contact) face up is
    the hard start: 23/11/2026, 0.10,0.15,0.25,0.50."""
    text = os.environ.get("DUKKI_STANDUP_MIX")
    if not text:
        return None
    values = [float(v) for v in text.split(",")]
    if len(values) != 4 or min(values) < 0.0 or abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"DUKKI_STANDUP_MIX must be four shares summing to 1; got {text!r}")
    return {f"{name}_prob": v for name, v in zip(STANDUP_STARTS, values)}


def dukki_stand_expert_coef() -> float:
    """``DUKKI_STAND_EXPERT_COEF``: the weight of the behaviour cloning from the stand expert
    (default 1.0; 0 turns it off). Pollen's alpha_stand gets up from face up only through thighs
    that pass through the floor: on the corrected ground contact it got up 10.5% of the time
    with gear play and 1-4x friction (face down 95%, as before), 23/11/2026."""
    coef = float(os.environ.get("DUKKI_STAND_EXPERT_COEF", "1.0"))
    if coef < 0.0:
        raise ValueError(f"DUKKI_STAND_EXPERT_COEF must be >= 0; got {coef}")
    return coef


def make_dukki_standup_distill_rl_cfg(base):
    """``base`` (the stand-up runner cfg) with PPO plus behaviour cloning from the stand
    expert on tilted frames: no walk anchor. The expert path is ``DUKKI_STAND_EXPERT``, its
    weight ``DUKKI_STAND_EXPERT_COEF``."""
    from dataclasses import fields, replace

    from mjlab_microduck.tasks.distill import PpoWithExpertBcCfg, default_bc_cfg

    algo = {f.name: getattr(base.algorithm, f.name) for f in fields(base.algorithm) if f.name != "class_name"}
    bc_cfg = {
        **default_bc_cfg(),
        "wandb_run_path": None,
        "checkpoint_path": os.environ.get("DUKKI_STAND_EXPERT", STAND_EXPERT_DEFAULT),
        "coef": dukki_stand_expert_coef(),
        "gate_tilt_deg": STAND_EXPERT_GATE_DEG,
        "anchor_coef": 0.0,
        "anchor_wandb_run_path": None,
    }
    return replace(base, algorithm=PpoWithExpertBcCfg(**algo, bc_cfg=bc_cfg), run_name="dukki_stand_distill")
