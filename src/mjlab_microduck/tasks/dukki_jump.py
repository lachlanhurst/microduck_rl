"""Dukki jump: stand balanced, jump once when asked, land back in a balanced stand.

Episode (EPISODE_S): the robot spawns standing. For WAIT_S (random) the jump command is off and
it should stand still; then the command is on for WINDOW_S and it should jump once; then it
stands again. The command rides in the twist obs slot (vx = 1 while the jump is asked for, 0
otherwise), so the 61D actor obs layout is the family's and a deployed policy jumps on request.

Feasibility (scripts/dukki_jump_probe.py, 27/10/2026, J288 BAM, open loop): a crouch of 0.6 to
1.2 rad of knee bend (hip and ankle pitch half that, same signs as the knee) followed by a short
hard extension gets both feet up to 45 mm off the ground for 0.1 to 0.25 s, at the 1.0 N.m motor
clamp. Only 7 of 80 open-loop jumps landed upright: the leaving is easy, the landing is the task.

State (JumpCommand, updated every step after physics):
  flight     both feet off the ground, the trunk near standing height and tilted under
             FLIGHT_MAX_TILT_DEG (a robot toppling over lifts its feet too; that is not flight)
  first      the first flight that starts inside the window (or GRACE_S after it)
  launch_ok  latched at that take-off: the trunk stayed under TAKEOFF_MAX_TILT_DEG from the request
             through take-off. A jump launched from a collapse does not pay (the backflip branch's
             launch-attitude lesson). Judged over the whole approach, not at the instant the feet
             leave: a squat-and-push pitches the trunk to about 20 deg by then (the probe's jump,
             2-4 deg in the squat), and a 15 deg check there failed every one of them
  landed     the first flight has ended

Teacher (make_dukki_jump_rl_cfg): the first run, unaided from scratch, found that collapsing into a
crouch at 60 mm in the first 0.4 s avoids the fall penalty, and the narrow stand product gave a
collapsed robot no gradient back: no stand, so never a jump. The Dukki stand-up policy (distilled
from Pollen's stand expert, iteration 3500, which gets up from anything and stands still at kp 1)
now teaches every frame without a jump request and every frame tilted past 35 deg, by behaviour
cloning on the student's own frames; the request window, upright, is PPO's.

Run 2 (taught, 27/10/2026) jumped: trunk 36 to 44 mm above standing, one jump, balanced landings.
But it did not squat at all (lowest point before take-off within 1 mm of standing): it jumped off
nearly straight legs by throwing the heavy head, head pitch and yaw at the torque clamp, and the
sideways part twisted the body, 47 to 77 deg of heading per jump on two robots in three. So for
run 3: the heading is held after landing, the head may not yaw or roll during the jump, a second
hop costs about what the first earns, and a squat of trunk and head before a real jump pays.

Rewards (scaled by dt like every mjlab term):
  jump_stand      height x upright x leg pose, multiplied (a sagging pose at the right height must
                  not score; the backflip lesson again), paid before the window and once landed
  jump_push       upward trunk speed in the window before take-off, upright only: the gradient
                  that finds the push
  jump_flight     per step of the first flight, if launch_ok: 1 plus the trunk height above
                  standing in units of FLIGHT_HEIGHT_UNIT
  jump_squat      once, at the landing of a first flight from an upright take-off: the squat
                  before take-off, the mean drop of the trunk and the head below their heights
                  when the jump was asked for, full at SQUAT_FULL_M. A squat with no jump earns
                  nothing, so it cannot become a place to rest (run 1's collapse)
  jump_heading    once landed, every step: the heading's error from when the jump was asked for
  jump_head_twist head yaw and roll speed, squared, from the request until landing (neck and head
                  pitch stay free: the head going down is part of the squat)
  jump_extra      any airborne step that is not the first flight
  jump_missed     after the window, if it never flew
  fall            the fall termination
Regularisers stay from the stand-up task: action rate, torque rate, angular velocity and
momentum, joint limits, self collisions, Dukki's over-limit penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.envs import mdp
from mjlab.managers import RewardTermCfg, TerminationTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_standup_env_cfg import STAND_Z, make_microduck_standup_env_cfg

EPISODE_S = 5.0
WAIT_S = (1.0, 2.0)
WINDOW_S = 1.0
GRACE_S = 0.3  # a take-off this soon after the window still counts as the jump
FLIGHT_MAX_TILT_DEG = 30.0
FLIGHT_MIN_Z = STAND_Z - 0.02
TAKEOFF_MAX_TILT_DEG = 25.0
FLIGHT_HEIGHT_UNIT = 0.02  # m of trunk height above standing per extra unit of flight reward
FALL_TILT_DEG = 60.0
SQUAT_FULL_M = 0.025  # trunk and head squat for the full bonus: about the probe's best crouch
HEAD_BODY = "jaw_soft"  # top of the neck chain, 225 mm when standing


def _tilt_deg(robot) -> torch.Tensor:
    g = robot.data.projected_gravity_b
    return torch.rad2deg(torch.arccos((-g[:, 2] / g.norm(dim=1).clamp_min(1e-6)).clamp(-1.0, 1.0)))


class JumpCommand(microduck_mdp.VelocityCommandCommandOnly):
    """Jump request in the twist slot, and the jump's state for the rewards."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self._env_ref = env
        self.t = torch.zeros(n, device=dev)
        self.t_jump = torch.zeros(n, device=dev)
        self.flight = torch.zeros(n, dtype=torch.bool, device=dev)
        self.first = torch.zeros(n, dtype=torch.bool, device=dev)
        self.launch_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.landed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.flights = torch.zeros(n, dtype=torch.long, device=dev)
        # Latched when the jump is asked for: heading, trunk and head heights.
        self.requested = torch.zeros(n, dtype=torch.bool, device=dev)
        self.heading_ref = torch.zeros(n, device=dev)
        self.trunk_ref = torch.zeros(n, device=dev)
        self.head_ref = torch.zeros(n, device=dev)
        # The squat: the lowest trunk and head since the request, until take-off.
        self.trunk_low = torch.zeros(n, device=dev)
        self.head_low = torch.zeros(n, device=dev)
        # True on the step a first flight from an upright take-off lands.
        self.landed_now = torch.zeros(n, dtype=torch.bool, device=dev)
        # The trunk's worst tilt from the request until take-off.
        self.approach_tilt = torch.zeros(n, device=dev)
        self._head_id = None

    @property
    def in_window(self) -> torch.Tensor:
        return (self.t >= self.t_jump) & (self.t < self.t_jump + WINDOW_S)

    @property
    def after_window(self) -> torch.Tensor:
        return self.t >= self.t_jump + WINDOW_S

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        lo, hi = WAIT_S
        self.t[env_ids] = 0.0
        self.t_jump[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(lo, hi)
        for flag in (self.flight, self.first, self.launch_ok, self.landed, self.requested, self.landed_now):
            flag[env_ids] = False
        self.approach_tilt[env_ids] = 0.0
        self.flights[env_ids] = 0
        self.vel_command_b[env_ids] = 0.0

    @property
    def squat_depth(self) -> torch.Tensor:
        """Mean drop of the trunk and the head below their heights at the request, m."""
        return 0.5 * ((self.trunk_ref - self.trunk_low) + (self.head_ref - self.head_low)).clamp_min(0.0)

    def _update_command(self) -> None:
        env = self._env_ref
        robot = env.scene["robot"]
        if self._head_id is None:
            self._head_id = list(robot.body_names).index(HEAD_BODY)
        self.t += env.step_dt
        z_trunk = robot.data.root_link_pos_w[:, 2]
        z_head = robot.data.body_link_pos_w[:, self._head_id, 2]
        newly = self.in_window & ~self.requested
        self.requested |= newly
        self.heading_ref = torch.where(newly, robot.data.heading_w, self.heading_ref)
        self.trunk_ref = torch.where(newly, z_trunk, self.trunk_ref)
        self.head_ref = torch.where(newly, z_head, self.head_ref)
        tilt = _tilt_deg(robot)
        self.approach_tilt = torch.where(newly, tilt, self.approach_tilt)
        self.trunk_low = torch.where(newly, z_trunk, self.trunk_low)
        self.head_low = torch.where(newly, z_head, self.head_low)
        squatting = self.requested & (self.flights == 0)
        self.trunk_low = torch.where(squatting, torch.minimum(self.trunk_low, z_trunk), self.trunk_low)
        self.head_low = torch.where(squatting, torch.minimum(self.head_low, z_head), self.head_low)
        self.approach_tilt = torch.where(squatting, torch.maximum(self.approach_tilt, tilt), self.approach_tilt)
        feet_down = env.scene.sensors["feet_ground_contact"].data.found[:, :2].bool().any(dim=1)
        z = robot.data.root_link_pos_w[:, 2]
        flight_now = ~feet_down & (z > FLIGHT_MIN_Z) & (tilt < FLIGHT_MAX_TILT_DEG)
        takeoff = flight_now & ~self.flight
        touchdown = ~flight_now & self.flight
        eligible = (self.t >= self.t_jump) & (self.t < self.t_jump + WINDOW_S + GRACE_S) & (self.flights == 0)
        starts_first = takeoff & eligible
        self.first |= starts_first
        self.launch_ok |= starts_first & (torch.maximum(self.approach_tilt, tilt) < TAKEOFF_MAX_TILT_DEG)
        ended_first = touchdown & self.first
        self.landed_now = ended_first & self.launch_ok
        self.landed |= ended_first
        self.first &= ~ended_first
        self.flights += takeoff.long()
        self.flight = flight_now
        self.vel_command_b[:] = 0.0
        self.vel_command_b[:, 0] = self.in_window.float()

    def _update_metrics(self) -> None:
        pass


@dataclass(kw_only=True)
class JumpCommandCfg(microduck_mdp.VelocityCommandCommandOnlyCfg):
    class_type: type = JumpCommand

    def build(self, env) -> JumpCommand:
        return JumpCommand(self, env)


def _jump(env) -> JumpCommand:
    return env.command_manager.get_term("twist")


def jump_stand(env, height_std: float = 0.03, tilt_std_deg: float = 15.0, pose_std: float = 0.3) -> torch.Tensor:
    """Balanced stand, as a product: height x upright x leg pose. Before the window, and once
    landed (and on the ground)."""
    cmd = _jump(env)
    robot = env.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2]
    height = torch.exp(-(((z - STAND_Z) / height_std) ** 2))
    upright = torch.exp(-((_tilt_deg(robot) / tilt_std_deg) ** 2))
    act = robot.actuators[0]
    ids = act._target_ids_list
    err = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
    legs = [i for i, n in enumerate(act.target_names) if not n.startswith(("neck", "head"))]
    pose = torch.exp(-(err[:, legs].pow(2).mean(dim=1)) / pose_std**2)
    active = (~cmd.in_window & ~cmd.after_window) | (cmd.landed & ~cmd.flight)
    return height * upright * pose * active.float()


def jump_push(env, max_speed: float = 1.0) -> torch.Tensor:
    """Upward trunk speed in the window before take-off, while upright."""
    cmd = _jump(env)
    robot = env.scene["robot"]
    vz = robot.data.root_link_lin_vel_w[:, 2].clamp(0.0, max_speed)
    upright = _tilt_deg(robot) < TAKEOFF_MAX_TILT_DEG
    return vz * (cmd.in_window & (cmd.flights == 0) & upright).float()


def jump_flight(env) -> torch.Tensor:
    """Per step of the first flight, from an upright take-off: 1 plus the height gained."""
    cmd = _jump(env)
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    gain = ((z - STAND_Z) / FLIGHT_HEIGHT_UNIT).clamp(0.0, 3.0)
    return (1.0 + gain) * (cmd.first & cmd.flight & cmd.launch_ok).float()


def jump_extra(env) -> torch.Tensor:
    """Airborne steps that are not the first flight: hopping again, or outside the window."""
    cmd = _jump(env)
    return (cmd.flight & ~cmd.first).float()


def jump_squat(env) -> torch.Tensor:
    """Once, at the landing of a first flight from an upright take-off: the squat before it, as a
    share of SQUAT_FULL_M (capped at 1)."""
    cmd = _jump(env)
    return (cmd.squat_depth / SQUAT_FULL_M).clamp(0.0, 1.0) * cmd.landed_now.float()


def jump_heading(env) -> torch.Tensor:
    """Once landed and on the ground, every step: |heading - heading when the jump was asked|, rad."""
    cmd = _jump(env)
    d = env.scene["robot"].data.heading_w - cmd.heading_ref
    err = torch.atan2(torch.sin(d), torch.cos(d)).abs()
    return err * (cmd.landed & ~cmd.flight).float()


def jump_head_twist(env) -> torch.Tensor:
    """Head yaw and roll speed squared, from the request until landing."""
    cmd = _jump(env)
    robot = env.scene["robot"]
    act = robot.actuators[0]
    names = list(act.target_names)
    ids = [act._target_ids_list[names.index(n)] for n in ("head_yaw", "head_roll")]
    speed2 = robot.data.joint_vel[:, ids].pow(2).sum(dim=1)
    return speed2 * (cmd.requested & ~cmd.landed).float()


def jump_missed(env) -> torch.Tensor:
    """After the window, if it never flew."""
    cmd = _jump(env)
    return (cmd.after_window & (cmd.flights == 0) & ~cmd.flight).float()


def make_dukki_jump_env_cfg(play: bool = False):
    """The stand-up env (all collisions, 61D obs, DR) turned into the jump task."""
    cfg = make_microduck_standup_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_S

    # Always start standing, no pushes; the head and body commands stay at zero.
    ground = cfg.events["set_ground_state"]
    for key in ("face_down", "face_up", "sitting", "standing"):
        ground.params[f"{key}_prob"] = 1.0 if key == "standing" else 0.0
    cfg.events.pop("push_robot", None)
    for name in ("head_pose", "body_pose"):
        if name in cfg.commands:
            term = cfg.commands[name]
            term.ranges = tuple((0.0, 0.0) for _ in term.ranges)

    # The jump request and state, in the twist slot.
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (1e6, 1e6)
    cfg.commands["twist"] = JumpCommandCfg(**{k: v for k, v in vars(twist).items() if k != "class_type"})

    # Rewards: keep the regularisers, replace the stand-up task terms.
    keep = {
        "body_ang_vel", "angular_momentum", "dof_pos_limits", "action_rate_l2",
        "joint_torque_rate_l2", "self_collisions", "action_over_limit",
    }
    for name in [k for k in cfg.rewards if k not in keep]:
        del cfg.rewards[name]
    cfg.rewards["jump_stand"] = RewardTermCfg(func=jump_stand, weight=2.0)
    cfg.rewards["jump_push"] = RewardTermCfg(func=jump_push, weight=10.0)
    cfg.rewards["jump_flight"] = RewardTermCfg(func=jump_flight, weight=50.0)
    cfg.rewards["jump_squat"] = RewardTermCfg(func=jump_squat, weight=10.0 / 0.02)  # once, undoing dt
    cfg.rewards["jump_heading"] = RewardTermCfg(func=jump_heading, weight=-5.0)
    cfg.rewards["jump_head_twist"] = RewardTermCfg(func=jump_head_twist, weight=-0.05)
    cfg.rewards["jump_extra"] = RewardTermCfg(func=jump_extra, weight=-100.0)
    cfg.rewards["jump_missed"] = RewardTermCfg(func=jump_missed, weight=-2.0)
    cfg.rewards["fall"] = RewardTermCfg(func=mdp.is_terminated, weight=-10.0 / 0.02)  # once, undoing dt

    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(FALL_TILT_DEG)}
    )

    # Curricula left pointing at removed rewards, events or commands go.
    for name in list(cfg.curriculum):
        p = cfg.curriculum[name].params or {}
        if (
            ("reward_name" in p and p["reward_name"] not in cfg.rewards)
            or ("event_name" in p and p["event_name"] not in cfg.events)
            or ("command_name" in p and p["command_name"] in ("head_pose", "body_pose"))
            or p.get("event_name") == "set_ground_state"
        ):
            del cfg.curriculum[name]
    return cfg


#: The stand expert: the Dukki stand-up policy's rsl_rl checkpoint (same 61D actor). Kept outside
#: the repository; on blackbox a copy of ..._dukki-standup-distill-kp1-kd0p05/model_3500.pt.
JUMP_EXPERT_DEFAULT = "logs/experts/dukki_standup_3500.pt"


def make_dukki_jump_rl_cfg(base):
    """``base`` (the stand-up runner cfg) with the stand expert teaching every frame without a
    jump request and every frame tilted past 35 deg. The path is ``DUKKI_JUMP_EXPERT``."""
    import os
    from dataclasses import fields, replace

    from mjlab_microduck.tasks.distill import PpoWithExpertBcCfg, default_bc_cfg

    algo = {f.name: getattr(base.algorithm, f.name) for f in fields(base.algorithm) if f.name != "class_name"}
    bc_cfg = {
        **default_bc_cfg(),
        "wandb_run_path": None,
        "checkpoint_path": os.environ.get("DUKKI_JUMP_EXPERT", JUMP_EXPERT_DEFAULT),
        "gate_tilt_deg": 35.0,
        "teach_command_zero": True,
        "anchor_coef": 0.0,
        "anchor_wandb_run_path": None,
    }
    return replace(
        base,
        algorithm=PpoWithExpertBcCfg(**algo, bc_cfg=bc_cfg),
        experiment_name="dukki_jump",
        run_name="dukki_jump",
        max_iterations=3000,
    )
