"""Dukki jump: when asked, squat with the legs and the neck, extend up out of the squat into one
jump, and land back in a balanced stand facing the same way.

Episode (EPISODE_S): the robot spawns standing. For WAIT_S (random) the jump command is off and
it should stand still; then the command is on for WINDOW_S and it should jump once; then it
stands again. The command rides in the twist obs slot (vx = 1 while the jump is asked for, 0
otherwise), so the 61D actor obs layout is the family's and a deployed policy jumps on request.

Feasibility (scripts/dukki_jump_probe.py, 27/10/2026, J288 BAM, open loop): a crouch of 0.6 to
1.2 rad of knee bend (hip and ankle pitch half that, same signs as the knee) followed by a short
hard extension gets both feet up to 45 mm off the ground for 0.1 to 0.25 s, at the 1.0 N.m motor
clamp. Only 7 of 80 open-loop jumps landed upright: the leaving is easy, the landing is the task.
The legs can lower the trunk about 43 mm (knee 1.4 rad, feet flat). The neck: equal changes of
neck_pitch and head_pitch keep the head level; +0.6 rad each lowers it 18 mm on the trunk (and
24 mm back), and the head is tallest near -0.4 rad, from where a level drop goes forward.

Teacher (make_dukki_jump_rl_cfg): unaided from scratch, the first run collapsed into a crouch and
never stood. The Dukki stand-up policy (distilled from Pollen's stand expert) teaches every frame
without a jump request and every frame tilted past 35 deg, by behaviour cloning on the student's
own frames; the request window, upright, is PPO's.

Lessons from runs 1 to 3p (27/10 to 30/10/2026; their code is in git up to ebbbb16, and each run's
log directory keeps its own):
  1. A squat paid without a jump becomes a place to rest (run 1 collapsed into a 60 mm crouch).
  2. A per-step price after the request that a noisy policy's flailing runs up must start near 0
     and ramp in, or falling before the request becomes the cheapest episode (runs 3, 3b, 3c, 3e).
  3. Unloaded feet are not a jump (run 3f dropped so fast they lifted): a take-off is real only
     with the trunk rising, and height counts only above standing. A take-off bar the policy
     cannot reach yet stops it taking off at all (run 3g), so the bar ramps.
  4. Paying in the air buys a jump and a fall (run 3h). Paying only once the landing has settled
     while most jumps still fall makes trying worse than not trying: it stopped jumping within
     30 iterations, twice (runs 3i, 3k).
  5. A bonus paid at touchdown buys extra hops (run 3j: a 60 ms blip to collect it, then the real
     jump), and, with anything priced per step after landing, a fall straight after touchdown
     (run 3o collapsed that way).
  6. A part of the movement paid for on its own is bought without the jump: a 43 mm squat into a
     4.6 mm hop (run 3m), the head flung past standing for an extension bonus (run 3n), the neck
     joints swung apart between level checks (run 3p), the feet lifted 13 mm for a height paid per
     step in the air (run 3l), a nod for a head squat (run 3d).
  7. Without a squat it jumps off straight legs by throwing the head (runs 2, 3d).

Run 4 (from scratch, 30/10/2026): one outcome, paid once per jump, the style as multipliers on it.
  jump_outcome = 10 x height x style x heading
    height   the first flight's peak trunk height above standing, in FLIGHT_HEIGHT_UNITs, up to
             FLIGHT_UNITS_MAX (20 mm is 1, so 40 mm is 2)
    style    style_floor + (1 - style_floor) x (LEG_WEIGHT x leg squat + NECK_WEIGHT x neck squat)
             leg squat: the trunk's drop from the request until take-off, full at SQUAT_FULL_M
             neck squat: the smaller of the head's drop on the trunk (trunk frame, so the neck's
             doing) from the request until take-off and how much of that drop is back at take-off,
             full at NECK_FULL_M: the neck must go down and come back up in the push
    heading  exp(-(heading change since the request / heading_std_deg)^2)
  paid (1 - settle_share) at touchdown and settle_share once the landing settles (SETTLE_S on the
  ground, upright, no second hop), all from a real, upright take-off in the window. No part of
  the movement pays without the jump (lessons 1, 6): a squat, the neck or the heading is worth
  what it adds to a jump, and squatting harder pays only by jumping higher. settle_share ramps
  from 0, as run 3h to 3j did by hand, and stops short of 1 so a jump that falls still beats no
  jump (lesson 4); with no per-step price after landing, nothing is saved by falling (lesson 5).
  The neck is one action (CoupledNeckAction): the head_pitch target follows neck_pitch's offset
  from standing, so the head is level by construction and cannot be swung (lesson 6). A deployed
  jump policy needs the same: head_pitch target = its default + the neck_pitch target's offset.
Shaping kept from run 3, where it worked: jump_push (upward trunk speed in the window before
take-off, the gradient that finds the push), jump_stand, jump_hop_early, jump_missed, the fall;
and, ramped in (lesson 2), jump_extra per airborne step after the request that is not the first
flight, jump_retake per extra real take-off, jump_head_twist on head yaw and roll speed.

State (JumpCommand, updated every step after physics):
  flight     both feet off the ground, the trunk near standing height and tilted under
             FLIGHT_MAX_TILT_DEG (a robot toppling over lifts its feet too; that is not flight)
  first      the first real flight that starts inside the window (or GRACE_S after it)
  launch_ok  latched at that take-off: the trunk stayed under TAKEOFF_MAX_TILT_DEG from the
             request through take-off (worst over the approach: a squat-and-push pitches the trunk
             to about 20 deg by the time the feet leave)
  landed     the first flight has ended; settled: SETTLE_S later it was still up
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_standup_env_cfg import STAND_Z, make_microduck_standup_env_cfg

EPISODE_S = 5.0
WAIT_S = (1.0, 2.0)
WINDOW_S = 1.0
GRACE_S = 0.3  # a take-off this soon after the window still counts as the jump
FLIGHT_MAX_TILT_DEG = 30.0
FLIGHT_MIN_Z = STAND_Z - 0.02
TAKEOFF_MAX_TILT_DEG = 25.0
FALL_TILT_DEG = 60.0
SETTLE_S = 0.5  # s on the ground and upright after landing before the settle share is paid
SETTLE_MAX_TILT_DEG = 20.0
HEAD_BODY = "jaw_soft"  # top of the neck chain, 225 mm when standing
NECK_JOINTS = ("neck_pitch", "head_pitch")  # equal changes from standing keep the head level

# The outcome (jump_outcome).
OUTCOME_SCALE = 10.0  # per FLIGHT_HEIGHT_UNIT of height, at full style and heading
FLIGHT_HEIGHT_UNIT = 0.02  # m of trunk height above standing per unit
FLIGHT_UNITS_MAX = 3.0
SQUAT_FULL_M = 0.040  # trunk drop for the full leg squat: the legs reach about 43 mm
NECK_FULL_M = 0.015  # head drop on the trunk, down and back up, for the full neck squat
LEG_WEIGHT = 0.6
NECK_WEIGHT = 0.4

# A take-off is real only with the trunk rising this fast, m/s, by training iteration (lesson 3).
# Height is paid directly, so the bar only tells a jump from unloaded feet; 0.3 m/s is 4.6 mm.
TAKEOFF_VZ_STAGES = ((0, 0.1), (1000, 0.2), (1500, 0.3))
TAKEOFF_MIN_VZ = TAKEOFF_VZ_STAGES[-1][1]

# Reward weights by training iteration, (iteration, weight): the prices a noisy policy's flailing
# runs up after the request start low (lesson 2).
STAGES = {
    "jump_extra": ((0, -20.0), (1500, -50.0), (2000, -100.0)),
    "jump_retake": ((0, 0.0), (1000, -5.0 / 0.02), (1500, -10.0 / 0.02)),  # once per take-off
    "jump_head_twist": ((0, 0.0), (750, -0.05), (1500, -0.1)),
}
# jump_outcome's parameters by training iteration, (iteration, value): first any jump pays, then
# the squat and the heading come to matter, and the payment moves towards the settle.
OUTCOME_STAGES = {
    "style_floor": ((0, 1.0), (750, 0.5), (1500, 0.2)),
    "heading_std_deg": ((0, 90.0), (750, 40.0), (1500, 20.0)),
    "settle_share": ((0, 0.0), (1000, 0.4), (1500, 0.6), (2000, 0.75)),
}
STEPS_PER_ITERATION = 24  # the runner's num_steps_per_env; curricula count env steps


def _tilt_deg(robot) -> torch.Tensor:
    g = robot.data.projected_gravity_b
    return torch.rad2deg(torch.arccos((-g[:, 2] / g.norm(dim=1).clamp_min(1e-6)).clamp(-1.0, 1.0)))


class CoupledNeckAction(JointPositionAction):
    """Joint position action with head_pitch's target set to its default plus neck_pitch's offset,
    so the two move equally and the head stays level. The policy's own head_pitch output is
    ignored."""

    def __init__(self, cfg, env):
        super().__init__(cfg=cfg, env=env)
        names = list(self._target_names)
        self._neck, self._head = (names.index(j) for j in NECK_JOINTS)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        p, off = self._processed_actions, self._offset
        p[:, self._head] = off[:, self._head] + (p[:, self._neck] - off[:, self._neck])


@dataclass(kw_only=True)
class CoupledNeckActionCfg(JointPositionActionCfg):
    def build(self, env) -> CoupledNeckAction:
        return CoupledNeckAction(self, env)


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
        # Real take-offs since the request.
        self.flights = torch.zeros(n, dtype=torch.long, device=dev)
        # Latched when the jump is asked for: heading, trunk and head heights, head on the trunk.
        self.requested = torch.zeros(n, dtype=torch.bool, device=dev)
        self.heading_ref = torch.zeros(n, device=dev)
        self.trunk_ref = torch.zeros(n, device=dev)
        self.head_ref = torch.zeros(n, device=dev)
        self.head_rel_ref = torch.zeros(n, device=dev)
        # The squat, from the request until take-off: lowest trunk and head, and the head's
        # largest drop on the trunk; and how much of that drop was back at take-off.
        self.trunk_low = torch.zeros(n, device=dev)
        self.head_low = torch.zeros(n, device=dev)
        self.neck_drop = torch.zeros(n, device=dev)
        self.neck_ext = torch.zeros(n, device=dev)
        # The trunk's worst tilt from the request until take-off.
        self.approach_tilt = torch.zeros(n, device=dev)
        # The first flight's peak trunk height above standing.
        self.flight_peak = torch.zeros(n, device=dev)
        # The worst gap between the neck pitch joints' changes, request to landing (for the eval;
        # CoupledNeckAction keeps it small).
        self.level_err_max = torch.zeros(n, device=dev)
        # True on the step a first flight from an upright take-off lands; landed so; time of it.
        self.landed_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.landed_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t_land = torch.zeros(n, device=dev)
        # The settle check, once, SETTLE_S after landing: made; passed on this step; passed.
        self.settle_checked = torch.zeros(n, dtype=torch.bool, device=dev)
        self.settled_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.settled = torch.zeros(n, dtype=torch.bool, device=dev)
        # True on the step of a real take-off after the first since the request.
        self.retake_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self._head_id = None
        self._neck_ids = None
        # Set by the takeoff_min_vz curriculum (TAKEOFF_VZ_STAGES).
        self.takeoff_min_vz = TAKEOFF_VZ_STAGES[0][1]

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
        for flag in (
            self.flight, self.first, self.launch_ok, self.landed, self.requested, self.landed_now,
            self.landed_ok, self.settle_checked, self.settled_now, self.settled, self.retake_now,
        ):
            flag[env_ids] = False
        for value in (
            self.approach_tilt, self.flight_peak, self.neck_drop, self.neck_ext, self.level_err_max,
        ):
            value[env_ids] = 0.0
        self.flights[env_ids] = 0
        self.vel_command_b[env_ids] = 0.0

    @property
    def squat_depth(self) -> torch.Tensor:
        """The trunk's drop below its height at the request, before take-off, m."""
        return (self.trunk_ref - self.trunk_low).clamp_min(0.0)

    @property
    def leg_share(self) -> torch.Tensor:
        """The leg squat as a share of SQUAT_FULL_M, 0 to 1."""
        return (self.squat_depth / SQUAT_FULL_M).clamp(0.0, 1.0)

    @property
    def neck_share(self) -> torch.Tensor:
        """The neck squat, down and back up by take-off, as a share of NECK_FULL_M, 0 to 1."""
        return (torch.minimum(self.neck_drop, self.neck_ext) / NECK_FULL_M).clamp(0.0, 1.0)

    def neck_offsets(self, robot) -> tuple[torch.Tensor, torch.Tensor]:
        """neck_pitch and head_pitch from standing, rad (positive lowers the head)."""
        if self._neck_ids is None:
            act = robot.actuators[0]
            names = list(act.target_names)
            self._neck_ids = [act._target_ids_list[names.index(j)] for j in NECK_JOINTS]
        d = robot.data.joint_pos[:, self._neck_ids] - robot.data.default_joint_pos[:, self._neck_ids]
        return d[:, 0], d[:, 1]

    def head_on_trunk(self, robot) -> torch.Tensor:
        """The head's height in the trunk frame, m: the neck's doing alone."""
        rel = robot.data.body_link_pos_w[:, self._head_id] - robot.data.root_link_pos_w
        return quat_apply_inverse(robot.data.root_link_quat_w, rel)[:, 2]

    def _update_command(self) -> None:
        env = self._env_ref
        robot = env.scene["robot"]
        if self._head_id is None:
            self._head_id = list(robot.body_names).index(HEAD_BODY)
        self.t += env.step_dt
        z = robot.data.root_link_pos_w[:, 2]
        z_head = robot.data.body_link_pos_w[:, self._head_id, 2]
        head_rel = self.head_on_trunk(robot)
        tilt = _tilt_deg(robot)

        newly = self.in_window & ~self.requested
        self.requested |= newly
        # Flights count from the request: a stumble off the ground before it does not void the jump.
        self.flights = torch.where(newly, torch.zeros_like(self.flights), self.flights)
        self.heading_ref = torch.where(newly, robot.data.heading_w, self.heading_ref)
        for ref, low, now in ((self.trunk_ref, self.trunk_low, z), (self.head_ref, self.head_low, z_head)):
            ref.copy_(torch.where(newly, now, ref))
            low.copy_(torch.where(newly, now, low))
        self.head_rel_ref = torch.where(newly, head_rel, self.head_rel_ref)
        self.approach_tilt = torch.where(newly, tilt, self.approach_tilt)

        # The squat, from the request until the first real take-off.
        squatting = self.requested & (self.flights == 0)
        self.trunk_low = torch.where(squatting, torch.minimum(self.trunk_low, z), self.trunk_low)
        self.head_low = torch.where(squatting, torch.minimum(self.head_low, z_head), self.head_low)
        head_drop = (self.head_rel_ref - head_rel).clamp_min(0.0)
        self.neck_drop = torch.where(squatting, torch.maximum(self.neck_drop, head_drop), self.neck_drop)
        self.approach_tilt = torch.where(squatting, torch.maximum(self.approach_tilt, tilt), self.approach_tilt)
        dn, dh = self.neck_offsets(robot)
        level = (dn - dh).abs()
        self.level_err_max = torch.where(
            self.requested & ~self.landed, torch.maximum(self.level_err_max, level), self.level_err_max
        )

        # Flight. A real take-off has the trunk rising (lesson 3); only real take-offs can be the
        # jump or count towards ``flights``; any other airborne step after the request is an
        # extra hop.
        feet_down = env.scene.sensors["feet_ground_contact"].data.found[:, :2].bool().any(dim=1)
        flight_now = ~feet_down & (z > FLIGHT_MIN_Z) & (tilt < FLIGHT_MAX_TILT_DEG)
        takeoff = flight_now & ~self.flight
        real = takeoff & (robot.data.root_link_lin_vel_w[:, 2] >= self.takeoff_min_vz)
        touchdown = ~flight_now & self.flight
        eligible = (self.t >= self.t_jump) & (self.t < self.t_jump + WINDOW_S + GRACE_S) & (self.flights == 0)
        starts_first = real & eligible
        self.first |= starts_first
        self.launch_ok |= starts_first & (torch.maximum(self.approach_tilt, tilt) < TAKEOFF_MAX_TILT_DEG)
        # How much of the neck's drop is back at take-off, at most the drop (past it is a fling,
        # lesson 6).
        back = torch.minimum((self.neck_drop - head_drop).clamp_min(0.0), self.neck_drop)
        self.neck_ext = torch.where(starts_first, back, self.neck_ext)
        in_first = self.first & flight_now & self.launch_ok
        self.flight_peak = torch.where(in_first, torch.maximum(self.flight_peak, z - STAND_Z), self.flight_peak)

        ended_first = touchdown & self.first
        self.landed_now = ended_first & self.launch_ok
        self.landed |= ended_first
        self.landed_ok |= self.landed_now
        self.t_land = torch.where(self.landed_now, self.t, self.t_land)
        self.first &= ~ended_first
        self.retake_now = real & self.requested & (self.flights >= 1)
        self.flights += real.long()
        self.flight = flight_now

        due = self.landed_ok & ~self.settle_checked & (self.t - self.t_land >= SETTLE_S)
        self.settled_now = due & ~flight_now & (self.flights == 1) & (tilt < SETTLE_MAX_TILT_DEG)
        self.settled |= self.settled_now
        self.settle_checked |= due

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


def heading_change(env) -> torch.Tensor:
    """|heading - heading when the jump was asked for|, rad."""
    cmd = _jump(env)
    d = env.scene["robot"].data.heading_w - cmd.heading_ref
    return torch.atan2(torch.sin(d), torch.cos(d)).abs()


def jump_outcome(
    env, style_floor: float = 1.0, heading_std_deg: float = 90.0, settle_share: float = 0.0
) -> torch.Tensor:
    """Once (1 - settle_share) at the touchdown of a first flight from an upright take-off, and
    once settle_share when that landing settles: height x style x heading (module docstring)."""
    cmd = _jump(env)
    height = (cmd.flight_peak / FLIGHT_HEIGHT_UNIT).clamp(0.0, FLIGHT_UNITS_MAX)
    style = style_floor + (1.0 - style_floor) * (LEG_WEIGHT * cmd.leg_share + NECK_WEIGHT * cmd.neck_share)
    heading = torch.exp(-((torch.rad2deg(heading_change(env)) / heading_std_deg) ** 2))
    share = (1.0 - settle_share) * cmd.landed_now.float() + settle_share * cmd.settled_now.float()
    return height * style * heading * share


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


def jump_extra(env) -> torch.Tensor:
    """Airborne after the request that is not the first flight: hopping again, or after the window."""
    cmd = _jump(env)
    return (cmd.flight & ~cmd.first & cmd.requested).float()


def jump_retake(env) -> torch.Tensor:
    """Once per real take-off after the first since the request."""
    return _jump(env).retake_now.float()


def jump_hop_early(env) -> torch.Tensor:
    """Airborne before the request: a stumble off the ground while it should be standing."""
    cmd = _jump(env)
    return (cmd.flight & ~cmd.requested).float()


def jump_head_twist(env) -> torch.Tensor:
    """Head yaw and roll speed squared, from the request until landing (run 2 twisted its body by
    throwing the head sideways)."""
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


def reward_param(env, env_ids, reward_name: str, param: str, stages: list[dict]) -> torch.Tensor:
    """Curriculum: sets a reward term's parameter from ``stages`` ({"step", "value"}), the latest
    stage whose step has passed, like microduck_mdp.reward_weight does for weights."""
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in stages:
        if env.common_step_counter > stage["step"]:
            term_cfg.params[param] = stage["value"]
    return torch.tensor([term_cfg.params[param]])


def takeoff_min_vz(env, env_ids, stages: list[dict]) -> torch.Tensor:
    """Curriculum: the jump command's take-off speed from ``stages`` ({"step", "value"})."""
    del env_ids
    cmd = _jump(env)
    for stage in stages:
        if env.common_step_counter > stage["step"]:
            cmd.takeoff_min_vz = stage["value"]
    return torch.tensor([cmd.takeoff_min_vz])


def _steps(stages, key):
    return [{"step": it * STEPS_PER_ITERATION, key: v} for it, v in stages]


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

    # The neck as one action: head_pitch follows neck_pitch.
    action = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = CoupledNeckActionCfg(**{f.name: getattr(action, f.name) for f in fields(action) if f.init})

    # Rewards: keep the regularisers, replace the stand-up task terms.
    keep = {
        "body_ang_vel", "angular_momentum", "dof_pos_limits", "action_rate_l2",
        "joint_torque_rate_l2", "self_collisions", "action_over_limit",
    }
    for name in [k for k in cfg.rewards if k not in keep]:
        del cfg.rewards[name]
    cfg.rewards["jump_outcome"] = RewardTermCfg(
        func=jump_outcome,
        weight=OUTCOME_SCALE / 0.02,  # once, undoing dt
        params={name: stages[0][1] for name, stages in OUTCOME_STAGES.items()},
    )
    cfg.rewards["jump_stand"] = RewardTermCfg(func=jump_stand, weight=2.0)
    cfg.rewards["jump_push"] = RewardTermCfg(func=jump_push, weight=10.0)
    for name, func in (("jump_extra", jump_extra), ("jump_retake", jump_retake), ("jump_head_twist", jump_head_twist)):
        cfg.rewards[name] = RewardTermCfg(func=func, weight=STAGES[name][0][1])
    # Run 2's price. Besides discouraging hopping unasked, it is what tells a noisy early policy to
    # act less randomly: its flailing hops are charged, and calmer actions are the cheapest fix, so
    # the action noise falls and the teacher's stance holds. Too strong and falling fast is cheaper
    # (run 3 at -100); too weak and the noise never falls (run 3b at -5).
    cfg.rewards["jump_hop_early"] = RewardTermCfg(func=jump_hop_early, weight=-20.0)
    cfg.rewards["jump_missed"] = RewardTermCfg(func=jump_missed, weight=-2.0)
    cfg.rewards["fall"] = RewardTermCfg(func=mdp.is_terminated, weight=-10.0 / 0.02)  # once, undoing dt

    for name, stages in STAGES.items():
        cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": name, "weight_stages": _steps(stages, "weight")},
        )
    for param, stages in OUTCOME_STAGES.items():
        cfg.curriculum[f"jump_outcome_{param}"] = CurriculumTermCfg(
            func=reward_param,
            params={"reward_name": "jump_outcome", "param": param, "stages": _steps(stages, "value")},
        )
    cfg.curriculum["takeoff_min_vz"] = CurriculumTermCfg(
        func=takeoff_min_vz, params={"stages": _steps(TAKEOFF_VZ_STAGES, "value")}
    )

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
    from dataclasses import replace

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
