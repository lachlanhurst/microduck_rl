#!/usr/bin/env python3
"""How well a jump checkpoint jumps, across many robots: one episode each, the task's own request.

Per robot, from the jump task's own state (JumpCommand) and the robot:

    jumped once   one flight, started in the request window from an upright take-off, and landed
    settled       of those, still upright on the ground 0.5 s after landing (when the jump is paid)
    squat         trunk and head drop below their heights at the request, before take-off
    neck squat    the head's drop on the trunk before take-off; how much of it is back at take-off
    head level    worst gap between the two neck pitch joints' changes, request to landing
    style         jump_outcome's leg and neck squat shares, 0 to 1
    rise          trunk peak in the first flight above its height at the request
    upright end   standing (trunk above 105 mm, tilt under 20 deg) at the end
    heading       |heading change| from the request to the end
    fell          the fall termination fired
    at limit      time with any joint's motor torque within 5% of the 1.0 N.m clamp
    head yaw      peak head-yaw motor torque

Pushes are off as in training. A take-off is judged at the final TAKEOFF_MIN_VZ (training ramps
up to it; ``--takeoff-vz`` to change). ``--scripted`` runs the open-loop jump of
scripts/dukki_jump_probe.py instead of a policy (crouch 0.8 rad of knee over 0.4 s, hold 0.2 s,
push 1.0 for 0.08 s, from the request), a baseline that also checks this script; run it stiff
(DUKKI_KP=4) as the probe explains.

    DUKKI_KP=1 DUKKI_KD=0.05 uv run scripts/dukki_jump_eval.py --checkpoint path/model_2500.pt
    DUKKI_KP=4 DUKKI_KD=0.05 uv run scripts/dukki_jump_eval.py --scripted --device cpu --envs 32
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks.dukki import check_cfg_matches_run
from mjlab_microduck.tasks.dukki_jump import TAKEOFF_MIN_VZ
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner


def scripted_policy(env):
    """The probe's best open-loop jump, timed from each robot's request."""
    spec = importlib.util.spec_from_file_location("probe", Path(__file__).with_name("dukki_jump_probe.py"))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    names = list(env.scene["robot"].actuators[0].target_names)
    unit = probe.crouch_direction(names, 1.0, 1.0, env.device)
    cmd = env.command_manager.get_term("twist")

    def policy(_obs):
        s = cmd.t - cmd.t_jump
        crouch = (s.clamp(0, 0.4) / 0.4) * 0.8
        push = (s >= 0.6) & (s < 0.68)
        scale = torch.where(push, torch.full_like(s, -1.0), torch.where(s >= 0.68, torch.zeros_like(s), crouch))
        return scale.unsqueeze(1) * unit

    return policy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-Jump-Flat-Dukki")
    ap.add_argument("--checkpoint")
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--takeoff-vz", type=float, default=TAKEOFF_MIN_VZ)
    args = ap.parse_args()
    if not args.scripted and not args.checkpoint:
        ap.error("--checkpoint or --scripted")

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.envs
    duration = cfg.episode_length_s - 0.1  # one episode, the task's own length and schedule
    if args.checkpoint:
        check_cfg_matches_run(cfg, args.checkpoint)
    torch.manual_seed(args.seed)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    rl_cfg = load_rl_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    if args.scripted:
        policy = scripted_policy(env)
    else:
        runner = (load_runner_cls(args.task) or OnPolicyRunner)(wrapped, asdict(rl_cfg), device=args.device)
        runner.load(args.checkpoint, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
    wrapped.reset()
    env.reset(seed=args.seed)
    obs = wrapped.get_observations()
    robot = env.scene["robot"]
    act = robot.actuators[0]
    cmd = env.command_manager.get_term("twist")
    cmd.takeoff_min_vz = args.takeoff_vz
    names = list(act.target_names)
    head_yaw = names.index("head_yaw")
    n, dt = args.envs, env.step_dt

    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    rise = torch.zeros(n, device=env.device)
    at_limit = torch.zeros(n, device=env.device)
    head_yaw_peak = torch.zeros(n, device=env.device)
    squat_trunk = torch.zeros(n, device=env.device)
    squat_head = torch.zeros(n, device=env.device)
    with torch.no_grad():
        for _ in range(int(duration / dt)):
            obs, _, dones, _ = wrapped.step(policy(obs))
            ended = dones.bool() & alive
            fell |= ended & env.termination_manager.terminated.bool()
            alive &= ~ended
            z = robot.data.root_link_pos_w[:, 2]
            live = alive & cmd.requested
            rise = torch.where(live & cmd.first & cmd.flight, torch.maximum(rise, z - cmd.trunk_ref), rise)
            squat_trunk = torch.where(live, torch.maximum(squat_trunk, cmd.trunk_ref - cmd.trunk_low), squat_trunk)
            squat_head = torch.where(live, torch.maximum(squat_head, cmd.head_ref - cmd.head_low), squat_head)
            tau = act.reported_torque.abs() if hasattr(act, "reported_torque") else torch.zeros(n, len(names), device=env.device)
            at_limit += (alive & (tau.amax(dim=1) > 0.95)).float() * dt
            head_yaw_peak = torch.where(alive, torch.maximum(head_yaw_peak, tau[:, head_yaw]), head_yaw_peak)
    g = robot.data.projected_gravity_b
    tilt = torch.rad2deg(torch.arccos((-g[:, 2] / g.norm(dim=1)).clamp(-1, 1)))
    upright = alive & (robot.data.root_link_pos_w[:, 2] > 0.105) & (tilt < 20)
    d = robot.data.heading_w - cmd.heading_ref
    heading = torch.rad2deg(torch.atan2(torch.sin(d), torch.cos(d)).abs())
    once = alive & cmd.landed & cmd.launch_ok & (cmd.flights == 1)
    extra = alive & (cmd.flights > 1)

    def pct(mask):
        return 100.0 * mask.float().mean().item()

    def med(x, mask):
        v = x[mask]
        return (v.median().item(), v.quantile(0.9).item()) if v.numel() else (math.nan, math.nan)

    label = "scripted jump" if args.scripted else args.checkpoint
    print(f"== {label} on {args.task}, {n} robots, one {cfg.episode_length_s:.0f} s episode each")
    print(f"  jumped once {pct(once):5.1f}%   of which settled {pct(once & cmd.settled):5.1f}%   more than once {pct(extra):5.1f}%   no jump {pct(alive & (cmd.flights == 0)):5.1f}%   fell {pct(fell):5.1f}%   upright at the end {pct(upright):5.1f}%")
    for title, x, mask, unit, scale in (
        ("rise (jumped once)", rise, once, "mm", 1000),
        ("squat, trunk", squat_trunk, once, "mm", 1000),
        ("squat, head", squat_head, once, "mm", 1000),
        ("neck squat (jumped once)", cmd.neck_drop, once, "mm", 1000),
        ("neck extension by take-off", cmd.neck_ext, once, "mm", 1000),
        ("leg squat share", cmd.leg_share, once, "", 1),
        ("neck squat share", cmd.neck_share, once, "", 1),
        ("head level error (jumped once)", cmd.level_err_max, once, "deg", 180 / math.pi),
        ("heading change (alive)", heading, alive & cmd.requested, "deg", 1),
        ("time at torque limit (alive)", at_limit, alive, "s", 1),
        ("head yaw peak torque (alive)", head_yaw_peak, alive, "N.m", 1),
    ):
        m, p90 = med(x, mask)
        print(f"  {title:30s} median {m * scale:7.2f} {unit}   90% {p90 * scale:7.2f} {unit}")
    env.close()


if __name__ == "__main__":
    main()
