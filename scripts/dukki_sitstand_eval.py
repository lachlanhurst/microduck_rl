#!/usr/bin/env python3
"""How often a sit-stand checkpoint does what it is told, by case, under training conditions.

Runs the task as trained (random seated or standing starts, posture commands resampled every
few seconds, pushes on) for many robots and, for every commanded segment, records the posture
it started in and whether the body reached the commanded height in time:

    sit from standing, sit while seated, stand from seated, stand while standing

A segment counts as reached when the root height is within REACH_MM of the command's target
height at any point from SETTLE_S after the command until the segment ends. Also reports the
peak motor torque seen during each case, so the costly transitions show, and, when the task
has the head-ground sensor (Dukki, dukki.make_head_off_ground), the share of segments in which
the neck or head touched the ground and the largest head contact force.

    DUKKI_KP=1 DUKKI_KD=0.05 uv run scripts/dukki_sitstand_eval.py \\
        --task Mjlab-SitStand-Flat-Dukki --checkpoint path/model_2999.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks.dukki import HEAD_GROUND_SENSOR, check_cfg_matches_run
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

SIT_Z = 0.060  # sit-stand cfg's targets [m]
STAND_Z = 0.115
REACH_MM = 15.0
SETTLE_S = 2.5  # the command ramps over 2 s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-SitStand-Flat-Dukki")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--duration", type=float, default=60.0, help="[s]")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pushes", action="store_true", help="as dukki_render: pushes and their curricula off")
    ap.add_argument("--head-zero", action="store_true", help="as dukki_render: head and body pose commands held at 0")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.envs
    if args.no_pushes:
        removed = [k for k in cfg.events if "push" in k or "topple" in k]
        for name in removed:
            del cfg.events[name]
        for name in [k for k, t in cfg.curriculum.items() if (t.params or {}).get("event_name") in removed]:
            del cfg.curriculum[name]
    check_cfg_matches_run(cfg, args.checkpoint)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    rl_cfg = load_rl_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = (load_runner_cls(args.task) or OnPolicyRunner)(wrapped, asdict(rl_cfg), device=args.device)
    runner.load(args.checkpoint, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    torch.manual_seed(args.seed)
    wrapped.reset()
    env.reset(seed=args.seed)
    obs = wrapped.get_observations()
    robot = env.scene["robot"]
    act = robot.actuators[0]
    twist = env.command_manager.get_term("twist")
    n, dt = args.envs, env.step_dt

    sit = twist.vel_command_b[:, 0] > 0.5
    z = robot.data.root_link_pos_w[:, 2]
    started_tall = z > 0.5 * (SIT_Z + STAND_Z)
    since = torch.zeros(n, device=env.device)
    reached = torch.zeros(n, dtype=torch.bool, device=env.device)
    peak = torch.zeros(n, device=env.device)
    head_sensor = env.scene.sensors.get(HEAD_GROUND_SENSOR) if hasattr(env.scene.sensors, "get") else None
    head_touched = torch.zeros(n, dtype=torch.bool, device=env.device)
    head_peak = torch.zeros(n, device=env.device)

    counts = defaultdict(lambda: [0, 0])  # case -> [reached, segments]
    peaks = defaultdict(float)
    head_counts = defaultdict(int)  # case -> segments with head-ground contact
    head_peaks = defaultdict(float)

    def case_of(s, tall):
        return ("sit" if s else "stand") + (" from standing" if tall else " from seated")

    def close(mask):
        for i in torch.nonzero(mask).flatten().tolist():
            if since[i] < SETTLE_S:  # cut short by a reset or the end; too short to judge
                continue
            c = case_of(bool(sit[i]), bool(started_tall[i]))
            counts[c][0] += int(reached[i])
            counts[c][1] += 1
            peaks[c] = max(peaks[c], float(peak[i]))
            head_counts[c] += int(head_touched[i])
            head_peaks[c] = max(head_peaks[c], float(head_peak[i]))

    with torch.no_grad():
        for _ in range(int(args.duration / dt)):
            if args.head_zero:
                for name in ("head_pose", "body_pose"):
                    if name in env.command_manager.active_terms:
                        env.command_manager.get_term(name)._command[:] = 0.0
            obs, _, dones, _ = wrapped.step(policy(obs))
            now_sit = twist.vel_command_b[:, 0] > 0.5
            z = robot.data.root_link_pos_w[:, 2]
            ended = (now_sit != sit) | dones.bool()
            close(ended)
            tau = act.reported_torque.abs().amax(dim=1) if hasattr(act, "reported_torque") else torch.zeros(n, device=env.device)
            # New segments start from here.
            started_tall = torch.where(ended, z > 0.5 * (SIT_Z + STAND_Z), started_tall)
            sit = now_sit
            since = torch.where(ended, torch.zeros_like(since), since + dt)
            reached = torch.where(ended, torch.zeros_like(reached), reached)
            peak = torch.where(ended, torch.zeros_like(peak), torch.maximum(peak, tau))
            if head_sensor is not None:
                d = head_sensor.data
                touch = (d.found.view(n, -1) > 0).any(dim=-1)
                force = torch.norm(d.force.view(n, -1, 3).sum(dim=1), dim=-1)
                head_touched = torch.where(ended, touch, head_touched | touch)
                head_peak = torch.where(ended, force, torch.maximum(head_peak, force))
            target = torch.where(sit, torch.full_like(z, SIT_Z), torch.full_like(z, STAND_Z))
            reached |= (since >= SETTLE_S) & ((z - target).abs() * 1000 < REACH_MM)
    close(torch.ones(n, dtype=torch.bool, device=env.device))

    extras = [f for f, on in (("no pushes", args.no_pushes), ("head held at 0", args.head_zero)) if on]
    print(f"== {args.checkpoint} on {args.task}, {n} robots for {args.duration:.0f} s on {args.device}{', ' + ', '.join(extras) if extras else ''}")
    for c in ("sit from standing", "sit from seated", "stand from seated", "stand from standing"):
        ok, total = counts[c]
        share = 100 * ok / total if total else float("nan")
        line = f"  {c:20s} reached {ok:5d} of {total:5d} ({share:5.1f}%)  peak motor torque {peaks[c]:.2f} N.m"
        if head_sensor is not None:
            head_share = 100 * head_counts[c] / total if total else float("nan")
            line += f"  head touched {head_share:5.1f}%  peak head force {head_peaks[c]:5.1f} N"
        print(line)
    env.close()


if __name__ == "__main__":
    main()
