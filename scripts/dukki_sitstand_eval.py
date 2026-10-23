#!/usr/bin/env python3
"""How often a sit-stand checkpoint does what it is told, by case, under training conditions.

Runs the task as trained (random seated or standing starts, posture commands resampled every
few seconds, pushes on) for many robots and, for every commanded segment, records the posture
it started in and whether the body reached the commanded height in time:

    sit from standing, sit while seated, stand from seated, stand while standing

A segment counts as reached when the root height is within REACH_MM of the command's target
height at any point from SETTLE_S after the command until the segment ends. Also reports the
peak motor torque seen during each case, so the costly transitions show.

    DUKKI_KP=1 DUKKI_KD=0.05 uv run scripts/dukki_sitstand_eval.py \\
        --task Mjlab-SitStand-Flat-Dukki --checkpoint path/model_2999.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks.dukki import check_cfg_matches_run
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
    args = ap.parse_args()

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.envs
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

    counts = defaultdict(lambda: [0, 0])  # case -> [reached, segments]
    peaks = defaultdict(float)

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

    with torch.no_grad():
        for _ in range(int(args.duration / dt)):
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
            target = torch.where(sit, torch.full_like(z, SIT_Z), torch.full_like(z, STAND_Z))
            reached |= (since >= SETTLE_S) & ((z - target).abs() * 1000 < REACH_MM)
    close(torch.ones(n, dtype=torch.bool, device=env.device))

    print(f"== {args.checkpoint} on {args.task}, {n} robots for {args.duration:.0f} s")
    for c in ("sit from standing", "sit from seated", "stand from seated", "stand from standing"):
        ok, total = counts[c]
        share = 100 * ok / total if total else float("nan")
        print(f"  {c:20s} reached {ok:5d} of {total:5d} ({share:5.1f}%)  peak motor torque {peaks[c]:.2f} N.m")
    env.close()


if __name__ == "__main__":
    main()
