#!/usr/bin/env python3
"""How reliably a stand-up checkpoint gets up, from each starting posture, across many robots.

For each start (face down, face up, sitting) the task is run with that start forced, pushes off,
for --duration seconds with no resets, and every robot is judged:

    upright    trunk above UPRIGHT_MM and tilted less than UPRIGHT_DEG (a body propped on its
               head reaches standing height pitched well over, so height alone is not enough)
    got up     upright at some point; time to the first upright step
    stays up   upright for the whole last second
    at limit   time with any joint's motor torque within 5% of the 1.0 N.m clamp
    spin       fastest yaw rate

    DUKKI_KP=1 DUKKI_KD=0.05 uv run scripts/dukki_standup_eval.py \\
        --task Mjlab-StandUp-Flat-Dukki-Distill --checkpoint path/model_3999.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks.dukki import check_cfg_matches_run
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

UPRIGHT_MM = 105.0
UPRIGHT_DEG = 20.0
STARTS = ("face_down", "face_up", "sitting")


def build(task: str, start: str, envs: int, duration: float, device: str):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = envs
    cfg.episode_length_s = duration + 10.0
    removed = [k for k in cfg.events if "push" in k or "topple" in k]
    for name in removed:
        del cfg.events[name]
    ground = cfg.events["set_ground_state"]
    for key in ("face_down", "face_up", "sitting", "standing"):
        ground.params[f"{key}_prob"] = 1.0 if key == start else 0.0
    for name in [
        k
        for k, t in cfg.curriculum.items()
        if (t.params or {}).get("event_name") in [*removed, "set_ground_state"]
    ]:
        del cfg.curriculum[name]
    return cfg, ManagerBasedRlEnv(cfg=cfg, device=device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-StandUp-Flat-Dukki-Distill")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--duration", type=float, default=6.0, help="[s]")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"== {args.checkpoint} on {args.task}, {args.envs} robots per start, {args.duration:.0f} s, pushes off")
    for start in STARTS:
        torch.manual_seed(args.seed)
        cfg, env = build(args.task, start, args.envs, args.duration, args.device)
        check_cfg_matches_run(cfg, args.checkpoint)
        rl_cfg = load_rl_cfg(args.task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
        runner = (load_runner_cls(args.task) or OnPolicyRunner)(wrapped, asdict(rl_cfg), device=args.device)
        runner.load(args.checkpoint, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        wrapped.reset()
        env.reset(seed=args.seed)
        obs = wrapped.get_observations()
        robot = env.scene["robot"]
        act = robot.actuators[0]
        n, dt = args.envs, env.step_dt
        steps = int(args.duration / dt)
        last_second = int(1.0 / dt)

        first_up = torch.full((n,), float("nan"), device=env.device)
        up_at_end = torch.ones(n, dtype=torch.bool, device=env.device)
        at_limit = torch.zeros(n, device=env.device)
        spin = torch.zeros(n, device=env.device)
        with torch.no_grad():
            for k in range(steps):
                obs, *_ = wrapped.step(policy(obs))
                z = robot.data.root_link_pos_w[:, 2] * 1000
                g = robot.data.projected_gravity_b
                tilt = torch.rad2deg(torch.arccos((-g[:, 2] / g.norm(dim=1)).clamp(-1, 1)))
                up = (z > UPRIGHT_MM) & (tilt < UPRIGHT_DEG)
                first_up = torch.where(up & first_up.isnan(), torch.full_like(first_up, (k + 1) * dt), first_up)
                if k >= steps - last_second:
                    up_at_end &= up
                at_limit += (act.reported_torque.abs().amax(dim=1) > 0.95).float() * dt
                spin = torch.maximum(spin, robot.data.root_link_ang_vel_b[:, 2].abs())
        got = ~first_up.isnan()
        t = first_up[got]
        q = lambda p: float(t.quantile(p)) if t.numel() else float("nan")  # noqa: E731
        print(
            f"  {start:9s} got up {int(got.sum()):4d}/{n} ({100 * got.float().mean():5.1f}%)"
            f"  stays up {100 * up_at_end.float().mean():5.1f}%"
            f"  time to upright median {q(0.5):.2f} s, 90% {q(0.9):.2f} s"
            f"  at limit median {float(at_limit.median()):.2f} s, 90% {float(at_limit.quantile(0.9)):.2f} s"
            f"  spin median {float(spin.median()):.1f}, 90% {float(spin.quantile(0.9)):.1f} rad/s"
        )
        env.close()


if __name__ == "__main__":
    main()
