#!/usr/bin/env python3
"""Does walking pay more than standing? The reward margin of a walk checkpoint (25/11/2026).

For each command, two groups of robots under the task's full reward, scored against that command: one
walks (the policy sees the command), the other stands (the policy sees a zero twist command, so the same
policy does what it does when told to stand, while the rewards still read the real command). Reported per
command: each group's reward per second, summed over every term and split by term where they differ most,
and falls. If standing scores close to walking, a run under these rewards drifts towards standing; a lower
KL only slows it (the user's review, 25/11/2026). The reward set is whatever the DUKKI_* switches make it
(DUKKI_YAW_ONLY, DUKKI_LIN_PROGRESS, DUKKI_BODY_ANG_VEL, DUKKI_FINE_LIN_VEL, ...), so run it once per set.

    DUKKI_KP=1 DUKKI_KD=0.05 DUKKI_KD_RANGE=0.07,0.07 DUKKI_CALIBRATION_SPREAD=0 DUKKI_FRICTION=1,3 \\
    DUKKI_FINE_LIN_VEL=2.0,0.1 DUKKI_FINE_ANG_VEL=1.5,0.3 [DUKKI_SOFT_GROUND=0.06,0.06,1] \\
        uv run scripts/dukki_walk_margin.py --checkpoint model_N.pt [--task ...]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402

import dukki_eval as de  # noqa: E402

DEFAULT_CMDS = ["fwd0.15:0.15:0:0", "back0.15:-0.15:0:0", "fwd0.30:0.30:0:0", "back0.30:-0.30:0:0", "turn1.0:0:0:1.0", "turn0.6:0:0:0.6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-Backlash-Dukki")
    ap.add_argument("--cmds", nargs="+", default=DEFAULT_CMDS, help="name:vx:vy:yaw")
    ap.add_argument("--per", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cmds = [(c.split(":")[0], [float(v) for v in c.split(":")[1:]]) for c in args.cmds]
    de.SCENARIOS = {f"{name}/{g}": tuple(v) for name, v in cmds for g in ("walk", "stand")}
    env, wrapped, runner, cmd = de.build(args.task, args.per, args.seconds + args.settle, device=args.device)
    runner.load(args.checkpoint, map_location=env.device)
    policy = runner.get_inference_policy(device=env.device)
    n = env.num_envs
    stand = torch.tensor([g == "stand" for name, v in cmds for g in ("walk", "stand") for _ in range(args.per)], device=env.device)

    om = env.observation_manager
    names, dims = om.active_terms["actor"], om.group_obs_term_dim["actor"]
    start = 0
    for t, d in zip(names, dims):
        size = 1
        for x in d:
            size *= x
        if t == "command":
            sl = slice(start, start + size)
        start += size

    rm = env.reward_manager
    terms = list(rm._term_names)
    wrapped.reset()
    env.reset(seed=0)
    obs = wrapped.get_observations()
    acc = torch.zeros(n, len(terms), device=env.device)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    settle, steps = int(args.settle / env.step_dt), int((args.seconds + args.settle) / env.step_dt)
    with torch.no_grad():
        for k in range(steps):
            de.pin_commands(env, cmd)
            o = obs["actor"].clone()
            o[stand, sl] = 0.0
            obs["actor"] = o
            obs, _, dones, extras = wrapped.step(policy(obs))
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            alive &= ~(dones.bool() & ~timeouts.bool())
            if k >= settle:
                acc += rm._step_reward * alive[:, None].float()
    rate = acc / (steps - settle)  # reward per second, per term
    print(f"== {args.label} {os.path.basename(args.checkpoint)} on {args.task}"
          f" soft={os.environ.get('DUKKI_SOFT_GROUND', '-')}")
    print(f"{'command':<10} {'walk /s':>8} {'stand /s':>9} {'margin':>8} {'falls w/s':>10}   biggest differences (walk - stand)")
    for i, (name, v) in enumerate(cmds):
        w = slice((2 * i) * args.per, (2 * i + 1) * args.per)
        s = slice((2 * i + 1) * args.per, (2 * i + 2) * args.per)
        rw, rs = rate[w].mean(0), rate[s].mean(0)
        diff = rw - rs
        top = torch.argsort(-diff.abs())[:4]
        fw, fs = int((~alive[w]).sum()), int((~alive[s]).sum())
        print(f"{name:<10} {rw.sum().item():8.2f} {rs.sum().item():9.2f} {(rw.sum() - rs.sum()).item():+8.2f} {fw:>4}/{fs:<4}   "
              + "  ".join(f"{terms[j]} {diff[j].item():+.2f}" for j in top))


if __name__ == "__main__":
    main()
