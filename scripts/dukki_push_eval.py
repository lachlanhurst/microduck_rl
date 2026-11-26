#!/usr/bin/env python3
"""Push recovery of a Dukki walk checkpoint: one kick of a given size and direction, robot-like (25/11/2026).

Robots stand 3 s, then stand on or ramp over 1 s to their command (standing, walking forwards or backwards at
--speed) and hold it 2 s; then the trunk gets one velocity kick, as training's pushes do (added to the root
velocity), of each --sizes [m/s], backwards, forwards or sideways relative to the robot's heading. Reported per
command, direction and size: falls in the --after seconds following (the task's fall termination), and how many of
those fell backwards (trunk pitched back). Training kicks are +-0.3 m/s in a random world direction every 3-6 s.

    DUKKI_KP=1 DUKKI_KD=0.05 DUKKI_KD_RANGE=0.07,0.07 DUKKI_CALIBRATION_SPREAD=0 DUKKI_FRICTION=1,3 ROBOT_FILTERS=1 \\
        uv run scripts/dukki_push_eval.py --checkpoint model_N.pt [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402

import dukki_eval as de  # noqa: E402

DIRS = {"back": (-1.0, 0.0), "fwd": (1.0, 0.0), "side": (0.0, 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-Backlash-Dukki")
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--speed", type=float, default=0.15)
    ap.add_argument("--per", type=int, default=24)
    ap.add_argument("--after", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json")
    args = ap.parse_args()

    load = de.load_env_cfg

    def cfg_for_eval(task):
        cfg = load(task)
        if os.environ.get("ROBOT_FILTERS"):
            from mjlab_microduck.tasks.lowpass_action import LowPassJointPositionActionCfg

            a = cfg.actions["joint_pos"]
            kept = {f.name: getattr(a, f.name) for f in fields(a) if f.init and f.name != "lowpass"}
            cfg.actions["joint_pos"] = LowPassJointPositionActionCfg(
                **kept, lowpass={r".*(hip|knee|ankle).*": 0.7, r"(neck|head)_.*": 0.5}
            )
        return cfg

    de.load_env_cfg = cfg_for_eval
    cmds = {"stand": 0.0, "walk fwd": args.speed, "walk back": -args.speed}
    layout = [(c, d, s) for c in cmds for d in DIRS for s in args.sizes]
    de.SCENARIOS = {f"{i}": (0.0, 0.0, 0.0) for i in range(len(layout))}
    stand_s, ramp_s, hold_s = 3.0, 1.0, 2.0
    env, wrapped, runner, cmd = de.build(args.task, args.per, stand_s + ramp_s + hold_s + args.after, device=args.device)
    if args.checkpoint.endswith(".onnx"):
        policy = de.OnnxPolicy(args.checkpoint)
    else:
        runner.load(args.checkpoint, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
    n, dt = env.num_envs, env.step_dt
    rep = lambda vals: torch.tensor([v for v in vals for _ in range(args.per)], device=env.device)  # noqa: E731
    vx = rep([cmds[c] for c, d, s in layout])
    kdir = torch.tensor([DIRS[d] for c, d, s in layout for _ in range(args.per)], device=env.device)
    ksize = rep([s for c, d, s in layout])
    target = torch.zeros(n, 3, device=env.device)
    target[:, 0] = vx

    wrapped.reset()
    env.reset(seed=args.seed)
    robot = env.scene["robot"]
    obs = wrapped.get_observations()
    alive_before = torch.ones(n, dtype=torch.bool, device=env.device)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    fell_back = torch.zeros(n, dtype=torch.bool, device=env.device)
    kick_k = int((stand_s + ramp_s + hold_s) / dt)
    steps = kick_k + int(args.after / dt)
    with torch.no_grad():
        for k in range(steps):
            share = min(max(k - int(stand_s / dt), 0) / int(ramp_s / dt), 1.0)
            de.pin_commands(env, target * share)
            if k == kick_k:
                yaw = robot.data.heading_w
                c, s = torch.cos(yaw), torch.sin(yaw)
                vel = robot.data.root_link_vel_w.clone()
                vel[:, 0] += ksize * (c * kdir[:, 0] - s * kdir[:, 1])
                vel[:, 1] += ksize * (s * kdir[:, 0] + c * kdir[:, 1])
                robot.write_root_link_velocity_to_sim(vel)
            g = robot.data.projected_gravity_b.clone()
            obs, _, dones, extras = wrapped.step(policy(obs))
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            down = dones.bool() & ~timeouts.bool()
            if k < kick_k:
                alive_before &= ~down
            else:
                new = down & ~fell & alive_before
                fell_back |= new & (g[:, 0] < 0.0)  # gravity's body x negative: the trunk pitched back
                fell |= new

    results = {}
    print(f"falls after one kick, of robots up at the kick ({args.per} per cell); (b) = fell backwards")
    print(f"{'command':<10} {'kick':<5} " + "".join(f"{f'{s:.1f} m/s':>12}" for s in args.sizes))
    for c in cmds:
        for d in DIRS:
            row = []
            for s in args.sizes:
                i = layout.index((c, d, s))
                sl = slice(i * args.per, (i + 1) * args.per)
                up = int(alive_before[sl].sum())
                f, fb = int(fell[sl].sum()), int(fell_back[sl].sum())
                results[f"{c}/{d}/{s}"] = {"up": up, "fell": f, "fell_back": fb}
                row.append(f"{f}/{up} ({fb}b)")
            print(f"{c:<10} {d:<5} " + "".join(f"{r:>12}" for r in row))
    tot = sum(v["fell"] for v in results.values())
    print(f"TOTAL falls {tot}/{int(alive_before.sum())}, backwards {sum(v['fell_back'] for v in results.values())}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"checkpoint": args.checkpoint, "results": results}, fh, indent=1)


if __name__ == "__main__":
    main()
