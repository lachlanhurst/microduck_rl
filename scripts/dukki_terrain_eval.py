#!/usr/bin/env python3
"""Dukki walking over its terrain (tasks/dukki_terrain.py), robot-like, pushes off.

Every terrain type at ``--levels`` difficulty rows (row r spans difficulty r/levels to
(r+1)/levels: steps and obstacles up to that share of DUKKI_TERRAIN_STEP, slopes of
DUKKI_TERRAIN_SLOPE), robots starting on each tile's 1 m centre platform facing a random way,
standing ``--stand`` s, ramping over ``--ramp`` s and then commanded straight forwards or
straight backwards for ``--seconds`` (walk_test.py's shape). Reported per type and row:
falls, and progress (distance from the start over the commanded distance). The tiles are the
same for every checkpoint (``--seed``).

    DUKKI_KP=1 DUKKI_KD=0.07 DUKKI_CALIBRATION_SPREAD=0 DUKKI_FRICTION=1,3 ROBOT_FILTERS=1 \\
        uv run scripts/dukki_terrain_eval.py --checkpoint path/model_N.pt [--json out.json]

ROBOT_FILTERS=1 adds robotd's target low-pass (legs 0.7, head 0.5) as walk_profile.py does.
DUKKI_SOFT_GROUND=tc,tc,1 puts every robot on ground of that softness.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Terrain-Backlash-Dukki")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--per", type=int, default=12, help="robots per type, row and direction")
    ap.add_argument("--speed", type=float, default=0.15)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--stand", type=float, default=3.0, help="seconds standing before the walk, as walk_test.py")
    ap.add_argument("--ramp", type=float, default=1.0, help="seconds ramping up to the speed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json")
    ap.add_argument("--plane", action="store_true", help="control: the same protocol on a plane (one 'flat' type)")
    ap.add_argument("--iterations", type=int, help="override the solver iterations (terrain task: 30)")
    ap.add_argument("--speeds", action="store_true",
                    help="also report actuated joint speeds while walking (alive robots): each robot's peak, and the "
                         "share of robot-time above 8, 10.8 (XL330-M288 no-load at 5 V) and 18.6 rad/s (J288 identified)")
    ap.add_argument("--pushes", action="store_true", help="keep the task's random pushes (the eval removes them)")
    ap.add_argument("--edge", type=float, default=0.0,
                    help="start this far [m] from the tile centre along x (the hills' slope axis), on the start "
                         "platform's edge: facing out to walk forwards, backed up to the terrain to walk backwards. "
                         "0: the old start, the centre at a random heading (most of a walk on the platform)")
    args = ap.parse_args()

    load = de.load_env_cfg

    def cfg_for_eval(task):
        cfg = load(task)
        cfg.curriculum.pop("terrain_levels", None)
        if args.pushes:  # dukki_eval.build drops every event named *push*; rename the task's to keep it
            for k in [k for k in cfg.events if "push" in k]:
                cfg.events[k.replace("push", "shove")] = cfg.events.pop(k)
                for c in cfg.curriculum.values():
                    if (c.params or {}).get("event_name") == k:
                        c.params["event_name"] = k.replace("push", "shove")
        if args.iterations:
            cfg.sim.mujoco.iterations = args.iterations
        if args.plane:
            cfg.scene.terrain.terrain_type = "plane"
            cfg.scene.terrain.terrain_generator = None
        else:
            gen = cfg.scene.terrain.terrain_generator
            gen.curriculum = True
            gen.num_rows = args.levels
            gen.seed = args.seed
        if os.environ.get("ROBOT_FILTERS"):
            from mjlab_microduck.tasks.lowpass_action import LowPassJointPositionActionCfg

            a = cfg.actions["joint_pos"]
            kept = {f.name: getattr(a, f.name) for f in fields(a) if f.init and f.name != "lowpass"}
            legs = float(os.environ.get("LEGS_ALPHA", "0.7"))
            cfg.actions["joint_pos"] = LowPassJointPositionActionCfg(
                **kept, lowpass={r".*(hip|knee|ankle).*": legs, r"(neck|head)_.*": 0.5}
            )
        return cfg

    de.load_env_cfg = cfg_for_eval
    types = ["plane"] if args.plane else list(cfg_for_eval(args.task).scene.terrain.terrain_generator.sub_terrains)
    dirs = (args.speed, -args.speed)
    layout = [(t, r, d) for t in range(len(types)) for r in range(args.levels) for d in range(len(dirs))]
    de.SCENARIOS = {f"{i}": (0.0, 0.0, 0.0) for i in range(len(layout))}
    env, wrapped, runner, cmd = de.build(args.task, args.per, args.seconds + args.stand + args.ramp, device=args.device)
    if args.checkpoint.endswith(".onnx"):
        policy = de.OnnxPolicy(args.checkpoint)
    else:
        runner.load(args.checkpoint, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)

    terrain = env.scene.terrain
    n = env.num_envs
    t_of = torch.tensor([t for (t, r, d) in layout for _ in range(args.per)], device=env.device)
    r_of = torch.tensor([r for (t, r, d) in layout for _ in range(args.per)], device=env.device)
    d_of = torch.tensor([d for (t, r, d) in layout for _ in range(args.per)], device=env.device)
    vx = torch.tensor(dirs, device=env.device)[d_of]
    cmd = torch.zeros(n, 3, device=env.device)
    cmd[:, 0] = vx
    if not args.plane:
        terrain.terrain_levels[:] = r_of
        terrain.terrain_types[:] = t_of
        terrain.env_origins[:] = terrain.terrain_origins[r_of, t_of]

    wrapped.reset()
    env.reset(seed=args.seed)
    robot = env.scene["robot"]
    if args.edge > 0.0:
        # Out along +x or -x (alternating), facing out to walk forwards, in to walk backwards.
        out = torch.where(torch.arange(n, device=env.device) % 2 == 0, 1.0, -1.0)
        pose = robot.data.root_link_pose_w.clone()
        pose[:, 0] = env.scene.env_origins[:, 0] + args.edge * out
        pose[:, 1] = env.scene.env_origins[:, 1]
        facing_out = vx > 0
        heading = torch.where((out > 0) == facing_out, 0.0, math.pi)
        pose[:, 3] = torch.cos(heading / 2)
        pose[:, 4:6] = 0.0
        pose[:, 6] = torch.sin(heading / 2)
        robot.write_root_link_pose_to_sim(pose)
    obs = wrapped.get_observations()
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    fall_t = torch.full((n,), -1.0, device=env.device)
    reach = torch.zeros(n, device=env.device)
    stand_steps, ramp_steps = int(args.stand / env.step_dt), int(args.ramp / env.step_dt)
    steps = stand_steps + ramp_steps + int(args.seconds / env.step_dt)
    p0 = robot.data.root_link_pos_w[:, :2].clone()
    act = robot.actuators[0]
    names = list(act.target_names)
    peak = torch.zeros(n, len(names), device=env.device)
    over = {b: torch.zeros(n, device=env.device) for b in (8.0, 10.8, 18.6)}
    walk_steps = torch.zeros(n, device=env.device)
    with torch.no_grad():
        for k in range(steps):
            share = min(max(k - stand_steps, 0) / max(ramp_steps, 1), 1.0)
            de.pin_commands(env, cmd * share)
            obs, _, dones, extras = wrapped.step(policy(obs))
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            fell = dones.bool() & ~timeouts.bool() & alive
            fall_t = torch.where(fell, torch.full_like(fall_t, k * env.step_dt), fall_t)
            alive &= ~fell
            if k == stand_steps - 1:
                p0 = robot.data.root_link_pos_w[:, :2].clone()
            if k >= stand_steps and args.speeds:
                qd = robot.data.joint_vel[:, act._target_ids_list].abs()
                live = alive.float()
                peak = torch.maximum(peak, qd * live[:, None])
                top = qd.max(dim=1).values
                for b in over:
                    over[b] += (top > b).float() * live
                walk_steps += live
            if k >= stand_steps:
                d = torch.linalg.norm(robot.data.root_link_pos_w[:, :2] - p0, dim=1)
                reach = torch.where(alive, torch.maximum(reach, d), reach)
    standing_falls = int(((fall_t >= 0) & (fall_t < args.stand)).sum().item())
    print(f"falls while standing (first {args.stand:.0f} s): {standing_falls}")
    progress = reach / (args.speed * (args.seconds + args.ramp / 2))

    results = {}
    print(f"{'terrain':>10} {'row':>4} " + "".join(f"{'fwd falls':>11}{'fwd prog':>10}{'back falls':>12}{'back prog':>10}"))
    for t, name in enumerate(types):
        for r in range(args.levels):
            row = {}
            for d, label in enumerate(("fwd", "back")):
                ids = (t_of == t) & (r_of == r) & (d_of == d)
                row[label] = {
                    "falls": int((~alive[ids]).sum().item()),
                    "n": int(ids.sum().item()),
                    "progress": float(progress[ids].mean().item()),
                }
            results[f"{name}/{r}"] = row
            print(
                f"{name:>10} {r:>4} "
                f"{row['fwd']['falls']:>6}/{row['fwd']['n']:<4}{row['fwd']['progress']:>10.2f}"
                f"{row['back']['falls']:>7}/{row['back']['n']:<4}{row['back']['progress']:>10.2f}"
            )
    if args.speeds:
        rp = peak.max(dim=1).values
        q = torch.quantile(rp, torch.tensor([0.5, 0.9, 0.99], device=env.device))
        print(f"SPEEDS peak joint speed per robot: median {q[0]:.1f}, 90% {q[1]:.1f}, 99% {q[2]:.1f}, max {rp.max():.1f} rad/s"
              f" ({names[int(peak.max(dim=0).values.argmax())]}); robot-time above 8 / 10.8 / 18.6 rad/s: "
              + " / ".join(f"{100 * (over[b].sum() / walk_steps.sum().clamp(min=1)).item():.2f}%" for b in over)
              + "; robots ever above: " + " / ".join(f"{100 * (peak.max(dim=1).values > b).float().mean().item():.0f}%" for b in over))
        jm = peak.max(dim=0).values
        print("SPEEDS per joint max: " + ", ".join(f"{nm} {v:.1f}" for nm, v in zip(names, jm.tolist())))
    total_falls = int((~alive).sum().item())
    print(f"TOTAL falls {total_falls}/{n}, mean progress {progress.mean().item():.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "types": types, "levels": args.levels, "results": results,
                       "total_falls": total_falls, "standing_falls": standing_falls, "n": n}, f, indent=1)


if __name__ == "__main__":
    main()
