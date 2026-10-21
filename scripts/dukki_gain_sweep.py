#!/usr/bin/env python3
"""Gain sweep for the Dukki (J288) velocity task.

Trains Mjlab-Velocity-Flat-Dukki once per (kp, kd) pair, one HF job each, so the
gains can be chosen from training results rather than on the robot. Gains are
passed as DUKKI_KP / DUKKI_KD (tasks/dukki.py), which --hf-jobs forwards.

Prints the commands by default. --launch submits them (each job bills GPU time).
--local runs them one after another on this machine instead, logged to wandb
(project --wandb-project, tagged gain-sweep) with their seeds fixed alike.

    uv run scripts/dukki_gain_sweep.py                       # print the grid
    uv run scripts/dukki_gain_sweep.py --launch --namespace <you>
    uv run scripts/dukki_gain_sweep.py --local --launch
    uv run scripts/dukki_gain_sweep.py --kp 1 2 3 --kd 0.05 --iterations 2000

Compare runs in wandb on the tracking terms (track_linear_velocity,
track_angular_velocity), episode length and action rate at equal iterations.

The bench identified kp 0.97 to 4.03 N.m/rad and kd 0.02 to 0.05 N.m.s/rad
(dukki docs/j288-testing.md section 15). kd above 0.05 extrapolates the fitted
range; the kd term is the firmware's own law on the filtered speed, so it should
extrapolate well, but check the chosen pair on the bench (Mjlab-Testbench-J288).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

TASK = "Mjlab-Velocity-Flat-Dukki"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kp", type=float, nargs="+", default=[1.0, 2.0, 4.0], help="kp values [N.m/rad]")
    ap.add_argument("--kd", type=float, nargs="+", default=[0.05, 0.08], help="kd values [N.m.s/rad]")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--flavor", default="l4x1", help="HF Jobs hardware flavor")
    ap.add_argument("--timeout", default="8h")
    ap.add_argument("--namespace", help="HF namespace to run under")
    ap.add_argument("--local", action="store_true", help="train here, one run after another")
    ap.add_argument("--wandb-project", default="dukki", help="wandb project for --local runs")
    ap.add_argument("--seed", type=int, default=42, help="seed for every --local run")
    ap.add_argument("--launch", action="store_true", help="submit the jobs (default: print only)")
    args = ap.parse_args()

    runs = []
    for kp in args.kp:
        for kd in args.kd:
            name = f"dukki-kp{kp:g}-kd{kd:g}".replace(".", "p")
            env = {"DUKKI_KP": f"{kp:g}", "DUKKI_KD": f"{kd:g}"}
            cmd = [
                "uv", "run", "train", args.task,
                "--env.scene.num-envs", str(args.num_envs),
                "--agent.max-iterations", str(args.iterations),
                "--agent.run-name", name,
            ]
            if args.local:
                cmd += [
                    "--agent.seed", str(args.seed),
                    "--agent.wandb-project", args.wandb_project,
                    "--agent.wandb-tags", "gain-sweep", f"kp{kp:g}", f"kd{kd:g}",
                ]
            else:
                cmd += [
                    "--hf-jobs", "--detach",
                    "--flavor", args.flavor,
                    "--timeout", args.timeout,
                    "--run-name", name,
                ]
                if args.namespace:
                    cmd += ["--namespace", args.namespace]
            runs.append((env, cmd))

    failed = []
    for env, cmd in runs:
        line = " ".join(f"{k}={v}" for k, v in env.items()) + " " + shlex.join(cmd)
        print(line, flush=True)
        if args.launch:
            result = subprocess.run(cmd, env={**os.environ, **env})
            if result.returncode != 0 and args.local:
                # A failed local run shouldn't cost the rest of the sweep.
                print(f"run failed ({result.returncode}), continuing", file=sys.stderr)
                failed.append(line)
            elif result.returncode != 0:
                print(f"submission failed ({result.returncode}), stopping", file=sys.stderr)
                return result.returncode
    if failed:
        print("failed runs:\n" + "\n".join(failed), file=sys.stderr)
        return 1
    if not args.launch:
        print(f"\n{len(runs)} runs. Add --launch to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
