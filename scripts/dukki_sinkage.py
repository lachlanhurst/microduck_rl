#!/usr/bin/env python3
"""How far Dukki sinks into the ground for each contact time constant (tasks/dukki_terrain.py, 1).

Groups of robots, one per time constant, stand for 3 s and then walk at vx for 4 s under a
walking checkpoint; each group's trunk height against the hardest group's is the sinkage. The
user's carpet gives up to about 5 mm under a foot (24/11/2026).

    DUKKI_KP=1 DUKKI_KD=0.07 DUKKI_CALIBRATION_SPREAD=0 DUKKI_FRICTION=1,3 \\
        uv run scripts/dukki_sinkage.py --checkpoint path/model_N.pt [--tc 0.02 0.04 0.06 0.1 0.15]
"""

from __future__ import annotations

import argparse
import os

os.environ["DUKKI_SOFT_GROUND"] = os.environ.get("DUKKI_SOFT_GROUND", "0.02,0.02,0")

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402

import dukki_eval as de  # noqa: E402
from mjlab_microduck.tasks.dukki_terrain import randomize_contact_softness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-Backlash-Dukki")
    ap.add_argument("--tc", type=float, nargs="+", default=[0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15, 0.2])
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--vx", type=float, default=0.15)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    de.SCENARIOS = {f"tc{tc}": (0.0, 0.0, 0.0) for tc in args.tc}
    env, wrapped, runner, cmd = de.build(args.task, args.per, 10.0, device=args.device)
    if args.checkpoint.endswith(".onnx"):
        policy = de.OnnxPolicy(args.checkpoint)
    else:
        runner.load(args.checkpoint, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
    wrapped.reset()
    env.reset(seed=0)
    n = env.num_envs
    tcs = torch.tensor([tc for tc in args.tc for _ in range(args.per)], device=env.device)

    def apply_tc():
        for tc in args.tc:
            ids = torch.nonzero(tcs == tc).flatten()
            randomize_contact_softness(env, ids, (tc, tc), 1.0)

    apply_tc()
    robot = env.scene["robot"]
    obs = wrapped.get_observations()
    phases = [("stand", 3.0, 0.0), ("walk", 4.0, args.vx), ("back", 4.0, -args.vx)]
    out = {}
    falls = torch.zeros(n, device=env.device)
    with torch.no_grad():
        for name, secs, vx in phases:
            c = cmd.clone()
            c[:, 0] = vx
            steps = int(secs / env.step_dt)
            z = torch.zeros(n, device=env.device)
            zmin = torch.full((n,), 1e9, device=env.device)
            k0 = steps // 3
            for k in range(steps):
                de.pin_commands(env, c)
                obs, _, dones, extras = wrapped.step(policy(obs))
                timeouts = extras.get("time_outs", torch.zeros_like(dones))
                fell = dones.bool() & ~timeouts.bool()
                if fell.any():
                    falls += fell.float()
                    apply_tc()
                if k >= k0:
                    zz = robot.data.root_link_pos_w[:, 2]
                    z += zz
                    zmin = torch.minimum(zmin, zz)
            out[name] = (z / (steps - k0), zmin)
    ref = None
    print(f"{'tc [s]':>8} " + " ".join(f"{p[0] + ' sink mm':>16}" for p in phases) + "   falls")
    for tc in args.tc:
        ids = tcs == tc
        means = [out[p[0]][0][ids].mean().item() for p in phases]
        if ref is None:
            ref = means
        sink = [(r - m) * 1000 for r, m in zip(ref, means)]
        print(f"{tc:8.3f} " + " ".join(f"{s:16.2f}" for s in sink) + f"   {int(falls[ids].sum().item())}/{args.per}")


if __name__ == "__main__":
    main()
