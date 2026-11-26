#!/usr/bin/env python3
"""Does the robot stand still after it stops walking? Per ground softness (26/11/2026).

On the robot (dukki docs/policy-testing.md, recordings scratch/policies/shake_*.json, 26/11/2026),
stopping on carpet leaves the trunk twisting at about 6 Hz, 1 rad/s RMS, in 70 to 75% of the
half-second windows, whichever network stands (the walk policy or the stand-up policy in robotd's
stand slot). On a hard floor the same policy settles 0.4 to 1.3 s after the stop and then stands
dead still (0.02 rad/s). None of the soft-ground evaluations measured this: they count falls.

Groups of robots, one per contact time constant (tasks/dukki_terrain.py, 1: 0.02 s is the hard
floor, 0.06 s about 5 mm of sinkage walking, the user's carpet), each: stand STAND_S, walk at the
command for WALK_S (ramped in over 0.3 s), stop over STOP_S, then stand AFTER_S. As robotd does,
the walk network runs while the command's magnitude is 0.05 or more and the stand network (if
given) below it. Reported for the stand after the stop, per group, with the robot's numbers to
compare: trunk rate RMS about each axis from 1 s after the stop (the robot on carpet: yaw 1.0,
hard floor 0.02 rad/s), the share of half-second windows whose yaw RMS is over 0.6 rad/s (carpet
71-75%, hard 22%, all in the first second), the settle time (the last |yaw rate| over 0.3 rad/s),
the leg target change a tick after the filter (carpet 0.14-0.20 rad, hard 0.002), the main
frequency of the yaw rate, and falls.

    cd microduck_rl && DUKKI_KP=1 DUKKI_KD=0.07 DUKKI_CALIBRATION_SPREAD=0 DUKKI_FRICTION=1,1.5 \\
        ROBOT_FILTERS=1 uv run python scripts/dukki_stillness_eval.py \\
        --walk ../policies/dukki_walk_mpush2_12750.onnx [--stand ../policies/dukki_standup_gentle_10250.onnx]

Switches (environment), as walk_profile: ROBOT_FILTERS=1 (robotd's low-pass, legs LEGS_ALPHA
default 0.7, head HEAD_ALPHA default 0.5), TASK (default Mjlab-Velocity-Flat-Backlash-Dukki),
EXTRA_DELAY=n adds n 5 ms physics steps to the servos' command delay (training: 1 to 2),
DAMP=r sets the contact's damping ratio (solref[1]; training always 1, critical) for a springier
ground that gives energy back, as carpet pile does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ["DUKKI_SOFT_GROUND"] = os.environ.get("DUKKI_SOFT_GROUND", "0.02,0.02,0")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import dukki_eval as de  # noqa: E402
from mjlab_microduck.tasks.dukki_terrain import randomize_contact_softness  # noqa: E402

STAND_S, WALK_S, AFTER_S = 2.0, 3.0, 6.0
_load = de.load_env_cfg


def _cfg(task):
    cfg = _load(task)
    if os.environ.get("ROBOT_FILTERS"):
        from dataclasses import fields

        from mjlab_microduck.tasks.lowpass_action import LowPassJointPositionActionCfg

        legs = float(os.environ.get("LEGS_ALPHA", "0.7"))
        head = float(os.environ.get("HEAD_ALPHA", "0.5"))
        a = cfg.actions["joint_pos"]
        kept = {f.name: getattr(a, f.name) for f in fields(a) if f.init and f.name != "lowpass"}
        cfg.actions["joint_pos"] = LowPassJointPositionActionCfg(
            **kept, lowpass={r".*(hip|knee|ankle).*": legs, r"(neck|head)_.*": head})
    extra = int(os.environ.get("EXTRA_DELAY", "0"))
    if extra:
        from copy import deepcopy

        robot = cfg.scene.entities["robot"]
        acts = []
        for act in robot.articulation.actuators:
            act = deepcopy(act)
            act.delay_min_lag += extra
            act.delay_max_lag += extra
            acts.append(act)
        robot.articulation.actuators = tuple(acts)
    return cfg


de.load_env_cfg = _cfg


class Policy:
    def __init__(self, path, runner, device):
        if path.endswith(".onnx"):
            self.f = de.OnnxPolicy(path)
        else:
            runner.load(path, map_location=device)
            self.f = runner.get_inference_policy(device=device)

    def __call__(self, obs):
        return self.f(obs)


def _feet_forces(sensor, robot):
    """Per robot: [internal horizontal force N, left foot sideways N, right foot sideways N, total vertical N].
    Internal = the part of the two feet's horizontal ground forces that cancels (0.5 (|L| + |R| - |L + R|)):
    the feet pushing against each other through the floor, which holds no weight up. Sideways is along the
    trunk's left (+y, heading frame), as the sensor reports it (net contact force, global frame)."""
    f = sensor.data.force  # (n, 2, 3), LEFT then RIGHT
    h = f[..., :2]
    internal = 0.5 * (h[:, 0].norm(dim=-1) + h[:, 1].norm(dim=-1) - (h[:, 0] + h[:, 1]).norm(dim=-1))
    q = robot.data.root_link_quat_w  # w, x, y, z
    w, x, y, z = q.unbind(-1)
    left = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z)], -1)
    left = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    side = (h * left[:, None, :]).sum(-1)  # (n, 2)
    return torch.stack([internal, side[:, 0], side[:, 1], f[..., 2].sum(-1)], -1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walk", required=True)
    ap.add_argument("--stand", help="robotd's stand slot; without it the walk network stands too")
    ap.add_argument("--tc", nargs="+", default=["0.02", "0.04", "0.06", "0.09", "0.12"],
                    help="groups: contact time constant, or tc:k:d to also set the sole twist hinges' stiffness N.m/rad and damping N.m.s/rad (needs DUKKI_SOLE_TWIST set, tasks/dukki_terrain.py 7)")
    ap.add_argument("--per", type=int, default=32)
    ap.add_argument("--cmd", default="0.3,0,0", help="vx,vy,yaw of the walk before the stop")
    ap.add_argument("--stop-s", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    walk_cmd = torch.tensor([float(v) for v in args.cmd.split(",")])
    groups = [tuple(float(v) for v in g.split(":")) for g in args.tc]
    de.SCENARIOS = {f"g{i}": (0.0, 0.0, 0.0) for i in range(len(groups))}
    torch.manual_seed(args.seed)
    total = STAND_S + WALK_S + args.stop_s + AFTER_S
    env, wrapped, runner, cmd = de.build(os.environ.get("TASK", "Mjlab-Velocity-Flat-Backlash-Dukki"),
                                         args.per, total, device=args.device)
    walk = Policy(args.walk, runner, env.device)
    stand = Policy(args.stand, runner, env.device) if args.stand else walk
    wrapped.reset()
    env.reset(seed=args.seed)
    n = env.num_envs
    gid = torch.tensor([i for i in range(len(groups)) for _ in range(args.per)], device=env.device)

    damp = float(os.environ.get("DAMP", "1.0"))

    def apply_tc():
        for i, g in enumerate(groups):
            ids = torch.nonzero(gid == i).flatten()
            randomize_contact_softness(env, ids, (g[0], g[0]), 1.0)
            if len(g) >= 3:
                from mjlab_microduck.tasks.dukki_terrain import set_sole_shear, set_sole_twist

                set_sole_twist(env, ids, torch.full((len(ids),), g[1], device=env.device),
                               torch.full((len(ids),), g[2], device=env.device))
                if len(g) == 5:
                    set_sole_shear(env, ids, torch.full((len(ids),), g[3], device=env.device),
                                   torch.full((len(ids),), g[4], device=env.device))
        if damp != 1.0:  # a springier contact: MuJoCo's solref damping ratio below critical
            geoms = env._dukki_soft_geoms
            solref = env.sim.model.geom_solref
            block = solref[:, geoms]
            block[..., 1] = damp
            solref[:, geoms] = block

    apply_tc()
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    names = list(term.target_names)
    legs = [i for i, nm in enumerate(names) if any(s in nm for s in ("hip", "knee", "ankle"))]
    dt = env.step_dt
    obs = wrapped.get_observations()
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    gyro, dleg, t_log, twist, brace, feet = [], [], [], [], [], []
    feet_sensor = env.scene["feet_ground_contact"]
    # Bracing: how far the hip roll targets sit past the joints while standing (the robot on carpet,
    # settled: 6-7 deg each, pushing the feet the same mirrored way; hard floor 2-3 deg).
    roll_names = ["left_hip_roll", "right_hip_roll"]
    roll_cols = [names.index(nm) for nm in roll_names]
    roll_jids = [list(robot.joint_names).index(nm) for nm in roll_names]
    tw_idx = [list(robot.joint_names).index(nm) for nm in ("passive_left_sole_twist", "passive_right_sole_twist")
              if nm in robot.joint_names]
    prev_target = None
    t_stop = STAND_S + WALK_S
    steps = int(round(total / dt))
    with torch.no_grad():
        for k in range(steps):
            t = k * dt
            if t < STAND_S:
                share = 0.0
            elif t < STAND_S + 0.3:
                share = (t - STAND_S) / 0.3
            elif t < t_stop:
                share = 1.0
            elif t < t_stop + args.stop_s:
                share = 1.0 - (t - t_stop) / args.stop_s
            else:
                share = 0.0
            c = cmd.clone()
            c[:] = walk_cmd.to(env.device) * share
            de.pin_commands(env, c)
            net = walk if float(torch.linalg.norm(c[0])) >= 0.05 else stand
            obs, _, dones, extras = wrapped.step(net(obs))
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            f = dones.bool() & ~timeouts.bool()
            if f.any():
                fell |= f
                apply_tc()
            target = term._processed_actions.clone()
            if prev_target is not None:
                dleg.append((target[:, legs] - prev_target[:, legs]).abs().max(1).values.cpu().numpy())
            else:
                dleg.append(np.zeros(n))
            prev_target = target
            gyro.append(robot.data.root_link_ang_vel_b.cpu().numpy().copy())
            brace.append((target[:, roll_cols] - robot.data.joint_pos[:, roll_jids]).cpu().numpy())
            feet.append(_feet_forces(feet_sensor, robot))
            if tw_idx:
                twist.append(robot.data.joint_pos[:, tw_idx].abs().max(1).values.cpu().numpy())
            t_log.append(t)
    gyro = np.stack(gyro)            # (T, n, 3)
    dleg = np.stack(dleg)            # (T, n)
    t_log = np.array(t_log)
    fell = fell.cpu().numpy()
    after = t_log >= t_stop + args.stop_s
    late = t_log >= t_stop + args.stop_s + 1.0
    before = (t_log >= 1.0) & (t_log < STAND_S)
    t0 = t_stop + args.stop_s
    win = int(round(0.5 / dt))
    print(f"walk {os.path.basename(args.walk)}, stand {os.path.basename(args.stand) if args.stand else '(walk)'}; "
          f"cmd {args.cmd} for {WALK_S} s, stop over {args.stop_s} s; filters "
          f"{'legs ' + os.environ.get('LEGS_ALPHA', '0.7') + ' head ' + os.environ.get('HEAD_ALPHA', '0.5') if os.environ.get('ROBOT_FILTERS') else 'off'}; "
          f"extra delay {os.environ.get('EXTRA_DELAY', '0')} x 5 ms; contact damping {os.environ.get('DAMP', '1.0')}; friction {os.environ.get('DUKKI_FRICTION', 'task default')}")
    print(f"{'group':>14} {'falls':>6} | before walk yaw | after stop, from 1 s: {'yaw':>5} {'roll':>5} {'pitch':>5} rad/s RMS"
          f" | windows >0.6 | settle s (median, p90) | leg dtarget (median, p90) | yaw Hz | sole twist deg walking, standing")
    twist = np.stack(twist) if twist else None
    brace = np.degrees(np.stack(brace))  # (T, n, 2)
    feet = np.stack(feet)  # (T, n, 4)
    walking = (t_log >= STAND_S + 0.5) & (t_log < t_stop)
    out = {}
    for i, label in enumerate(args.tc):
        grp = (gid == i).cpu().numpy()
        ids = grp & ~fell
        if not ids.any():
            print(f"{label:>14} {int(fell[grp].sum()):3d}/{args.per}  all fell")
            continue
        g = gyro[:, ids]
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
        yaw_b = rms(g[before, :, 2])
        yaw, roll, pitch = rms(g[late, :, 2]), rms(g[late, :, 0]), rms(g[late, :, 1])
        first = after & ~late
        f1 = np.sqrt(np.mean(g[first, :, 2] ** 2, axis=0))  # per robot, the first second after the stop
        first_s = f"first 1 s yaw RMS median {np.median(f1):.2f} p90 {np.percentile(f1, 90):.2f}, >0.6 {np.mean(f1 > 0.6) * 100:.0f}%"
        ga = g[after, :, 2]
        nwin = ga.shape[0] // win
        wr = np.sqrt(np.mean(ga[: nwin * win].reshape(nwin, win, -1) ** 2, axis=1))  # (nwin, robots)
        shaking = float(np.mean(wr > 0.6) * 100)
        settle = []
        for r in range(ga.shape[1]):
            above = np.flatnonzero(np.abs(ga[:, r]) > 0.3)
            settle.append((above[-1] + 1) * dt if len(above) else 0.0)
        settle = np.array(settle)
        dl = dleg[late][:, ids]
        spec = np.abs(np.fft.rfft((ga - ga.mean(0)) * np.hanning(len(ga))[:, None], axis=0)).mean(1)
        fr = np.fft.rfftfreq(len(ga), dt)
        band = fr > 1.0  # the shake, not the slow drift
        hz = float(fr[band][np.argmax(spec[band])])
        nfall = int(fell[grp].sum())
        print(f"{label:>14} {nfall:3d}/{args.per} | {yaw_b:15.3f} | {'':21}{yaw:5.2f} {roll:5.2f} {pitch:5.2f}          "
              f" | {shaking:10.0f}% | {np.median(settle):5.1f} {np.percentile(settle, 90):5.1f}            "
              f" | {np.median(dl):.4f} {np.percentile(dl, 90):.4f}         | {hz:4.1f}"
              + (f"   | {np.degrees(twist[walking][:, ids].max()):5.1f} {np.degrees(twist[late][:, ids].max()):5.1f}" if twist is not None else "")
              + f"   | {first_s}"
              + f" | hip roll target-joint deg L {brace[late][:, ids, 0].mean():+.1f} R {brace[late][:, ids, 1].mean():+.1f}"
              + f" | feet: squeeze {feet[late][:, ids, 0].mean():.2f} N (p90 {np.percentile(feet[late][:, ids, 0], 90):.2f}),"
                f" sideways L {feet[late][:, ids, 1].mean():+.2f} R {feet[late][:, ids, 2].mean():+.2f}, vertical {feet[late][:, ids, 3].mean():.1f} N;"
                f" first 1 s squeeze {feet[first][:, ids, 0].mean():.2f} (p90 {np.percentile(feet[first][:, ids, 0], 90):.2f}),"
                f" walking {feet[walking][:, ids, 0].mean():.2f} (p90 {np.percentile(feet[walking][:, ids, 0], 90):.2f})")
        out[label] = dict(falls=nfall, yaw_before=yaw_b, yaw=yaw, roll=roll, pitch=pitch, shaking_pct=shaking,
                            settle_median=float(np.median(settle)), settle_p90=float(np.percentile(settle, 90)),
                            dleg_median=float(np.median(dl)), first_yaw_median=float(np.median(f1)), first_yaw_p90=float(np.percentile(f1, 90)), dleg_p90=float(np.percentile(dl, 90)), yaw_hz=hz,
                          brace_l=float(brace[late][:, ids, 0].mean()), brace_r=float(brace[late][:, ids, 1].mean()),
                          squeeze_n=float(feet[late][:, ids, 0].mean()), squeeze_p90=float(np.percentile(feet[late][:, ids, 0], 90)),
                          side_l=float(feet[late][:, ids, 1].mean()), side_r=float(feet[late][:, ids, 2].mean()),
                          vertical_n=float(feet[late][:, ids, 3].mean()),
                          squeeze_first_n=float(feet[first][:, ids, 0].mean()), squeeze_walk_n=float(feet[walking][:, ids, 0].mean()),
                          squeeze_walk_p90=float(np.percentile(feet[walking][:, ids, 0], 90)))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"args": vars(args), "env": {k: v for k, v in os.environ.items() if k.startswith(("DUKKI_", "ROBOT_", "LEGS_", "HEAD_", "EXTRA_", "TASK", "DAMP"))},
                       "results": out}, fh, indent=1)
    env.close()


if __name__ == "__main__":
    main()
