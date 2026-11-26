#!/usr/bin/env python3
"""Can Dukki rise slowly from the sit-stand task's seat at all? An open-loop scripted rise (25/11/2026).

Two modes. ``--replay CKPT`` records a sit-stand policy's own rise and plays it back open loop, slowed
by each ``--stretch``: if the slowed rises still stand, a gentler rise exists along the policy's own path.
Without it, a scripted blend of the leg targets SIT -> HOME (below), which tipped every robot (25/11/2026:
past 45 deg 0.3 to 1.2 s into the blend, any lean), so it says little on its own.

The user wants the current sitting position kept, and three penalty fine-tunes of the rise (soft, soft10,
softrise, 22/11/2026) stayed at 0.3 to 0.4 m/s and 6.5 to 8 rad/s, the servos at their 1.0 N.m clamp.
If no slow open-loop rise works from this seat at kp 1, a gentle rise needs momentum and the penalties
were fighting the physics; if one does, the policy can be taught it.

Robots start seated (the task's SIT keyframe reset, its joint and tilt noise), hold the SIT targets for
``--settle`` s, then blend the leg targets to HOME over T s (smoothstep), optionally leaning forward on
the way (``--lean`` rad added to both hip pitches, a half-sine bump over the blend), then hold HOME for
2 s. Neck and head stay at HOME. For each T and lean: share standing at the end (trunk above 105 mm,
tilt under 20 deg), fell (the task's termination), peak upward trunk speed, peak trunk rate, time with a
joint within 5% of the torque clamp. The gear-play sit-stand task, pushes off, as eval-night.sh.

    DUKKI_KP=1 DUKKI_KD=0.05 DUKKI_KD_RANGE=0.07,0.07 DUKKI_CALIBRATION_SPREAD=0 DUKKI_FRICTION=2,2 \\
        uv run scripts/dukki_sitstand_rise_probe.py [--times 0.3 0.6 1 1.5 2 3] [--leans 0 0.1 0.2 0.3]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402

import dukki_eval as de  # noqa: E402
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SITTING_TARGET_OVERRIDES  # noqa: E402

TASK = "Mjlab-SitStand-Flat-Backlash-Dukki"
LEFT_HIP_PITCH, RIGHT_HIP_PITCH = 2, 11  # servo order; left forward lean is +, right -
LEGS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--leans", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--per", type=int, default=32)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--replay", help="a sit-stand checkpoint: record its rise from the seat, then replay it open loop "
                    "time-stretched by each of --stretch")
    ap.add_argument("--stretch", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0])
    args = ap.parse_args()

    load = de.load_env_cfg

    def seated(task):
        cfg = load(task)
        ground = cfg.events["set_ground_state"]
        for k in ("face_down", "face_up", "standing"):
            ground.params[f"{k}_prob"] = 0.0
        ground.params["sitting_prob"] = 1.0
        for name in [k for k, t in cfg.curriculum.items() if (t.params or {}).get("event_name") == "set_ground_state"]:
            del cfg.curriculum[name]
        gone = [k for k in cfg.events if "jerk" in k or "tip" in k]
        for name in gone:
            del cfg.events[name]
        for name in [k for k, t in cfg.curriculum.items() if (t.params or {}).get("event_name") in gone]:
            del cfg.curriculum[name]
        return cfg

    de.load_env_cfg = seated
    if args.replay:
        return replay(args)
    combos = [(t, l) for t in args.times for l in args.leans]
    de.SCENARIOS = {f"{t}/{l}": (0.0, 0.0, 0.0) for t, l in combos}
    longest = args.settle + max(args.times) + 2.0
    env, wrapped, runner, cmd = de.build(TASK, args.per, longest, device=args.device)
    n = env.num_envs
    T = torch.tensor([t for t, l in combos for _ in range(args.per)], device=env.device)
    lean = torch.tensor([l for t, l in combos for _ in range(args.per)], device=env.device)

    wrapped.reset()
    env.reset(seed=0)
    term = env.action_manager.get_term("joint_pos")
    home = term._offset.clone()  # (n, 14), default joint positions in servo order
    sit = home.clone()
    for i, v in SITTING_TARGET_OVERRIDES.items():
        sit[:, i] = v
    scale = term._scale
    robot = env.scene["robot"]
    act = robot.actuators[0]
    dt = env.step_dt
    settle_steps = int(args.settle / dt)
    steps = int(longest / dt)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    peak_vz = torch.zeros(n, device=env.device)
    peak_rate = torch.zeros(n, device=env.device)
    clamp_t = torch.zeros(n, device=env.device)
    tip_t = torch.full((n,), -1e9, device=env.device)
    twist = env.command_manager.get_term("twist")
    with torch.no_grad():
        for k in range(steps):
            # Posture flag (twist vx): stand, so the task's own terminations and resets read a rise.
            twist.vel_command_b[:] = 0.0
            t = (k - settle_steps) * dt
            s = torch.clamp(torch.full((n,), t, device=env.device) / T, 0.0, 1.0)
            s = s * s * (3 - 2 * s)  # smoothstep
            target = sit + s[:, None] * (home - sit)
            bump = lean * torch.sin(math.pi * s)
            target[:, LEFT_HIP_PITCH] += bump
            target[:, RIGHT_HIP_PITCH] -= bump
            target[:, [5, 6, 7, 8]] = home[:, [5, 6, 7, 8]]
            raw = (target - home) / scale
            _, _, dones, extras = wrapped.step(raw)
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            alive &= ~(dones.bool() & ~timeouts.bool())
            gk = robot.data.projected_gravity_b
            tk = torch.rad2deg(torch.arccos((-gk[:, 2] / gk.norm(dim=1)).clamp(-1, 1)))
            tip_t = torch.where((tk > 45.0) & (tip_t < -1e8), torch.full_like(tip_t, (k - settle_steps) * dt), tip_t)
            if k == settle_steps - 1:
                settle_tilt = tk.clone()
                settle_z = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).clone()
            if k >= settle_steps:
                vz = robot.data.root_link_lin_vel_w[:, 2]
                rate = torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=1)
                peak_vz = torch.where(alive, torch.maximum(peak_vz, vz), peak_vz)
                peak_rate = torch.where(alive, torch.maximum(peak_rate, rate), peak_rate)
                tau, limit = de.motor_torque_and_limit(env, act)
                clamp_t += ((tau > 0.95 * limit).any(dim=1) & alive).float() * dt
    z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    g = robot.data.projected_gravity_b
    tilt = torch.rad2deg(torch.arccos((-g[:, 2] / g.norm(dim=1)).clamp(-1, 1)))
    up = alive & (z > 0.105) & (tilt < 20.0)
    alive &= tip_t < -1e8  # the sit-stand task does not end a toppled episode; count tipping past 45 deg as a fall
    if os.environ.get("PROBE_DEBUG"):
        q = robot.data.joint_pos[:, term._target_ids]
        print("final z", z[:: args.per * len(args.leans)].tolist(), "tilt", tilt[:: args.per * len(args.leans)].tolist())
        print("after settle: tilt", settle_tilt.tolist(), "z", settle_z.tolist())
        print("tipped past 45 deg at (s from rise start)", tip_t.tolist())
        print("target-home max", (target - home).abs().max().item(), "joint err", (q - target)[0].tolist())
    print(f"{'T s':>5} {'lean':>5} {'stood':>7} {'fell':>6} {'peak vz m/s':>13} {'peak rate rad/s':>16} {'at clamp s':>11}")
    for t, l in combos:
        ids = (T == t) & (lean == l)
        ok = ids & up
        med = lambda x, m: float(x[m].median().item()) if m.any() else float("nan")  # noqa: E731
        print(
            f"{t:5.1f} {l:5.2f} {100 * up[ids].float().mean().item():6.1f}% {100 * (~alive[ids]).float().mean().item():5.1f}%"
            f" {med(peak_vz, ok):13.2f} {med(peak_rate, ok):16.1f} {med(clamp_t, ok):11.2f}"
        )


def replay(args):
    """Record the policy's rise (stand command from the seat) for every robot, then reset to the same robots
    and play each one's own actions back open loop, stretched in time by its group's factor."""
    stretches = args.stretch
    de.SCENARIOS = {f"x{f}": (0.0, 0.0, 0.0) for f in stretches}
    rise_s = 3.0
    longest = args.settle + rise_s * max(stretches) + 2.0
    env, wrapped, runner, cmd = de.build(TASK, args.per, longest, device=args.device)
    runner.load(args.replay, map_location=env.device)
    policy = runner.get_inference_policy(device=env.device)
    n, dt = env.num_envs, env.step_dt
    F = torch.tensor([f for f in stretches for _ in range(args.per)], device=env.device)
    robot = env.scene["robot"]
    act = robot.actuators[0]
    twist = env.command_manager.get_term("twist")
    settle_steps, rec_steps = int(args.settle / dt), int(rise_s / dt)

    def tilt_z():
        g = robot.data.projected_gravity_b
        tilt = torch.rad2deg(torch.arccos((-g[:, 2] / g.norm(dim=1)).clamp(-1, 1)))
        return tilt, robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]

    # Pass 1: the policy, sit command while settling, then stand.
    wrapped.reset()
    env.reset(seed=0)
    obs = wrapped.get_observations()
    rec = []
    with torch.no_grad():
        for k in range(settle_steps + rec_steps):
            twist.vel_command_b[:] = 0.0
            twist.vel_command_b[:, 0] = 1.0 if k < settle_steps else 0.0
            a = policy(obs)
            if k < settle_steps:
                first = a.clone()
            else:
                rec.append(a.clone())
            obs, *_ = wrapped.step(a)
    tilt, z = tilt_z()
    policy_up = (z > 0.105) & (tilt < 20.0)
    rec = torch.stack(rec)  # (steps, n, 14)

    # Pass 2: the same robots, the recorded actions stretched.
    wrapped.reset()
    env.reset(seed=0)
    obs = wrapped.get_observations()
    tipped = torch.zeros(n, dtype=torch.bool, device=env.device)
    peak_vz = torch.zeros(n, device=env.device)
    peak_rate = torch.zeros(n, device=env.device)
    clamp_t = torch.zeros(n, device=env.device)
    total = settle_steps + int(rise_s * max(stretches) / dt) + int(2.0 / dt)
    idx = torch.arange(n, device=env.device)
    with torch.no_grad():
        for k in range(total):
            twist.vel_command_b[:] = 0.0
            twist.vel_command_b[:, 0] = 1.0 if k < settle_steps else 0.0
            if k < settle_steps:
                a = policy(obs)  # the same settle as pass 1 (the policy, seated)
            else:
                u = (k - settle_steps) / F  # recorded step, fractional
                i0 = torch.clamp(u.floor().long(), max=rec_steps - 1)
                i1 = torch.clamp(i0 + 1, max=rec_steps - 1)
                w = (u - u.floor()).clamp(0, 1)[:, None]
                a = rec[i0, idx] * (1 - w) + rec[i1, idx] * w
            obs, *_ = wrapped.step(a)
            t, _ = tilt_z()
            tipped |= t > 45.0
            if k >= settle_steps:
                ok = ~tipped
                peak_vz = torch.where(ok, torch.maximum(peak_vz, robot.data.root_link_lin_vel_w[:, 2]), peak_vz)
                peak_rate = torch.where(ok, torch.maximum(peak_rate, torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=1)), peak_rate)
                tau, limit = de.motor_torque_and_limit(env, act)
                clamp_t += ((tau > 0.95 * limit).any(dim=1) & ok).float() * dt
    tilt, z = tilt_z()
    up = ~tipped & (z > 0.105) & (tilt < 20.0)
    print(f"policy {args.replay}: rose from the seat {100 * policy_up.float().mean().item():.1f}% (closed loop, pass 1)")
    print(f"{'stretch':>8} {'stood':>7} {'tipped':>7} {'peak vz m/s':>13} {'peak rate rad/s':>16} {'at clamp s':>11}")
    for f in stretches:
        ids = F == f
        ok = ids & up
        med = lambda x, m: float(x[m].median().item()) if m.any() else float("nan")  # noqa: E731
        print(f"{f:8.1f} {100 * up[ids].float().mean().item():6.1f}% {100 * tipped[ids].float().mean().item():6.1f}%"
              f" {med(peak_vz, ok):13.2f} {med(peak_rate, ok):16.1f} {med(clamp_t, ok):11.2f}")


if __name__ == "__main__":
    main()
