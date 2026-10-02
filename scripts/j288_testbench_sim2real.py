#!/usr/bin/env python3
"""Sim2real validation of a Mjlab-Testbench-J288 policy on the J288 pendulum rig.

Runs one ONNX policy on a fixed schedule of target angles, either in the mjlab
training env (J288 BAM M6) or on the real servo over Unitree's USB to
single-bus adapter, logs both at 200 Hz, and plots one against the other.

    # 1) Sim, the same env the policy trained in:
    uv run scripts/j288_testbench_sim2real.py --mode sim --onnx policy.onnx --out sim.npz
    # 2) Real: the rig from dukki docs/j288-testing.md section 15, the arm and
    #    mass matching the setup the policy trained on (DUKKI_BENCH_SETUP), the
    #    hanging zero from `python -m bam.unitree.calibrate`:
    uv run scripts/j288_testbench_sim2real.py --mode real --onnx policy.onnx --out real.npz \\
        --port /dev/cu.usbmodem3744CBC819741 --zero-file zero_150mm.json
    # 3) Compare:
    uv run scripts/j288_testbench_sim2real.py --compare sim.npz real.npz --out-plot j288.png

Observation (must match the env): [joint_pos, joint_vel, last_action, command],
joint_pos from the arm hanging, joint_vel the position difference over one
50 Hz policy tick (tasks/dukki.py), as the robot's bridge would compute it.
Action: position offset [rad] from 0, scale 1.

On the real servo the policy runs at 50 Hz and the held command is resent
every 5 ms between ticks: the reply to each frame is the 200 Hz log, and the
traffic keeps the servo's frame timeout from latching. The gains go out in
every frame, kp/kd as the env trained with (DUKKI_KP / DUKKI_KD, or
--kp/--kd).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

CONTROL_DT = 0.02  # decimation 4 x 5 ms
SIM_DT = 0.005
DECIM = 4
MAX_ANGLE = math.radians(80.0)
LOG_KEYS = ("t", "target", "q", "qd", "action", "goal", "torque", "winding_temp")


def make_target_schedule(total_time: float, hold_time: float = 4.0, seed: int = 0) -> np.ndarray:
    """One target angle per policy step, held for hold_time."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(total_time / CONTROL_DT))
    per_hold = int(round(hold_time / CONTROL_DT))
    targets = np.zeros(n_steps, dtype=np.float32)
    for i in range(0, n_steps, per_hold):
        targets[i : i + per_hold] = rng.uniform(-MAX_ANGLE, MAX_ANGLE)
    return targets


class Policy:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(onnx_path)
        self.in_name = self.session.get_inputs()[0].name
        print(f"policy {onnx_path}: input {self.in_name} {self.session.get_inputs()[0].shape}")
        self.last_action = 0.0

    def __call__(self, obs: np.ndarray) -> float:
        out = self.session.run(None, {self.in_name: obs.astype(np.float32)[None, :]})[0]
        self.last_action = float(np.asarray(out).reshape(-1)[0])
        return self.last_action


def _empty_log(n: int) -> dict:
    return {k: np.zeros(n, dtype=np.float32) for k in LOG_KEYS}


# ---------------------------------------------------------------------------
# Sim: the mjlab training env, stepped by hand to log at the physics rate
# ---------------------------------------------------------------------------


def rollout_sim(onnx_path: str, total_time: float, seed: int, kp, kd, smooth=True, setup=None) -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from mjlab_microduck.tasks.j288_testbench_env_cfg import make_j288_testbench_env_cfg

    cfg = make_j288_testbench_env_cfg(play=True, setup=setup, kp=kp, kd=kd, smooth=smooth)
    cfg.scene.num_envs = 1
    cfg.commands["target_angle"].resampling_time_range = (1e6, 1e6)
    cfg.episode_length_s = max(total_time + 10.0, cfg.episode_length_s)
    cfg.observations["actor"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
    env.reset(seed=seed)
    cmd = env.command_manager.get_term("target_angle")
    robot = env.scene["robot"]
    act = robot.actuators[0]

    policy = Policy(onnx_path)
    targets = make_target_schedule(total_time, seed=seed)
    rec = _empty_log(len(targets) * DECIM)
    t, i = 0.0, 0
    for target in targets:
        cmd._target[0, 0] = float(target)
        obs = env.observation_manager.compute(update_history=True)["actor"][0].cpu().numpy()
        action = policy(obs)
        env.action_manager.process_action(torch.tensor([[action]], device=env.device))
        for _ in range(DECIM):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()  # runs the J288 actuator
            rec["t"][i], rec["target"][i] = t, target
            rec["q"][i] = robot.data.joint_pos[0, 0].item()
            rec["qd"][i] = robot.data.joint_vel[0, 0].item()
            rec["action"][i], rec["goal"][i] = action, action
            rec["torque"][i] = act.reported_torque[0, 0].item()
            i += 1
            env.sim.step()
            env.scene.update(dt=env.physics_dt)
            t += env.physics_dt
    env.close()
    return rec


# ---------------------------------------------------------------------------
# Real: the J288 over Unitree's USB adapter (bam fork driver)
# ---------------------------------------------------------------------------


def rollout_real(
    onnx_path: str,
    total_time: float,
    seed: int,
    port: str,
    motor_id: int,
    zero_file: str,
    kp: float,
    kd: float,
    limit: float,
    max_winding_temp: float,
) -> dict:
    from bam.unitree.digital_servo import MODE_FOC, DigitalServo, sent_gains
    from bam.unitree.record import load_zero

    servo = DigitalServo(port, motor_id)
    state = servo.recover()
    zero, _ = load_zero(servo, zero_file)
    # Pre-flight: braked (mode 0) and hanging, the arm must read about 0, or
    # the calibration is stale (arm remounted on another spline tooth).
    hanging = (servo.stop() or state)["position"] - zero
    if abs(hanging) > 0.1:
        servo.close()
        raise SystemExit(
            f"arm reads {hanging:+.3f} rad at rest; let it hang still, or recalibrate "
            "with python -m bam.unitree.calibrate"
        )
    kp_sent, kd_sent = sent_gains(kp, kd)
    print(f"zero (hanging) at rotor position {zero:+.4f} rad; kp {kp_sent:.4f}, kd {kd_sent:.5f}")

    policy = Policy(onnx_path)
    targets = make_target_schedule(total_time, seed=seed)
    rec = _empty_log(len(targets) * DECIM)

    def send(goal: float) -> dict:
        nonlocal state
        reply = servo.command(MODE_FOC, position=goal + zero, kp=kp, kd=kd)
        if reply is not None:
            state = reply
        return state

    # Start hanging, holding 0, so the first tick's position difference is clean.
    for _ in range(50):
        send(0.0)
        time.sleep(SIM_DT)
    q_prev = state["position"] - zero
    t0 = time.perf_counter()
    i = 0
    try:
        for k, target in enumerate(targets):
            tick = t0 + k * CONTROL_DT
            while time.perf_counter() < tick:
                pass
            q = state["position"] - zero
            qd = (q - q_prev) / CONTROL_DT
            q_prev = q
            obs = np.array([q, qd, policy.last_action, target], dtype=np.float32)
            goal = policy(obs)
            for j in range(DECIM):
                sub = tick + j * SIM_DT
                while time.perf_counter() < sub:
                    pass
                s = send(goal)
                rec["t"][i] = time.perf_counter() - t0
                rec["target"][i] = target
                rec["q"][i] = s["position"] - zero
                rec["qd"][i] = s["speed"]
                rec["action"][i], rec["goal"][i] = goal, goal
                rec["torque"][i] = s["torque"]
                rec["winding_temp"][i] = s["winding_temp"]
                i += 1
            if abs(q) > limit:
                raise RuntimeError(f"arm at {q:+.3f} rad, beyond the {limit} rad limit")
            if s["winding_temp"] > max_winding_temp:
                raise RuntimeError(f"winding at {s['winding_temp']} degC")
            if k % 50 == 0 or target != targets[k - 1]:
                print(
                    f"t={rec['t'][i - 1]:6.2f}s target={math.degrees(target):+6.1f}° "
                    f"q={math.degrees(q):+6.1f}° torque={s['torque']:+.3f} N.m "
                    f"winding={s['winding_temp']} °C"
                )
    except RuntimeError as e:
        print(f"ABORTED: {e}; keeping the {i} samples logged so far")
    finally:
        servo.close()  # mode 0
    return {key: v[:i] for key, v in rec.items()}


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def compare_and_plot(sim_file: str, real_file: str, out_path: str) -> None:
    import matplotlib.pyplot as plt

    sim, real = dict(np.load(sim_file)), dict(np.load(real_file))
    n = min(len(sim["t"]), len(real["t"]))
    t = sim["t"][:n]

    def mae(a, b):
        return float(np.mean(np.abs(a[:n] - b[:n])))

    print(f"samples compared      : {n}")
    print(f"position, sim vs real : {mae(sim['q'], real['q']):.4f} rad")
    print(f"tracking error, sim   : {mae(sim['q'], sim['target']):.4f} rad")
    print(f"tracking error, real  : {mae(real['q'], real['target']):.4f} rad")
    print(f"torque, sim vs real   : {mae(sim['torque'], real['torque']):.4f} N.m")
    print(f"action, sim vs real   : {mae(sim['action'], real['action']):.4f} rad")
    if real["winding_temp"].any():
        print(f"winding, real         : {real['winding_temp'][0]:.0f} -> {real['winding_temp'][n - 1]:.0f} °C")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, sim["target"][:n], "k-", lw=1, alpha=0.4, label="target")
    axes[0].plot(t, sim["q"][:n], "b-", lw=1.2, label="sim")
    axes[0].plot(t, real["q"][:n], "r-", lw=1.2, label="real")
    axes[0].set_ylabel("position [rad]")
    axes[0].set_title(f"J288 testbench, position sim vs real {mae(sim['q'], real['q']):.4f} rad")
    axes[1].plot(t, sim["qd"][:n], "b-", lw=1, label="sim (true)")
    axes[1].plot(t, real["qd"][:n], "r-", lw=1, label="real (reported)")
    axes[1].set_ylabel("speed [rad/s]")
    axes[2].plot(t, sim["torque"][:n], "b-", lw=1, label="sim")
    axes[2].plot(t, real["torque"][:n], "r-", lw=1, label="real")
    axes[2].set_ylabel("reported torque [N.m]")
    axes[3].plot(t, sim["action"][:n], "b-", lw=1, label="sim")
    axes[3].plot(t, real["action"][:n], "r-", lw=1, label="real")
    axes[3].set_ylabel("action [rad]")
    axes[3].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


def main() -> None:
    from mjlab_microduck.tasks.dukki import dukki_gains_from_env

    env_kp, env_kd = dukki_gains_from_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "real"])
    ap.add_argument("--onnx")
    ap.add_argument("--out")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kp", type=float, default=env_kp, help="[N.m/rad], as trained")
    ap.add_argument("--kd", type=float, default=env_kd, help="[N.m.s/rad], as trained")
    ap.add_argument("--setup", help="sim rig, a J288_BENCH_SETUPS key (default DUKKI_BENCH_SETUP or l150_m108)")
    ap.add_argument("--port", default="/dev/cu.usbmodem3744CBC819741")
    ap.add_argument("--motor-id", type=int, default=0)
    ap.add_argument("--zero-file", help="hanging-angle calibration from bam.unitree.calibrate")
    ap.add_argument("--limit", type=float, default=1.8, help="abort beyond this angle [rad]")
    ap.add_argument("--max-winding-temp", type=float, default=80.0, help="abort above [°C]")
    ap.add_argument("--compare", nargs=2, metavar=("SIM_NPZ", "REAL_NPZ"))
    ap.add_argument("--out-plot", default="j288_testbench_sim2real.png")
    args = ap.parse_args()

    if args.compare:
        compare_and_plot(*args.compare, args.out_plot)
        return
    if not (args.mode and args.onnx and args.out):
        ap.error("--mode, --onnx and --out are needed for a rollout")
    if args.mode == "sim":
        rec = rollout_sim(args.onnx, args.duration, args.seed, args.kp, args.kd, setup=args.setup)
    else:
        if not args.zero_file:
            ap.error("--zero-file is needed on the real servo")
        rec = rollout_real(
            args.onnx, args.duration, args.seed, args.port, args.motor_id,
            args.zero_file, args.kp, args.kd, args.limit, args.max_winding_temp,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **rec)
    err = float(np.mean(np.abs(rec["q"] - rec["target"])))
    print(f"saved {len(rec['t'])} samples to {out}; tracking error {err:.4f} rad ({math.degrees(err):.2f}°)")


if __name__ == "__main__":
    main()
