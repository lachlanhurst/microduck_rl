#!/usr/bin/env python3
"""Fixed-command evaluation of Dukki walking checkpoints, on CPU.

Each checkpoint walks the same set of commands on the same robot model, with
the training domain randomization kept (friction, masses, armature, supply,
delay, encoder bias) but pushes off, so checkpoints and robot models can be
compared like for like. Every env re-resets with the same seed before each
checkpoint, so all checkpoints see identical randomization.

    uv run scripts/dukki_eval.py --task Mjlab-Velocity-Flat-Dukki \\
        --checkpoints flat_2999=path/model_2999.pt bl_2999=path/model_2999.pt

A checkpoint may also be an exported ``.onnx`` policy (61-D obs in, 14 actions
out), e.g. Pollen's XL330 ``alpha_walking.onnx`` on Mjlab-Velocity-Flat-MicroDuck.

Per scenario, after a settling period: net speed (distance covered over the
window) and net yaw rate (heading turned over the window), which separate real
motion from the body's sway, the mean velocity tracking error, falls, motor
torque (peak, RMS of the worst joint, and the share of time the worst joint
spends within 5% of its limit: the J288's 1.0 N.m clamp, or for a
voltage-controlled servo like the XL330 its stall torque vin * kt / R, capped by
any firmware current limit), and
jitter (mean |change in action| per policy step, and joint speed RMS while
standing, where a steady policy should be near zero).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks.dukki import check_cfg_matches_run
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

# name: (vx, vy, yaw rate)
SCENARIOS = {
    "stand": (0.0, 0.0, 0.0),
    "walk_0.15": (0.15, 0.0, 0.0),
    "walk_0.30": (0.30, 0.0, 0.0),
    "back_0.20": (-0.20, 0.0, 0.0),
    "side_0.20": (0.0, 0.20, 0.0),
    "turn_1.0": (0.0, 0.0, 1.0),
}
CLAMP = 1.0  # J288 motor torque clamp [N.m]
# Fastest joint speed in the J288 BAM identification (dukki docs/j288-testing.md
# section 15.4): above it the actuator model is extrapolating.
IDENTIFIED_SPEED = 18.6  # [rad/s]


class OnnxPolicy:
    """An exported policy, run one environment at a time (batch 1 graphs)."""

    def __init__(self, path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path)
        self.name = self.session.get_inputs()[0].name

    def __call__(self, obs):
        x = obs["actor"].cpu().numpy().astype("float32")
        out = [self.session.run(None, {self.name: x[i : i + 1]})[0] for i in range(len(x))]
        import numpy as np

        return torch.as_tensor(np.concatenate(out, axis=0), device=obs["actor"].device)


def motor_torque_and_limit(env, act):
    """|motor torque| per env and joint, and the torque limit to compare it with."""
    if hasattr(act, "reported_torque"):
        return act.reported_torque.abs(), torch.full_like(act.reported_torque, CLAMP)
    tau = env.sim.data.qfrc_actuator[:, act._dof_ids].abs()
    bam = act._bam_model
    limit = act.vin_tensor * bam.kt.value / bam.R.value  # (N, 1), voltage-limited at rest
    max_current = getattr(bam.actuator, "max_current", None)
    if max_current:  # firmware current limit, e.g. the XL330's 1.75 A
        limit = torch.clamp(limit, max=max_current * bam.kt.value)
    return tau, limit.expand_as(tau)


def build(task: str, envs_per_scenario: int, duration: float, device: str = "cpu"):
    cfg = load_env_cfg(task)
    n = envs_per_scenario * len(SCENARIOS)
    cfg.scene.num_envs = n
    cfg.episode_length_s = duration + 10.0
    removed = [k for k in cfg.events if "push" in k or "topple" in k]
    for name in removed:
        del cfg.events[name]
    # Curricula that adjust a removed event by name (sit-stand, stand-up) would fail on start.
    for name in [k for k, t in cfg.curriculum.items() if (t.params or {}).get("event_name") in removed]:
        del cfg.curriculum[name]
    for term in cfg.commands.values():
        term.resampling_time_range = (1e6, 1e6)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)
    runner_cls = load_runner_cls(task) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(load_rl_cfg(task)), device=device)
    cmd = torch.tensor([v for v in SCENARIOS.values() for _ in range(envs_per_scenario)], device=device)
    return env, wrapped, runner, cmd


def pin_commands(env, cmd):
    twist = env.command_manager.get_term("twist")
    for flag in ("is_standing_env", "is_heading_env", "is_world_env"):
        if hasattr(twist, flag):
            getattr(twist, flag)[:] = False
    twist.vel_command_b[:] = cmd
    for name in ("head_pose", "body_pose"):
        if name in env.command_manager.active_terms:
            env.command_manager.get_term(name)._command[:] = 0.0


def evaluate(env, wrapped, runner, cmd, ckpt, duration, settle, seed):
    if ckpt.endswith(".onnx"):
        policy = OnnxPolicy(ckpt)
    else:
        runner.load(ckpt, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
    wrapped.reset()
    env.reset(seed=seed)
    pin_commands(env, cmd)
    obs = wrapped.get_observations()
    robot = env.scene["robot"]
    act = robot.actuators[0]
    n, j = cmd.shape[0], len(act.target_names)
    steps, settle_steps = int(duration / env.step_dt), int(settle / env.step_dt)
    acc = {k: torch.zeros(n, device=env.device) for k in ("lin_err", "yaw_err", "samples", "falls", "dact", "standvel")}
    peak = torch.zeros(n, j, device=env.device)
    sq = torch.zeros(n, j, device=env.device)
    near_clamp = torch.zeros(n, j, device=env.device)
    peak_qd = torch.zeros(n, j, device=env.device)
    fast = torch.zeros(n, device=env.device)
    prev_action = None
    p0 = h_prev = None
    yaw_turned = torch.zeros(n, device=env.device)
    with torch.no_grad():
        for k in range(steps):
            pin_commands(env, cmd)
            action = policy(obs)
            obs, _, dones, extras = wrapped.step(action)
            if k < settle_steps:
                prev_action = action.clone()
                continue
            heading = robot.data.heading_w.clone()
            if p0 is None:
                p0, h_prev = robot.data.root_link_pos_w[:, :2].clone(), heading
            yaw_turned += torch.atan2(torch.sin(heading - h_prev), torch.cos(heading - h_prev))
            h_prev = heading
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            acc["falls"] += (dones.bool() & ~timeouts.bool()).float()
            v = robot.data.root_link_lin_vel_b[:, :2]
            w = robot.data.root_link_ang_vel_b[:, 2]
            acc["lin_err"] += torch.linalg.norm(v - cmd[:, :2], dim=1)
            acc["yaw_err"] += (w - cmd[:, 2]).abs()
            acc["samples"] += 1
            acc["dact"] += (action - prev_action).abs().mean(dim=1)
            prev_action = action.clone()
            tau, limit = motor_torque_and_limit(env, act)
            peak = torch.maximum(peak, tau)
            sq += tau**2
            near_clamp += (tau > 0.95 * limit).float()
            qd = robot.data.joint_vel[:, act._target_ids_list]
            acc["standvel"] += qd.pow(2).mean(dim=1)
            peak_qd = torch.maximum(peak_qd, qd.abs())
            fast += (qd.abs() > IDENTIFIED_SPEED).any(dim=1).float()
    s = acc["samples"]
    window = (steps - settle_steps) * env.step_dt
    net_speed = torch.linalg.norm(robot.data.root_link_pos_w[:, :2] - p0, dim=1) / window
    net_yaw = yaw_turned / window
    names = list(act.target_names)
    out = {}
    per = n // len(SCENARIOS)
    for i, sc in enumerate(SCENARIOS):
        sl = slice(i * per, (i + 1) * per)
        rms = (sq[sl] / s[sl, None]).sqrt().mean(0)
        worst = int(peak[sl].max(0).values.argmax())
        clamp_pct = 100 * (near_clamp[sl] / s[sl, None]).mean(0)
        clamp_joint = int(clamp_pct.argmax())
        out[sc] = {
            "net_speed": float(net_speed[sl].mean()),
            "net_yaw": float(net_yaw[sl].mean()),
            "lin_err": float((acc["lin_err"][sl] / s[sl]).mean()),
            "yaw_err": float((acc["yaw_err"][sl] / s[sl]).mean()),
            "falls": int(acc["falls"][sl].sum()),
            "peak_torque": float(peak[sl].max()),
            "peak_joint": names[worst],
            "rms_torque_max_joint": float(rms.max()),
            "clamp_pct_worst_joint": float(clamp_pct[clamp_joint]),
            "clamp_joint": names[clamp_joint],
            "action_change": float((acc["dact"][sl] / s[sl]).mean()),
            "joint_speed_rms": float((acc["standvel"][sl] / s[sl]).sqrt().mean()),
            "peak_joint_speed": float(peak_qd[sl].max()),
            "peak_speed_joint": names[int(peak_qd[sl].max(0).values.argmax())],
            "pct_above_identified_speed": float(100 * (fast[sl] / s[sl]).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="robot model to evaluate on, e.g. Mjlab-Velocity-Flat-Dukki")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="label=path/model_N.pt ...")
    ap.add_argument("--envs-per-scenario", type=int, default=8)
    ap.add_argument("--duration", type=float, default=12.0, help="[s]")
    ap.add_argument("--settle", type=float, default=2.0, help="ignored at the start [s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write results here")
    ap.add_argument("--device", default="cpu", help="simulate on this device, e.g. cuda:0 as in training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)  # startup randomization is drawn while building
    env, wrapped, runner, cmd = build(args.task, args.envs_per_scenario, args.duration, args.device)
    for item in args.checkpoints:  # all up front, before any long evaluation
        check_cfg_matches_run(env.cfg, item.split("=", 1)[1])
    results = {}
    for item in args.checkpoints:
        label, path = item.split("=", 1)
        results[label] = evaluate(env, wrapped, runner, cmd, path, args.duration, args.settle, args.seed)
        print(f"\n== {label} on {args.task}")
        print(f"  {'scenario':10s} {'net m/s':>8s} {'net r/s':>8s} {'lin err':>8s} {'yaw err':>8s} {'falls':>5s} {'peak tau':>9s} {'joint':>15s} {'rms max':>8s} {'clamp %':>8s} {'at clamp':>15s} {'d action':>9s} {'qd rms':>7s} {'peak qd':>8s} {'qd joint':>15s} {'>18.6 %':>8s}")
        for sc, r in results[label].items():
            print(
                f"  {sc:10s} {r['net_speed']:8.3f} {r['net_yaw']:+8.2f} {r['lin_err']:8.3f} {r['yaw_err']:8.3f} {r['falls']:5d} {r['peak_torque']:9.3f} "
                f"{r['peak_joint']:>15s} {r['rms_torque_max_joint']:8.3f} {r['clamp_pct_worst_joint']:8.1f} {r['clamp_joint']:>15s} "
                f"{r['action_change']:9.4f} {r['joint_speed_rms']:7.3f} {r['peak_joint_speed']:8.2f} {r['peak_speed_joint']:>15s} {r['pct_above_identified_speed']:8.2f}"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"task": args.task, "scenarios": SCENARIOS, "results": results}, f, indent=1)
    env.close()


if __name__ == "__main__":
    main()
