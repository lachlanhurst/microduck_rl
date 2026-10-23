#!/usr/bin/env python3
"""Render a video of a Dukki walking checkpoint on CPU.

One robot, a scripted sequence of commands, a camera that follows the trunk,
and an overlay with the command, the measured velocity and the most loaded
joints' motor torque (the J288 clamps at 1.0 N.m). Training domain
randomization is kept, pushes are off.

    uv run scripts/dukki_render.py --task Mjlab-Velocity-Flat-Dukki \\
        --checkpoint path/model_2999.pt --out dukki.mp4

--sequence takes name:vx:vy:yaw:seconds segments, e.g.
    stand:0:0:0:4 walk:0.2:0:0:5 turn:0:0:1.0:5
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks.dukki import check_cfg_matches_run
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

DEFAULT_SEQUENCE = ["stand:0:0:0:4", "walk:0.2:0:0:5", "turn:0:0:1.0:5"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-Dukki")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="dukki.mp4")
    ap.add_argument("--sequence", nargs="+", default=DEFAULT_SEQUENCE)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--distance", type=float, default=0.75, help="camera distance [m]")
    ap.add_argument("--azimuth", type=float, default=135.0, help="[deg], relative to the trunk heading")
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--label", default="", help="title line in the overlay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--start",
        choices=("task", "sitting", "standing"),
        default="task",
        help="starting posture where the task mixes them (sit-stand): its own random mix, or forced",
    )
    args = ap.parse_args()

    segments = []
    for item in args.sequence:
        name, vx, vy, yaw, secs = item.split(":")
        segments.append((name, float(vx), float(vy), float(yaw), float(secs)))
    total = sum(s[4] for s in segments)

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = total + 10.0
    ground = cfg.events.get("set_ground_state")
    if args.start != "task":
        if ground is None:
            raise SystemExit(f"--start {args.start}: {args.task} has no set_ground_state reset to force")
        sitting = args.start == "sitting"
        for key in ("face_down_prob", "face_up_prob"):
            ground.params[key] = 0.0
        ground.params["sitting_prob"] = 1.0 if sitting else 0.0
        ground.params["standing_prob"] = 0.0 if sitting else 1.0
    removed = [k for k in cfg.events if "push" in k or "topple" in k]
    for name in removed:
        del cfg.events[name]
    # Curricula that adjust a removed event by name (sit-stand, stand-up) would fail on start.
    for name in [k for k, t in cfg.curriculum.items() if (t.params or {}).get("event_name") in removed]:
        del cfg.curriculum[name]
    for term in cfg.commands.values():
        term.resampling_time_range = (1e6, 1e6)
    check_cfg_matches_run(cfg, args.checkpoint)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    rl_cfg = load_rl_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = (load_runner_cls(args.task) or OnPolicyRunner)(wrapped, asdict(rl_cfg), device="cpu")
    runner.load(args.checkpoint, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    env.reset(seed=args.seed)
    robot = env.scene["robot"]
    act = robot.actuators[0]
    names = [n.replace("left_", "L ").replace("right_", "R ") for n in act.target_names]
    twist = env.command_manager.get_term("twist")

    def pin(vx, vy, yaw):
        for flag in ("is_standing_env", "is_heading_env", "is_world_env"):
            if hasattr(twist, flag):
                getattr(twist, flag)[:] = False
        twist.vel_command_b[:] = torch.tensor([vx, vy, yaw])
        for name in ("head_pose", "body_pose"):
            if name in env.command_manager.active_terms:
                env.command_manager.get_term(name)._command[:] = 0.0

    mj_model = env.sim.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = args.distance, args.elevation
    trunk = robot.indexing.body_ids[0]  # root body
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
    except OSError:
        font = ImageFont.load_default()

    fps = int(round(1.0 / env.step_dt))
    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    obs = wrapped.get_observations()
    look = None
    with torch.no_grad():
        for name, vx, vy, yaw, secs in segments:
            for _ in range(int(round(secs / env.step_dt))):
                pin(vx, vy, yaw)
                obs, *_ = wrapped.step(policy(obs))

                mj_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                mj_data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                mujoco.mj_forward(mj_model, mj_data)
                pos = mj_data.xpos[trunk].copy()
                look = pos if look is None else 0.9 * look + 0.1 * pos  # smooth the follow
                heading = np.degrees(robot.data.heading_w[0].item())
                cam.lookat[:] = [look[0], look[1], 0.09]
                cam.azimuth = heading + args.azimuth
                renderer.update_scene(mj_data, camera=cam)
                frame = Image.fromarray(renderer.render())

                v = robot.data.root_link_lin_vel_b[0, :2].numpy()
                w = robot.data.root_link_ang_vel_b[0, 2].item()
                tau = act.reported_torque[0].abs().numpy()
                top = np.argsort(-tau)[:3]
                lines = [
                    args.label,
                    f"{name:6s} command vx {vx:+.2f} vy {vy:+.2f} m/s  yaw {yaw:+.2f} rad/s",
                    f"       measured vx {v[0]:+.2f} vy {v[1]:+.2f} m/s  yaw {w:+.2f} rad/s",
                    "motor torque (clamp 1.00 N.m): "
                    + "  ".join(f"{names[i]} {tau[i]:.2f}" for i in top),
                ]
                draw = ImageDraw.Draw(frame)
                draw.rectangle([0, 0, args.width, 26 * len(lines) + 14], fill=(0, 0, 0))
                for i, line in enumerate(lines):
                    colour = (255, 120, 90) if i == 3 and tau[top[0]] > 0.95 else (235, 235, 235)
                    draw.text((12, 8 + 26 * i), line, fill=colour, font=font)
                writer.append_data(np.asarray(frame))
    writer.close()
    env.close()
    print(f"wrote {args.out} ({total:.0f} s at {fps} fps)")


if __name__ == "__main__":
    main()
