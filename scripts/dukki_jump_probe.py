#!/usr/bin/env python3
"""Can the robot jump at all? Open-loop scripted jumps, many at once, no policy.

A feasibility check before designing a jump task: each robot runs one scripted motion from the
standing spawn, with the task's actuators (J288 BAM for Dukki) and pushes off:

    hold HOME (HOLD_S), crouch to ``depth`` over CROUCH_S, hold it (SETTLE_S), then push: target
    ``ext`` x the crouch direction past HOME for ``push`` seconds, then back to HOME.

A crouch bends the knees by ``depth`` rad and the hip and ankle pitch by half of it, each with a
sign; ``--find-squat`` first tries the four hip and ankle sign combinations and reports which
lowers the trunk while staying upright and over the feet. The jump grid then reports, per
setting: foot clearance (the higher foot's lowest point above its standing height, while both are
up), airborne time (both feet up), the trunk's rise, whether it ends upright, and peak motor
torque.

    DUKKI_KP=4 DUKKI_KD=0.05 uv run scripts/dukki_jump_probe.py --find-squat
    DUKKI_KP=4 DUKKI_KD=0.05 uv run scripts/dukki_jump_probe.py --hip-sign 1 --ankle-sign 1

Run it stiff (kp 4 here) so an open-loop script reaches the torques a policy would by aiming
past its target; the motor torque clamp is the same at any kp. At kp 1 a held crouch sags until
the error makes the torque and every squat collapsed. On Dukki the squat is hip +1, ankle +1.
"""

from __future__ import annotations

import argparse
import itertools

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

HOLD_S, CROUCH_S, SETTLE_S, AFTER_S = 0.5, 0.4, 0.3, 1.2
UP_MM = 2.0  # a foot counts as up this far above its standing height
LEFT = {"hip_pitch": "left_hip_pitch", "knee": "left_knee", "ankle": "left_ankle"}
RIGHT = {"hip_pitch": "right_hip_pitch", "knee": "right_knee", "ankle": "right_ankle"}


def build(task: str, n: int, device: str) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = n
    cfg.episode_length_s = 30.0
    removed = [k for k in cfg.events if "push" in k or "topple" in k]
    for name in removed:
        del cfg.events[name]
    ground = cfg.events.get("set_ground_state")
    if ground is not None:
        for key in ("face_down", "face_up", "sitting", "standing"):
            ground.params[f"{key}_prob"] = 1.0 if key == "standing" else 0.0
    for name in [
        k for k, t in cfg.curriculum.items() if (t.params or {}).get("event_name") in [*removed, "set_ground_state"]
    ]:
        del cfg.curriculum[name]
    if "terminations" in dir(cfg):
        for name in [k for k in cfg.terminations if k != "time_out"]:
            del cfg.terminations[name]
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def crouch_direction(names: list[str], hip_sign: float, ankle_sign: float, device) -> torch.Tensor:
    """Offsets from HOME per rad of knee bend; the right side mirrors the left."""
    d = torch.zeros(len(names), device=device)
    for side, sign in ((LEFT, 1.0), (RIGHT, -1.0)):
        d[names.index(side["knee"])] = sign
        d[names.index(side["hip_pitch"])] = sign * hip_sign * 0.5
        d[names.index(side["ankle"])] = sign * ankle_sign * 0.5
    return d


def run(env, actions_fn, steps: int):
    """Steps the env with actions_fn(k) -> (n, 14); returns per-step trunk z, foot z, tilt, torque."""
    robot = env.scene["robot"]
    act = robot.actuators[0]
    # The lowest bodies: the ankle links carry the soles (Microduck and Dukki models).
    feet = [i for i, b in enumerate(robot.body_names) if b in ("ankle_left", "ankle_right")]
    log = {"z": [], "foot": [], "tilt": [], "x": [], "tau": []}
    for k in range(steps):
        env.step(actions_fn(k))
        g = robot.data.projected_gravity_b
        log["tilt"].append(torch.rad2deg(torch.arccos((-g[:, 2] / g.norm(dim=1)).clamp(-1, 1))))
        log["z"].append(robot.data.root_link_pos_w[:, 2])
        log["x"].append(robot.data.root_link_pos_w[:, :2])
        log["foot"].append(robot.data.body_link_pos_w[:, feet, 2])
        log["tau"].append(act.reported_torque.abs().amax(dim=1) if hasattr(act, "reported_torque") else torch.zeros_like(g[:, 0]))
    return {k: torch.stack(v) for k, v in log.items()}, [robot.body_names[i] for i in feet]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-StandUp-Flat-Dukki")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--find-squat", action="store_true")
    ap.add_argument("--hip-sign", type=float, default=1.0)
    ap.add_argument("--ankle-sign", type=float, default=1.0)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--exts", type=float, nargs="+", default=[0.2, 0.4, 0.7, 1.0])
    ap.add_argument("--pushes", type=float, nargs="+", default=[0.08, 0.12, 0.18, 0.25])
    args = ap.parse_args()

    if args.find_squat:
        combos = list(itertools.product((1.0, -1.0), (1.0, -1.0)))
        env = build(args.task, len(combos), args.device)
        env.reset(seed=0)
        names = list(env.scene["robot"].actuators[0].target_names)
        dirs = torch.stack([crouch_direction(names, h, a, env.device) for h, a in combos])
        dt = env.step_dt
        n_hold, n_c = int(HOLD_S / dt), int(CROUCH_S / dt)
        depth = 0.8

        def actions(k):
            f = min(max((k - n_hold) / n_c, 0.0), 1.0)
            return dirs * depth * f

        log, feet = run(env, actions, n_hold + n_c + int(1.0 / dt))
        x0 = log["x"][n_hold - 1]
        print(f"squat search, knee bend {depth} rad; feet bodies {feet}")
        for i, (h, a) in enumerate(combos):
            print(
                f"  hip {h:+.0f} ankle {a:+.0f}: trunk {log['z'][n_hold - 1, i] * 1000:.0f} -> {log['z'][-1, i] * 1000:.0f} mm,"
                f" tilt {log['tilt'][-1, i]:.0f} deg, moved {(log['x'][-1, i] - x0[i]).norm() * 1000:.0f} mm"
            )
        return

    grid = list(itertools.product(args.depths, args.exts, args.pushes))
    env = build(args.task, len(grid), args.device)
    env.reset(seed=0)
    names = list(env.scene["robot"].actuators[0].target_names)
    unit = crouch_direction(names, args.hip_sign, args.ankle_sign, env.device)
    dt = env.step_dt
    n_hold, n_c, n_s = int(HOLD_S / dt), int(CROUCH_S / dt), int(SETTLE_S / dt)
    depth = torch.tensor([g[0] for g in grid], device=env.device)
    ext = torch.tensor([g[1] for g in grid], device=env.device)
    push = torch.tensor([g[2] for g in grid], device=env.device)
    n_push_max = int(max(args.pushes) / dt) + 1
    steps = n_hold + n_c + n_s + n_push_max + int(AFTER_S / dt)
    t_push0 = n_hold + n_c + n_s

    def actions(k):
        f = min(max((k - n_hold) / n_c, 0.0), 1.0)
        crouch = depth * f
        pushing = (k >= t_push0) & ((k - t_push0) * dt < push)
        after = (k - t_push0) * dt >= push
        scale = torch.where(pushing, -ext, torch.where(after, torch.zeros_like(crouch), crouch))
        return scale.unsqueeze(1) * unit

    log, feet = run(env, actions, steps)
    foot0 = log["foot"][n_hold - 1]  # standing foot heights
    rise = (log["foot"] - foot0) * 1000  # (T, n, 2) mm
    both_up = (rise > UP_MM).all(dim=2)
    clearance = torch.where(both_up, rise.amin(dim=2), torch.zeros_like(both_up, dtype=torch.float)).amax(dim=0)
    airborne = both_up.float().sum(dim=0) * dt
    trunk_rise = (log["z"][t_push0:].amax(dim=0) - log["z"][n_hold - 1]) * 1000
    upright_end = log["tilt"][-1] < 20
    peak_tau = log["tau"][t_push0:].amax(dim=0)
    order = sorted(range(len(grid)), key=lambda i: -float(clearance[i]))
    print(f"jump grid, hip {args.hip_sign:+.0f} ankle {args.ankle_sign:+.0f}; best first (feet {feet})")
    print("  depth   ext  push   clearance  airborne  trunk rise  ends upright  peak torque")
    for i in order[:15]:
        d, e, p = grid[i]
        print(
            f"  {d:5.2f} {e:5.2f} {p:5.2f}   {clearance[i]:6.1f} mm  {airborne[i]:5.2f} s   {trunk_rise[i]:6.1f} mm"
            f"   {'yes' if upright_end[i] else 'no ':3s}           {peak_tau[i]:.2f}"
        )
    print(f"  {int((clearance > 0).sum())} of {len(grid)} settings left the ground; {int(((clearance > 0) & upright_end).sum())} of them ended upright")


if __name__ == "__main__":
    main()
