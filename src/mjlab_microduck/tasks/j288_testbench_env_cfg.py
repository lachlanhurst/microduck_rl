"""J288 test-bench RL environment (task ``Mjlab-Testbench-J288``).

The XL330 testbench task (testbench_env_cfg.py) on the J288 pendulum rig the
servo was identified on: one joint, arm hanging down at 0, target angles in
[-80, 80] degrees, no domain randomization. A policy trained here is run on the
real servo with scripts/j288_testbench_sim2real.py, which checks the whole
chain (actuator model, gains, delay, joint velocity observation, ONNX export,
a 50 Hz host loop) before there is a robot.

The joint velocity observation is the position difference over one policy step
(tasks/dukki.py), as the robot will use; the sim2real script computes the
same. ``DUKKI_BENCH_SETUP``, ``DUKKI_KP`` and ``DUKKI_KD`` pick the rig and the
gains when the task is registered.

Smooth by default. With only the XL330 recipe's regularisation, the policy
learns to move bang-bang: the 1.0 N.m clamp on every target change and about
27 rad/s, past the 18.6 rad/s the identification ever reached (above about
20 rad/s the model rests on an assumed back-EMF constant) and hard on the
windings. ``smooth=True`` adds a soft speed limit at 15 rad/s and a 100x
torque penalty, so the policy stays where the model was fitted.
``Mjlab-Testbench-J288-Fast`` (``smooth=False``) keeps the bang-bang policy,
for deliberately exercising the voltage-limited region.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.j288_testbench_constants import (
    TESTBENCH_JOINT,
    bench_setup_from_env,
    make_j288_testbench_robot_cfg,
)
from mjlab_microduck.tasks.dukki import dukki_gains_from_env, joint_vel_position_difference
from mjlab_microduck.tasks.testbench_env_cfg import (
    MicroduckTestbenchRlCfg,
    make_testbench_env_cfg,
)


#: Soft joint speed limit [rad/s], inside the identified range (peak 18.6).
SPEED_LIMIT = 15.0


def joint_speed_over_limit(
    env, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """sum(max(|qd| - limit, 0)^2): zero below the limit, steep above it."""
    asset: Entity = env.scene[asset_cfg.name]
    excess = torch.clamp(asset.data.joint_vel[:, asset_cfg.joint_ids].abs() - limit, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def make_j288_testbench_env_cfg(
    play: bool = False,
    setup: str | None = None,
    kp: float | None = None,
    kd: float | None = None,
    smooth: bool = True,
) -> ManagerBasedRlEnvCfg:
    env_kp, env_kd = dukki_gains_from_env()
    robot_cfg = make_j288_testbench_robot_cfg(
        setup=setup or bench_setup_from_env(),
        kp=env_kp if kp is None else kp,
        kd=env_kd if kd is None else kd,
    )
    cfg = make_testbench_env_cfg(
        play=play,
        robot_cfg=robot_cfg,
        joint_name=TESTBENCH_JOINT,
        joint_vel_func=joint_vel_position_difference,
        joint_vel_lag=0,
    )
    if smooth:
        cfg.rewards["joint_speed_over_limit"] = RewardTermCfg(
            func=joint_speed_over_limit, weight=-0.5, params={"limit": SPEED_LIMIT}
        )
        cfg.rewards["joint_torques_l2"].weight = -0.1
    return cfg


J288TestbenchRlCfg = deepcopy(MicroduckTestbenchRlCfg)
J288TestbenchRlCfg.experiment_name = "testbench_j288"
J288TestbenchRlCfg.run_name = "testbench_j288"
