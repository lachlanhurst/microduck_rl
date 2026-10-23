"""Dukki (J288) robot, actuator and task variants."""

from __future__ import annotations

import pytest
import torch

# Tasks first, as mjlab's plugin entry point does: importing the robot modules
# first would make mjlab load the task package while they are half-imported.
import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot import dukki_constants as dk
from mjlab_microduck.robot import microduck_constants as mc

EXTRA = dk.NUM_SERVOS * dk.SERVO_EXTRA_MASS + dk.BATTERY_EXTRA_MASS + dk.HEAD_EXTRA_MASS


@pytest.mark.parametrize(
    "base, dukki",
    [
        (mc.get_walk_spec, dk.get_dukki_walk_spec),
        (mc.get_standup_spec, dk.get_dukki_standup_spec),
        (mc.get_allcollisions_spec, dk.get_dukki_allcollisions_spec),
        (mc.get_walk_backlash_spec, dk.get_dukki_walk_backlash_spec),
        (mc.get_backlash_spec, dk.get_dukki_backlash_spec),
        (mc.get_allcollisions_backlash_spec, dk.get_dukki_allcollisions_backlash_spec),
    ],
)
def test_dukki_mass_added_on_every_model(base, dukki):
    a, b = base().compile(), dukki().compile()
    assert b.body_mass.sum() == pytest.approx(a.body_mass.sum() + EXTRA, abs=1e-6)
    head = a.body(dk.HEAD_BODY_NAME).id
    # Head: the electronics plus its two servos (head_roll and the jaw).
    assert b.body_mass[head] == pytest.approx(
        a.body_mass[head] + dk.HEAD_EXTRA_MASS + 2 * dk.SERVO_EXTRA_MASS, abs=1e-6
    )
    # Inertias stay physical (positive, triangle inequality) after the edit.
    inertia = b.body_inertia[1:]
    assert (inertia > 0).all()
    s = inertia.sum(axis=1, keepdims=True)
    assert (2 * inertia <= s + 1e-12).all()


def test_dukki_robot_is_about_1p18_kg():
    assert dk.get_dukki_walk_spec().compile().body_mass.sum() == pytest.approx(1.184, abs=0.002)


def _bench_env(**kw):
    from mjlab.envs import ManagerBasedRlEnv

    from mjlab_microduck.tasks.j288_testbench_env_cfg import make_j288_testbench_env_cfg

    cfg = make_j288_testbench_env_cfg(play=True, **kw)
    cfg.scene.num_envs = 2
    for group in cfg.observations.values():
        group.enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset(seed=0)
    return env


@pytest.fixture(scope="module")
def bench_env():
    env = _bench_env()
    yield env
    env.close()


def test_bench_uses_j288_m6(bench_env):
    from mjlab_microduck.actuator import J288DRActuator

    act = bench_env.scene["robot"].actuators[0]
    assert isinstance(act, J288DRActuator)
    assert torch.allclose(act.kp, torch.full_like(act.kp, dk.J288_KP))
    assert torch.allclose(act.kd, torch.full_like(act.kd, dk.J288_KD))
    assert bench_env.sim.mj_model.dof_armature[0] == pytest.approx(6.27e-4, rel=1e-2)


def test_friction_scale_applied_once(bench_env):
    """BAM applies friction_scale itself; the mixin must not apply it again."""
    act = bench_env.scene["robot"].actuators[0]
    motor = torch.full((2, 1), 0.3)
    external = torch.full((2, 1), 0.1)
    stribeck = torch.zeros((2, 1))
    act.reset_friction_scale(slice(None))
    base = act._compute_friction_budget(motor, external, stribeck)
    act.set_friction_scale(slice(None), torch.full((2, 1), 2.0))
    scaled = act._compute_friction_budget(motor, external, stribeck)
    act.reset_friction_scale(slice(None))
    assert torch.allclose(scaled, 2.0 * base)


def test_position_difference_velocity(bench_env):
    env = bench_env
    robot = env.scene["robot"]
    q_prev = robot.data.joint_pos[:, 0].clone()
    for _ in range(3):
        obs, *_ = env.step(torch.tensor([[0.5], [-0.5]]))
        q = robot.data.joint_pos[:, 0].clone()
        expected = (q - q_prev) / env.step_dt
        assert torch.allclose(obs["actor"][:, 1], expected, atol=1e-5)
        # Recomputing within the same step does not change it.
        again = env.observation_manager.compute()["actor"][:, 1]
        assert torch.allclose(again, expected, atol=1e-5)
        q_prev = q
    env.reset(env_ids=torch.tensor([0]))
    vel = env.observation_manager.compute()["actor"][:, 1]
    assert vel[0] == 0.0 and vel[1] != 0.0


def test_per_joint_gains_and_reset():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.entity import EntityArticulationInfoCfg

    from mjlab_microduck.robot.j288_testbench_constants import TESTBENCH_JOINT
    from mjlab_microduck.tasks.j288_testbench_env_cfg import make_j288_testbench_env_cfg

    cfg = make_j288_testbench_env_cfg(play=True)
    robot = cfg.scene.entities["robot"]
    act_cfg = dk.make_j288_actuators(
        target_names_expr=(TESTBENCH_JOINT,),
        joint_gains=((TESTBENCH_JOINT, 3.0, 0.07),),
        delay_min_lag=0,
        delay_max_lag=0,
    )
    robot.articulation = EntityArticulationInfoCfg(
        actuators=(act_cfg,), soft_joint_pos_limit_factor=1.0
    )
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset(seed=0)
    act = env.scene["robot"].actuators[0]
    assert torch.allclose(act.kp, torch.full_like(act.kp, 3.0))
    assert torch.allclose(act.kd, torch.full_like(act.kd, 0.07))
    act.set_joint_gains(torch.tensor([1]), kp=0.5)
    env.reset(env_ids=torch.tensor([1]))
    assert act.kp[1, 0] == pytest.approx(3.0)
    env.close()


def test_dukki_tasks_registered_with_microduck_obs_layout():
    from mjlab.tasks.registry import load_env_cfg

    from mjlab_microduck.tasks.dukki import joint_vel_position_difference

    for dukki_id, base_id in [
        ("Mjlab-Velocity-Flat-Dukki", "Mjlab-Velocity-Flat-MicroDuck"),
        ("Mjlab-Velocity-Flat-Backlash-Dukki", "Mjlab-Velocity-Flat-Backlash-MicroDuck"),
        ("Mjlab-VelStand-Flat-Dukki", "Mjlab-VelStand-Flat-MicroDuck"),
        ("Mjlab-StandUp-Flat-Dukki", "Mjlab-StandUp-Flat-MicroDuck"),
        ("Mjlab-SitStand-Flat-Dukki", "Mjlab-SitStand-Flat-MicroDuck"),
    ]:
        d, b = load_env_cfg(dukki_id), load_env_cfg(base_id)
        assert list(d.observations["actor"].terms) == list(b.observations["actor"].terms)
        jv = d.observations["actor"].terms["joint_vel"]
        assert jv.func is joint_vel_position_difference
        assert jv.delay_max_lag == 0
        acts = d.scene.entities["robot"].articulation.actuators
        assert type(acts[0]).__name__ == "J288DRActuatorCfg"
    # The J288 backlash twin keeps rotor-side joint obs (no encoder-through-backlash).
    d = load_env_cfg("Mjlab-Velocity-Flat-Backlash-Dukki")
    assert d.observations["actor"].terms["joint_pos"].func.__name__ == "joint_pos_rel"
    assert d.observations["critic"].terms["joint_vel"].func.__name__ == "joint_vel_rel"


def test_dukki_tasks_penalise_commands_past_the_stop():
    """The first Dukki runs jammed a hip_yaw against its stop at the torque clamp."""
    from mjlab.tasks.registry import load_env_cfg

    from mjlab_microduck.tasks.dukki import OVER_LIMIT_TORQUE, make_dukki_variant
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

    for task in ("Mjlab-Velocity-Flat-Dukki", "Mjlab-Velocity-Flat-Backlash-Dukki"):
        term = load_env_cfg(task).rewards["action_over_limit"]
        assert term.weight < 0
        assert term.params["overshoot"] == pytest.approx(OVER_LIMIT_TORQUE / dk.J288_KP)
    # The allowance follows the gains, so a gain sweep keeps the same torque cap.
    cfg = make_dukki_variant(make_microduck_velocity_env_cfg(), kp=4.0)
    assert cfg.rewards["action_over_limit"].params["overshoot"] == pytest.approx(OVER_LIMIT_TORQUE / 4.0)


def test_onnx_expert_matches_onnxruntime():
    """The stand expert rebuilt from Pollen's ONNX must give the exported policy's actions."""
    import pathlib

    import numpy as np
    import pytest
    from tensordict import TensorDict

    path = pathlib.Path("logs/hf/microduck-policies/alpha_stand.onnx")
    if not path.exists():
        pytest.skip("alpha_stand.onnx not downloaded")
    import onnxruntime as ort

    from mjlab_microduck.tasks.distill import OnnxActorExpert

    expert = OnnxActorExpert(path, ["actor"])
    session = ort.InferenceSession(str(path))
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=(1, 61)).astype(np.float32)
        want = session.run(None, {"obs": x})[0]
        got = expert(TensorDict({"actor": torch.from_numpy(x)}, batch_size=[1])).numpy()
        assert np.allclose(got, want, atol=1e-5)
