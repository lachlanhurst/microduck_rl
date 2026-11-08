"""Dukki (J288) robot, actuator and task variants."""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Tasks first, as mjlab's plugin entry point does: importing the robot modules
# first would make mjlab load the task package while they are half-imported.
import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot import dukki_constants as dk
from mjlab_microduck.robot import microduck_constants as mc

CAD_SPECS = [
    mc.get_walk_spec,
    mc.get_standup_spec,
    mc.get_allcollisions_spec,
    mc.get_walk_backlash_spec,
    mc.get_backlash_spec,
    mc.get_allcollisions_backlash_spec,
]
DUKKI_SPECS = [
    dk.get_dukki_walk_spec,
    dk.get_dukki_standup_spec,
    dk.get_dukki_allcollisions_spec,
    dk.get_dukki_walk_backlash_spec,
    dk.get_dukki_backlash_spec,
    dk.get_dukki_allcollisions_backlash_spec,
]
CAD_MASS = 1.0738  # the Fusion inertials, without cables and fasteners


def _physical(inertia):
    """Inertias positive and meeting the triangle inequality."""
    s = inertia.sum(axis=1, keepdims=True)
    return (inertia > 0).all() and (2 * inertia <= s + 1e-12).all()


@pytest.mark.parametrize("spec_fn", CAD_SPECS)
def test_cad_mass_on_every_model(spec_fn):
    """Every MJCF carries the Fusion inertials: 1.074 kg, without cables and fasteners."""
    model = spec_fn().compile()
    assert model.body_mass.sum() == pytest.approx(CAD_MASS, abs=0.001)
    assert _physical(model.body_inertia[1:])


@pytest.mark.parametrize("cad_fn, dukki_fn", list(zip(CAD_SPECS, DUKKI_SPECS)))
def test_unmodelled_mass_spread_by_mass(cad_fn, dukki_fn):
    """Cables and fasteners scale every body alike; no centre of mass moves."""
    a, b = cad_fn().compile(), dukki_fn().compile()
    k = 1.0 + dk.UNMODELLED_MASS / a.body_mass.sum()
    assert b.body_mass.sum() == pytest.approx(CAD_MASS + dk.UNMODELLED_MASS, abs=0.001)
    np.testing.assert_allclose(b.body_mass, a.body_mass * k, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(b.body_inertia, a.body_inertia * k, rtol=1e-6, atol=1e-15)
    np.testing.assert_allclose(b.body_ipos, a.body_ipos, atol=1e-12)
    assert _physical(b.body_inertia[1:])


def test_unmodelled_mass_reverts_at_zero():
    cad = mc.get_walk_spec().compile()
    model = dk.add_unmodelled_mass(mc.get_walk_spec(), mass=0.0).compile()
    np.testing.assert_array_equal(model.body_mass, cad.body_mass)
    np.testing.assert_array_equal(model.body_inertia, cad.body_inertia)


def test_dukki_models_use_the_dukki_specs():
    for name, (spec_fn, _) in dk._DUKKI_MODELS.items():
        assert spec_fn in DUKKI_SPECS, name


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


def test_dukki_sitstand_keeps_head_off_ground():
    from mjlab.tasks.registry import load_env_cfg

    from mjlab_microduck.tasks import dukki

    cfg = load_env_cfg("Mjlab-SitStand-Flat-Dukki")
    sensor = next(s for s in cfg.scene.sensors if s.name == dukki.HEAD_GROUND_SENSOR)
    assert sensor.primary.mode == "subtree" and sensor.primary.pattern == "neck"
    assert cfg.rewards["head_ground_contact"].weight == dukki.HEAD_GROUND_CONTACT_WEIGHT < 0
    assert cfg.rewards["head_ground_force"].weight == dukki.HEAD_GROUND_FORCE_WEIGHT < 0
    assert "sample_still" in cfg.events  # still practice kept
    assert cfg.rewards["sit_stall"].func is dukki.SitStall and cfg.rewards["sit_stall"].weight < 0
    assert cfg.rewards["stand_stall"].params["posture"] == "stand" and cfg.rewards["stand_stall"].weight < 0
    assert cfg.rewards["rise_speed_squared"].weight == dukki.SOFT_TRANSITION_WEIGHT < 0
    assert cfg.rewards["descent_speed_squared"].weight == dukki.SOFT_TRANSITION_WEIGHT
    # Pollen's microduck sit-stand is unchanged.
    base = load_env_cfg("Mjlab-SitStand-Flat-MicroDuck")
    assert "head_ground_contact" not in base.rewards and "sit_stall" not in base.rewards
    assert all(s.name != dukki.HEAD_GROUND_SENSOR for s in base.scene.sensors)


def test_sit_stall_counts_from_the_sit_command():
    from types import SimpleNamespace

    from mjlab_microduck.tasks import dukki

    n, dt = 3, 0.02
    cmd = torch.zeros(n, 3)
    z = torch.tensor([0.115, 0.115, 0.062])
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=dt,
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(vel_command_b=cmd)),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.stack([torch.zeros(n), torch.zeros(n), z], 1)))},
    )
    term = dukki.SitStall(None, env)
    cmd[:2, 0] = 1.0  # robots 0 and 1 told to sit, robot 2 seated and told to sit too
    cmd[2, 0] = 1.0
    steps = int(dukki.SIT_STALL_GRACE_S / dt)
    for _ in range(steps):
        assert term(env).sum() == 0  # within the grace period
    term(env)  # age reaches the grace period exactly; the cost starts past it
    out = term(env)
    assert out.tolist() == [1.0, 1.0, 0.0]  # still standing after it; the seated one is fine
    cmd[1, 0] = 0.0  # robot 1 told to stand: no cost
    assert term(env).tolist() == [1.0, 0.0, 0.0]
    cmd[1, 0] = 1.0  # sit again: the grace period starts over
    assert term(env).tolist() == [1.0, 0.0, 0.0]


def test_stand_stall_and_vertical_speed_squared():
    from types import SimpleNamespace

    from mjlab_microduck.tasks import dukki

    n, dt = 2, 0.02
    cmd = torch.zeros(n, 3)  # both told to stand
    z = torch.tensor([0.062, 0.115])  # robot 0 still seated, robot 1 standing
    vz = torch.tensor([0.5, -0.3])
    data = SimpleNamespace(
        root_link_pos_w=torch.stack([torch.zeros(n), torch.zeros(n), z], 1),
        root_link_lin_vel_w=torch.stack([torch.zeros(n), torch.zeros(n), vz], 1),
    )
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=dt,
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(vel_command_b=cmd)),
        scene={"robot": SimpleNamespace(data=data)},
    )
    term = dukki.SitStall(None, env)
    for _ in range(int(dukki.SIT_STALL_GRACE_S / dt) + 2):
        out = term(env, posture="stand")
    assert out.tolist() == [1.0, 0.0]
    rise = dukki.rise_speed_squared(env)
    assert rise[0] == pytest.approx((0.5 - dukki.SOFT_RISE_MAX_VZ) ** 2)
    assert rise[1] == 0.0
    drop = dukki.descent_speed_squared(env)
    assert drop[0] == 0.0
    assert drop[1] == pytest.approx((0.3 - dukki.SOFT_DESCENT_MAX_VZ) ** 2)
