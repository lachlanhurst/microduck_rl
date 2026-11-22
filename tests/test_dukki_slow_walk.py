"""Slow walking under high friction (dukki.py, item 8): friction ramp, smoothed fine tracking, slow-walk bucket."""

import math
from types import SimpleNamespace

import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import dukki as dk
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_all_off_by_default():
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Dukki")
    assert "track_lin_vel_fine" not in cfg.rewards
    assert cfg.commands["twist"].rel_slow_walk_envs == 0.0
    assert not any(name.endswith("_ramp") for name in cfg.curriculum)


@pytest.mark.parametrize("name", ["DUKKI_FRICTION_RAMP", "DUKKI_FINE_LIN_VEL", "DUKKI_SLOW_WALK"])
def test_bad_settings_refused(monkeypatch, name):
    monkeypatch.setenv(name, {"DUKKI_FRICTION_RAMP": "5,1,100", "DUKKI_FINE_LIN_VEL": "1,0",
                              "DUKKI_SLOW_WALK": "1.5"}[name])
    reader = {"DUKKI_FRICTION_RAMP": dk.dukki_friction_ramp, "DUKKI_FINE_LIN_VEL": dk.dukki_fine_lin_vel,
              "DUKKI_SLOW_WALK": dk.dukki_slow_walk}[name]
    with pytest.raises(ValueError):
        reader()


def test_friction_ramp_restarts_when_a_resume_restores_the_counter():
    """The env resets at step 0, then the runner loads the checkpoint and the counter jumps to the
    run's step: the ramp must start there, not count the jump as progress (run
    *_dukki-walk-backlash-* went straight to its end range)."""
    term = SimpleNamespace(params={"scale_range": (1.0, 3.0)})
    env = SimpleNamespace(common_step_counter=0, event_manager=SimpleNamespace(get_term_cfg=lambda n: term))
    ramp = dict(event_name="f", start_range=(1.0, 3.0), end_range=(1.0, 5.0), ramp_steps=100 * 24)
    dk.friction_ramp_curriculum(env, None, **ramp)
    env.common_step_counter = 9250 * 24
    dk.friction_ramp_curriculum(env, None, **ramp)
    assert term.params["scale_range"] == pytest.approx((1.0, 3.0))
    env.common_step_counter += 50 * 24
    dk.friction_ramp_curriculum(env, None, **ramp)
    assert term.params["scale_range"] == pytest.approx((1.0, 4.0))


def test_friction_ramp_counts_from_this_run():
    term = SimpleNamespace(params={"scale_range": (1.0, 3.0)})
    env = SimpleNamespace(common_step_counter=9250 * 24, event_manager=SimpleNamespace(get_term_cfg=lambda n: term))
    ramp = dict(event_name="f", start_range=(1.0, 3.0), end_range=(1.0, 5.0), ramp_steps=100 * 24)
    assert float(dk.friction_ramp_curriculum(env, None, **ramp)) == pytest.approx(3.0)
    assert term.params["scale_range"] == pytest.approx((1.0, 3.0))
    env.common_step_counter += 50 * 24
    dk.friction_ramp_curriculum(env, None, **ramp)
    assert term.params["scale_range"] == pytest.approx((1.0, 4.0))
    for _ in range(20):
        env.common_step_counter += 25 * 24
        dk.friction_ramp_curriculum(env, None, **ramp)
    assert term.params["scale_range"] == pytest.approx((1.0, 5.0))


def _fake_env(cmd, vel, length):
    robot = SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=vel))
    return SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda n: cmd),
        episode_length_buf=length,
        step_dt=0.02,
    )


def test_smoothed_tracking_pays_mean_speed_not_the_stride_swing():
    """A gait whose speed swings ±0.15 m/s around 0.12 for a 0.15 command scores far above
    standing still once averaged, where the stock instantaneous term scores them alike."""
    cmd = torch.tensor([[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]])
    length = torch.tensor([5, 5])
    env = _fake_env(cmd, None, length)
    for k in range(200):
        swing = 0.12 + 0.15 * math.sin(2 * math.pi * k * 0.02 / 0.5)
        vel = torch.tensor([[swing, 0.0, 0.0], [0.0, 0.0, 0.0]])
        env.scene["robot"].data.root_link_lin_vel_b = vel
        out = dk.track_lin_vel_smoothed(env, std=0.1)
    walking, standing = out.tolist()
    assert walking > 0.8 and standing < 0.15
    # Reset envs start over from zero error.
    env.episode_length_buf = torch.tensor([1, 5])
    out = dk.track_lin_vel_smoothed(env, std=0.1)
    assert out[0] > 0.95


def test_switches_reach_the_task(monkeypatch):
    monkeypatch.setenv("DUKKI_FRICTION", "1,3")
    monkeypatch.setenv("DUKKI_FRICTION_RAMP", "1,5,500")
    monkeypatch.setenv("DUKKI_FINE_LIN_VEL", "1.0,0.1")
    monkeypatch.setenv("DUKKI_SLOW_WALK", "0.2")
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Dukki")
    cfg = dk.use_slow_walk_training(dk.use_friction_ramp(cfg))
    fine = cfg.rewards["track_lin_vel_fine"]
    assert fine.func is dk.track_lin_vel_smoothed and fine.weight == 1.0 and fine.params["std"] == 0.1
    assert "track_linear_velocity" in cfg.rewards  # beside the stock term, not instead
    assert cfg.commands["twist"].rel_slow_walk_envs == 0.2
    assert cfg.commands["twist"].slow_walk_speed == (0.1, 0.25)
    ramp = cfg.curriculum["randomize_joint_friction_ramp"].params
    assert ramp["start_range"] == (1.0, 3.0) and ramp["end_range"] == (1.0, 5.0)
    assert ramp["ramp_steps"] == 500 * dk.STEPS_PER_ITERATION


def test_slow_walk_bucket_in_a_running_env():
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Dukki", play=True)
    cfg.scene.num_envs = 64
    cfg = dk.use_slow_walk_training(cfg, fine=(1.0, 0.1), slow_share=0.5)
    cfg.commands["twist"].rel_turn_in_place_envs = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        twist = env.command_manager.get_term("twist")
        assert isinstance(twist, microduck_mdp.VelocityCommandCommandOnly)
        twist._resample_command(torch.arange(64))
        cmd = twist.vel_command_b
        slow = (cmd[:, 1] == 0) & (cmd[:, 2] == 0) & (cmd[:, 0].abs() >= 0.1) & (cmd[:, 0].abs() <= 0.25)
        assert 16 <= int(slow.sum()) <= 52
        assert not twist.is_standing_env[slow].any()
        env.step(torch.zeros(64, env.action_manager.total_action_dim))
        assert "track_lin_vel_fine" in env.reward_manager.active_terms
    finally:
        env.close()


def test_slow_walk_speed_range(monkeypatch):
    monkeypatch.setenv("DUKKI_SLOW_WALK", "0.5,0.1,0.3")
    assert dk.dukki_slow_walk() == (0.5, (0.1, 0.3))
    cfg = dk.use_slow_walk_training(load_env_cfg("Mjlab-Velocity-Flat-Dukki"))
    assert cfg.commands["twist"].rel_slow_walk_envs == 0.5
    assert cfg.commands["twist"].slow_walk_speed == (0.1, 0.3)
    monkeypatch.setenv("DUKKI_SLOW_WALK", "0.5,0.3")
    with pytest.raises(ValueError):
        dk.dukki_slow_walk()


def test_action_rate_weight_replaces_the_ramp(monkeypatch):
    base = load_env_cfg("Mjlab-Velocity-Flat-Dukki")
    stages = base.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[-1]["weight"] == -1.0
    assert dk.use_action_rate_weight(base) is base  # unset: unchanged
    assert base.curriculum["action_rate_weight"].params["weight_stages"] == stages
    monkeypatch.setenv("DUKKI_ACTION_RATE", "-0.4")
    cfg = dk.use_action_rate_weight(load_env_cfg("Mjlab-Velocity-Flat-Dukki"))
    assert cfg.rewards["action_rate_l2"].weight == -0.4
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"] == [{"step": 0, "weight": -0.4}]
    monkeypatch.setenv("DUKKI_ACTION_RATE", "0.4")
    with pytest.raises(ValueError):
        dk.dukki_action_rate()


def test_kd_range(monkeypatch):
    assert "randomize_j288_kd" not in load_env_cfg("Mjlab-Velocity-Flat-Dukki").events
    monkeypatch.setenv("DUKKI_KD_RANGE", "0.05,0.08")
    assert dk.dukki_kd_range() == (0.05, 0.08)
    cfg = dk.use_kd_range(load_env_cfg("Mjlab-Velocity-Flat-Dukki", play=True))
    assert cfg.events["randomize_j288_kd"].params["kd_range"] == (0.05, 0.08)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        act = env.scene["robot"].actuators[0]
        assert act.kd.shape == act.default_kd.shape
        assert float(act.kd.min()) >= 0.05 - 1e-6 and float(act.kd.max()) <= 0.08 + 1e-6
        assert float(act.kd.std()) > 0.0, "joints drawn independently"
    finally:
        env.close()
    monkeypatch.setenv("DUKKI_KD_RANGE", "0.08,0.05")
    with pytest.raises(ValueError):
        dk.dukki_kd_range()


def test_fine_yaw_and_turn_bucket(monkeypatch):
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Dukki")
    assert "track_ang_vel_fine" not in cfg.rewards
    monkeypatch.setenv("DUKKI_FINE_ANG_VEL", "1.5,0.3")
    monkeypatch.setenv("DUKKI_TURN_IN_PLACE", "0.2,0.2")
    cfg = dk.use_slow_walk_training(load_env_cfg("Mjlab-Velocity-Flat-Dukki"))
    term = cfg.rewards["track_ang_vel_fine"]
    assert term.func is dk.track_ang_vel_smoothed and term.weight == 1.5 and term.params["std"] == 0.3
    assert "track_angular_velocity" in cfg.rewards
    twist = cfg.commands["twist"]
    assert twist.rel_turn_in_place_envs == 0.2 and twist.turn_in_place_min_frac == pytest.approx(0.2)


def test_smoothed_yaw_tracking_pays_turning_not_standing():
    cmd = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    env = _fake_env(cmd, None, torch.tensor([5, 5]))
    for k in range(200):
        wobble = 0.45 + 0.4 * math.sin(2 * math.pi * k * 0.02 / 0.5)
        env.scene["robot"].data.root_link_ang_vel_b = torch.tensor([[0.0, 0.0, wobble], [0.0, 0.0, 0.0]])
        out = dk.track_ang_vel_smoothed(env, std=0.3)
    turning, standing = out.tolist()
    assert turning > 0.9 and standing < 0.1
