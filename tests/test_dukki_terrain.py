"""tasks/dukki_terrain.py: soft ground, sole torsion, Dukki's terrain, the speed range (CPU)."""

import mujoco
import numpy as np
import pytest

from mjlab_microduck.tasks import dukki_terrain as dt


def test_soft_ground_parse(monkeypatch):
    monkeypatch.delenv("DUKKI_SOFT_GROUND", raising=False)
    assert dt.dukki_soft_ground() is None
    monkeypatch.setenv("DUKKI_SOFT_GROUND", "0.02,0.09")
    assert dt.dukki_soft_ground() == (0.02, 0.09, dt.DEFAULT_SOFT_SHARE)
    monkeypatch.setenv("DUKKI_SOFT_GROUND", "0.02,0.09,0.7")
    assert dt.dukki_soft_ground() == (0.02, 0.09, 0.7)
    for bad in ("0.09,0.02", "0,0.1", "0.02,0.09,1.5", "0.02"):
        monkeypatch.setenv("DUKKI_SOFT_GROUND", bad)
        with pytest.raises(ValueError):
            dt.dukki_soft_ground()


def test_soft_ground_event_only_when_asked(monkeypatch):
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-Backlash-Dukki")
    assert "soft_ground" not in cfg.events and "sole_torsion" not in cfg.events
    cfg = dt.use_soft_ground(cfg, (0.02, 0.08, 0.5))
    term = cfg.events["soft_ground"]
    assert term.mode == "reset" and term.params["time_constant_range"] == (0.02, 0.08)
    assert "geom_solref" in dt.randomize_contact_softness.model_fields


def test_speed_range(monkeypatch):
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-Backlash-Dukki")
    assert dt.use_speed_range(cfg, (-0.4, 0.6)).commands["twist"].ranges.lin_vel_x == (-0.4, 0.6)
    monkeypatch.setenv("DUKKI_LIN_VEL_X", "0.1,0.6")
    with pytest.raises(ValueError):
        dt.dukki_lin_vel_x()


def test_terrain_task_registered_with_dukki_terrain():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Terrain-Backlash-Dukki")
    gen = cfg.scene.terrain.terrain_generator
    assert gen.curriculum, "rows must be difficulty levels for the terrain curriculum"
    assert set(gen.sub_terrains) == {"flat", "steps", "stairs", "obstacles", "hills"}
    assert not any("Hf" in type(t).__name__ for t in gen.sub_terrains.values()), "heightfields are slow in Warp"
    assert cfg.curriculum["terrain_levels"].func is dt.terrain_levels_dukki
    assert (cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations) == dt.TERRAIN_ITERATIONS
    assert cfg.sim.nconmax == dt.TERRAIN_NCONMAX
    assert "height_scan" not in cfg.observations["actor"].terms  # blind, as the flat walk


def test_terrain_heights_capped():
    gen = dt.make_dukki_terrains_cfg(step=0.03, slope=0.27)
    assert gen.sub_terrains["steps"].grid_height_range[1] * 2 <= 0.03 + 1e-9
    assert gen.sub_terrains["stairs"].step_height_range[1] <= 0.03
    assert gen.sub_terrains["obstacles"].max_height <= 0.03


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_hills_surface_continuous(seed):
    cfg = dt.BoxHillsTerrainCfg(proportion=1.0, slope_range=(0.02, 0.27))
    cfg.size = (8.0, 8.0)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    out = cfg.function(1.0, spec, np.random.default_rng(seed))
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    xs = np.arange(0.02, 7.99, 0.01)
    z = np.array([2.0 - mujoco.mj_ray(m, d, np.array([x, 4.0, 2.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, np.zeros(1, np.int32)) for x in xs])
    assert np.abs(np.diff(z)).max() <= 0.27 * 0.01 + 1e-4, "a ledge where two boxes meet"
    assert abs(np.abs(z).max() - 0.27) < 1e-3
    assert out.origin[2] == 0.0 and abs(z[np.argmin(abs(xs - 4.0))]) < 1e-6
