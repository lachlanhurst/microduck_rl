"""Dukki on soft ground and on uneven terrain (25/11/2026). All off by default.

1. Soft ground. ``DUKKI_SOFT_GROUND="lo,hi[,share]"``: every episode, ``share`` of the robots
   (default 0.5) get a contact time constant drawn in lo..hi seconds for every one of their
   collision geoms, the rest get ``lo``. MuJoCo's contact is a spring-damper the geoms sink
   into; its time constant (solref[0]) sets how soft, at critical damping (solref[1] = 1), so
   a soft contact sinks further and gives sideways too, as carpet pile does. The default
   floor is 0.02 s, about 1 mm under Dukki's weight. The robot's geoms are given a large
   solmix, so their solref, not the floor's or the terrain boxes', sets the contact: the
   terrain then needs no per-robot memory of its own (one copy of geom_solref per robot is
   allocated either way, for every geom). On the robot the carpet gives up to about 5 mm
   under a foot (the user, 24/11/2026); ``scripts/dukki_sinkage.py`` maps time constant to
   sinkage.

2. Torsional friction at the soles. ``DUKKI_SOLE_TORSION="lo,hi"`` gives the two sole geoms
   condim 4 (dukki_constants.sole_condim, at load) and draws their torsional friction in
   lo..hi [m] per episode. MuJoCo takes the larger of the two geoms' coefficients, and the
   floor's is 0.005.

3. Terrain sized for Dukki: ``make_dukki_terrains_cfg`` for the ``Mjlab-Velocity-Terrain-...-Dukki``
   tasks. Steps and obstacles up to ``DUKKI_TERRAIN_STEP`` (default 0.03 m) at the hardest
   row, slopes to ``DUKKI_TERRAIN_SLOPE`` (rise over run, default 0.27, about 15 degrees).
   The stock terrain curriculum promotes a robot that walks half a tile (4 m) in an episode,
   which Dukki at 0.1 to 0.4 m/s with commands changing direction never does;
   ``terrain_levels_dukki`` promotes a robot that lasts the episode having left its start by
   ``TERRAIN_UP_DISTANCE`` and demotes one that fell.

4. Faster walking. ``DUKKI_LIN_VEL_X="lo,hi"`` widens the forward/backward command range from the
   recipe's -0.4..0.4 m/s (the walk reaches about 0.25 m/s at a 0.30 command). robotd's own speed
   limit has to follow for the robot to be asked for more.

5. Walking must pay more than standing (25/11/2026, the user's review). The stock
   ``track_angular_velocity`` is exp(-((yaw_cmd - yaw)^2 + roll_rate^2 + pitch_rate^2) / std^2): a robot
   standing still keeps the whole term at a zero yaw command, a stepping trunk rocks and loses most of it,
   so the term pays standing over walking at almost every command, and more so on soft ground and steps
   (every fine-tune of 25/11 drifted to standing). ``DUKKI_YAW_ONLY=1`` scores yaw alone (same weight and
   std); roll and pitch are priced by ``body_ang_vel``, ``DUKKI_BODY_ANG_VEL=weight`` (recipe -0.05).
   ``DUKKI_LIN_PROGRESS=weight`` adds a term linear in the achieved fraction of the commanded speed (velocity
   along the command over its magnitude, clipped to 0..1, zero below 0.05 m/s commanded): unlike the fine
   Gaussian (std 0.1), it has a gradient from standing still.

6. Bigger pushes, mostly backwards. ``DUKKI_PUSH="max,back_share[,interval_lo,interval_hi]"``: every 3 to 6 s
   (or the interval given) the trunk gets a horizontal kick up to ``max`` m/s in the robot's own frame,
   ``back_share`` of them backwards. The recipe's pushes are +-0.3 m/s on each world axis; the robot's walk
   fell to a single 0.7 m/s backward kick a third of the time (scripts/dukki_push_eval.py, 25/11/2026).

7. Twist (and shear) under the soles. ``DUKKI_SOLE_TWIST="klo,khi,dlo,dhi[,share]"``: robot/dukki_constants.py
   puts a passive hinge in each foot about the sole's normal; every episode ``share`` of the robots
   (default 0.5) get its stiffness drawn in klo..khi [N.m/rad] and damping in dlo..dhi [N.m.s/rad],
   the rest a stiff one (``SOLE_TWIST_HARD``). On carpet the robot shook after stopping, the whole
   robot twisting on its feet at about 6 Hz with the hip yaws still (26/11/2026, dukki
   scratch/policies/shake_*.json); the soft contact above sinks but barely twists, and training on it
   left simulation dead still where the robot shook (runs/eval/stillness/). ``DUKKI_SOLE_SHEAR`` adds two
   slides in each sole's plane; as "klo,khi,dlo,dhi" (N/m, N.s/m) the soft-soled robots draw them per
   episode too (the hard ones get ``SOLE_SHEAR_HARD``). Of everything tried, only soft shear kept the
   simulated robot twisting after a stop (runs/eval/stillness/NOTES.md, carpet search). A fine-tune of the
   robot's walk on these soles with robotd's filter randomised (``DUKKI_LOWPASS_RANGE``) did not make it
   stiller after a stop and lost turning on the spot. The robot's shake turned out to be the stand-up
   network doing the standing (robotd's ``stand`` slot); the walk network standing at legs 0.84 is still
   on carpet (dukki docs/policy-testing.md section 13).

The policy stays blind (no height scan): it meets the terrain through its legs, as on the robot.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mjlab.terrains as terrain_gen
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput

_ROBOT = SceneEntityCfg("robot")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Soft ground
# ─────────────────────────────────────────────────────────────────────────────

#: solmix given to the robot's collision geoms with soft ground on: the robot's solref then
#: carries 99.99% of the mix against a geom at MuJoCo's default 1.
ROBOT_SOLMIX = 1.0e4
DEFAULT_SOFT_SHARE = 0.5


def dukki_soft_ground() -> tuple[float, float, float] | None:
    """``DUKKI_SOFT_GROUND`` ("lo,hi[,share]") parsed, or None when unset."""
    text = os.environ.get("DUKKI_SOFT_GROUND")
    if not text:
        return None
    parts = [float(v) for v in text.split(",")]
    if len(parts) not in (2, 3):
        raise ValueError(f"DUKKI_SOFT_GROUND must be lo,hi[,share]; got {text!r}")
    lo, hi = parts[0], parts[1]
    share = parts[2] if len(parts) == 3 else DEFAULT_SOFT_SHARE
    if not (0.0 < lo <= hi) or not (0.0 <= share <= 1.0):
        raise ValueError(f"DUKKI_SOFT_GROUND needs 0 < lo <= hi and share in 0..1; got {text!r}")
    return lo, hi, share


def _robot_collision_geoms(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    ids = env.scene[asset_cfg.name].indexing.geom_ids
    m = env.sim.mj_model
    keep = [int(g) for g in ids.tolist() if m.geom_contype[int(g)] or m.geom_conaffinity[int(g)]]
    return torch.tensor(keep, dtype=torch.long, device=env.device)


@requires_model_fields("geom_solref")
def randomize_contact_softness(
    env,
    env_ids: torch.Tensor | None,
    time_constant_range: tuple[float, float],
    soft_share: float = DEFAULT_SOFT_SHARE,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
    """Per episode, the robot's contact time constant: lo..hi for ``soft_share`` of the robots,
    lo for the rest (module docstring, 1)."""
    geoms = getattr(env, "_dukki_soft_geoms", None)
    if geoms is None:
        geoms = _robot_collision_geoms(env, asset_cfg)
        env._dukki_soft_geoms = geoms
        env.sim.model.geom_solmix[:, geoms] = ROBOT_SOLMIX
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, dtype=torch.long)
    lo, hi = time_constant_range
    n = len(env_ids)
    tc = torch.rand(n, device=env.device) * (hi - lo) + lo
    tc = torch.where(torch.rand(n, device=env.device) < soft_share, tc, torch.full_like(tc, lo))
    solref = env.sim.model.geom_solref
    rows, cols = torch.meshgrid(env_ids, geoms, indexing="ij")
    block = solref[rows, cols]
    block[..., 0] = tc[:, None]
    block[..., 1] = 1.0
    solref[rows, cols] = block
    env._dukki_contact_tc = getattr(env, "_dukki_contact_tc", torch.full((env.num_envs,), lo, device=env.device))
    env._dukki_contact_tc[env_ids] = tc


def use_soft_ground(cfg: ManagerBasedRlEnvCfg, spec: tuple[float, float, float] | None = None) -> ManagerBasedRlEnvCfg:
    spec = dukki_soft_ground() if spec is None else spec
    if spec is None:
        return cfg
    lo, hi, share = spec
    cfg.events["soft_ground"] = EventTermCfg(
        func=randomize_contact_softness,
        mode="reset",
        params={"time_constant_range": (lo, hi), "soft_share": share},
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2. Torsional friction at the soles
# ─────────────────────────────────────────────────────────────────────────────

SOLE_GEOMS = ("left_foot_collision", "right_foot_collision")


def dukki_sole_torsion() -> tuple[float, float] | None:
    """``DUKKI_SOLE_TORSION`` ("lo,hi") parsed, or None when unset."""
    text = os.environ.get("DUKKI_SOLE_TORSION")
    if not text:
        return None
    lo, hi = (float(v) for v in text.split(","))
    if not (0.0 < lo <= hi):
        raise ValueError(f"DUKKI_SOLE_TORSION needs 0 < lo <= hi; got {text!r}")
    return lo, hi


def use_sole_torsion(cfg: ManagerBasedRlEnvCfg, rng: tuple[float, float] | None = None) -> ManagerBasedRlEnvCfg:
    rng = dukki_sole_torsion() if rng is None else rng
    if rng is None:
        return cfg
    cfg.events["sole_torsion"] = EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=SOLE_GEOMS),
            "ranges": rng,
            "operation": "abs",
            "axes": [1],
        },
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 3. Terrain sized for Dukki
# ─────────────────────────────────────────────────────────────────────────────

TERRAIN_STEP = float(os.environ.get("DUKKI_TERRAIN_STEP", "0.03"))
TERRAIN_SLOPE = float(os.environ.get("DUKKI_TERRAIN_SLOPE", "0.27"))
#: Share of robots on the flat column (``DUKKI_TERRAIN_FLAT``, default 0.10); the other types keep their
#: proportions between them. Turning on the spot trains poorly on steps and slopes: on the terrain the
#: walk's yaw tracking error ran 2.4-2.7 against 1.8 on flat soft ground, and turn in place fell from
#: 0.72 to 0.06 rad/s within 250 iterations (25/11/2026).
TERRAIN_FLAT = float(os.environ.get("DUKKI_TERRAIN_FLAT", "0.10"))
#: Obstacle blocks' top height and count range (``DUKKI_OBSTACLE_HEIGHT``, default the step height;
#: ``DUKKI_OBSTACLE_COUNT`` "lo,hi", default 40,120) and the steps field (``DUKKI_STEPS_DENSE=1``: 25 cm cells in a
#: 2.5 m ring around the start, in place of 45 cm cells over the whole tile). The user, 25/11 evening: the obstacle
#: and step tests were still too sparse and could be a little higher.
OBSTACLE_HEIGHT = float(os.environ.get("DUKKI_OBSTACLE_HEIGHT", "0") or 0) or None
OBSTACLE_COUNT = tuple(int(v) for v in os.environ.get("DUKKI_OBSTACLE_COUNT", "40,120").split(","))
STEPS_DENSE = os.environ.get("DUKKI_STEPS_DENSE", "0") not in ("", "0")
#: Distance from the start [m] a robot must reach, without falling, to be promoted.
TERRAIN_UP_DISTANCE = float(os.environ.get("DUKKI_TERRAIN_UP", "1.0"))


@dataclass(kw_only=True)
class BoxHillsTerrainCfg(SubTerrainCfg):
    """A hill (or, on half the tiles, a valley) either side of the start platform, along x: from
    the platform's edge a ramp of ``ramp_length``, a flat crest ``crest_length`` long, a ramp back
    down to the platform's level, then flat to the tile's edge. The ramps' gradient (rise over
    run) goes from ``slope_range[0]`` to ``[1]`` with difficulty. Every surface is the top face of a
    box; the faces meet edge to edge, so there is no ledge at a crease. Walking along y crosses
    the slopes sideways."""

    slope_range: tuple[float, float] = (0.02, 0.27)
    platform_width: float = 1.0
    ramp_length: float = 1.0
    crest_length: float = 0.4
    thickness: float = 0.5

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        body = spec.body("terrain")
        sx, sy = self.size
        cx, cy = sx / 2, sy / 2
        slope = self.slope_range[0] + difficulty * (self.slope_range[1] - self.slope_range[0])
        rise = slope * self.ramp_length * (1.0 if rng.uniform() < 0.5 else -1.0)
        t = self.thickness
        geoms = []

        def flat(x0, x1, z):
            g = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=((x1 - x0) / 2, sy / 2, t / 2),
                pos=((x0 + x1) / 2, cy, z - t / 2),
            )
            geoms.append(TerrainGeometry(geom=g, color=(0.5, 0.5, 0.5, 1.0)))

        def ramp(x0, z0, x1, z1):
            a = math.atan2(z1 - z0, x1 - x0)
            length = math.hypot(x1 - x0, z1 - z0)
            nx, nz = -math.sin(a), math.cos(a)  # the top face's normal
            g = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(length / 2, sy / 2, t / 2),
                pos=((x0 + x1) / 2 - nx * t / 2, cy, (z0 + z1) / 2 - nz * t / 2),
                quat=(math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0),
            )
            geoms.append(TerrainGeometry(geom=g, color=(0.45, 0.55, 0.75, 1.0)))

        p = self.platform_width / 2
        flat(cx - p, cx + p, 0.0)
        for side in (1.0, -1.0):
            # Breakpoints outward from the platform's edge: (distance from centre, height).
            pts = [
                (p, 0.0),
                (p + self.ramp_length, rise),
                (p + self.ramp_length + self.crest_length, rise),
                (p + 2 * self.ramp_length + self.crest_length, 0.0),
                (sx / 2, 0.0),
            ]
            for (d0, z0), (d1, z1) in zip(pts[:-1], pts[1:]):
                xa, xb = cx + side * d0, cx + side * d1
                (xa, za), (xb, zb) = sorted([(xa, z0), (xb, z1)])
                if z0 == z1:
                    flat(xa, xb, z0)
                else:
                    ramp(xa, za, xb, zb)
        return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geoms)


@dataclass(kw_only=True)
class DenseObstaclesTerrainCfg(SubTerrainCfg):
    """Blocks scattered in a ring from just off the start platform to ``outer_radius`` (where a robot gets to
    in an episode), ``count_range`` of them by difficulty, ``max_height`` x (0.3..1, by difficulty and per
    block) tall, 5 to 25 cm across, any yaw. At full difficulty about a fifth of the ring is covered."""

    count_range: tuple[int, int] = (40, 120)
    max_height: float = 0.03
    platform_width: float = 1.0
    outer_radius: float = 2.5
    size_range: tuple[float, float] = (0.05, 0.25)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        body = spec.body("terrain")
        sx, sy = self.size
        cx, cy = sx / 2, sy / 2
        floor = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(sx / 2, sy / 2, 0.25), pos=(cx, cy, -0.25))
        geoms = [TerrainGeometry(geom=floor, color=(0.5, 0.5, 0.5, 1.0))]
        n = int(self.count_range[0] + difficulty * (self.count_range[1] - self.count_range[0]))
        inner = self.platform_width / 2 * math.sqrt(2) + 0.05
        for _ in range(n):
            r = math.sqrt(rng.uniform(inner**2, self.outer_radius**2))  # uniform over the ring's area
            a = rng.uniform(0, 2 * math.pi)
            w, l = rng.uniform(*self.size_range), rng.uniform(*self.size_range)
            h = self.max_height * rng.uniform(0.3, 1.0) * (0.3 + 0.7 * difficulty)
            yaw = rng.uniform(0, math.pi)
            g = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(w / 2, l / 2, h / 2),
                pos=(cx + r * math.cos(a), cy + r * math.sin(a), h / 2),
                quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
            )
            geoms.append(TerrainGeometry(geom=g, color=(0.45, 0.55, 0.75, 1.0)))
        return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geoms)


@dataclass(kw_only=True)
class DenseStepsTerrainCfg(SubTerrainCfg):
    """Square cells of ``cell`` m in a ring from just off the start platform to ``outer_radius``, each raised 0 to
    max_step x difficulty above the floor (so neighbours differ by up to that, up or down), flat floor beyond and on
    the platform. About 300 cells."""

    max_step: float = 0.03
    cell: float = 0.20
    platform_width: float = 1.0
    outer_radius: float = 2.5

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        body = spec.body("terrain")
        sx, sy = self.size
        cx, cy = sx / 2, sy / 2
        floor = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(sx / 2, sy / 2, 0.25), pos=(cx, cy, -0.25))
        geoms = [TerrainGeometry(geom=floor, color=(0.5, 0.5, 0.5, 1.0))]
        top_max = self.max_step * max(difficulty, 0.1)
        p = self.platform_width / 2
        n = int(self.outer_radius / self.cell) + 1
        for i in range(-n, n):
            for j in range(-n, n):
                x0, y0 = i * self.cell, j * self.cell
                xc, yc = x0 + self.cell / 2, y0 + self.cell / 2
                if math.hypot(xc, yc) > self.outer_radius:
                    continue
                if max(abs(x0), abs(x0 + self.cell)) <= p and max(abs(y0), abs(y0 + self.cell)) <= p:
                    continue  # the start platform stays at 0
                if abs(xc) < p and abs(yc) < p:
                    continue
                top = rng.uniform(0.0, top_max)
                g = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(self.cell / 2, self.cell / 2, (top + 0.2) / 2),
                    pos=(cx + xc, cy + yc, (top - 0.2) / 2),
                )
                geoms.append(TerrainGeometry(geom=g, color=(0.45, 0.55, 0.75, 1.0)))
        return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geoms)


def make_dukki_terrains_cfg(step: float = TERRAIN_STEP, slope: float = TERRAIN_SLOPE) -> TerrainGeneratorCfg:
    """Rows are difficulty 0..1 (the curriculum moves robots between them), one column per terrain
    type, robots spread over the columns by proportion. 8 m tiles; the robot starts on a 1 m
    platform at each tile's centre. About 3,500 geoms: every per-robot geom field (friction,
    solref) and the foot-height sensor's ray-cast scene hold every geom once per robot."""
    gen = _terrains_cfg(step, slope)
    rest = sum(t.proportion for k, t in gen.sub_terrains.items() if k != "flat")
    for k, t in gen.sub_terrains.items():
        t.proportion = TERRAIN_FLAT if k == "flat" else t.proportion * (1.0 - TERRAIN_FLAT) / rest
    return gen


def _terrains_cfg(step: float, slope: float) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        # Curriculum layout: one column per type, rows in rising difficulty, so the terrain
        # curriculum's levels mean something. The generator's default (off) scatters 200 tiles
        # of random type and difficulty: 17,900 geoms, and the foot-height sensor's ray-cast
        # scene over every geom for every robot ran out of the 16 GB at 4096 robots.
        curriculum=True,
        sub_terrains={
            "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.10),
            # Steps up and down: each 0.45 m cell at a height in +-bound, so neighbours differ
            # by up to 2 x bound. bound = step / 2 at the hardest row.
            "steps": DenseStepsTerrainCfg(proportion=0.25, max_step=step) if STEPS_DENSE else terrain_gen.BoxRandomGridTerrainCfg(
                proportion=0.25,
                grid_width=0.45,
                grid_height_range=(0.002, step / 2),
                platform_width=1.0,
            ),
            # Stairs down from the platform (and up on the way back), the whole step each time.
            "stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
                proportion=0.20,
                step_height_range=(0.005, step),
                step_width=0.30,
                platform_width=1.0,
                border_width=1.0,
            ),
            # Fixed clutter where the robot walks (DenseObstaclesTerrainCfg). The stock scattered boxes
            # (40 over the whole 8 m tile) covered a few percent of the ground, and a robot walking 1-2 m
            # from the start rarely met one (the user, from the videos, 25/11/2026).
            "obstacles": DenseObstaclesTerrainCfg(
                proportion=0.20, max_height=OBSTACLE_HEIGHT or step, count_range=OBSTACLE_COUNT
            ),
            # Hills and valleys of tilted boxes (BoxHillsTerrainCfg). MuJoCo Warp's heightfield
            # collisions cost 120 of 196 ms a step at 4096 robots (Pollen's heightfield pyramid
            # and a heightfield of lumps, 25/11/2026); boxes cost next to nothing.
            "hills": BoxHillsTerrainCfg(proportion=0.25, slope_range=(0.02, slope)),
        },
        add_lights=False,
    )


def terrain_levels_dukki(env, env_ids: torch.Tensor, command_name: str = "twist", asset_cfg: SceneEntityCfg = _ROBOT):
    """Promote robots that lasted the episode and left their start by TERRAIN_UP_DISTANCE, demote
    the ones that fell (module docstring, 3)."""
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    timed_out = env.termination_manager.time_outs[env_ids]
    move_up = timed_out & (distance > TERRAIN_UP_DISTANCE)
    move_down = ~timed_out
    terrain.update_env_origins(env_ids, move_up, move_down)
    levels = terrain.terrain_levels.float()
    return {"mean": levels.mean(), "max": levels.max()}


#: Solver iterations on Dukki's terrain. Pollen's rough recipe raises them to 30 (line search 50)
#: against NaNs on box edges; every Dukki policy trained at the flat recipe's 10 (20), and the
#: walk policy gearplay 10000 falls twice as often at 30 on the same terrain (131 against 60 of
#: 720 in dukki_terrain_eval.py, 25/11/2026), 4 of 120 against 0 even on a plane. So 10, as the
#: policies the robot runs were trained; the NaN termination stays. DUKKI_TERRAIN_ITERATIONS
#: overrides ("iterations,ls_iterations").
TERRAIN_ITERATIONS = tuple(int(v) for v in os.environ.get("DUKKI_TERRAIN_ITERATIONS", "10,20").split(","))


#: Contacts per robot. Pollen's rough recipe's 200 ran out of GPU memory creating the CUDA graph at
#: 4096 robots (25/11/2026); 64 fits (11 GB) and only the soles collide in the walk model.
TERRAIN_NCONMAX = int(os.environ.get("DUKKI_TERRAIN_NCONMAX", "64"))


def use_dukki_terrain(cfg: ManagerBasedRlEnvCfg, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Swap a rough velocity cfg's terrain for Dukki's and its curriculum for terrain_levels_dukki,
    at the flat recipe's solver iterations (TERRAIN_ITERATIONS). Robots start in the bottom four
    rows; the stock start is any row."""
    cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations = TERRAIN_ITERATIONS
    cfg.sim.nconmax = TERRAIN_NCONMAX
    cfg.scene.terrain.max_init_terrain_level = 3
    gen = make_dukki_terrains_cfg()
    if play:
        gen.num_rows = 5
    cfg.scene.terrain.terrain_generator = gen
    if os.environ.get("DUKKI_TURN_FLAT_ONLY", "0") not in ("", "0") and "twist" in cfg.commands:
        # Turn-in-place commands only on the flat tiles: every terrain run of 25/11 lost turning on the spot.
        cfg.commands["twist"].turn_in_place_flat_only = True
    if "terrain_levels" in cfg.curriculum:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_dukki, params={"command_name": "twist"}
        )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 4. Faster walking
# ─────────────────────────────────────────────────────────────────────────────


def dukki_lin_vel_x() -> tuple[float, float] | None:
    """``DUKKI_LIN_VEL_X`` ("lo,hi") parsed, or None when unset."""
    text = os.environ.get("DUKKI_LIN_VEL_X")
    if not text:
        return None
    lo, hi = (float(v) for v in text.split(","))
    if not lo < 0.0 < hi:
        raise ValueError(f"DUKKI_LIN_VEL_X needs lo < 0 < hi; got {text!r}")
    return lo, hi


def use_speed_range(cfg: ManagerBasedRlEnvCfg, rng: tuple[float, float] | None = None) -> ManagerBasedRlEnvCfg:
    rng = dukki_lin_vel_x() if rng is None else rng
    twist = cfg.commands.get("twist")
    if rng is None or twist is None or "track_linear_velocity" not in cfg.rewards:
        return cfg
    twist.ranges.lin_vel_x = rng
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 5. Walking must pay more than standing
# ─────────────────────────────────────────────────────────────────────────────

PROGRESS_MIN_CMD = 0.05  # [m/s] below this the command counts as standing and the progress term pays 0


def track_yaw_rate_only(env, std: float, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """exp(-(yaw_cmd - yaw rate)^2 / std^2): the stock term without the roll and pitch rates."""
    w = env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(env.command_manager.get_command(command_name)[:, 2] - w) / std**2)


def linear_progress(env, command_name: str = "twist", asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Velocity along the commanded xy direction over the commanded speed, clipped to 0..1; 0 when the
    command is under PROGRESS_MIN_CMD."""
    cmd = env.command_manager.get_command(command_name)[:, :2]
    speed = torch.linalg.norm(cmd, dim=1)
    v = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]
    frac = torch.sum(v * cmd, dim=1) / torch.clamp(speed, min=1e-6) ** 2
    return torch.where(speed >= PROGRESS_MIN_CMD, frac.clamp(0.0, 1.0), torch.zeros_like(frac))


def use_walk_margin(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """DUKKI_YAW_ONLY, DUKKI_BODY_ANG_VEL and DUKKI_LIN_PROGRESS (module docstring, 5). All off by default."""
    if "track_angular_velocity" not in cfg.rewards:
        return cfg
    if os.environ.get("DUKKI_YAW_ONLY", "0") not in ("", "0"):
        cfg.rewards["track_angular_velocity"].func = track_yaw_rate_only
    w = os.environ.get("DUKKI_BODY_ANG_VEL")
    if w and "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = float(w)
    w = os.environ.get("DUKKI_LIN_PROGRESS")
    if w:
        cfg.rewards["lin_progress"] = RewardTermCfg(func=linear_progress, weight=float(w), params={"command_name": "twist"})
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bigger pushes, mostly backwards
# ─────────────────────────────────────────────────────────────────────────────


def dukki_push() -> tuple[float, float, float, float] | None:
    """``DUKKI_PUSH`` ("max,back_share[,interval_lo,interval_hi]") parsed, or None when unset. Recipe pushes are
    +-0.3 m/s on each world axis every 3-6 s; the robot's walk falls to a single 0.7 m/s backward kick a third of
    the time and to 0.9 m/s almost always, backwards (dukki_push_eval.py, 25/11/2026)."""
    text = os.environ.get("DUKKI_PUSH")
    if not text:
        return None
    v = [float(x) for x in text.split(",")]
    if len(v) not in (2, 4) or not (v[0] > 0 and 0.0 <= v[1] <= 1.0):
        raise ValueError(f"DUKKI_PUSH must be max,back_share[,interval_lo,interval_hi]; got {text!r}")
    return (v[0], v[1], v[2], v[3]) if len(v) == 4 else (v[0], v[1], 3.0, 6.0)


def push_body_frame(env, env_ids: torch.Tensor, max_speed: float, back_share: float, asset_cfg: SceneEntityCfg = _ROBOT) -> None:
    """Add a horizontal kick to the root velocity, uniform in size up to ``max_speed``: ``back_share`` of them
    backwards relative to the robot's heading (within +-30 deg), the rest in any direction."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    size = torch.rand(n, device=env.device) * max_speed
    heading = asset.data.heading_w[env_ids]
    back = torch.rand(n, device=env.device) < back_share
    angle = torch.where(
        back,
        heading + math.pi + (torch.rand(n, device=env.device) - 0.5) * (math.pi / 3),
        torch.rand(n, device=env.device) * 2 * math.pi,
    )
    vel = asset.data.root_link_vel_w[env_ids].clone()
    vel[:, 0] += size * torch.cos(angle)
    vel[:, 1] += size * torch.sin(angle)
    asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def use_big_pushes(cfg: ManagerBasedRlEnvCfg, spec: tuple[float, float, float, float] | None = None) -> ManagerBasedRlEnvCfg:
    spec = dukki_push() if spec is None else spec
    if spec is None or "push_robot" not in cfg.events or "track_linear_velocity" not in cfg.rewards:
        return cfg
    term = cfg.events["push_robot"]
    term.func = push_body_frame
    term.params = {"max_speed": spec[0], "back_share": spec[1]}
    term.interval_range_s = (spec[2], spec[3])
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 7. Twist and shear under the soles
# ─────────────────────────────────────────────────────────────────────────────

#: Stiffness [N.m/rad] and damping [N.m.s/rad] of the sole hinges for the robots on a hard floor.
SOLE_TWIST_HARD = (20.0, 0.5)
DEFAULT_TWIST_SHARE = 0.5


def dukki_sole_twist() -> tuple[float, float, float, float, float] | None:
    """``DUKKI_SOLE_TWIST`` ("klo,khi,dlo,dhi[,share]") parsed, or None when unset."""
    text = os.environ.get("DUKKI_SOLE_TWIST")
    if not text:
        return None
    parts = [float(v) for v in text.split(",")]
    if len(parts) not in (4, 5):
        raise ValueError(f"DUKKI_SOLE_TWIST must be klo,khi,dlo,dhi[,share]; got {text!r}")
    share = parts[4] if len(parts) == 5 else DEFAULT_TWIST_SHARE
    klo, khi, dlo, dhi = parts[:4]
    if not (0.0 < klo <= khi) or not (0.0 <= dlo <= dhi) or not (0.0 <= share <= 1.0):
        raise ValueError(f"DUKKI_SOLE_TWIST needs 0 < klo <= khi, 0 <= dlo <= dhi, share in 0..1; got {text!r}")
    return klo, khi, dlo, dhi, share


def _sole_twist_ids(env, asset_cfg: SceneEntityCfg):
    from mjlab_microduck.robot.dukki_constants import SOLE_TWIST_JOINTS

    robot = env.scene[asset_cfg.name]
    idx = [list(robot.joint_names).index(n) for n in SOLE_TWIST_JOINTS]
    jnt = robot.indexing.joint_ids[idx].to(env.device)
    dof = robot.indexing.joint_v_adr[idx].to(env.device)
    return jnt, dof


@requires_model_fields("jnt_stiffness", "dof_damping")
def set_sole_shear(env, env_ids: torch.Tensor, stiffness: torch.Tensor, damping: torch.Tensor,
                   asset_cfg: SceneEntityCfg = _ROBOT) -> None:
    """The shear slides (DUKKI_SOLE_SHEAR, robot/dukki_constants.sole_twist) of robots ``env_ids``:
    stiffness N/m and damping N.s/m, one value per robot."""
    from mjlab_microduck.robot.dukki_constants import SOLE_TWIST_JOINTS

    robot = env.scene[asset_cfg.name]
    names = [f"{n}_shear_{t}" for n in SOLE_TWIST_JOINTS for t in ("u", "w")]
    idx = [list(robot.joint_names).index(n) for n in names]
    jnt = robot.indexing.joint_ids[idx].tolist()
    dof = robot.indexing.joint_v_adr[idx].tolist()
    env_ids = env_ids.to(env.device, dtype=torch.long)
    k = env.sim.model.jnt_stiffness
    d = env.sim.model.dof_damping
    for j, v in zip(jnt, dof):
        k[env_ids, j] = stiffness.to(k.dtype)
        d[env_ids, v] = damping.to(d.dtype)


@requires_model_fields("jnt_stiffness", "dof_damping")
def set_sole_twist(env, env_ids: torch.Tensor, stiffness: torch.Tensor, damping: torch.Tensor,
                   asset_cfg: SceneEntityCfg = _ROBOT) -> None:
    """Both sole hinges of robots ``env_ids`` to these stiffness and damping values (one per robot)."""
    jnt, dof = _sole_twist_ids(env, asset_cfg)
    env_ids = env_ids.to(env.device, dtype=torch.long)
    k = env.sim.model.jnt_stiffness
    d = env.sim.model.dof_damping
    for j, v in zip(jnt.tolist(), dof.tolist()):
        k[env_ids, j] = stiffness.to(k.dtype)
        d[env_ids, v] = damping.to(d.dtype)


@requires_model_fields("jnt_stiffness", "dof_damping")
def randomize_sole_twist(env, env_ids: torch.Tensor | None, stiffness_range: tuple[float, float],
                         damping_range: tuple[float, float], soft_share: float = DEFAULT_TWIST_SHARE,
                         asset_cfg: SceneEntityCfg = _ROBOT) -> None:
    """Per episode (module docstring, 6)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    k = torch.rand(n, device=env.device) * (stiffness_range[1] - stiffness_range[0]) + stiffness_range[0]
    d = torch.rand(n, device=env.device) * (damping_range[1] - damping_range[0]) + damping_range[0]
    soft = torch.rand(n, device=env.device) < soft_share
    k = torch.where(soft, k, torch.full_like(k, SOLE_TWIST_HARD[0]))
    d = torch.where(soft, d, torch.full_like(d, SOLE_TWIST_HARD[1]))
    set_sole_twist(env, env_ids, k, d, asset_cfg)
    shear = dukki_sole_shear()
    if shear is not None:  # the same robots on soft (sheared) soles as on twisting ones
        klo, khi, dlo, dhi = shear
        ks = torch.rand(n, device=env.device) * (khi - klo) + klo
        ds = torch.rand(n, device=env.device) * (dhi - dlo) + dlo
        ks = torch.where(soft, ks, torch.full_like(ks, SOLE_SHEAR_HARD[0]))
        ds = torch.where(soft, ds, torch.full_like(ds, SOLE_SHEAR_HARD[1]))
        set_sole_shear(env, env_ids, ks, ds, asset_cfg)


#: Shear slide stiffness [N/m] and damping [N.s/m] for the robots on a hard floor.
SOLE_SHEAR_HARD = (5000.0, 20.0)


def dukki_sole_shear() -> tuple[float, float, float, float] | None:
    """``DUKKI_SOLE_SHEAR`` as "klo,khi,dlo,dhi" (N/m, N.s/m) drawn per episode; a bare flag ("1") only adds
    the slides (robot/dukki_constants.sole_twist) for an evaluation to set."""
    text = os.environ.get("DUKKI_SOLE_SHEAR")
    if not text or "," not in text:
        return None
    v = [float(x) for x in text.split(",")]
    if len(v) != 4 or not (0 < v[0] <= v[1]) or not (0 <= v[2] <= v[3]):
        raise ValueError(f"DUKKI_SOLE_SHEAR must be klo,khi,dlo,dhi; got {text!r}")
    return v[0], v[1], v[2], v[3]


def use_sole_twist(cfg: ManagerBasedRlEnvCfg, spec: tuple[float, float, float, float, float] | None = None) -> ManagerBasedRlEnvCfg:
    spec = dukki_sole_twist() if spec is None else spec
    if spec is None:
        return cfg
    from dataclasses import replace

    from mjlab_microduck.robot.dukki_constants import SOLE_TWIST_JOINTS

    robot = cfg.scene.entities["robot"]
    init = robot.init_state
    extra = {n: 0.0 for n in SOLE_TWIST_JOINTS}
    if os.environ.get("DUKKI_SOLE_SHEAR"):
        extra.update({f"{n}_shear_{t}": 0.0 for n in SOLE_TWIST_JOINTS for t in ("u", "w")})
    robot.init_state = replace(init, joint_pos={**init.joint_pos, **extra})
    klo, khi, dlo, dhi, share = spec
    cfg.events["sole_twist"] = EventTermCfg(
        func=randomize_sole_twist,
        mode="reset",
        params={"stiffness_range": (klo, khi), "damping_range": (dlo, dhi), "soft_share": share},
    )
    return cfg
