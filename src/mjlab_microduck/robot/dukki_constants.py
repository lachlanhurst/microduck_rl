"""Dukki: the microduck reworked on Unitree J288 servos.

Same kinematics, meshes, joint names and order as the microduck (the J288 has
the XL330's 20 x 34 x 26 mm envelope, so the CAD and joint frames are kept), with
three mass changes and the J288 actuator model.

Mass changes, applied to the MJCF inertials by ``apply_dukki_masses``:

* each of the 15 servos: J288 39 g against XL330 18 g, so +21 g per servo, as a
  uniform box filling the servo housing's bounding box, in the body that
  carries the housing;
* battery: the 6S pack weighs 132 g against about 100 g for the NP-F550, so
  +32 g at the pack's place in the trunk;
* head: compute module, carrier, buck, heatsink, amplifier, speaker and
  microphone, estimated at 80 to 120 g net (dukki docs/hardware.md 11.1). +100 g
  as a 55 x 40 x 20 mm box at the head body's current centre of mass. Replace
  with the weighed head, and its real centre of mass, once built.

The whole robot goes from 0.737 kg to about 1.18 kg. Each extra mass is added to
its body's existing inertial with the parallel-axis theorem; the CAD masses are
otherwise unchanged.

Actuators: ``J288DRActuatorCfg`` with the M6 fit from the bench
(``bam/params/unitree_j288/m6.json`` in the bam fork). The reflected rotor
inertia (6.3e-4 kg.m^2), friction, torque clamp (1.0 N.m motor side) and
speed filter (15 ms) come from that file; this module sets the gains, supply
voltage range and command delay.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.actuator import J288DRActuatorCfg
from mjlab_microduck.robot.microduck_constants import (
    BACKLASH_HOME_FRAME,
    FULL_COLLISION,
    HOME_FRAME,
    SERVO_MESH_NAME,
    get_allcollisions_backlash_spec,
    get_allcollisions_spec,
    get_backlash_spec,
    get_standup_spec,
    get_walk_backlash_spec,
    get_walk_spec,
)

# ── Masses [kg] ──────────────────────────────────────────────────────────────
J288_MASS = 0.039
XL330_MASS = 0.018
SERVO_EXTRA_MASS = J288_MASS - XL330_MASS
NUM_SERVOS = 15

BATTERY_MESH_NAME = "np_f970"
BATTERY_EXTRA_MASS = 0.132 - 0.100

HEAD_BODY_NAME = "jaw_soft"
HEAD_EXTRA_MASS = 0.100
HEAD_EXTRA_SIZE = (0.055, 0.040, 0.020)  # box [m], head body axes


def _box_inertia(mass: float, size) -> np.ndarray:
    """Inertia tensor of a uniform box about its centre, box axes [kg.m^2]."""
    x, y, z = size
    return mass / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def _quat_to_mat(quat) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


class _Inertial:
    """A body's inertial in the body frame: mass, centre of mass, tensor about it."""

    def __init__(self, mass: float, com: np.ndarray, inertia: np.ndarray):
        self.mass, self.com, self.inertia = mass, com, inertia

    def add(self, mass: float, com: np.ndarray, inertia: np.ndarray) -> None:
        """Add a rigid mass (inertia about its own centre, body axes)."""
        total = self.mass + mass
        new_com = (self.mass * self.com + mass * com) / total

        def shifted(m, c, inertia_c):
            d = c - new_com
            return inertia_c + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        self.inertia = shifted(self.mass, self.com, self.inertia) + shifted(
            mass, com, inertia
        )
        self.mass, self.com = total, new_com


def apply_dukki_masses(
    spec: mujoco.MjSpec,
    servo_extra_mass: float = SERVO_EXTRA_MASS,
    battery_extra_mass: float = BATTERY_EXTRA_MASS,
    head_extra_mass: float = HEAD_EXTRA_MASS,
) -> mujoco.MjSpec:
    """Add the J288, 6S pack and head-electronics masses to a microduck spec.

    Works on every microduck model (walk, groundcontact, allcollisions and
    their backlash twins): servo housings are found by mesh name on visual
    geoms (no collision), which every model has exactly once per servo.
    """
    model = spec.copy().compile()

    # Compiled geoms follow the spec's geom order; check it while pairing.
    spec_geoms = list(spec.geoms)
    assert len(spec_geoms) == model.ngeom
    inertials: dict[str, _Inertial] = {}

    def inertial(body_id: int) -> _Inertial:
        name = model.body(body_id).name
        if name not in inertials:
            rot = _quat_to_mat(model.body_iquat[body_id])
            inertials[name] = _Inertial(
                float(model.body_mass[body_id]),
                model.body_ipos[body_id].copy(),
                rot @ np.diag(model.body_inertia[body_id]) @ rot.T,
            )
        return inertials[name]

    def add_mesh_box(geom_id: int, mass: float) -> None:
        """A uniform box filling the geom's bounding box (geom frame)."""
        rot = _quat_to_mat(model.geom_quat[geom_id])
        centre = model.geom_pos[geom_id] + rot @ model.geom_aabb[geom_id, :3]
        inertia = rot @ _box_inertia(mass, 2.0 * model.geom_aabb[geom_id, 3:]) @ rot.T
        inertial(model.geom_bodyid[geom_id]).add(mass, centre, inertia)

    servo_mesh = model.mesh(SERVO_MESH_NAME).id
    battery_mesh = model.mesh(BATTERY_MESH_NAME).id
    n_servos = n_batteries = 0
    for gid, sgeom in enumerate(spec_geoms):
        assert sgeom.parent.name == model.body(model.geom_bodyid[gid]).name
        visual = model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0
        if not visual or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_dataid[gid] == servo_mesh:
            add_mesh_box(gid, servo_extra_mass)
            n_servos += 1
        elif model.geom_dataid[gid] == battery_mesh:
            add_mesh_box(gid, battery_extra_mass)
            n_batteries += 1
    assert n_servos == NUM_SERVOS, f"found {n_servos} servo housings"
    assert n_batteries == 1, f"found {n_batteries} battery geoms"

    head = inertial(model.body(HEAD_BODY_NAME).id)
    head.add(head_extra_mass, head.com.copy(), _box_inertia(head_extra_mass, HEAD_EXTRA_SIZE))

    for body in spec.bodies:
        if body.name in inertials:
            new = inertials[body.name]
            i = new.inertia
            body.explicitinertial = True
            body.mass = new.mass
            body.ipos = new.com
            body.fullinertia = [i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]
    return spec


def get_dukki_walk_spec() -> mujoco.MjSpec:
    return apply_dukki_masses(get_walk_spec())


def get_dukki_standup_spec() -> mujoco.MjSpec:
    return apply_dukki_masses(get_standup_spec())


def get_dukki_allcollisions_spec() -> mujoco.MjSpec:
    return apply_dukki_masses(get_allcollisions_spec())


def get_dukki_walk_backlash_spec() -> mujoco.MjSpec:
    return apply_dukki_masses(get_walk_backlash_spec())


def get_dukki_backlash_spec() -> mujoco.MjSpec:
    return apply_dukki_masses(get_backlash_spec())


def get_dukki_allcollisions_backlash_spec() -> mujoco.MjSpec:
    return apply_dukki_masses(get_allcollisions_backlash_spec())


# ── J288 actuators ───────────────────────────────────────────────────────────
# Gains [N.m/rad, N.m.s/rad], output side. Inside the identified range (kp 0.97
# to 4.03, kd 0.02 to 0.05; the model validated at twice the stiffest fitted
# gain). For scale: the XL330 at kp_fw 200 and 7.4 V is about 0.59 N.m/rad at
# stall, with about 0.05 N.m.s/rad of unfiltered back-EMF damping. A starting
# point for the gain sweep (tasks/dukki.py), not a tuned value.
J288_KP = 2.0
J288_KD = 0.05

# 6S LiPo, 21.0 V (3.5 V/cell) to 25.2 V full. The J288's torque only becomes
# voltage-limited above about 20 rad/s, so this mostly moves the speed ceiling.
J288_VIN_RANGE = (21.0, 25.2)
J288_VIN_MIN = 18.0

# Command delay [5 ms physics steps], measured 21/10/2026 on the bench (dukki
# hardware.md 5.5, compute-setup.md 5.9): from a servo's sample in the frame a
# tick reads to that servo acting on the tick's command, 4.25 ms at the median
# (3.75 to 5.25 ms for 90%) through the bridge with protocol 2, all fifteen
# servos moving, plus 0.15 ms of inference on the CM4's Cortex-A72. About one
# step; two covers the slow tail and scheduling jitter (robotd has no real-time
# priority). Was a 1 to 4 step placeholder before the bridge existed.
J288_DELAY_MIN_LAG = 1
J288_DELAY_MAX_LAG = 2

_J288_ACTUATOR_KWARGS = dict(
    motor_name="unitree_j288",
    model="m6",
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=J288_KP,
    kd_fw=J288_KD,
    vin_range=J288_VIN_RANGE,
    # Pack + distribution + harness [Ohm]. A guess: a healthy 6S pack is about
    # 0.02 to 0.05 Ohm, plus wiring.
    vin_drop_resistance_range=(0.02, 0.1),
    vin_min=J288_VIN_MIN,
    delay_min_lag=J288_DELAY_MIN_LAG,
    delay_max_lag=J288_DELAY_MAX_LAG,
)
j288_actuators = J288DRActuatorCfg(**_J288_ACTUATOR_KWARGS)


def make_j288_actuators(**overrides) -> J288DRActuatorCfg:
    """The Dukki actuator cfg with some fields replaced (gain sweeps, tests)."""
    return J288DRActuatorCfg(**{**_J288_ACTUATOR_KWARGS, **overrides})


# Model name -> (spec_fn, init state). Mirrors the microduck robot cfgs.
_DUKKI_MODELS = {
    "walk": (get_dukki_walk_spec, HOME_FRAME),
    "standup": (get_dukki_standup_spec, HOME_FRAME),
    "allcollisions": (get_dukki_allcollisions_spec, HOME_FRAME),
    "walk_backlash": (get_dukki_walk_backlash_spec, BACKLASH_HOME_FRAME),
    "backlash": (get_dukki_backlash_spec, BACKLASH_HOME_FRAME),
    "allcollisions_backlash": (get_dukki_allcollisions_backlash_spec, BACKLASH_HOME_FRAME),
}


def make_dukki_robot_cfg(
    model: str = "walk", actuators: J288DRActuatorCfg = j288_actuators
) -> EntityCfg:
    """A Dukki robot cfg on one of the microduck models.

    The backlash models use the plain J288 actuator: its PD closes on the
    rotor, so it reads the servo joint only.
    """
    spec_fn, init_state = _DUKKI_MODELS[model]
    return EntityCfg(
        spec_fn=spec_fn,
        init_state=init_state,
        collisions=(FULL_COLLISION,),
        articulation=EntityArticulationInfoCfg(
            actuators=(actuators,),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


DUKKI_WALK_ROBOT_CFG = make_dukki_robot_cfg("walk")
DUKKI_STANDUP_ROBOT_CFG = make_dukki_robot_cfg("standup")
DUKKI_ALLCOLLISIONS_ROBOT_CFG = make_dukki_robot_cfg("allcollisions")
DUKKI_WALK_BACKLASH_ROBOT_CFG = make_dukki_robot_cfg("walk_backlash")
DUKKI_BACKLASH_ROBOT_CFG = make_dukki_robot_cfg("backlash")
DUKKI_ALLCOLLISIONS_BACKLASH_ROBOT_CFG = make_dukki_robot_cfg("allcollisions_backlash")
