"""Dukki: the microduck reworked on Unitree J288 servos.

Same kinematics, joint names and order as the microduck. The MJCFs carry the
Dukki itself: geometry from the Fusion model (J288 housings, 6S battery,
bridge, head electronics) and every body's inertial from it too, the printed
parts from their PA12 SLS / TPU materials and the bought parts weighed
(microduck-unitree scripts/f360_to_mjcf, masses.json). The robot is 1.074 kg
there, without cables and fasteners.

This module adds two things the MJCFs do not have:

* Cables and fasteners, ``UNMODELLED_MASS`` (an estimate, 100 g), spread
  over every body in proportion to its mass by ``add_unmodelled_mass``, so
  1.174 kg in all. Each body's mass and inertia scale by the same factor;
  centres of mass, and so the robot's, do not move. Applied when the Dukki
  robot cfgs load the MJCFs, not written into them: set ``UNMODELLED_MASS``
  to 0 to go back to the CAD masses, and a rebuild from Fusion leaves it be.
* The J288 actuator model.

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
    get_allcollisions_backlash_spec,
    get_allcollisions_spec,
    get_backlash_spec,
    get_standup_spec,
    get_walk_backlash_spec,
    get_walk_spec,
)

# ── Unmodelled mass ──────────────────────────────────────────────────────────
# Cables (servo daisy chains, battery and head leads) and fasteners, which the
# Fusion model has none of. About 100 g, estimated 08/11/2026; replace with
# the weighed robot minus the MJCF's 1.074 kg. 0.0 reverts to the CAD masses.
UNMODELLED_MASS = 0.100  # [kg]


def add_unmodelled_mass(spec: mujoco.MjSpec, mass: float | None = None) -> mujoco.MjSpec:
    """Spread ``mass`` (default ``UNMODELLED_MASS``) over every body in proportion to its own mass.

    Each body's mass and inertia (about its centre of mass) scale by
    1 + mass / total, which is the same as uniformly denser parts: body
    centres of mass and principal axes stay put.
    """
    if mass is None:
        mass = UNMODELLED_MASS
    if mass == 0.0:
        return spec
    total = float(spec.copy().compile().body_mass.sum())
    k = 1.0 + mass / total
    for body in spec.bodies:
        if body.mass > 0.0:
            body.mass *= k
            body.inertia = np.asarray(body.inertia) * k
            body.fullinertia = np.asarray(body.fullinertia) * k  # NaN when unset, stays NaN
    return spec


def get_dukki_walk_spec() -> mujoco.MjSpec:
    return add_unmodelled_mass(get_walk_spec())


def get_dukki_standup_spec() -> mujoco.MjSpec:
    return add_unmodelled_mass(get_standup_spec())


def get_dukki_allcollisions_spec() -> mujoco.MjSpec:
    return add_unmodelled_mass(get_allcollisions_spec())


def get_dukki_walk_backlash_spec() -> mujoco.MjSpec:
    return add_unmodelled_mass(get_walk_backlash_spec())


def get_dukki_backlash_spec() -> mujoco.MjSpec:
    return add_unmodelled_mass(get_backlash_spec())


def get_dukki_allcollisions_backlash_spec() -> mujoco.MjSpec:
    return add_unmodelled_mass(get_allcollisions_backlash_spec())


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
