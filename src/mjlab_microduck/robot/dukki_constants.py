"""Dukki: the microduck reworked on Unitree J288 servos.

Same kinematics, joint names and order as the microduck. The MJCFs carry the
Dukki itself: geometry from the Fusion model (J288 housings, 6S battery,
bridge, head electronics) and every body's inertial from it too, the printed
parts from their PA12 SLS / TPU materials and the bought parts weighed
(microduck-unitree scripts/f360_to_mjcf, masses.json). The robot is 1.074 kg
without cables and fasteners. This module adds the J288 actuator model.

Actuators: ``J288DRActuatorCfg`` with the M6 fit from the bench
(``bam/params/unitree_j288/m6.json`` in the bam fork). The reflected rotor
inertia (6.3e-4 kg.m^2), friction, torque clamp (1.0 N.m motor side) and
speed filter (15 ms) come from that file; this module sets the gains, supply
voltage range and command delay.
"""

from __future__ import annotations

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
    "walk": (get_walk_spec, HOME_FRAME),
    "standup": (get_standup_spec, HOME_FRAME),
    "allcollisions": (get_allcollisions_spec, HOME_FRAME),
    "walk_backlash": (get_walk_backlash_spec, BACKLASH_HOME_FRAME),
    "backlash": (get_backlash_spec, BACKLASH_HOME_FRAME),
    "allcollisions_backlash": (get_allcollisions_backlash_spec, BACKLASH_HOME_FRAME),
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
