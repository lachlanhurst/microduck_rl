"""Dukki: the microduck reworked on Unitree J288 servos.

Same kinematics, joint names and order as the microduck. The MJCFs carry the
Dukki itself: geometry from the Fusion model (J288 housings, 6S battery,
bridge, head electronics) and every body's inertial from it too, the printed
parts from their PA12 SLS / TPU materials and the bought parts weighed
(microduck-unitree scripts/f360_to_mjcf, masses.json). The robot is 1.074 kg
there, without cables and fasteners.

This module adds four things the MJCFs do not have:

* Cables and fasteners, ``UNMODELLED_MASS`` (an estimate, 100 g), spread
  over every body in proportion to its mass by ``add_unmodelled_mass``, so
  1.174 kg in all. Each body's mass and inertia scale by the same factor;
  centres of mass, and so the robot's, do not move. Applied when the Dukki
  robot cfgs load the MJCFs, not written into them: set ``UNMODELLED_MASS``
  to 0 to go back to the CAD masses, and a rebuild from Fusion leaves it be.
* A tighter head roll (left to right tilt) range, ``HEAD_ROLL_LIMIT_DEG``
  (15 degrees either way, a design constraint from 09/11/2026; the MJCF has
  25), set by ``limit_head_roll`` the same way: ``None`` goes back to the
  MJCF's range.
* The real robot's joint stops, ``JOINT_TRAVEL_DEG`` (measured by hand on
  the assembled robot 21/11/2026, against the calibration jig's zeros), set
  by ``limit_joint_travel``: each joint's range narrows to where the robot
  meets something, never widens. ``APPLY_JOINT_TRAVEL = False`` reverts.
  ``target_clip_rad`` is the firmware's target clamp, which tasks/dukki.py
  applies to the policy's targets.
* The J288 actuator model.

Actuators: ``J288DRActuatorCfg`` with the M6 fit from the bench
(``bam/params/unitree_j288/m6.json`` in the bam fork). The reflected rotor
inertia (6.3e-4 kg.m^2), friction, torque clamp (1.0 N.m motor side) and
speed filter (15 ms) come from that file; this module sets the gains, supply
voltage range and command delay.
"""

from __future__ import annotations

import os

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
# Fusion model has none of. Weighed 19/11/2026: 1,011 g without the battery,
# plus the 143 g battery (heat shrunk, 21/11/2026), minus the MJCF's 1.0848 kg. The 100 g estimate of
# 08/11/2026 trained the 09/11 policies. 0.0 reverts to the CAD masses.
UNMODELLED_MASS = 0.069  # [kg]


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


# ── Head roll limit ──────────────────────────────────────────────────────────
# Left to right head tilt, cut from the MJCF's ±25 degrees for a design
# constraint (09/11/2026). None keeps the MJCF's range.
HEAD_ROLL_LIMIT_DEG: float | None = 15.0
HEAD_ROLL_JOINT = "head_roll"


def head_roll_limit_rad(limit_deg: float | None = None) -> float | None:
    """``limit_deg`` (default ``HEAD_ROLL_LIMIT_DEG``) in radians, or None when unlimited."""
    if limit_deg is None:
        limit_deg = HEAD_ROLL_LIMIT_DEG
    return None if limit_deg is None else float(np.radians(limit_deg))


def limit_head_roll(spec: mujoco.MjSpec, limit_deg: float | None = None) -> mujoco.MjSpec:
    """Narrow the head_roll joint to ±``limit_deg`` (default ``HEAD_ROLL_LIMIT_DEG``).

    Only narrows: a limit wider than the MJCF's leaves it as it is.
    """
    limit = head_roll_limit_rad(limit_deg)
    if limit is None:
        return spec
    joint = spec.joint(HEAD_ROLL_JOINT)
    lo, hi = joint.range
    joint.range = [max(lo, -limit), min(hi, limit)]
    return spec


# ── Joint stops ──────────────────────────────────────────────────────────────
# Where each joint first meets something, either way, in degrees: measured by
# hand on the assembled robot with every servo limp (21/11/2026), against the
# zeros the calibration jig gave (averaged over four seatings) (dukki hardware.md section 12,
# hardware/calibration/). The same table as the firmware's
# duck-control model::JOINT_TRAVEL_DEG; keep the two together. The mouth is
# not in the MJCF.
JOINT_TRAVEL_DEG: dict[str, tuple[float, float]] = {
    "left_hip_yaw": (-44.8, 43.2),
    "left_hip_roll": (-19.8, 27.6),
    "left_hip_pitch": (-228.3, 82.1),
    "left_knee": (-117.4, 87.9),
    "left_ankle": (-77.9, 71.8),
    "neck_pitch": (-150.8, 34.4),
    "head_pitch": (-85.6, 95.6),
    "head_yaw": (-177.2, 168.3),
    "head_roll": (-20.4, 20.9),
    "right_hip_yaw": (-36.6, 45.6),
    "right_hip_roll": (-25.2, 24.5),
    "right_hip_pitch": (-82.0, 200.2),
    "right_knee": (-85.0, 118.7),
    "right_ankle": (-73.2, 76.7),
}
#: The firmware holds every target this far inside the travel
#: (duck-control model::STOP_MARGIN_DEG), so the servo never presses a stop.
STOP_MARGIN_DEG = 3.0
#: The firmware's actuator bound on every target (duck-control safety::ACTUATOR_MIN/MAX).
ACTUATOR_LIMIT_RAD = float(np.pi)
#: False keeps the MJCF's ranges and clips no target but head roll's.
APPLY_JOINT_TRAVEL = True


def limit_joint_travel(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Narrow each joint's range to the real robot's stops (``JOINT_TRAVEL_DEG``).

    Only narrows: where the robot travels further than the MJCF allows (the hip
    yaws, the hip pitches' far side), the MJCF's range stays, so the training
    distribution does not grow.
    """
    if not APPLY_JOINT_TRAVEL:
        return spec
    for name, (lo_deg, hi_deg) in JOINT_TRAVEL_DEG.items():
        joint = spec.joint(name)
        lo, hi = joint.range
        joint.range = [max(lo, np.radians(lo_deg)), min(hi, np.radians(hi_deg))]
    return spec


def target_clip_rad() -> dict[str, tuple[float, float]]:
    """Each joint's target range as the firmware's safety layer clamps it, radians.

    ``STOP_MARGIN_DEG`` inside the measured travel, within the actuator bound,
    and head roll within its limit: duck-control's safety::target_ranges. Only
    head roll's limit when ``APPLY_JOINT_TRAVEL`` is off.
    """
    clip: dict[str, tuple[float, float]] = {}
    if APPLY_JOINT_TRAVEL:
        for name, (lo_deg, hi_deg) in JOINT_TRAVEL_DEG.items():
            clip[name] = (
                max(float(np.radians(lo_deg + STOP_MARGIN_DEG)), -ACTUATOR_LIMIT_RAD),
                min(float(np.radians(hi_deg - STOP_MARGIN_DEG)), ACTUATOR_LIMIT_RAD),
            )
    roll = head_roll_limit_rad()
    if roll is not None:
        lo, hi = clip.get(HEAD_ROLL_JOINT, (-roll, roll))
        clip[HEAD_ROLL_JOINT] = (max(lo, -roll), min(hi, roll))
    return clip


def sole_condim(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """With ``DUKKI_SOLE_TORSION`` set, the sole geoms get condim 4 (torsional friction), which
    tasks/dukki_terrain.py then randomises. Off: the MJCF's condim 3."""
    if not os.environ.get("DUKKI_SOLE_TORSION"):
        return spec
    for name in ("left_foot_collision", "right_foot_collision"):
        spec.geom(name).condim = 4
    return spec


#: The passive hinges under the soles that DUKKI_SOLE_TWIST adds (tasks/dukki_terrain.py, 7).
SOLE_TWIST_JOINTS = ("passive_left_sole_twist", "passive_right_sole_twist")
#: Their armature [kg.m^2]: enough to keep the stiffest spring tasks/dukki_terrain.py uses stable at
#: the 5 ms physics step, small against the trunk's yaw inertia.
SOLE_TWIST_ARMATURE = 2.0e-4


def sole_twist(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """With ``DUKKI_SOLE_TWIST`` set, a passive hinge in each foot about its sole's normal, through the
    sole's centre: carpet pile lets a planted foot twist (and spring back) under the leg, which a rigid
    contact cannot (26/11/2026). Its stiffness and damping are set per episode by
    tasks/dukki_terrain.py; here a stiff default. Off: the model is unchanged."""
    if not os.environ.get("DUKKI_SOLE_TWIST"):
        return spec
    import numpy as np

    model = spec.compile()
    for side, name in zip(("left", "right"), SOLE_TWIST_JOINTS):
        g = model.geom(f"{side}_foot_collision")
        mesh = int(model.geom_dataid[g.id])
        a, n = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        v = model.mesh_vert[a : a + n].astype(float)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g.id])
        v = v @ rot.reshape(3, 3).T + model.geom_pos[g.id]  # in the foot body's frame
        centre = v.mean(0)
        # The sole is a thin plate: its normal is the direction it varies least along.
        _, _, vt = np.linalg.svd(v - centre, full_matrices=False)
        axis = vt[2]
        if np.dot(axis, -centre) < 0:  # point from the sole towards the ankle
            axis = -axis
        body = spec.body(model.body(int(model.geom_bodyid[g.id])).name)
        body.add_joint(
            name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis.tolist(), pos=centre.tolist(),
            stiffness=20.0, damping=0.5, armature=SOLE_TWIST_ARMATURE, frictionloss=0.0,
            limited=mujoco.mjtLimited.mjLIMITED_TRUE, range=[-0.35, 0.35],
        )
        if os.environ.get("DUKKI_SOLE_SHEAR"):
            # Two slides in the sole's plane: pile shears under a planted foot as well as twisting.
            u = np.cross(axis, [1.0, 0.0, 0.0])
            if np.linalg.norm(u) < 0.5:
                u = np.cross(axis, [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            w = np.cross(axis, u)
            for tag, ax in (("u", u), ("w", w)):
                body.add_joint(
                    name=f"{name}_shear_{tag}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=ax.tolist(),
                    pos=centre.tolist(), stiffness=5000.0, damping=20.0, armature=SOLE_SHEAR_ARMATURE,
                    frictionloss=0.0, limited=mujoco.mjtLimited.mjLIMITED_TRUE, range=[-0.01, 0.01],
                )
    return spec


#: Armature of the shear slides [kg]: about 1% of the robot's mass.
SOLE_SHEAR_ARMATURE = 0.01


def gear_play(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Evaluation knobs for the backlash models' gear play (26/11/2026, the carpet shake search; off by
    default): ``DUKKI_PLAY_DEG`` sets every passive *_backlash joint's play to +-deg (the MJCF's is +-1),
    ``DUKKI_PLAY_SOLREF`` the time constant of the play's end stops (the MJCF's 0.01 s; larger is a
    springier stop: give in the gearbox or frame past the dead zone)."""
    deg = os.environ.get("DUKKI_PLAY_DEG")
    tc = os.environ.get("DUKKI_PLAY_SOLREF")
    if not deg and not tc:
        return spec
    import math

    for j in spec.joints:
        if not (j.name.startswith("passive_") and j.name.endswith("_backlash")):
            continue
        if deg:
            r = math.radians(float(deg))
            j.range = [-r, r]
        if tc:
            j.solref_limit = [float(tc), 1.0]
    return spec


def dukki_spec(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """The Dukki changes to a loaded MJCF: unmodelled mass, the head roll limit, the joint stops
    and, when asked for, torsional friction at the soles and twist under them."""
    return gear_play(sole_twist(sole_condim(limit_joint_travel(limit_head_roll(add_unmodelled_mass(spec))))))


def get_dukki_walk_spec() -> mujoco.MjSpec:
    return dukki_spec(get_walk_spec())


# ── Whole-body ground contact ────────────────────────────────────────────────
# The ground-contact models (stand-up, sit-stand and their backlash twins) collide only the
# parts Pollen's Onshape export gave collision geoms, carried over to the Fusion parts: the
# soles, hip_l, the lower legs, the head shells and jaw, and a box for the battery. The trunk
# shells, the thighs, the feet above the soles, the hip and neck brackets and the J288 housings
# outside the head pass through the floor. On the robot, lying face up and rolled onto its
# left side, the left thigh props it at 34 degrees, where the simulated one sank 25 mm into the
# floor and rolled on to 70 (23/11/2026). These parts' visual meshes are added as collision
# geoms (MuJoCo collides their convex hulls). DUKKI_BODY_CONTACT=0 leaves the MJCF's as they are.
BODY_CONTACT_MESHES = (
    "left_shell", "right_shell", "j288", "upper_leg_left", "upper_leg_right",
    "upper_leg_rigidity_plate", "upper_leg_rigidity_plate_right", "foot_left", "foot_right",
    "yaw2roll", "neck", "neck_pitch", "yaw_roll_motion",
)
#: Bodies left alone: the head's J288s sit inside its shells, which already collide.
BODY_CONTACT_SKIP_BODIES = ("jaw_soft",)
#: Body pairs never collided with each other, where the hulls touch in ordinary poses (MuJoCo
#: already skips a body and its parent): the neck pitch bracket's hull reaches 2 mm into the
#: head shell in every pose; the feet's hulls reach 7.6 mm into hip_l at the SIT keyframe (the
#: soles alone, 1.3 mm).
BODY_CONTACT_EXCLUDES: tuple[tuple[str, str], ...] = (
    ("neck_pitch", "jaw_soft"),
    ("ankle_left", "hip_l"),
    ("ankle_right", "hip_l_2"),
)


def body_contact_enabled() -> bool:
    """DUKKI_BODY_CONTACT, on unless set to 0."""
    return os.environ.get("DUKKI_BODY_CONTACT", "1") != "0"


def add_body_contact(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Give every ``BODY_CONTACT_MESHES`` visual mesh a colliding twin (group 3, condim 3), as hip_l's."""
    if not body_contact_enabled():
        return spec
    for body in spec.bodies:
        if body.name in BODY_CONTACT_SKIP_BODIES:
            continue
        for g in list(body.geoms):
            if g.type != mujoco.mjtGeom.mjGEOM_MESH or g.meshname not in BODY_CONTACT_MESHES:
                continue
            if g.contype or g.conaffinity:
                continue
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.meshname, pos=g.pos, quat=g.quat,
                contype=1, conaffinity=1, group=3, condim=3,
            )
    for a, b in BODY_CONTACT_EXCLUDES:
        spec.add_exclude(bodyname1=a, bodyname2=b)
    return spec


def get_dukki_standup_spec() -> mujoco.MjSpec:
    return add_body_contact(dukki_spec(get_standup_spec()))


def get_dukki_allcollisions_spec() -> mujoco.MjSpec:
    return dukki_spec(get_allcollisions_spec())


def get_dukki_walk_backlash_spec() -> mujoco.MjSpec:
    return dukki_spec(get_walk_backlash_spec())


def get_dukki_backlash_spec() -> mujoco.MjSpec:
    return add_body_contact(dukki_spec(get_backlash_spec()))


def get_dukki_allcollisions_backlash_spec() -> mujoco.MjSpec:
    return dukki_spec(get_allcollisions_backlash_spec())


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
    # The groundcontact model's backlash twin under the name its base task uses (StandUp, SitStand).
    "standup_backlash": (get_dukki_backlash_spec, BACKLASH_HOME_FRAME),
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
