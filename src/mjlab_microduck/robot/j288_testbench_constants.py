"""J288 pendulum testbench entity, for sim2real validation before the robot exists.

The rig the J288 was identified on (dukki docs/j288-testing.md section 15): the
servo in a bracket clamped to the bench, an arm on the output, a mass at the
tip. The joint's zero is the arm hanging straight down. Inertials come from
``bam.testbench_mujoco.Pendulum`` plus ``mass_inertia``, exactly as
``bam.unitree.mjlab.J288Simulator`` builds them, so the dynamics are the ones
BAM was fitted against: a uniform rod plus a point mass at the tip, plus the
mass's own inertia about its centre.

The setups are the ones recorded. ``DUKKI_BENCH_SETUP`` picks one when the task
is registered (default ``l150_m108``: 150 mm arm, 108 g total, 0.146 N.m of
gravity torque at horizontal and 2.2e-3 kg.m^2 of load inertia, about a leg
joint's).
"""

from __future__ import annotations

import os

import mujoco
from bam.testbench_mujoco import Pendulum
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.actuator import J288DRActuatorCfg
from mjlab_microduck.robot.dukki_constants import J288_KD, J288_KP

TESTBENCH_JOINT = "j288"

# name -> (arm length [m], end mass [kg], arm mass [kg], mass_inertia [kg.m^2]),
# from j288-testing.md section 15.1.
J288_BENCH_SETUPS: dict[str, tuple[float, float, float, float]] = {
    "l75_m70": (0.075, 0.055, 0.015, 2.08e-05),
    "l75_m246": (0.075, 0.231, 0.015, 8.73e-05),
    "l75_m415": (0.075, 0.400, 0.015, 1.51e-04),
    "l75_m586": (0.075, 0.571, 0.015, 2.16e-04),
    "l150_m108": (0.150, 0.091, 0.017, 3.44e-05),
    "l150_m333": (0.150, 0.316, 0.017, 1.19e-04),
    "l150_m417": (0.150, 0.400, 0.017, 1.51e-04),
    # 02/10/2026: the 108 g setup with a bolt added to hold the end mass on,
    # weighed at 163 g total. Disc inertia m r^2 / 2 at r = 27.5 mm, the bolt
    # counted in it.
    "l150_m163": (0.150, 0.146, 0.017, 5.52e-05),
}
DEFAULT_BENCH_SETUP = "l150_m108"


def bench_setup_from_env() -> str:
    setup = os.environ.get("DUKKI_BENCH_SETUP", DEFAULT_BENCH_SETUP)
    if setup not in J288_BENCH_SETUPS:
        raise ValueError(f"DUKKI_BENCH_SETUP must be one of {sorted(J288_BENCH_SETUPS)}, got {setup!r}")
    return setup


def bench_log(setup: str) -> dict:
    """The setup as a BAM log header (the fields the pendulum testbench reads)."""
    length, mass, arm_mass, mass_inertia = J288_BENCH_SETUPS[setup]
    return {"length": length, "mass": mass, "arm_mass": arm_mass, "mass_inertia": mass_inertia}


def make_j288_testbench_spec(setup: str) -> mujoco.MjSpec:
    """One hinge about +x at the origin, arm hanging along -z at q = 0."""
    log = bench_log(setup)
    total_mass, com_z, inertia_x = Pendulum(log).inertial_params()
    inertia_x += log["mass_inertia"]

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
    # The pivot sits above mjlab's ground plane, clear of the arm's reach.
    base = spec.worldbody.add_body(name="bench", pos=[0.0, 0.0, log["length"] + 0.15])
    base.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.013, 0.017, 0.010],
        rgba=[0.3, 0.3, 0.3, 1.0], contype=0, conaffinity=0,
    )
    arm = base.add_body(name="arm")
    arm.add_joint(name=TESTBENCH_JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1.0, 0.0, 0.0])
    arm.mass = total_mass
    arm.ipos = [0.0, 0.0, com_z]
    # Only rotation about x (the hinge) matters; keep the others physical-ish.
    arm.inertia = [inertia_x, inertia_x, max(inertia_x * 1e-2, 1e-9)]
    arm.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.0, 0.0, 0.0, 0.0, 0.0, -log["length"]], size=[0.004, 0.0, 0.0],
        rgba=[0.55, 0.35, 0.2, 1.0], contype=0, conaffinity=0,
    )
    arm.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0.0, 0.0, -log["length"]],
        quat=[0.7071068, 0.0, 0.7071068, 0.0], size=[0.0275, 0.006, 0.0],
        rgba=[0.7, 0.7, 0.75, 1.0], contype=0, conaffinity=0,
    )
    return spec


def make_j288_testbench_robot_cfg(
    setup: str = DEFAULT_BENCH_SETUP,
    kp: float = J288_KP,
    kd: float = J288_KD,
) -> EntityCfg:
    actuators = J288DRActuatorCfg(
        motor_name="unitree_j288",
        model="m6",
        target_names_expr=(TESTBENCH_JOINT,),
        kp_fw=kp,
        kd_fw=kd,
        # The bench ran on the 6S pack at 24.0 to 24.5 V.
        vin=24.2,
        # The USB adapter's delay fitted at 0.02 ms, but the policy loop on the
        # host adds up to a step of scheduling jitter.
        delay_min_lag=0,
        delay_max_lag=1,
    )
    return EntityCfg(
        spec_fn=lambda: make_j288_testbench_spec(setup),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={TESTBENCH_JOINT: 0.0}, joint_vel={".*": 0.0}
        ),
        collisions=(),
        articulation=EntityArticulationInfoCfg(
            actuators=(actuators,),
            soft_joint_pos_limit_factor=1.0,
        ),
    )
