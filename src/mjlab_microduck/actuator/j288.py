"""Unitree J288 actuator for the Dukki robot (microduck rework on J288 servos).

Wraps ``bam.unitree.mjlab.J288BamActuator`` (the bam fork, branch
``unitree-j288``) with the two things the microduck tasks need on top:

* per-episode friction DR, through the same ``FrictionDRMixin`` hook the XL330
  actuator uses (``randomize_bam_friction`` in tasks/mdp.py);
* per-joint default gains. The J288 takes kp [N.m/rad] and kd [N.m.s/rad],
  output side, in every command frame, so each joint can have its own pair.
  ``joint_gains`` maps joint-name regexes to (kp, kd), first match wins, and
  joints that match nothing take ``kp_fw`` / ``kd_fw``. Gains are restored to
  these defaults on reset, so an event or a limp mode that changes them for an
  episode does not leak into the next.

The firmware law, the speed filter, the torque and voltage clamps, mode 0 and
the reported torque and speed are all BAM's (see the fork's
``bam/unitree/actuator.py``). The firmware PD closes on the rotor side of the
gearbox, so on the backlash models it reads the servo joint alone, not servo
plus backlash (the opposite of the XL330's output-side encoder): the plain
actuator is the right one for those models too.

Parameters for the Dukki robot are in robot/dukki_constants.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from bam.unitree.mjlab import J288BamActuator, J288BamActuatorCfg

from mjlab_microduck.actuator.friction_dr_bam import FrictionDRMixin


class J288DRActuator(FrictionDRMixin, J288BamActuator):
    """J288BamActuator + friction DR hooks + per-joint default gains."""

    cfg: "J288DRActuatorCfg"

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        for j, name in enumerate(self._target_names):
            for pattern, kp, kd in self.cfg.joint_gains:
                if re.fullmatch(pattern, name):
                    self.kp[:, j] = kp
                    self.kd[:, j] = kd
                    break
        self.default_kp = self.kp.clone()
        self.default_kd = self.kd.clone()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        if self.kp is not None:
            self.kp[ids] = self.default_kp[ids]
            self.kd[ids] = self.default_kd[ids]


@dataclass(kw_only=True)
class J288DRActuatorCfg(J288BamActuatorCfg):
    """J288BamActuatorCfg with per-joint (kp, kd) defaults.

    ``joint_gains``: ``((regex, kp, kd), ...)``, matched against the actuated
    joint names with ``re.fullmatch``, first match wins.
    """

    joint_gains: tuple[tuple[str, float, float], ...] = field(default_factory=tuple)

    def build(self, entity, target_ids, target_names) -> J288DRActuator:
        return J288DRActuator(self, entity, target_ids, target_names)
