"""Joint position action with robotd's target low-pass, so a policy trains on what the robot runs.

robotd (microduck `robotd/src/control.rs`) turns each action into targets as
`default + scale * action`, then low-passes them per joint group,
`target = alpha * target + (1 - alpha) * previous`, then clamps them in its safety layer. The first
tick after a reset goes out unfiltered (no previous target yet). This does the same: scale and
offset, the low-pass on the unclipped targets, then the clip.

`lowpass` maps joint-name regexes to alpha (1.0 is no filter); joints no pattern matches are not
filtered. The observed last action stays the raw action, as robotd's is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class LowPassJointPositionActionCfg(JointPositionActionCfg):
    lowpass: dict[str, float] = field(default_factory=dict)
    #: Optional joint-name regex -> (lo, hi): that group's alpha is drawn per robot at each reset,
    #: uniformly in lo..hi (26/11/2026: the robot's filter setting matters on carpet, so the policy
    #: trains across a range of it rather than one value or none). Overrides ``lowpass`` there.
    lowpass_range: dict[str, tuple[float, float]] = field(default_factory=dict)

    def build(self, env) -> "LowPassJointPositionAction":
        return LowPassJointPositionAction(self, env)


class LowPassJointPositionAction(JointPositionAction):
    def __init__(self, cfg: LowPassJointPositionActionCfg, env):
        super().__init__(cfg=cfg, env=env)
        alpha = torch.ones(len(self._target_names), device=self.device)
        for pattern, value in cfg.lowpass.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"low-pass alpha for {pattern!r} must be in (0, 1], got {value}")
            hits = [i for i, name in enumerate(self._target_names) if re.fullmatch(pattern, name)]
            if not hits:
                raise ValueError(f"low-pass pattern {pattern!r} matches no joint of {self._target_names}")
            alpha[hits] = value
        self._alpha = alpha.unsqueeze(0).repeat(self.num_envs, 1)
        self._ranges = []
        for pattern, (lo, hi) in cfg.lowpass_range.items():
            if not 0.0 < lo <= hi <= 1.0:
                raise ValueError(f"low-pass range for {pattern!r} must be 0 < lo <= hi <= 1, got {(lo, hi)}")
            hits = [i for i, name in enumerate(self._target_names) if re.fullmatch(pattern, name)]
            if not hits:
                raise ValueError(f"low-pass pattern {pattern!r} matches no joint of {self._target_names}")
            self._ranges.append((torch.tensor(hits, device=self.device), lo, hi))
        self._draw_alpha(slice(None))
        self._previous = torch.zeros(self.num_envs, len(self._target_names), device=self.device)
        self._has_previous = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        target = self._raw_actions * self._scale + self._offset
        filtered = self._alpha * target + (1.0 - self._alpha) * self._previous
        target = torch.where(self._has_previous, filtered, target)
        self._previous = target.clone()
        self._has_previous[:] = True
        if self.cfg.clip is not None:
            target = torch.clamp(target, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self._processed_actions = target

    def _draw_alpha(self, env_ids) -> None:
        for hits, lo, hi in self._ranges:
            n = self._alpha[env_ids].shape[0]
            a = torch.rand(n, 1, device=self.device) * (hi - lo) + lo
            block = self._alpha[env_ids]
            block[:, hits] = a.expand(-1, len(hits))
            self._alpha[env_ids] = block

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._has_previous[env_ids] = False
        self._draw_alpha(env_ids)
