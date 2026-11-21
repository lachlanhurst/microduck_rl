"""LowPassJointPositionAction filters as robotd does: unfiltered first tick, alpha blend after, clip last."""

import dataclasses, torch, pytest
import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.lowpass_action import LowPassJointPositionActionCfg

def test_lowpass_matches_robotd():
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Dukki", play=True); cfg.scene.num_envs = 2
    a = cfg.actions["joint_pos"]
    fields = {f.name: getattr(a, f.name) for f in dataclasses.fields(a) if f.init}
    cfg.actions["joint_pos"] = LowPassJointPositionActionCfg(**fields, lowpass={r".*(hip|knee|ankle).*": 0.7})
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        term = env.action_manager.get_term("joint_pos")
        names = term.target_names
        legs = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
        head = [i for i in range(len(names)) if i not in legs]
        term.reset(None)
        a1 = torch.full((2, term.action_dim), 0.1); term.process_actions(a1)
        t1 = term._processed_actions.clone()
        u1 = a1 * term._scale + term._offset
        assert torch.allclose(t1, torch.clamp(u1, term._clip[:, :, 0], term._clip[:, :, 1])), "first tick unfiltered"
        a2 = torch.full((2, term.action_dim), -0.1); term.process_actions(a2)
        u2 = a2 * term._scale + term._offset
        want = u2.clone(); want[:, legs] = 0.7 * u2[:, legs] + 0.3 * u1[:, legs]
        want = torch.clamp(want, term._clip[:, :, 0], term._clip[:, :, 1])
        assert torch.allclose(term._processed_actions, want, atol=1e-6)
        assert torch.allclose(term._processed_actions[:, head], torch.clamp(u2, term._clip[:, :, 0], term._clip[:, :, 1])[:, head])
        assert torch.equal(term._raw_actions, a2)
        term.reset(torch.tensor([0])); term.process_actions(a1)
        assert torch.allclose(term._processed_actions[0], torch.clamp(u1, term._clip[:, :, 0], term._clip[:, :, 1])[0]), "unfiltered after reset"
        env.reset(); env.step(torch.zeros(2, term.action_dim))
    finally:
        env.close()
