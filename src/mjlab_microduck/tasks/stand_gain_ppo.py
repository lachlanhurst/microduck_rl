"""PPO with a penalty on the policy's gain while standing (26/11/2026, the carpet shake).

On carpet the robot shakes after stopping: a loop between the policy and the legs at 5.5 to 7 Hz, the
servos far past their bandwidth there (dukki scratch/policies shake_*.json, runs/eval/stillness/NOTES.md).
Simulation does not shake, so no reward about stillness has anything to work on there (the shear-carpet and
feet-squeeze fine-tunes changed nothing). This lowers the loop's gain directly instead: for the samples
whose velocity command is zero, the actor's mean action is compared between the observation and a copy
with small Gaussian noise on the sensor inputs (gyro, gravity, joint positions and velocities, last action;
never the commands), and the mean squared difference is added to the PPO loss. A policy that answers small
sensor changes with big target swings pays for it; walking is left alone.

    DUKKI_STAND_GAIN="coef[,scale[,threshold]]"  coef: weight on the mean squared action change (rad^2);
                                                 scale: multiplies the noise sizes in NOISE (default 1);
                                                 threshold: |command| below which a sample counts as standing (0.05).
    uv run train ... --agent.algorithm.class-name mjlab_microduck.tasks.stand_gain_ppo:StandGainPPO

The loss is added to the PPO minibatch's gradients just before its optimiser step (after PPO's own clip,
then the actor's gradients are clipped again), so the two are optimised together. Logged as
Loss/stand_gain (the standing samples' action change RMS, rad) and Loss/walk_gain (the same for moving
samples, not penalised; for comparison).

Result (26/11/2026, from mpush2 12750, whose gain is 0.055 rad standing): coef 10 halved the gain in 150
iterations and the policy lost standing balance (simulated stands fell 1 to 19 times in 32 where 12750 fell
0 or 1). Coef 3 took 45% off; its 13500 walked almost as 12750 and on the robot's carpet stood stiller than
the stand-up network, with two sideways falls. The walk network unchanged, standing at legs 0.84, was better
than both (dukki docs/policy-testing.md section 13): not used.
"""
from __future__ import annotations

import os

import torch
from rsl_rl.algorithms.ppo import PPO
from torch import nn

#: Noise standard deviation per observation term, matched by substring of the term's name (first match wins).
#: Sizes of the robot's own small motions while standing: a few hundredths of a radian, tenths of a rad/s.
NOISE = (
    ("ang_vel", 0.1),     # gyro, rad/s
    ("gravity", 0.02),    # projected gravity, unit vector
    ("joint_pos", 0.02),  # rad
    ("joint_vel", 0.5),   # rad/s
    ("action", 0.03),     # last action, rad
)


class StandGainPPO(PPO):
    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg = PPO.construct_algorithm(obs, env, cfg, device)  # resolves class_name to this class
        alg._setup_stand_gain(env)
        return alg

    def _setup_stand_gain(self, env) -> None:
        text = os.environ.get("DUKKI_STAND_GAIN", "0")
        v = [float(x) for x in text.split(",")]
        self.gain_coef = v[0]
        scale = v[1] if len(v) > 1 else 1.0
        self.gain_threshold = v[2] if len(v) > 2 else 0.05
        om = env.unwrapped.observation_manager
        names = om.active_terms["actor"]
        dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim["actor"]]
        sigma, cmd, start = [], None, 0
        layout = []
        for name, d in zip(names, dims):
            s = 0.0
            if "command" not in name:
                for key, value in NOISE:
                    if key in name:
                        s = value * scale
                        break
            if name in ("velocity_commands", "command", "twist_command") or (cmd is None and "command" in name and "head" not in name):
                cmd = slice(start, start + d)
            sigma += [s] * d
            layout.append(f"{name}[{start}:{start + d}] noise {s:g}")
            start += d
        if cmd is None:
            raise ValueError(f"StandGainPPO: no velocity command term in the actor observations {names}")
        self.gain_sigma = torch.tensor(sigma, device=self.device)
        self.gain_cmd = cmd
        print(f"[StandGainPPO] coef {self.gain_coef:g}, standing below |command| {self.gain_threshold:g}, "
              f"command {cmd}; " + "; ".join(layout))

    def _gain(self, obs):
        x = obs["actor"]
        standing = x[:, self.gain_cmd].norm(dim=-1) < self.gain_threshold
        out = {}
        for key, mask in (("stand_gain", standing), ("walk_gain", ~standing)):
            if int(mask.sum()) < 16:
                out[key] = None
                continue
            td = obs[mask]
            noisy = td.clone()
            noisy["actor"] = td["actor"] + torch.randn_like(td["actor"]) * self.gain_sigma
            if key == "stand_gain":
                out[key] = (self.actor(noisy) - self.actor(td)).pow(2).mean()
            else:
                with torch.no_grad():
                    out[key] = (self.actor(noisy) - self.actor(td)).pow(2).mean()
        return out

    def update(self) -> dict[str, float]:
        storage = self.storage
        gen_orig = storage.mini_batch_generator
        current = {}

        def gen(*a, **k):
            for batch in gen_orig(*a, **k):
                current["obs"] = batch.observations
                yield batch

        step_orig = self.optimizer.step
        log = {"stand_gain": [], "walk_gain": []}

        def step(*a, **k):
            g = self._gain(current["obs"])
            if g["stand_gain"] is not None:
                if self.gain_coef > 0:
                    (self.gain_coef * g["stand_gain"]).backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                log["stand_gain"].append(g["stand_gain"].detach().sqrt().item())
            if g["walk_gain"] is not None:
                log["walk_gain"].append(g["walk_gain"].sqrt().item())
            return step_orig(*a, **k)

        storage.mini_batch_generator = gen
        self.optimizer.step = step
        try:
            out = super().update()
        finally:
            storage.mini_batch_generator = gen_orig
            self.optimizer.step = step_orig
        for key, values in log.items():
            out[key] = sum(values) / len(values) if values else 0.0
        return out
