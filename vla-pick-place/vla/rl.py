"""PPO fine-tuning of the behaviour-cloned policy.

Design choices:
  * Actor: the BC policy itself. Its first predicted action is the mean of a
    Gaussian with a learned, state-independent std.
  * Asymmetric actor-critic: the critic is a small MLP on privileged simulator
    state (tip, target block and pad positions, grasp state). It is used only in
    training, so the deployed policy still needs only camera, instruction and joints.
  * Anchor to the BC policy: a penalty on the distance between the fine-tuned and
    the frozen BC action means keeps RL from drifting away from what imitation
    already does well.
  * Potential-based reward shaping (distance to the block, then block to pad) plus a
    success bonus and penalties for wrong placements, disturbed distractors and
    safety-filter interventions.
  * Model selection: every `val_every` iterations the deterministic policy is run on
    validation layouts (separate seeds from the test benchmark) and the best
    checkpoint is kept, starting with the BC policy itself, so RL is only adopted
    if it measurably helps.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import denormalize_action
from .env import BLOCK_HALF, PickPlaceEnv
from .evaluate import VAL_SEED, run_split, torch_policy
from .language import TRAIN_TEMPLATES, train_pairs
from .model import load_checkpoint, save_checkpoint

RL_SEED = 50_000


def privileged_state(env: PickPlaceEnv) -> np.ndarray:
    tip = env.tip_pos()
    blk = env.block_pos(env.task.target_block)
    pad = env.pad_pos(env.task.target_pad)
    held_target = float(env.held == env.task.target_block)
    held_other = float(env.held is not None and not held_target)
    return np.array([*tip, *blk, *pad, *(blk[:2] - pad), *(tip - blk), held_target, held_other,
                     float(env.suction), env.t / env.max_steps], dtype=np.float32)


def potential(env: PickPlaceEnv) -> float:
    blk = env.block_pos(env.task.target_block)
    if env.held == env.task.target_block:
        return 1.0 - float(np.linalg.norm(blk[:2] - env.pad_pos(env.task.target_pad)))
    top = blk + np.array([0, 0, BLOCK_HALF])
    return -float(np.linalg.norm(env.tip_pos() - top))


class Critic(nn.Module):
    def __init__(self, n_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class VecEnv:
    """N environments stepped in lockstep with automatic reset."""

    def __init__(self, n: int, seed: int, rcfg: dict):
        self.envs = [PickPlaceEnv() for _ in range(n)]
        self.rng = np.random.default_rng(seed)
        self.rcfg = rcfg
        self.obs = [self._reset(e) for e in self.envs]
        self.flags = [(False, False) for _ in self.envs]

    def _reset(self, env: PickPlaceEnv) -> dict:
        return env.reset(env.sample_task(self.rng, train_pairs(), TRAIN_TEMPLATES), self.rng)

    def batch(self):
        return (torch.from_numpy(np.stack([o["image"] for o in self.obs])),
                torch.from_numpy(np.stack([o["tokens"] for o in self.obs])),
                torch.from_numpy(np.stack([o["proprio"] for o in self.obs])),
                torch.from_numpy(np.stack([privileged_state(e) for e in self.envs])))

    def step(self, actions: np.ndarray):
        rewards, dones, successes = [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions, strict=True)):
            phi = potential(env)
            obs, _, done, info = env.step(denormalize_action(a))
            wrong_before, dist_before = self.flags[i]
            r = self.rcfg["gamma"] * potential(env) - phi
            r -= self.rcfg["intervention_penalty"] * info.safety_interventions
            if info.wrong_placement and not wrong_before:
                r -= self.rcfg["wrong_placement_penalty"]
            if info.distractor_disturbed and not dist_before:
                r -= self.rcfg["disturb_penalty"]
            if info.success:
                r += self.rcfg["success_bonus"]
            self.flags[i] = (info.wrong_placement, info.distractor_disturbed)
            # Only success is a true termination; hitting the step limit is a time-out,
            # so the value of the last state is still bootstrapped.
            dones.append(float(info.success))
            if done:
                successes.append(float(info.success))
                obs = self._reset(env)
                self.flags[i] = (False, False)
            self.obs[i] = obs
            rewards.append(r)
        return np.array(rewards, np.float32), np.array(dones, np.float32), successes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rl.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 2))

    actor = load_checkpoint(cfg["init_checkpoint"])
    actor.log_std.data.fill_(cfg["init_log_std"])
    ref = copy.deepcopy(actor).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    venv = VecEnv(cfg["n_envs"], RL_SEED + cfg["seed"], cfg["reward"])
    critic = Critic(len(privileged_state(venv.envs[0])))
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg["actor_lr"])
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg["critic_lr"])
    gamma, lam, T, N = cfg["reward"]["gamma"], cfg["gae_lambda"], cfg["horizon"], cfg["n_envs"]
    history, recent = [], []
    os.makedirs(cfg["out_dir"], exist_ok=True)
    val_env = PickPlaceEnv()

    def validate() -> float:
        return run_split(torch_policy(actor), "seen", cfg["val_episodes"], val_env, seed=VAL_SEED)["success_rate"]

    best = validate()
    best_iter = 0
    print(json.dumps({"iteration": 0, "val_success": best}), flush=True)
    save_checkpoint(os.path.join(cfg["out_dir"], "rl_policy.pt"), actor.eval(), {"iteration": 0, "val_success": best})

    for it in range(cfg["iterations"]):
        t0 = time.time()
        buf = {k: [] for k in ("img", "tok", "pro", "priv", "act", "logp", "val", "rew", "done")}
        actor.eval()
        with torch.no_grad():
            for _ in range(T):
                img, tok, pro, priv = venv.batch()
                mean = actor(img, tok, pro)[:, 0]
                dist = torch.distributions.Normal(mean, actor.log_std.exp())
                a = dist.sample()
                rew, done, succ = venv.step(a.clamp(-1, 1).numpy())
                recent.extend(succ)
                for k, v in zip(buf, (img, tok, pro, priv, a, dist.log_prob(a).sum(-1), critic(priv),
                                      torch.from_numpy(rew), torch.from_numpy(done)), strict=True):
                    buf[k].append(v)
            last_val = critic(venv.batch()[3])
        val = torch.stack(buf["val"])
        rew, done = torch.stack(buf["rew"]), torch.stack(buf["done"])
        adv = torch.zeros(T, N)
        gae = torch.zeros(N)
        for t in reversed(range(T)):
            nxt = last_val if t == T - 1 else val[t + 1]
            # Time-limit resets are treated as non-terminal; true terminations cut the bootstrap.
            delta = rew[t] + gamma * nxt * (1 - done[t]) - val[t]
            gae = delta + gamma * lam * (1 - done[t]) * gae
            adv[t] = gae
        ret = (adv + val).reshape(-1)
        adv = adv.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = {k: torch.cat(buf[k]) for k in ("img", "tok", "pro", "priv", "act", "logp")}

        actor.train()
        n = T * N
        stats = []
        for _ in range(cfg["ppo_epochs"]):
            perm = torch.randperm(n)
            for s in range(0, n, cfg["minibatch"]):
                idx = perm[s:s + cfg["minibatch"]]
                mean = actor(flat["img"][idx], flat["tok"][idx], flat["pro"][idx])[:, 0]
                dist = torch.distributions.Normal(mean, actor.log_std.exp())
                ratio = (dist.log_prob(flat["act"][idx]).sum(-1) - flat["logp"][idx]).exp()
                a_idx = adv[idx]
                pg = -torch.min(ratio * a_idx, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * a_idx).mean()
                with torch.no_grad():
                    ref_mean = ref(flat["img"][idx], flat["tok"][idx], flat["pro"][idx])[:, 0]
                anchor = ((mean - ref_mean) ** 2).mean()
                loss_a = pg + cfg["anchor_coef"] * anchor - cfg["entropy_coef"] * dist.entropy().sum(-1).mean()
                opt_a.zero_grad()
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt_a.step()
                actor.log_std.data.clamp_(cfg["min_log_std"], cfg["init_log_std"])
                loss_c = ((critic(flat["priv"][idx]) - ret[idx]) ** 2).mean()
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
                stats.append((pg.item(), anchor.item(), loss_c.item(), (ratio - 1).abs().mean().item()))
        recent = recent[-cfg["success_window"]:]
        pg_m, anc_m, vc_m, rdev = np.mean(stats, 0)
        val = None
        if (it + 1) % cfg["val_every"] == 0:
            val = validate()
            if val > best:
                best, best_iter = val, it + 1
                save_checkpoint(os.path.join(cfg["out_dir"], "rl_policy.pt"), actor.eval(),
                                {"iteration": it + 1, "val_success": val})
        rec = {"iteration": it + 1, "rollout_success": float(np.mean(recent)) if recent else None,
               "val_success": val, "best_val": best, "best_iteration": best_iter,
               "mean_reward": float(rew.mean()), "policy_loss": pg_m, "anchor": anc_m, "value_loss": vc_m,
               "ratio_dev": rdev, "std": actor.log_std.exp().tolist(), "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump(history, open(os.path.join(cfg["out_dir"], "rl_history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
