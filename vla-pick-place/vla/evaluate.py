"""Closed-loop benchmark: task success, generalisation and safety.

Splits (all use layouts from seeds never used for training data):
  seen              - block/pad pairs and phrasings from training
  unseen_pairs      - block/pad colour pairs never used as a training target
  unseen_phrasing   - a new instruction wording with out-of-vocabulary words
  more_distractors  - 4 blocks on the table (training had at most 3)

Safety metrics: the share of steps where the safety filter had to correct the
command (speed, joint or quill limits), wrong placements (a block released off the target pad)
and disturbed distractors (a non-target block moved by more than 2 cm).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Callable

import numpy as np
import torch

from .data import denormalize_action
from .env import PickPlaceEnv
from .language import HELDOUT_PAIRS, HELDOUT_TEMPLATES, TRAIN_TEMPLATES, train_pairs
from .model import VLAPolicy, load_checkpoint

SPLITS = {
    "seen": dict(pairs=train_pairs(), templates=TRAIN_TEMPLATES, n_blocks=(2, 3)),
    "unseen_pairs": dict(pairs=list(HELDOUT_PAIRS), templates=TRAIN_TEMPLATES, n_blocks=(2, 3)),
    "unseen_phrasing": dict(pairs=train_pairs(), templates=HELDOUT_TEMPLATES, n_blocks=(2, 3)),
    "more_distractors": dict(pairs=train_pairs(), templates=TRAIN_TEMPLATES, n_blocks=(4, 4)),
}
EVAL_SEED = 90_000          # test layouts; data collection uses seeds < 10_000
VAL_SEED = 70_000           # validation layouts for model selection during RL

PolicyFn = Callable[[dict], np.ndarray]   # observation -> normalised action (4,)


def torch_policy(model: VLAPolicy) -> PolicyFn:
    model.eval()

    @torch.no_grad()
    def act(obs: dict) -> np.ndarray:
        chunk = model(torch.from_numpy(obs["image"])[None], torch.from_numpy(obs["tokens"])[None],
                      torch.from_numpy(obs["proprio"])[None])
        return chunk[0, 0].numpy()          # receding horizon: execute the first action
    return act


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def run_split(policy: PolicyFn, split: str, episodes: int, env: PickPlaceEnv | None = None,
              seed: int = EVAL_SEED) -> dict:
    env = env or PickPlaceEnv()
    spec = SPLITS[split]
    rng = np.random.default_rng(seed + list(SPLITS).index(split))
    succ, steps, interventions, n_steps, wrong, disturbed, latency = 0, [], 0, 0, 0, 0, []
    for _ in range(episodes):
        task = env.sample_task(rng, spec["pairs"], spec["templates"], spec["n_blocks"])
        obs, done, info = env.reset(task, rng), False, None
        while not done:
            t0 = time.perf_counter()
            a = policy(obs)
            latency.append(time.perf_counter() - t0)
            obs, _, done, info = env.step(denormalize_action(a))
            interventions += info.safety_interventions
            n_steps += 1
        succ += info.success
        wrong += info.wrong_placement
        disturbed += info.distractor_disturbed
        if info.success:
            steps.append(env.t)
    lo, hi = wilson(succ, episodes)
    return {
        "split": split, "episodes": episodes, "success_rate": succ / episodes, "ci95": [lo, hi],
        "mean_steps_success": float(np.mean(steps)) if steps else None,
        "intervention_step_rate": interventions / max(n_steps, 1),
        "wrong_placement_rate": wrong / episodes, "disturbed_distractor_rate": disturbed / episodes,
        "policy_ms_median": 1000 * float(np.median(latency)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/bc_policy.pt")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--backend", choices=["torch", "cpp"], default="torch")
    ap.add_argument("--weights", default="checkpoints/policy.bin", help="C++ weights (backend=cpp)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.backend == "torch":
        policy = torch_policy(load_checkpoint(args.checkpoint))
    else:
        import vla_cpp  # built from cpp/ with -DVLA_BUILD_PYTHON=ON
        runtime = vla_cpp.Policy(args.weights)

        def policy(obs: dict) -> np.ndarray:
            return np.asarray(runtime.predict(obs["image"], obs["tokens"], obs["proprio"]))[:4]
    env = PickPlaceEnv()
    results = []
    for split in args.splits:
        r = run_split(policy, split, args.episodes, env)
        results.append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        json.dump({"checkpoint": args.checkpoint, "backend": args.backend, "results": results},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
