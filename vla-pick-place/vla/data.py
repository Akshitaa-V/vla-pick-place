"""Demonstration collection with noise injection, and the action-chunk dataset.

Each frame also stores where the target block, the target pad and the end of
the arm appear in the image (simulator state projected through the camera
model). These are labels for an auxiliary keypoint loss during training only;
the policy never receives them as input.

Noise injection (DART-style): the executed action is the expert action plus
Gaussian noise, but the stored label is always the clean expert action for the
state that was actually visited. The policy therefore sees states slightly off
the expert's path, labelled with how to recover from them.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import Dataset

from .env import ACTION_LIMITS, ARM_HEIGHT, BLOCK_HALF, IMG_H, IMG_W, PickPlaceEnv, project_to_image
from .expert import ScriptedExpert
from .language import TRAIN_TEMPLATES, train_pairs

NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3)   # std as a fraction of the per-step action limit
NOISE_PROB = 0.5                     # noise is applied on about half of the steps


def localisation_target(env: PickPlaceEnv) -> np.ndarray:
    """Training-only keypoint labels: where the target block (top face), the target pad and the
    end of the arm appear in the image, as (u, v) on the [-1, 1] grid of the stride-4 heatmaps."""
    block = env.block_pos(env.task.target_block) + np.array([0.0, 0.0, BLOCK_HALF])
    pad = np.array([*env.pad_pos(env.task.target_pad), 0.0])
    arm_end = np.array([*env.tip_pos()[:2], ARM_HEIGHT])
    uv = []
    for p in (block, pad, arm_end):
        row, col = project_to_image(p)
        uv += [2 * ((col - 1.5) / 4) / (IMG_W // 4 - 1) - 1, 2 * ((row - 1.5) / 4) / (IMG_H // 4 - 1) - 1]
    return np.array(uv, dtype=np.float32)


def normalize_action(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float32) / ACTION_LIMITS).astype(np.float32)


def denormalize_action(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) * ACTION_LIMITS


def _collect_worker(args) -> dict:
    seed, n_episodes = args
    rng = np.random.default_rng(seed)
    env, expert = PickPlaceEnv(max_steps=60), ScriptedExpert()
    images, tokens, proprio, actions, aux, episode, step = [], [], [], [], [], [], []
    kept = 0
    for ep in range(n_episodes):
        task = env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES)
        obs = env.reset(task, rng)
        sigma = NOISE_LEVELS[ep % len(NOISE_LEVELS)]
        ep_rows, done, info = [], False, None
        while not done:
            label = expert.act(env)
            executed = label.copy()
            if sigma > 0 and rng.random() < NOISE_PROB:
                executed[:3] += rng.normal(0, sigma, 3) * ACTION_LIMITS[:3]
            ep_rows.append((obs["image"], obs["tokens"], obs["proprio"], normalize_action(label),
                            localisation_target(env)))
            obs, _, done, info = env.step(executed)
        if not info.success:          # keep only demonstrations that solved the task
            continue
        eid = seed * 100000 + ep
        for t, (im, tok, pr, a, loc) in enumerate(ep_rows):
            images.append(im)
            tokens.append(tok)
            proprio.append(pr)
            actions.append(a)
            aux.append(loc)
            episode.append(eid)
            step.append(t)
        kept += 1
    return {"images": np.stack(images), "tokens": np.stack(tokens), "proprio": np.stack(proprio),
            "actions": np.stack(actions), "aux": np.stack(aux), "episode": np.array(episode),
            "step": np.array(step),
            "kept": kept, "tried": n_episodes}


def collect(n_episodes: int, seed: int, workers: int = 2) -> dict:
    per = [n_episodes // workers + (i < n_episodes % workers) for i in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(_collect_worker, [(seed * 1000 + i, n) for i, n in enumerate(per)])
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("images", "tokens", "proprio",
                                                             "actions", "aux", "episode", "step")}
    out["kept"] = sum(p["kept"] for p in parts)
    out["tried"] = sum(p["tried"] for p in parts)
    return out


class ChunkDataset(Dataset):
    """Each sample: observation at t, the next `chunk` expert actions (t .. t+chunk-1) and
    the auxiliary localisation target (zeros if the data has none).

    Past the end of an episode the chunk is padded with a 'hold still, suction off' action.
    """

    def __init__(self, data: dict, chunk: int, indices: np.ndarray | None = None):
        self.images = data["images"]
        self.tokens = data["tokens"]
        self.proprio = data["proprio"]
        self.actions = data["actions"]
        self.aux = data["aux"] if "aux" in data else np.zeros((len(data["images"]), 6), np.float32)
        self.episode = data["episode"]
        self.chunk = chunk
        self.indices = np.arange(len(self.images)) if indices is None else indices
        n = len(self.images)
        hold = np.array([0, 0, 0, -1], dtype=np.float32)
        chunks = np.empty((n, chunk, 4), dtype=np.float32)
        for k in range(chunk):
            idx = np.minimum(np.arange(n) + k, n - 1)
            same = self.episode[idx] == self.episode
            chunks[:, k] = np.where(same[:, None], self.actions[idx], hold)
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return (torch.from_numpy(self.images[j]), torch.from_numpy(self.tokens[j]),
                torch.from_numpy(self.proprio[j]), torch.from_numpy(self.chunks[j]),
                torch.from_numpy(self.aux[j]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect noise-injected expert demonstrations.")
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="data/demos.npz")
    args = ap.parse_args()
    t0 = time.time()
    data = collect(args.episodes, args.seed, args.workers)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
    print(f"kept {data['kept']}/{data['tried']} demonstrations, {len(data['images'])} frames "
          f"in {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
