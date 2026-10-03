"""Figures for the README: success rate by test split, and a camera filmstrip of one episode."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vla.data import denormalize_action  # noqa: E402
from vla.env import PickPlaceEnv  # noqa: E402
from vla.evaluate import EVAL_SEED, SPLITS  # noqa: E402
from vla.model import load_checkpoint  # noqa: E402

LABELS = {"seen": "Seen\ninstructions", "more_distractors": "4 blocks\n(more clutter)",
          "unseen_pairs": "Unseen colour\npairs", "unseen_phrasing": "Unseen\nphrasing"}
ORDER = ["seen", "more_distractors", "unseen_pairs", "unseen_phrasing"]
SERIES = [("results/eval_bc.json", "Imitation learning", "#2a78d6"),
          ("results/eval_rl.json", "+ PPO fine-tuning", "#eb6834")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e2db"


def success_chart(path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    width = 0.36
    for k, (file, name, color) in enumerate(SERIES):
        res = {r["split"]: r for r in json.load(open(file))["results"]}
        x = np.arange(len(ORDER)) + (k - 0.5) * (width + 0.02)
        rates = [100 * res[s]["success_rate"] for s in ORDER]
        lo = [100 * (res[s]["success_rate"] - res[s]["ci95"][0]) for s in ORDER]
        hi = [100 * (res[s]["ci95"][1] - res[s]["success_rate"]) for s in ORDER]
        ax.bar(x, rates, width, color=color, label=name, zorder=2)
        ax.errorbar(x, rates, yerr=[lo, hi], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2, zorder=3)
        for xi, r, h in zip(x, rates, hi, strict=True):
            ax.text(xi, r + h + 1.5, f"{r:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(ORDER)), [LABELS[s] for s in ORDER], fontsize=8, color=INK)
    ax.set_ylabel("Task success (%)", fontsize=9, color=INK)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Closed-loop success, 200 test episodes per split (95% Wilson CI)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path)


def filmstrip(path: str, checkpoint: str = "checkpoints/rl_policy.pt") -> None:
    model, env = load_checkpoint(checkpoint), PickPlaceEnv()
    rng = np.random.default_rng(EVAL_SEED + 3)
    spec = SPLITS["more_distractors"]
    for _ in range(20):                                   # first successful test episode
        task = env.sample_task(rng, spec["pairs"], spec["templates"], spec["n_blocks"])
        obs, frames, done, info = env.reset(task, rng), [], False, None
        while not done:
            frames.append(obs["image"])
            with torch.no_grad():
                a = model(torch.from_numpy(obs["image"])[None], torch.from_numpy(obs["tokens"])[None],
                          torch.from_numpy(obs["proprio"])[None])[0, 0].numpy()
            obs, _, done, info = env.step(denormalize_action(a))
        frames.append(obs["image"])
        if info.success:
            break
    picks = np.linspace(0, len(frames) - 1, 6).round().astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(10, 1.6), dpi=150)
    for ax, i in zip(axes, picks, strict=True):
        ax.imshow(frames[i], interpolation="nearest")
        ax.set_title(f"step {i}", fontsize=7, color=MUTED)
        ax.axis("off")
    fig.suptitle(f'"{task.instruction}"', fontsize=8, color=INK, y=0.98)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    success_chart("results/success_by_split.png")
    filmstrip("results/episode_filmstrip.png")
    print("wrote results/success_by_split.png and results/episode_filmstrip.png")
