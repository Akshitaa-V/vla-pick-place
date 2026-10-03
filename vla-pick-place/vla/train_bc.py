"""Behaviour cloning of the VLA policy from expert demonstrations.

Single process:   python -m vla.train_bc --config configs/bc.yaml
Data parallel:    torchrun --nproc_per_node 2 -m vla.train_bc --config configs/bc.yaml

Under torchrun the model is wrapped in DistributedDataParallel (gloo backend on
CPU, nccl on GPU) and each rank trains on its own shard via DistributedSampler.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import ChunkDataset
from .model import PolicyConfig, VLAPolicy, keypoint_loss, save_checkpoint


def random_shift(images: torch.Tensor, pad: int) -> torch.Tensor:
    """Random translation by up to `pad` pixels (replicate padding), per sample."""
    if pad == 0:
        return images
    b, h, w, _ = images.shape
    x = F.pad(images.permute(0, 3, 1, 2).float(), (pad, pad, pad, pad), mode="replicate")
    dy = torch.randint(0, 2 * pad + 1, (b,))
    dx = torch.randint(0, 2 * pad + 1, (b,))
    out = torch.stack([x[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(b)])
    return out.permute(0, 2, 3, 1).to(torch.uint8)


def chunk_loss(pred: torch.Tensor, target: torch.Tensor, suction_weight: float) -> torch.Tensor:
    weights = torch.ones(target.shape[-1])
    weights[3] = suction_weight
    return ((pred - target) ** 2 * weights).mean()


def split_by_episode(episode: np.ndarray, val_frac: float, seed: int):
    eps = np.unique(episode)
    rng = np.random.default_rng(seed)
    val_eps = set(rng.choice(eps, size=max(1, int(len(eps) * val_frac)), replace=False).tolist())
    is_val = np.array([e in val_eps for e in episode])
    return np.where(~is_val)[0], np.where(is_val)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bc.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // world))
    else:
        rank, world = 0, 1
    torch.manual_seed(cfg["seed"] + rank)

    data = dict(np.load(cfg["data"]))
    pcfg = PolicyConfig(**cfg.get("model", {}))
    train_idx, val_idx = split_by_episode(data["episode"], cfg["val_frac"], cfg["seed"])
    train_ds = ChunkDataset(data, pcfg.chunk, train_idx)
    val_ds = ChunkDataset(data, pcfg.chunk, val_idx)
    sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=cfg["seed"]) if distributed else None
    per_rank_batch = cfg["batch_size"] // world
    train_dl = DataLoader(train_ds, batch_size=per_rank_batch, sampler=sampler,
                          shuffle=sampler is None, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256)

    model = VLAPolicy(pcfg)
    model.log_std.requires_grad_(False)   # RL-only parameter; unused in behaviour cloning
    net = DDP(model) if distributed else model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(train_dl)
    warmup = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    history, best = [], float("inf")
    if rank == 0:
        print(f"train frames {len(train_ds)}, val frames {len(val_ds)}, params "
              f"{sum(p.numel() for p in model.parameters()):,}, world size {world}", flush=True)
    for epoch in range(cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        net.train()
        t0, run = time.time(), 0.0
        for images, tokens, proprio, chunks, aux in train_dl:
            images = random_shift(images, cfg["shift_pad"])
            pred, kps, logits = net(images, tokens, proprio, with_aux=True)
            loss = chunk_loss(pred, chunks, cfg["suction_weight"])
            loss = loss + cfg["aux_weight"] * keypoint_loss(model, kps, logits, aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        if rank == 0:
            model.eval()
            with torch.no_grad():
                vals = [chunk_loss(model(i, t, p), c, cfg["suction_weight"]).item() * len(i)
                        for i, t, p, c, _ in val_dl]
            val = sum(vals) / len(val_ds)
            rec = {"epoch": epoch + 1, "train_loss": run / len(train_dl), "val_loss": val,
                   "seconds": round(time.time() - t0, 1)}
            history.append(rec)
            print(json.dumps(rec), flush=True)
            if val < best:
                best = val
                save_checkpoint(os.path.join(cfg["out_dir"], "bc_policy.pt"), model,
                                {"epoch": epoch + 1, "val_loss": val})
            json.dump(history, open(os.path.join(cfg["out_dir"], "bc_history.json"), "w"), indent=1)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
