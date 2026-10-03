"""Export a trained policy to the flat binary format read by the C++ runtime.

Layout (little endian):
  "VLAP" | uint32 version | 15 x int32 config
  then each tensor as: uint32 element count | float32 data (row-major, PyTorch layout)
in the order produced by `tensor_order`. The RL exploration std is training-only
and is not exported.

It also writes parity fixtures: real observations from the simulator with the
PyTorch outputs, used by the C++ tests to check numerical agreement.
"""
from __future__ import annotations

import argparse
import struct

import numpy as np
import torch

from .env import PickPlaceEnv
from .language import TRAIN_TEMPLATES, train_pairs
from .model import VLAPolicy, load_checkpoint

MAGIC, VERSION = b"VLAP", 4
CONFIG_FIELDS = ("dim", "depth", "heads", "mlp_dim", "patch", "chunk", "action_dim",
                 "proprio_dim", "vocab", "max_tokens", "img_h", "img_w", "stem1", "stem2", "n_keypoints")


def tensor_order(model: VLAPolicy) -> list[torch.Tensor]:
    m = model
    out = [m.conv1.weight, m.conv1.bias, m.conv2.weight, m.conv2.bias, m.conv3.weight, m.conv3.bias,
           m.film.weight, m.film.bias, m.kp_conv.weight, m.kp_conv.bias, m.kp_log_scale.view(1),
           m.kp_embed.weight, m.kp_embed.bias,
           m.word_embed.weight,
           m.proprio_embed.weight, m.proprio_embed.bias, m.action_query, m.pos]
    for b in m.blocks:
        out += [b.ln1.weight, b.ln1.bias, b.qkv.weight, b.qkv.bias, b.proj.weight, b.proj.bias,
                b.ln2.weight, b.ln2.bias, b.fc1.weight, b.fc1.bias, b.fc2.weight, b.fc2.bias]
    return out + [m.ln_out.weight, m.ln_out.bias, m.action_head.weight, m.action_head.bias]


def export_weights(model: VLAPolicy, path: str) -> int:
    n_bytes = 0
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", VERSION))
        f.write(struct.pack(f"<{len(CONFIG_FIELDS)}i", *[getattr(model.cfg, k) for k in CONFIG_FIELDS]))
        for t in tensor_order(model):
            arr = t.detach().contiguous().view(-1).numpy().astype("<f4")
            f.write(struct.pack("<I", arr.size))
            f.write(arr.tobytes())
            n_bytes += arr.nbytes
    return n_bytes


@torch.no_grad()
def export_fixtures(model: VLAPolicy, path: str, n: int = 16, seed: int = 123) -> None:
    """uint32 n, then per sample: image uint8 | tokens int32 x12 | proprio f32 x5 | expected f32."""
    env, rng = PickPlaceEnv(), np.random.default_rng(seed)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            task = env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES)
            obs = env.reset(task, rng)
            for _ in range(i % 5):                       # some samples mid-motion
                obs, *_ = env.step(rng.uniform(-1, 1, 4) * [0.1, 0.1, 0.02, 1])
            out = model(torch.from_numpy(obs["image"])[None], torch.from_numpy(obs["tokens"])[None],
                        torch.from_numpy(obs["proprio"])[None])[0].reshape(-1).numpy()
            f.write(obs["image"].astype(np.uint8).tobytes())
            f.write(obs["tokens"].astype("<i4").tobytes())
            f.write(obs["proprio"].astype("<f4").tobytes())
            f.write(out.astype("<f4").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/bc_policy.pt")
    ap.add_argument("--out", default="checkpoints/policy.bin")
    ap.add_argument("--fixtures", default="cpp/tests/fixtures.bin")
    args = ap.parse_args()
    model = load_checkpoint(args.checkpoint)
    n = export_weights(model, args.out)
    export_fixtures(model, args.fixtures)
    print(f"wrote {n / 1e6:.2f} MB of weights to {args.out} and fixtures to {args.fixtures}")


if __name__ == "__main__":
    main()
