"""Vision-language-action Transformer policy.

Inputs: a 64 x 96 RGB camera image, the instruction (word ids) and the arm's
joint state. Output: a chunk of the next K normalised actions.

Image pathway
  * CoordConv stem: RGB + (x, y) coordinate channels, two stride-2 convolutions
    -> a 16 x 24 feature map (stride 4).
  * Language-conditioned keypoints: the feature map is modulated by the
    instruction with FiLM (per-channel scale and shift computed from the mean
    word embedding), and a 3x3 convolution produces one heatmap per keypoint.
    A spatial softmax turns each heatmap into exact (u, v) coordinates:
    keypoint 0 = the block named in the instruction, 1 = the named pad,
    2 = the gripper. These coordinates enter the Transformer as one token.
  * A third convolution (stride 2) gives an 8 x 12 grid of scene tokens.

Token sequence (1 + 1 + 1 + 12 + 96 = 111):
  [ACTION query] [proprioception] [keypoints] [instruction words x12] [scene x96]

A pre-norm Transformer encoder mixes all tokens; the ACTION token is read out
into the action chunk. Attention is written out explicitly (fused QKV, exact
GELU) so the C++ runtime in cpp/ mirrors it operation for operation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import IMG_H, IMG_W
from .language import MAX_TOKENS, VOCAB


@dataclass
class PolicyConfig:
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_dim: int = 256
    patch: int = 8              # stride of the scene-token grid
    chunk: int = 4
    action_dim: int = 4
    proprio_dim: int = 5
    vocab: int = len(VOCAB)
    max_tokens: int = MAX_TOKENS
    img_h: int = IMG_H
    img_w: int = IMG_W
    stem1: int = 32             # channels after the first stem convolution
    stem2: int = 64             # channels of the stride-4 feature map
    n_keypoints: int = 3        # target block, target pad, gripper

    @property
    def n_patches(self) -> int:
        return (self.img_h // self.patch) * (self.img_w // self.patch)

    @property
    def seq_len(self) -> int:
        return 3 + self.max_tokens + self.n_patches


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q = q.view(b, n, h, d // h).transpose(1, 2)
        k = k.view(b, n, h, d // h).transpose(1, 2)
        v = v.view(b, n, h, d // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // h)
        att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(out)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


def _grid(n: int) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, n)


class VLAPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        cfg = cfg or PolicyConfig()
        self.cfg = cfg
        d, k = cfg.dim, cfg.n_keypoints
        self.conv1 = nn.Conv2d(5, cfg.stem1, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(cfg.stem1, cfg.stem2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(cfg.stem2, d, 3, stride=2, padding=1)
        self.film = nn.Linear(d, 2 * cfg.stem2)
        self.kp_conv = nn.Conv2d(cfg.stem2, k, 3, stride=1, padding=1)
        # Learnable sharpness of the spatial softmax; starting sharp speeds up learning.
        self.kp_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.kp_embed = nn.Linear(2 * k, d)
        self.word_embed = nn.Embedding(cfg.vocab, d)
        self.proprio_embed = nn.Linear(cfg.proprio_dim, d)
        self.action_query = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, d))
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp_dim) for _ in range(cfg.depth))
        self.ln_out = nn.LayerNorm(d)
        self.action_head = nn.Linear(d, cfg.chunk * cfg.action_dim)
        # RL fine-tuning only: std of the Gaussian exploration noise.
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), -1.6))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.action_query, std=0.02)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        h, w = cfg.img_h, cfg.img_w
        coords = torch.stack([_grid(w).view(1, w).expand(h, w), _grid(h).view(h, 1).expand(h, w)])
        self.register_buffer("coords", coords[None], persistent=False)
        self.register_buffer("grid_u", _grid(w // 4), persistent=False)
        self.register_buffer("grid_v", _grid(h // 4), persistent=False)

    def stem(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 (B, H, W, 3) -> stride-4 feature map (B, stem2, H/4, W/4)."""
        x = images.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        return F.gelu(self.conv2(F.gelu(self.conv1(x))))

    def keypoints(self, feat: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Language-conditioned spatial-softmax keypoints (B, n_keypoints, 2) as (u, v) in [-1, 1],
        and the heatmap logits (B, n_keypoints, H/4 * W/4) used by the training loss."""
        words = self.word_embed(tokens)
        mask = (tokens != 0).float().unsqueeze(-1)
        lang = (words * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        gamma, beta = self.film(lang).chunk(2, dim=-1)
        x = feat * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        heat = self.kp_conv(F.gelu(x)) * self.kp_log_scale.exp()       # (B, K, h, w)
        b, k, hh, ww = heat.shape
        logits = heat.view(b, k, hh * ww)
        p = logits.softmax(-1).view(b, k, hh, ww)
        u = (p.sum(2) * self.grid_u).sum(-1)
        v = (p.sum(3) * self.grid_v).sum(-1)
        return torch.stack([u, v], dim=-1), logits

    def scene_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv3(feat).flatten(2).transpose(1, 2)            # (B, n_patches, dim)

    def encode(self, images, tokens, proprio):
        b = images.shape[0]
        feat = self.stem(images)
        kps, logits = self.keypoints(feat, tokens)
        seq = torch.cat([
            self.action_query.expand(b, -1, -1),
            self.proprio_embed(proprio).unsqueeze(1),
            self.kp_embed(kps.flatten(1)).unsqueeze(1),
            self.word_embed(tokens),
            self.scene_tokens(feat),
        ], dim=1) + self.pos
        key_mask = torch.ones(b, seq.shape[1], dtype=torch.bool, device=seq.device)
        key_mask[:, 3:3 + self.cfg.max_tokens] = tokens != 0           # ignore <pad> words
        for blk in self.blocks:
            seq = blk(seq, key_mask)
        return self.ln_out(seq[:, 0]), kps, logits

    def forward(self, images, tokens, proprio, with_aux: bool = False):
        """Action chunk (B, chunk, action_dim) in normalised units; with `with_aux`, also the
        keypoints (B, K, 2) and heatmap logits (B, K, cells) for the auxiliary loss."""
        h, kps, logits = self.encode(images, tokens, proprio)
        chunk = self.action_head(h).view(-1, self.cfg.chunk, self.cfg.action_dim)
        return (chunk, kps, logits) if with_aux else chunk

    def heatmap_targets(self, uv: torch.Tensor, sigma_cells: float = 0.75) -> torch.Tensor:
        """Soft one-hot targets: a Gaussian around each labelled (u, v), (B, K, cells)."""
        hh, ww = self.grid_v.numel(), self.grid_u.numel()
        col = (uv[..., 0] + 1) / 2 * (ww - 1)
        row = (uv[..., 1] + 1) / 2 * (hh - 1)
        cols = torch.arange(ww, dtype=uv.dtype).view(1, 1, 1, ww)
        rows = torch.arange(hh, dtype=uv.dtype).view(1, 1, hh, 1)
        g = torch.exp(-((cols - col[..., None, None]) ** 2 + (rows - row[..., None, None]) ** 2)
                      / (2 * sigma_cells ** 2)).flatten(2)
        return g / g.sum(-1, keepdim=True).clamp(min=1e-12)


def keypoint_loss(model: VLAPolicy, kps: torch.Tensor, logits: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Heatmap cross-entropy (fast to learn) plus L1 on the soft-argmax coordinates (precise)."""
    target = model.heatmap_targets(uv.view(kps.shape))
    ce = -(target * logits.log_softmax(-1)).sum(-1).mean()
    return ce + 10.0 * (kps - uv.view(kps.shape)).abs().mean()


def save_checkpoint(path: str, model: VLAPolicy, extra: dict | None = None) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: str) -> VLAPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = VLAPolicy(PolicyConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
