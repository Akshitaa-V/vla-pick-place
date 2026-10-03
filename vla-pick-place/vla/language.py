"""Instruction templates, task splits and a small word-level tokenizer."""
from __future__ import annotations

import re

BLOCK_COLORS = ("red", "green", "blue", "yellow")
PAD_COLORS = ("purple", "orange", "cyan")

# (block colour, pad colour) pairs never used as the *target* during training.
# They appear separately (as distractors, or with other pads) but never together.
HELDOUT_PAIRS = (("red", "cyan"), ("blue", "orange"), ("yellow", "purple"))

TRAIN_TEMPLATES = (
    "put the {b} block on the {p} pad",
    "place the {b} block onto the {p} pad",
    "move the {b} block to the {p} pad",
    "pick up the {b} block and drop it on the {p} pad",
)
# A phrasing never seen in training; "set" and "down" are out-of-vocabulary.
HELDOUT_TEMPLATES = ("set the {b} block down on the {p} pad",)

PAD, UNK = "<pad>", "<unk>"
MAX_TOKENS = 12


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_vocab() -> list[str]:
    words = set()
    for t in TRAIN_TEMPLATES:
        words.update(_words(t.format(b="", p="")))
    words.update(BLOCK_COLORS)
    words.update(PAD_COLORS)
    return [PAD, UNK] + sorted(words)


VOCAB = build_vocab()
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}


def tokenize(text: str, max_tokens: int = MAX_TOKENS) -> list[int]:
    ids = [WORD_TO_ID.get(w, WORD_TO_ID[UNK]) for w in _words(text)][:max_tokens]
    return ids + [WORD_TO_ID[PAD]] * (max_tokens - len(ids))


def train_pairs() -> list[tuple[str, str]]:
    return [(b, p) for b in BLOCK_COLORS for p in PAD_COLORS if (b, p) not in HELDOUT_PAIRS]
