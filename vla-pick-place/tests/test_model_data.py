import numpy as np
import torch

from vla.data import ChunkDataset, denormalize_action, normalize_action
from vla.evaluate import wilson
from vla.language import HELDOUT_PAIRS, MAX_TOKENS, UNK, WORD_TO_ID, tokenize, train_pairs
from vla.model import PolicyConfig, VLAPolicy


def small_model():
    torch.manual_seed(0)
    return VLAPolicy(PolicyConfig(dim=32, depth=2, heads=2, mlp_dim=64)).eval()


def batch(b=2):
    g = torch.Generator().manual_seed(0)
    return (torch.randint(0, 256, (b, 64, 96, 3), dtype=torch.uint8, generator=g),
            torch.tensor([tokenize("put the red block on the cyan pad")] * b),
            torch.randn(b, 5, generator=g))


def test_tokenizer():
    ids = tokenize("put the red block on the cyan pad")
    assert len(ids) == MAX_TOKENS and ids[-1] == 0
    assert tokenize("set the red block down")[0] == WORD_TO_ID[UNK]


def test_heldout_pairs_never_in_training():
    assert not set(HELDOUT_PAIRS) & set(train_pairs())
    assert len(train_pairs()) + len(HELDOUT_PAIRS) == 12


def test_output_shape():
    out = small_model()(*batch(3))
    assert out.shape == (3, 4, 4)


def test_padding_tokens_do_not_change_action():
    model = small_model()
    images, tokens, proprio = batch()
    before = model(images, tokens, proprio)
    with torch.no_grad():
        model.word_embed.weight[0].normal_(0, 10.0)               # perturb the <pad> embedding
    after = model(images, tokens, proprio)
    assert torch.allclose(before, after, atol=1e-6)


def test_instruction_changes_action():
    model = small_model()
    images, tokens, proprio = batch()
    other = torch.tensor([tokenize("put the blue block on the purple pad")] * 2)
    assert not torch.allclose(model(images, tokens, proprio), model(images, other, proprio))


def test_scene_tokens_are_spatial():
    model = small_model()
    img = torch.zeros(1, 64, 96, 3, dtype=torch.uint8)
    base = model.scene_tokens(model.stem(img))[0]
    img[0, 8:16, 16:24, 1] = 255                                  # touches grid cell (row 1, col 2)
    changed = (model.scene_tokens(model.stem(img))[0] - base).abs().sum(-1) > 1e-6
    assert base.shape == (96, 32)
    hot = changed.nonzero().flatten().tolist()
    assert 1 * 12 + 2 in hot and len(hot) < 20                    # local effect only
    assert not changed[7 * 12 + 11]                               # far corner unaffected


def test_spatial_softmax_finds_a_peak():
    model = small_model()
    feat = torch.zeros(1, model.cfg.stem2, 16, 24)
    with torch.no_grad():
        model.kp_conv.weight.zero_()
        model.kp_conv.bias.zero_()
        model.kp_conv.weight[:, 0, 1, 1] = 50.0                   # heatmap = 50 x channel 0
    feat[0, 0, 12, 6] = 1.0                                       # bright spot at row 12, column 6
    kp, _ = model.keypoints(feat, torch.tensor([tokenize("put the red block on the cyan pad")]))
    u, v = kp[0, 0].tolist()
    assert abs(u - (-1 + 2 * 6 / 23)) < 0.02 and abs(v - (-1 + 2 * 12 / 15)) < 0.02


def test_heatmap_targets_peak_at_label():
    model = small_model()
    uv = torch.tensor([[[-1 + 2 * 5 / 23, -1 + 2 * 9 / 15]]])    # cell (row 9, col 5)
    target = model.heatmap_targets(uv)
    assert torch.isclose(target.sum(), torch.tensor(1.0))
    assert target[0, 0].argmax().item() == 9 * 24 + 5


def test_chunks_stop_at_episode_boundary():
    n = 6
    data = {"images": np.zeros((n, 64, 96, 3), np.uint8), "tokens": np.zeros((n, 12), np.int64),
            "proprio": np.zeros((n, 5), np.float32),
            "actions": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
            "episode": np.array([0, 0, 0, 1, 1, 1])}
    ds = ChunkDataset(data, chunk=3)
    chunk = ds[1][3].numpy()
    assert np.array_equal(chunk[0], data["actions"][1]) and np.array_equal(chunk[1], data["actions"][2])
    assert np.array_equal(chunk[2], [0, 0, 0, -1])                # padded, not episode 1's action


def test_action_normalisation_round_trip():
    a = np.array([0.1, -0.05, 0.02, 1.0], np.float32)
    assert np.allclose(denormalize_action(normalize_action(a)), a)


def test_wilson_interval():
    lo, hi = wilson(90, 100)
    assert 0.82 < lo < 0.9 < hi < 0.96
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
