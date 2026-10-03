import struct

import numpy as np
import pytest
import torch

from vla.data import normalize_action
from vla.env import PickPlaceEnv
from vla.evaluate import run_split
from vla.expert import ScriptedExpert
from vla.export import export_weights
from vla.language import TRAIN_TEMPLATES, train_pairs
from vla.model import PolicyConfig, VLAPolicy
from vla.rl import Critic, potential, privileged_state


def test_export_size_and_header(tmp_path):
    model = VLAPolicy(PolicyConfig(dim=32, depth=2, heads=2, mlp_dim=64))
    path = tmp_path / "w.bin"
    n_bytes = export_weights(model, str(path))
    n_params = sum(p.numel() for p in model.parameters()) - model.log_std.numel()  # exploration std is RL-only
    assert n_bytes == 4 * n_params
    with open(path, "rb") as f:
        assert f.read(4) == b"VLAP" and struct.unpack("<I", f.read(4))[0] == 4


def test_cpp_runtime_matches_torch(tmp_path):
    vla_cpp = pytest.importorskip("vla_cpp")
    torch.manual_seed(1)
    model = VLAPolicy(PolicyConfig(dim=32, depth=2, heads=2, mlp_dim=64)).eval()
    export_weights(model, str(tmp_path / "w.bin"))
    runtime = vla_cpp.Policy(str(tmp_path / "w.bin"))
    env, rng = PickPlaceEnv(), np.random.default_rng(0)
    obs = env.reset(env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES), rng)
    expected = model(torch.from_numpy(obs["image"])[None], torch.from_numpy(obs["tokens"])[None],
                     torch.from_numpy(obs["proprio"])[None]).detach().numpy().reshape(-1)
    got = runtime.predict(obs["image"], obs["tokens"], obs["proprio"])
    assert np.abs(got - expected).max() < 1e-4


def test_benchmark_runs_with_expert():
    env, expert = PickPlaceEnv(), ScriptedExpert()
    result = run_split(lambda obs: normalize_action(expert.act(env)), "seen", episodes=3, env=env)
    assert result["success_rate"] == 1.0 and result["wrong_placement_rate"] == 0.0


def test_potential_jumps_on_grasp():
    env, expert, rng = PickPlaceEnv(), ScriptedExpert(), np.random.default_rng(4)
    env.reset(env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES), rng)
    before = potential(env)
    while env.held is None:
        env.step(expert.act(env))
    assert potential(env) > before + 0.5          # grasping switches to the (higher) carry potential


def test_critic_input_size():
    env, rng = PickPlaceEnv(), np.random.default_rng(0)
    env.reset(env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES), rng)
    s = torch.from_numpy(privileged_state(env))[None]
    assert Critic(s.shape[1])(s).shape == (1,)
