import numpy as np
import pytest

from vla.data import localisation_target
from vla.env import (
    ACTION_LIMITS,
    JOINT_LIMIT,
    PickPlaceEnv,
    forward_kinematics,
    inverse_kinematics,
    project_to_image,
)
from vla.expert import ScriptedExpert
from vla.language import TRAIN_TEMPLATES, train_pairs


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv()


@pytest.mark.parametrize("xy", [(0.4, 0.1), (0.3, -0.45), (0.55, 0.0), (0.2, 0.5)])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_inverse_kinematics_round_trip(xy, sign):
    t1, t2 = inverse_kinematics(*xy, sign)
    tip = forward_kinematics(np.array([t1, t2, 0.0]))
    assert np.allclose(tip[:2], xy, atol=1e-9)


def test_reset_is_deterministic(env):
    images = []
    for _ in range(2):
        rng = np.random.default_rng(7)
        task = env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES)
        images.append(env.reset(task, rng)["image"])
    assert images[0].shape == (64, 96, 3)
    assert np.array_equal(images[0], images[1])


def test_arm_tracks_joint_targets(env):
    rng = np.random.default_rng(1)
    env.reset(env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES), rng)
    start = env.arm_q()
    for _ in range(5):
        env.step(np.array([0.1, -0.1, -0.02, -1.0]))
    assert np.allclose(env.arm_q() - start, [0.5, -0.5, -0.1], atol=0.05)


def test_safety_filter_clips_and_counts(env):
    rng = np.random.default_rng(2)
    env.reset(env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES), rng)
    a, n = env.safety_filter(np.array([1.0, 0.0, 0.0, 1.0]))
    assert n == 1 and abs(a[0] - ACTION_LIMITS[0]) < 1e-9
    env.target_q[0] = JOINT_LIMIT - 0.05
    a, n = env.safety_filter(np.array([0.1, 0.0, 0.0, 1.0]))
    assert n == 1 and env.target_q[0] + a[0] <= JOINT_LIMIT + 1e-9
    a, n = env.safety_filter(np.array([0.0, 0.0, 0.1, 1.0]))     # quill cannot rise above its top
    assert n >= 1 and a[2] <= 1e-9


def test_suction_needs_contact(env):
    rng = np.random.default_rng(3)
    env.reset(env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES), rng)
    for _ in range(3):
        env.step(np.array([0.0, 0.0, 0.0, 1.0]))                  # suction on, far above blocks
    assert env.held is None


def test_expert_solves_tasks(env):
    rng, expert = np.random.default_rng(11), ScriptedExpert()
    for _ in range(5):
        env.reset(env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES), rng)
        done, info = False, None
        while not done:
            _, _, done, info = env.step(expert.act(env))
        assert info.success and not info.wrong_placement and info.safety_interventions == 0


def test_wrong_pad_release_is_flagged(env):
    rng, expert = np.random.default_rng(5), ScriptedExpert()
    task = env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES)
    env.reset(task, rng)
    for _ in range(60):                                           # carry the block, then drop early
        env.step(expert.act(env))
        if env.held is not None and env.tip_pos()[2] > 0.15:
            break
    assert env.held == task.target_block
    _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
    assert info.wrong_placement


def test_projection_matches_rendered_block(env):
    """The projected centre of the target block's top face must land on that block's colour."""
    rng = np.random.default_rng(9)
    task = env.sample_task(rng, train_pairs(), TRAIN_TEMPLATES)
    obs = env.reset(task, rng)
    centre = env.block_pos(task.target_block) + np.array([0, 0, 0.035])
    row, col = np.round(project_to_image(centre)).astype(int)
    rgb = obs["image"][row, col].astype(int)
    expected = {"red": 0, "green": 1, "blue": 2, "yellow": 0}[task.target_block]
    assert rgb[expected] == rgb.max()
    uv = localisation_target(env)
    assert np.all(np.abs(uv) <= 1.1)
