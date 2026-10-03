"""Scripted expert that uses privileged simulator state (object poses).

The expert is written as a *continuous* feedback controller rather than a
sequence of hard phase switches, because a policy trained by regression copies
a smooth expert far better than one with sharp decision boundaries:

  * joint targets come from analytic inverse kinematics of the current goal
    (the target block, or the target pad once the block is held), and the arm
    moves towards them with a proportional gain clipped at the speed limit;
  * the quill height follows a funnel: high while far from the goal, lowering
    smoothly as the horizontal error shrinks;
  * after a grasp, horizontal motion is scaled by how far the block has been
    lifted, so the block never sweeps across the table.

It is only used to generate demonstrations; the learned policy never sees
privileged state.
"""
from __future__ import annotations

import numpy as np

from .env import ACTION_LIMITS, BLOCK_HALF, JZ_RANGE, TIP_Z0, PickPlaceEnv, inverse_kinematics

HOVER_Z = TIP_Z0                         # tip height with the quill fully up
PLACE_Z = 2 * BLOCK_HALF + 0.03          # tip height at release (block ~2 cm above the table)
FUNNEL_START, FUNNEL_END = 0.06, 0.008   # descend between these horizontal errors (m)


class ScriptedExpert:
    def __init__(self, gain: float = 0.6, xy_tol: float = 0.012):
        self.gain = gain
        self.xy_tol = xy_tol

    def _ik(self, env: PickPlaceEnv, xy: np.ndarray) -> np.ndarray:
        q = env.target_q
        t1, t2 = inverse_kinematics(xy[0], xy[1], 1.0 if q[1] >= 0 else -1.0)
        d1 = (t1 - q[0] + np.pi) % (2 * np.pi) - np.pi
        return np.array([q[0] + d1, t2])

    @staticmethod
    def _funnel(xy_err: float, low: float) -> float:
        s = np.clip((xy_err - FUNNEL_END) / (FUNNEL_START - FUNNEL_END), 0.0, 1.0)
        return low + s * (HOVER_Z - low)

    def _action(self, env: PickPlaceEnv, q_xy: np.ndarray, tip_z: float, xy_scale: float,
                suction: bool) -> np.ndarray:
        q = env.target_q
        jz_target = float(np.clip(tip_z - TIP_Z0, *JZ_RANGE))
        delta = self.gain * np.array([*(q_xy - q[:2]) * xy_scale, jz_target - q[2]])
        delta = np.clip(delta, -ACTION_LIMITS[:3], ACTION_LIMITS[:3])
        return np.array([*delta, 1.0 if suction else -1.0], dtype=np.float32)

    def act(self, env: PickPlaceEnv) -> np.ndarray:
        task, tip = env.task, env.tip_pos()
        if env.held is None and env.is_success():                 # done: lift and hold still
            return self._action(env, env.target_q[:2], HOVER_Z, 1.0, False)
        if env.held is not None and env.held != task.target_block:
            return self._action(env, env.target_q[:2], PLACE_Z, 1.0, False)   # wrong block: let go
        if env.held is None:                                       # approach and grasp
            block = env.block_pos(task.target_block)
            err = float(np.linalg.norm(tip[:2] - block[:2]))
            grasp_z = block[2] + BLOCK_HALF - 0.006               # press slightly onto the top face
            return self._action(env, self._ik(env, block[:2]), self._funnel(err, grasp_z), 1.0,
                                suction=err < 0.02)
        pad = env.pad_pos(task.target_pad)                         # carry and place
        err = float(np.linalg.norm(tip[:2] - pad))
        lift = float(np.clip((tip[2] - PLACE_Z) / (HOVER_Z - PLACE_Z - 0.02), 0.0, 1.0))
        xy_scale = lift if err > FUNNEL_START else 1.0
        release = err < self.xy_tol and tip[2] < PLACE_Z + 0.01
        return self._action(env, self._ik(env, pad), self._funnel(err, PLACE_Z), xy_scale,
                            suction=not release)
