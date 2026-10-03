"""Language-conditioned pick-and-place with a SCARA arm in MuJoCo.

The arm has two revolute joints, a vertical quill (prismatic joint) and a
suction cup. A fixed overhead camera gives a 64x96 RGB image. Each episode
places 2-4 coloured blocks and two coloured pads on the table and issues an
instruction such as "put the red block on the cyan pad".

The suction cup is modelled as a kinematic attachment: when it is switched on
and the cup touches the top of a block, the block follows the cup until the
suction is switched off, then falls under gravity.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np

from .language import BLOCK_COLORS, PAD_COLORS, tokenize

L1, L2 = 0.35, 0.30            # link lengths (m)
ARM_HEIGHT = 0.40              # height of the arm plane above the table
QUILL_LEN = 0.24               # tip sits 0.24 m below the arm plane at jz = 0
TIP_Z0 = ARM_HEIGHT - QUILL_LEN
JZ_RANGE = (-0.12, 0.0)
JOINT_LIMIT = 2.6
BLOCK_HALF = 0.035
PAD_RADIUS = 0.07
GRASP_XY_TOL = 0.03
GRASP_Z_TOL = 0.012

CONTROL_DT = 0.1               # 10 Hz control
PHYSICS_DT = 0.002
SUBSTEPS = int(round(CONTROL_DT / PHYSICS_DT))

# Per-step action limits: dtheta1, dtheta2 (rad), dz (m); 4th entry is suction.
ACTION_LIMITS = np.array([0.15, 0.15, 0.03, 1.0], dtype=np.float32)
INTERVENTION_TOL = 0.1         # a correction above 10% of the step limit counts as an intervention
IMG_H, IMG_W = 64, 96
CAM_POS = np.array([0.36, 0.0, 1.3])   # overhead camera, looking straight down
CAM_FOVY = 32.0                          # degrees
PARK = np.array([3.0, 3.0])    # off-camera parking spot for unused objects

RGBA = {
    "red": "0.85 0.15 0.15 1", "green": "0.15 0.7 0.2 1",
    "blue": "0.15 0.3 0.9 1", "yellow": "0.95 0.85 0.1 1",
    "purple": "0.55 0.2 0.7 1", "orange": "1.0 0.5 0.05 1", "cyan": "0.1 0.85 0.85 1",
}


def _mjcf() -> str:
    blocks = "\n".join(
        f'''<body name="block_{c}" pos="{3 + i} 3 {BLOCK_HALF}">
              <freejoint name="free_{c}"/>
              <geom name="block_{c}" type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}"
                    mass="0.1" rgba="{RGBA[c]}" friction="1 0.005 0.0001"/>
            </body>''' for i, c in enumerate(BLOCK_COLORS))
    pads = "\n".join(
        f'<geom name="pad_{c}" type="cylinder" pos="{3 + i} -3 0.001" size="{PAD_RADIUS} 0.001" '
        f'rgba="{RGBA[c]}" contype="0" conaffinity="0"/>' for i, c in enumerate(PAD_COLORS))
    return f"""
<mujoco model="scara_pick_place">
  <compiler angle="radian"/>
  <option timestep="{PHYSICS_DT}" gravity="0 0 -9.81"/>
  <visual><global offwidth="{IMG_W}" offheight="{IMG_H}"/><quality shadowsize="0" offsamples="0"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.2 0.2 0.2" specular="0 0 0"/></visual>
  <worldbody>
    <light pos="0.35 0 1.6" dir="0 0 -1" diffuse="0.45 0.45 0.45" ambient="0.2 0.2 0.2" specular="0 0 0"/>
    <geom name="table" type="plane" size="2 2 0.01" rgba="0.45 0.43 0.4 1"/>
    <camera name="top" pos="{CAM_POS[0]} {CAM_POS[1]} {CAM_POS[2]}" xyaxes="0 -1 0 1 0 0" fovy="{CAM_FOVY}"/>
    {pads}
    <body name="base">
      <geom type="cylinder" fromto="0 0 0 0 0 {ARM_HEIGHT}" size="0.05" rgba="0.3 0.3 0.32 1"
            contype="0" conaffinity="0"/>
      <body name="link1" pos="0 0 {ARM_HEIGHT}">
        <joint name="j1" type="hinge" axis="0 0 1" range="-{JOINT_LIMIT} {JOINT_LIMIT}" damping="3"/>
        <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.03" mass="2" rgba="0.9 0.9 0.92 1"
              contype="0" conaffinity="0"/>
        <body name="link2" pos="{L1} 0 0">
          <joint name="j2" type="hinge" axis="0 0 1" range="-{JOINT_LIMIT} {JOINT_LIMIT}" damping="1.5"/>
          <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.025" mass="1" rgba="0.75 0.75 0.8 1"
                contype="0" conaffinity="0"/>
          <body name="quill" pos="{L2} 0 0">
            <joint name="jz" type="slide" axis="0 0 1" range="{JZ_RANGE[0]} {JZ_RANGE[1]}" damping="20"/>
            <geom type="cylinder" fromto="0 0 0 0 0 -{QUILL_LEN - 0.03}" size="0.012" mass="0.1"
                  rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
            <geom name="cup" type="sphere" pos="0 0 -{QUILL_LEN - 0.015}" size="0.015" mass="0.02"
                  rgba="0.05 0.05 0.05 1"/>
            <site name="tip" pos="0 0 -{QUILL_LEN}"/>
          </body>
        </body>
      </body>
    </body>
    {blocks}
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="400" ctrlrange="-{JOINT_LIMIT} {JOINT_LIMIT}" forcerange="-60 60"/>
    <position name="a2" joint="j2" kp="250" ctrlrange="-{JOINT_LIMIT} {JOINT_LIMIT}" forcerange="-40 40"/>
    <position name="az" joint="jz" kp="3000" ctrlrange="{JZ_RANGE[0]} {JZ_RANGE[1]}" forcerange="-80 80"/>
  </actuator>
</mujoco>"""


def project_to_image(p: np.ndarray) -> np.ndarray:
    """Pinhole projection of a world point (x, y, z) to (row, col) pixel coordinates.

    The camera looks straight down; image right is world -y and image up is world +x.
    """
    f = (IMG_H / 2) / np.tan(np.radians(CAM_FOVY) / 2)
    depth = CAM_POS[2] - p[2]
    col = IMG_W / 2 - f * (p[1] - CAM_POS[1]) / depth - 0.5
    row = IMG_H / 2 - f * (p[0] - CAM_POS[0]) / depth - 0.5
    return np.array([row, col])


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Tip (x, y, z) for joint vector (theta1, theta2, jz)."""
    t1, t2, jz = q[:3]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.array([x, y, TIP_Z0 + jz])


def inverse_kinematics(x: float, y: float, elbow_sign: float) -> tuple[float, float]:
    """Analytic 2-link IK; elbow_sign picks the elbow-up or elbow-down branch."""
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = float(np.clip(c2, -1.0, 1.0))
    t2 = elbow_sign * np.arccos(c2)
    t1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
    t1 = (t1 + np.pi) % (2 * np.pi) - np.pi
    return float(t1), float(t2)


@dataclass
class TaskSpec:
    target_block: str
    target_pad: str
    instruction: str
    blocks: list[str]
    pads: list[str]
    block_xy: dict = field(default_factory=dict)
    pad_xy: dict = field(default_factory=dict)


@dataclass
class StepInfo:
    success: bool
    safety_interventions: int      # 1 if the safety filter corrected this step's command
    wrong_placement: bool          # a block was released outside the target pad
    distractor_disturbed: bool     # a non-target block moved more than 2 cm


class PickPlaceEnv:
    """Gym-style environment: reset(task) -> obs, step(action) -> obs, reward, done, info."""

    def __init__(self, max_steps: int = 60, render: bool = True):
        self.model = mujoco.MjModel.from_xml_string(_mjcf())
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W) if render else None
        m = self.model
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "top")
        self.tip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self.block_qadr = {c: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"free_{c}")]
                           for c in BLOCK_COLORS}
        self.block_vadr = {c: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"free_{c}")]
                           for c in BLOCK_COLORS}
        self.pad_geom = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"pad_{c}") for c in PAD_COLORS}
        self.task: TaskSpec | None = None

    # ------------------------------------------------------------------ state
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    def tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site].copy()

    def block_pos(self, color: str) -> np.ndarray:
        a = self.block_qadr[color]
        return self.data.qpos[a:a + 3].copy()

    def pad_pos(self, color: str) -> np.ndarray:
        return self.model.geom_pos[self.pad_geom[color]][:2].copy()

    # ------------------------------------------------------------------ reset
    @staticmethod
    def sample_task(rng: np.random.Generator, pairs, templates, n_blocks=(2, 3)) -> TaskSpec:
        tb, tp = pairs[rng.integers(len(pairs))]
        k = int(rng.integers(n_blocks[0], n_blocks[1] + 1))
        others = [c for c in BLOCK_COLORS if c != tb]
        blocks = [tb] + list(rng.choice(others, size=k - 1, replace=False))
        pads = [tp, str(rng.choice([c for c in PAD_COLORS if c != tp]))]
        template = templates[rng.integers(len(templates))]
        spec = TaskSpec(tb, tp, template.format(b=tb, p=tp), blocks, pads)
        placed: list[np.ndarray] = []
        for name in [f"pad_{p}" for p in pads] + [f"block_{b}" for b in blocks]:
            for _ in range(1000):
                r, a = rng.uniform(0.32, 0.60), rng.uniform(-0.95, 0.95)
                xy = np.array([r * np.cos(a), r * np.sin(a)])
                if all(np.linalg.norm(xy - o) > 0.16 for o in placed):
                    break
            placed.append(xy)
            kind, color = name.split("_")
            (spec.pad_xy if kind == "pad" else spec.block_xy)[color] = xy
        return spec

    def reset(self, task: TaskSpec, rng: np.random.Generator) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self.task = task
        for i, c in enumerate(PAD_COLORS):
            xy = task.pad_xy.get(c, PARK + [i * 0.3, -6.0])
            self.model.geom_pos[self.pad_geom[c]][:2] = xy
        for i, c in enumerate(BLOCK_COLORS):
            a = self.block_qadr[c]
            xy = task.block_xy.get(c, PARK + [i * 0.3, 0.0])
            yaw = rng.uniform(-np.pi, np.pi) if c in task.block_xy else 0.0
            self.data.qpos[a:a + 3] = [xy[0], xy[1], BLOCK_HALF]
            self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        q0 = np.array([rng.uniform(-0.6, 0.6), rng.choice([-1, 1]) * rng.uniform(0.8, 1.6), 0.0])
        self.data.qpos[:3] = q0
        self.data.ctrl[:] = q0
        self.target_q = q0.copy()
        self.suction = False
        self.held: str | None = None
        self.held_offset = np.zeros(3)
        self.t = 0
        self.initial_blocks = {c: self.block_pos(c) for c in task.blocks}
        self.wrong_placement = False
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):                      # let blocks settle
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    # ------------------------------------------------------------------ observe
    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render().copy()

    def proprio(self) -> np.ndarray:
        q = self.arm_q()
        return np.array([q[0] / JOINT_LIMIT, q[1] / JOINT_LIMIT, q[2] / abs(JZ_RANGE[0]),
                         1.0 if self.suction else -1.0, 1.0 if self.held else -1.0], dtype=np.float32)

    def observe(self) -> dict:
        return {
            "image": self.render() if self.renderer is not None else None,
            "tokens": np.array(tokenize(self.task.instruction), dtype=np.int64),
            "proprio": self.proprio(),
        }

    # ------------------------------------------------------------------ step
    def safety_filter(self, action: np.ndarray) -> tuple[np.ndarray, int]:
        """Clip to the per-step speed limits, the joint limits and the quill's travel.

        Returns the safe action and 1 if the filter had to intervene on this step (it changed
        the command by more than INTERVENTION_TOL of the per-step limit), else 0.
        """
        raw = np.asarray(action, dtype=np.float64).copy()
        a = raw.copy()
        a[:3] = np.clip(a[:3], -ACTION_LIMITS[:3], ACTION_LIMITS[:3])
        lo = np.array([-JOINT_LIMIT, -JOINT_LIMIT, JZ_RANGE[0]])
        hi = np.array([JOINT_LIMIT, JOINT_LIMIT, JZ_RANGE[1]])
        a[:3] = np.clip(self.target_q + a[:3], lo, hi) - self.target_q
        corrected = np.abs(a[:3] - raw[:3]) > INTERVENTION_TOL * ACTION_LIMITS[:3]
        return a, int(corrected.any())

    def _attach_held(self) -> None:
        a, v = self.block_qadr[self.held], self.block_vadr[self.held]
        self.data.qpos[a:a + 3] = self.tip_pos() + self.held_offset
        self.data.qvel[v:v + 6] = 0.0

    def _try_grasp(self) -> None:
        tip = self.tip_pos()
        for c in self.task.blocks:
            p = self.block_pos(c)
            top = p[2] + BLOCK_HALF
            if np.linalg.norm(tip[:2] - p[:2]) < GRASP_XY_TOL and abs(tip[2] - top) < GRASP_Z_TOL:
                self.held = c
                self.held_offset = p - tip
                return

    def step(self, action: np.ndarray):
        a, interventions = self.safety_filter(action)
        self.target_q = self.target_q + a[:3]
        self.data.ctrl[:] = self.target_q
        want_suction = bool(action[3] > 0.0)
        if self.suction and not want_suction and self.held is not None:
            released = self.held
            self.held = None
            pad = self.pad_pos(self.task.target_pad)
            if released != self.task.target_block or \
                    np.linalg.norm(self.block_pos(released)[:2] - pad) > PAD_RADIUS:
                self.wrong_placement = True
        self.suction = want_suction
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            if self.suction and self.held is None:
                self._try_grasp()
            if self.held is not None:
                self._attach_held()
        mujoco.mj_forward(self.model, self.data)
        self.t += 1
        success = self.is_success()
        info = StepInfo(success, interventions, self.wrong_placement, self.distractor_disturbed())
        done = success or self.t >= self.max_steps
        return self.observe(), float(success), done, info

    def is_success(self) -> bool:
        if self.held is not None:
            return False
        p = self.block_pos(self.task.target_block)
        on_pad = np.linalg.norm(p[:2] - self.pad_pos(self.task.target_pad)) < PAD_RADIUS
        resting = p[2] < BLOCK_HALF + 0.01
        return bool(on_pad and resting)

    def distractor_disturbed(self) -> bool:
        for c in self.task.blocks:
            if c == self.task.target_block:
                continue
            if np.linalg.norm(self.block_pos(c)[:2] - self.initial_blocks[c][:2]) > 0.02:
                return True
        return False
