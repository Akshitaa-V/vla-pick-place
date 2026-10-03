"""Language-conditioned pick-and-place: VLA policy, imitation learning, RL fine-tuning."""
import os

# Load PyTorch's compiler stack before MuJoCo's OSMesa renderer: with OSMesa
# loaded first, importing torch._dynamo segfaults (both bundle LLVM).
import torch._dynamo  # noqa: F401,E402

os.environ.setdefault("MUJOCO_GL", "osmesa")
