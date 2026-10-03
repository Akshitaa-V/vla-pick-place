import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# The optional C++ binding is built into cpp/build (cmake -DVLA_BUILD_PYTHON=ON).
sys.path.insert(0, os.path.join(ROOT, "cpp", "build"))

import vla  # noqa: E402,F401  (sets the import order and MUJOCO_GL before tests import mujoco)
