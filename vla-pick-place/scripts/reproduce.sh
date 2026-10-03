#!/usr/bin/env bash
# Full pipeline: demonstrations -> imitation learning -> RL fine-tuning -> export -> benchmarks.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m vla.data --episodes 1600 --seed 1 --out data/demos.npz
torchrun --nproc_per_node 2 -m vla.train_bc --config configs/bc.yaml
python -m vla.evaluate --checkpoint checkpoints/bc_policy.pt --episodes 100 --out results/eval_bc.json
python -m vla.rl --config configs/rl.yaml
python -m vla.evaluate --checkpoint checkpoints/rl_policy.pt --episodes 100 --out results/eval_rl.json

python -m vla.export --checkpoint checkpoints/rl_policy.pt --out checkpoints/policy.bin
cmake -S cpp -B cpp/build -DVLA_BUILD_PYTHON=ON -DPython_EXECUTABLE="$(which python)"
cmake --build cpp/build -j"$(nproc)"
ctest --test-dir cpp/build --output-on-failure
./cpp/build/vla_bench checkpoints/policy.bin cpp/tests/fixtures.bin 500 | tee results/cpp_latency.json
PYTHONPATH=cpp/build python -m vla.evaluate --backend cpp --weights checkpoints/policy.bin \
  --episodes 100 --out results/eval_rl_cpp.json
