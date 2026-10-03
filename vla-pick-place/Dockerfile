FROM python:3.11-slim

ENV MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends libosmesa6 build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY vla ./vla
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[dev]"

COPY . .
RUN cmake -S cpp -B cpp/build -DVLA_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python) \
 && cmake --build cpp/build -j"$(nproc)"
ENV PYTHONPATH=/app/cpp/build

# Default: benchmark the released policy with the C++ runtime in the loop.
CMD ["python", "-m", "vla.evaluate", "--backend", "cpp", "--weights", "checkpoints/policy.bin", "--episodes", "50"]
