# syntax=docker/dockerfile:1
# FrameGenX API + UI container.
#
# Environment contract (mirrors the tested local env, see requirements.txt):
#   Python 3.14 + torch 2.13.0+cu130 wheels (self-contained CUDA runtime,
#   no nvidia/cuda base image needed — libcuda comes from the host driver
#   via NVIDIA Container Toolkit / WSL2 GPU passthrough).
#
# GPU: run with `docker compose up` (compose reserves the GPU) or:
#   docker run --gpus all ...

# ---------------------------------------------------------------------------
# Stage 1 — build the React/Vite glass-box UI
# ---------------------------------------------------------------------------
FROM node:22-alpine AS ui-build
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — Python runtime
# ---------------------------------------------------------------------------
FROM python:3.14-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/root/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OpenCV runtime deps: libGL + glib. (No compilers needed: every pinned dep
# ships prebuilt cp314 manylinux wheels.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first for maximal layer-cache reuse across code edits.
COPY requirements.txt .
# requirements.txt pins torch==2.13.0+cu130 and carries the cu130
# extra-index-url itself — do NOT add another torch install line here or the
# two installs will fight over different CUDA builds.
# The BuildKit cache mount keeps downloaded wheels across rebuilds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

# Serve the UI from the API process (src/api.py mounts frontend/dist).
COPY --from=ui-build /ui/dist /app/frontend/dist

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
