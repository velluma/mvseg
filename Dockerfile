# CUDA 12.1 runtime with cuDNN — matches torch cu121 wheels.
# For a different CUDA version, change the base image and the torch index in README.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

# System deps: python 3.11 + build/runtime libs for medical image IO
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# uv (fast, reproducible installs from the committed lock file)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Install deps first (better layer caching). Copy only lock + project metadata.
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
RUN uv sync --frozen --extra dev || uv sync --extra dev

# Copy the rest of the project
COPY . .

# nnU-Net environment variables (override at `docker run` as needed)
ENV nnUNet_raw=/workspace/nnUNet_raw \
    nnUNet_preprocessed=/workspace/nnUNet_preprocessed \
    nnUNet_results=/workspace/nnUNet_results \
    PYTHONPATH=/workspace/src

CMD ["uv", "run", "python", "-c", "import monai, lightning, wandb; print('mvseg image ready')"]
