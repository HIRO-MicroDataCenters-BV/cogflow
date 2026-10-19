# syntax=docker/dockerfile:1.4
# Framework runtime base for the PyTorch training-job image:
# ``hiroregistry/cogflow-runtime:torch-<TORCH_VERSION>``.
#
# Holds everything that is HEAVY and RARELY changes (Python, gcc for the
# triton JIT, the CUDA-enabled torch wheels — several GB). It contains no
# cogflow. The per-release variant ``docker_image_variants/torch/Dockerfile``
# builds FROM this image and adds only the cogflow wheel plus the light
# extras, so a cogflow release re-pushes a few MB and the nodes pull only
# that delta instead of the whole framework layer every time.
#
# Rebuilt only when this file (or TORCH_VERSION) changes — see
# .github/workflows/runtime-images.yml.
FROM python:3.10-slim

# gcc (with libc6-dev via recommends) for triton's JIT — same reasoning as
# docker_image_variants/latest/Dockerfile.
RUN apt-get update \
    && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

# PyPI's default torch wheels are CUDA-enabled and run on CPU too, so one
# image serves ``gpus=0`` and ``gpus>0`` jobs. Pinned: the tag names it.
ARG TORCH_VERSION
RUN test -n "${TORCH_VERSION}" || { echo "TORCH_VERSION build-arg is required" >&2; exit 1; } \
    && pip3 install --no-cache-dir "torch==${TORCH_VERSION}"

RUN python3 -c 'import torch; print("torch", torch.__version__)'
