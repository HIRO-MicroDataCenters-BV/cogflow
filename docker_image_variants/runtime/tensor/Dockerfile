# syntax=docker/dockerfile:1.4
# Framework runtime base for the TensorFlow training-job image:
# ``hiroregistry/cogflow-runtime:tensor-<TF_VERSION>``.
#
# Holds everything that is HEAVY and RARELY changes (Python, gcc,
# ``tensorflow[and-cuda]`` with its bundled CUDA libraries — several GB). It
# contains no cogflow. The per-release variant
# ``docker_image_variants/tensor/Dockerfile`` builds FROM this image and adds
# only the cogflow wheel plus the light extras, so a cogflow release
# re-pushes a few MB and the nodes pull only that delta.
#
# TensorFlow stays on the 2.17 line: the last one that accepts the
# ``protobuf 3.20.x`` the vendored kfp needs (``>=3.20.3,<5``). The protobuf
# pin itself is applied in the variant, AFTER cogflow is installed, because
# pip may bump it while resolving either package.
#
# Rebuilt only when this file (or TF_VERSION) changes — see
# .github/workflows/runtime-images.yml.
FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG TF_VERSION
RUN test -n "${TF_VERSION}" || { echo "TF_VERSION build-arg is required" >&2; exit 1; } \
    && pip3 install --no-cache-dir "tensorflow[and-cuda]==${TF_VERSION}"

RUN python3 -c 'import tensorflow as tf; print("tensorflow", tf.__version__)'
