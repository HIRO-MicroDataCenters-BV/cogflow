# syntax=docker/dockerfile:1.4
# ``# syntax=docker/dockerfile:1.4`` pins the BuildKit Dockerfile
# frontend to the 1.4 minor line (bare ``1`` would float to whatever
# is latest at build time). 1.4 is sufficient for
# ``RUN --mount=type=bind`` used below. docker/build-push-action@v6
# already uses BuildKit.

# Use the official Python image as the base image
FROM python:3.10-slim

# Triton JIT-compiles a small C extension for its CUDA utils on the first
# kernel launch (``triton/runtime/build.py::_build``). The slim base ships
# no compiler, so any GPU component that reaches a triton kernel dies with:
#
#   RuntimeError: Failed to find C compiler. Please specify via CC
#   environment variable or set triton.knobs.build.impl.
#
# No CUDA toolkit / ``nvcc`` is needed — triton bundles its own ``ptxas``
# and ``cuda.h``.
#
# Deliberately WITHOUT ``--no-install-recommends``. Debian's ``gcc`` only
# *Recommends* ``libc6-dev``, so suppressing recommends installs a compiler
# with no C standard library headers and triton fails one step later with:
#
#   cuda.h:56:10: fatal error: stdlib.h: No such file or directory
#
# Taking the recommends gives a toolchain that is actually usable, and is
# the exact configuration verified end to end on an L40: install gcc,
# pip install torch (2.13.0+cu130 / triton 3.7.1), compile and launch a
# kernel — passes.
#
# Kept as a separate layer before the pip steps so it stays cached across
# cogflow version bumps.
RUN apt-get update \
    && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG COGFLOW_VERSION=1.11.15

# Prefer a locally-available wheel when present; fall back to PyPI.
#
# Release workflow (tag-push): the ``build`` job ships a wheel as an
# artifact; ``docker-publish`` downloads it into ``./dist/`` and the
# wheel-install branch runs. That side-steps PyPI entirely, so the
# image can't race the CDN propagation of a freshly-uploaded version.
#
# Release workflow (workflow_dispatch): no artifact exists; ``./dist/``
# is created empty by the workflow and this step falls back to
# ``pip install cogflow==$VERSION`` from PyPI (the version is already
# long-published in this code path, so no race).
#
# Local ``docker build`` with no ``./dist/`` directory: the bind
# mount below fails. The release workflow always ``mkdir -p dist``s
# before building, so both CI paths are covered; local dev can do
# the same or just install from PyPI directly.
#
# Wheel matching:
#
# - Exact ``COGFLOW_VERSION`` match so a stale wheel from a different
#   version in the build context (e.g. a dev who ran ``python -m build``
#   earlier) can't silently install the wrong cogflow.
# - ``find`` is POSIX-sh portable (slim base uses ``dash`` — no
#   ``compgen``).
# - Match-count branching:
#     1   → install that wheel
#     >1  → hard fail (ambiguous build context)
#     0   → PyPI fallback
#
# BuildKit bind-mount (``--mount=type=bind``): mounts ``./dist/`` into
# the RUN read-only for this step only, so wheel bytes NEVER land in
# a committed image layer. Using ``COPY`` would bake them into an
# earlier layer that a later ``rm`` can't reclaim — Docker layers are
# immutable — bloating the image by the wheel size.
RUN --mount=type=bind,source=dist,target=/tmp/cogflow-dist \
    wheel_match="cogflow-${COGFLOW_VERSION}-*.whl"; \
    wheel_paths="$(find /tmp/cogflow-dist -maxdepth 1 -type f -name "${wheel_match}")"; \
    if [ -n "${wheel_paths}" ]; then \
        wheel_count="$(printf '%s\n' "${wheel_paths}" | wc -l)"; \
    else \
        wheel_count=0; \
    fi; \
    if [ "${wheel_count}" -eq 1 ]; then \
        echo "Installing cogflow from local wheel: ${wheel_paths}"; \
        pip3 install --no-cache-dir --ignore-installed "${wheel_paths}"; \
    elif [ "${wheel_count}" -gt 1 ]; then \
        echo "Ambiguous build context: expected exactly one ${wheel_match}, found ${wheel_count}" >&2; \
        printf '%s\n' "${wheel_paths}" >&2; \
        exit 1; \
    else \
        echo "No local wheel matching ${wheel_match}; installing cogflow==${COGFLOW_VERSION} from PyPI"; \
        pip3 install --no-cache-dir --ignore-installed cogflow==${COGFLOW_VERSION}; \
    fi

# Install additional Python packages (shap, xgboost).
# shap pinned to 0.44.* which publishes cp310 manylinux wheels; newer shap
# (0.47+) has a broken setup.py for the slim base and falls back to source
# builds that need g++.
RUN pip3 install --no-cache-dir "shap==0.44.*" xgboost

# Install compatible protobuf version first
RUN pip install --no-cache-dir protobuf==3.20.1

# Verify the installation of cogflow
RUN python3 -c 'import cogflow'
