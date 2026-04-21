# syntax=docker/dockerfile:1
# ``# syntax=docker/dockerfile:1`` enables the BuildKit Dockerfile
# frontend (1.2+), which is what supports ``RUN --mount=type=bind``
# below. docker/build-push-action@v6 already uses BuildKit; this
# directive just pins the Dockerfile syntax version explicitly.

# Use the official Python image as the base image
FROM python:3.10-slim

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
    wheel_count="$(find /tmp/cogflow-dist -maxdepth 1 -type f -name "${wheel_match}" | wc -l)"; \
    if [ "${wheel_count}" -eq 1 ]; then \
        wheel_path="$(find /tmp/cogflow-dist -maxdepth 1 -type f -name "${wheel_match}")"; \
        echo "Installing cogflow from local wheel: ${wheel_path}"; \
        pip3 install --ignore-installed "${wheel_path}"; \
    elif [ "${wheel_count}" -gt 1 ]; then \
        echo "Ambiguous build context: expected exactly one ${wheel_match}, found ${wheel_count}" >&2; \
        find /tmp/cogflow-dist -maxdepth 1 -type f -name "${wheel_match}" >&2; \
        exit 1; \
    else \
        echo "No local wheel matching ${wheel_match}; installing cogflow==${COGFLOW_VERSION} from PyPI"; \
        pip3 install --ignore-installed cogflow==${COGFLOW_VERSION}; \
    fi

# Install additional Python packages (shap, xgboost).
# shap pinned to 0.44.* which publishes cp310 manylinux wheels; newer shap
# (0.47+) has a broken setup.py for the slim base and falls back to source
# builds that need g++.
RUN pip3 install "shap==0.44.*" xgboost

# Install compatible protobuf version first
RUN pip install --no-cache-dir protobuf==3.20.1

# Verify the installation of cogflow
RUN python3 -c 'import cogflow'
