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
# Local ``docker build`` with no ``./dist/`` directory: the COPY of
# the empty/absent directory would fail, so the workflow always
# creates the directory first. Dev builds without the workflow can
# just ``mkdir -p dist`` in the build context or use the PyPI fallback.
COPY dist/ /tmp/cogflow-dist/
# POSIX-sh portable glob check — ``compgen`` is bash-only and
# ``python:3.10-slim``'s ``/bin/sh`` is dash. ``ls ... 2>/dev/null``
# returns the matching filenames on success or an empty string on
# no-match; ``-n`` tests for non-empty.
RUN if [ -n "$(ls /tmp/cogflow-dist/cogflow-*.whl 2>/dev/null)" ]; then \
        echo "Installing cogflow from local wheel(s)"; \
        ls /tmp/cogflow-dist/cogflow-*.whl; \
        pip3 install --ignore-installed /tmp/cogflow-dist/cogflow-*.whl; \
    else \
        echo "No local wheel found; installing cogflow==${COGFLOW_VERSION} from PyPI"; \
        pip3 install --ignore-installed cogflow==${COGFLOW_VERSION}; \
    fi \
    && rm -rf /tmp/cogflow-dist

# Install additional Python packages (shap, xgboost).
# shap pinned to 0.44.* which publishes cp310 manylinux wheels; newer shap
# (0.47+) has a broken setup.py for the slim base and falls back to source
# builds that need g++.
RUN pip3 install "shap==0.44.*" xgboost

# Install compatible protobuf version first
RUN pip install --no-cache-dir protobuf==3.20.1

# Verify the installation of cogflow
RUN python3 -c 'import cogflow'
