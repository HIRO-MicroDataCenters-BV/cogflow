# syntax=docker/dockerfile:1.4
# PyTorch training-job variant: ``hiroregistry/cogflow:<version>-torch``.
#
# Built FROM the framework runtime base (``cogflow-runtime:torch-<ver>``,
# see runtime/torch/Dockerfile) and adds only the cogflow wheel of this
# release plus the light extras. The multi-GB torch/CUDA layers come from
# the base and are shared by every cogflow release, so a release re-pushes
# (and a node re-pulls) only this thin top layer.
#
# Build context is ``docker_image_variants/`` (the release workflow passes
# ``file:`` explicitly) so the wheel bind-mount can read ``latest/dist/``,
# the same directory the main image's build uses.
#
# Consumed by the platform's Kubeflow ``PyTorchJob`` trainers
# (``python -m cogflow.trainers.pytorch.<name>``, submitted by CogAPI).
ARG RUNTIME_IMAGE=hiroregistry/cogflow-runtime:torch-2.14.1
FROM ${RUNTIME_IMAGE}

ARG COGFLOW_VERSION
RUN test -n "${COGFLOW_VERSION}" || { echo "COGFLOW_VERSION build-arg is required" >&2; exit 1; }

# Wheel-or-PyPI install, identical to latest/Dockerfile: the release
# workflow drops the freshly built wheel into latest/dist/ (tag push) or
# leaves it empty (workflow_dispatch rebuild → PyPI).
RUN --mount=type=bind,source=latest/dist,target=/tmp/cogflow-dist \
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

# Light extras the main image also carries, then the protobuf pin the
# vendored kfp needs (applied last: pip may bump it while resolving above).
RUN pip3 install --no-cache-dir "shap==0.44.*" xgboost \
    && pip3 install --no-cache-dir protobuf==3.20.1

# Verify the trainer entrypoint imports (cogflow + torch together).
RUN python3 -c 'import cogflow, torch; import cogflow.trainers.pytorch.tabular; print("cogflow", cogflow.__version__, "torch", torch.__version__)'
