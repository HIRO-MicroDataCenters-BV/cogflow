# syntax=docker/dockerfile:1.4
# TensorFlow training-job variant: ``hiroregistry/cogflow:<version>-tensor``.
#
# Built FROM the framework runtime base (``cogflow-runtime:tensor-<ver>``,
# see runtime/tensor/Dockerfile) and adds only the cogflow wheel of this
# release plus the light extras. The multi-GB TensorFlow/CUDA layers come
# from the base and are shared by every cogflow release, so a release
# re-pushes (and a node re-pulls) only this thin top layer.
#
# Build context is ``docker_image_variants/`` (the release workflow passes
# ``file:`` explicitly) so the wheel bind-mount can read ``latest/dist/``.
#
# Consumed by the platform's Kubeflow ``TFJob`` trainers
# (``python -m cogflow.trainers.tensorflow.<name>``, submitted by CogAPI).
ARG RUNTIME_IMAGE=hiroregistry/cogflow-runtime:tensor-2.17.1
FROM ${RUNTIME_IMAGE}

ARG COGFLOW_VERSION
RUN test -n "${COGFLOW_VERSION}" || { echo "COGFLOW_VERSION build-arg is required" >&2; exit 1; }

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

# protobuf pinned LAST: TensorFlow 2.17 accepts 3.20.3 and the vendored kfp
# needs the 3.20 line; cogflow's own resolution may have bumped it.
RUN pip3 install --no-cache-dir "shap==0.44.*" xgboost \
    && pip3 install --no-cache-dir protobuf==3.20.3

# Verify the trainer entrypoint imports (cogflow + tensorflow together).
RUN python3 -c 'import cogflow, tensorflow as tf; import cogflow.trainers.tensorflow.tabular; print("cogflow", cogflow.__version__, "tensorflow", tf.__version__)'
