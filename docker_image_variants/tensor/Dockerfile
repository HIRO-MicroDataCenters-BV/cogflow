# TensorFlow training-job variant: ``hiroregistry/cogflow:<version>-tensor``.
#
# Layered on the main cogflow image of the SAME version (pushed earlier in
# the same ``docker-publish`` job), so cogflow, gcc, shap, xgboost are
# inherited and the two images can never drift apart. Only TensorFlow is
# added.
#
# Consumed by the platform's Kubeflow ``TFJob`` trainers
# (``python -m cogflow.trainers.tensorflow.<name>``, submitted by CogAPI).
#
# ``tensorflow[and-cuda]`` bundles the CUDA runtime libraries so the same
# image serves ``gpus=0`` and ``gpus>0`` jobs; it is Linux/amd64-only, hence
# the amd64-only build policy for this variant (readme.md → Architecture
# policy). Pinned to the 2.17 line: it is the last line that accepts the
# ``protobuf 3.20.x`` the vendored kfp needs (``>=3.20.3,<5``); 2.18+
# requires protobuf >= 5 and would break pipeline compilation in the image.
# protobuf is re-pinned AFTER the TensorFlow install because pip may bump
# it while resolving TensorFlow's dependencies.
ARG COGFLOW_VERSION
FROM hiroregistry/cogflow:${COGFLOW_VERSION}

RUN pip3 install --no-cache-dir "tensorflow[and-cuda]==2.17.*" \
    && pip3 install --no-cache-dir protobuf==3.20.3

# Verify the trainer entrypoint imports (cogflow + tensorflow together).
RUN python3 -c 'import cogflow, tensorflow; import cogflow.trainers.tensorflow.tabular'
