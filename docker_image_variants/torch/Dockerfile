# PyTorch training-job variant: ``hiroregistry/cogflow:<version>-torch``.
#
# Layered on the main cogflow image of the SAME version (pushed earlier in
# the same ``docker-publish`` job), so cogflow, gcc (triton JIT), shap,
# xgboost and the protobuf pin are inherited and the two images can never
# drift apart. Only torch is added.
#
# Consumed by the platform's Kubeflow ``PyTorchJob`` trainers
# (``python -m cogflow.trainers.pytorch.<name>``, submitted by CogAPI).
#
# torch comes from PyPI's default index: CUDA-enabled wheels that also run
# on CPU, so one image serves both ``gpus=0`` and ``gpus>0`` jobs. That is
# the reason this variant is amd64-only on every channel (the CUDA wheels
# have no arm64 build) — see docker_image_variants/readme.md →
# Architecture policy.
ARG COGFLOW_VERSION
FROM hiroregistry/cogflow:${COGFLOW_VERSION}

RUN pip3 install --no-cache-dir torch

# Verify the trainer entrypoint imports (cogflow + torch together).
RUN python3 -c 'import cogflow, torch; import cogflow.trainers.pytorch.tabular'
