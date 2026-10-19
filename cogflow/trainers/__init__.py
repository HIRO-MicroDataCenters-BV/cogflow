"""
CogFlow trainers — pod-side entrypoints for platform training jobs.

Each module under ``cogflow.trainers.<framework>`` is a ``python -m``
entrypoint that a Kubeflow ``PyTorchJob`` / ``TFJob`` runs inside a
cogflow image. A trainer:

1. downloads the dataset through cogflow (``cogflow.datasets.download_dataset``),
2. prepares features/targets (:mod:`cogflow.trainers._common`),
3. trains under an MLflow run tagged with the platform's training-job id,
4. registers the model with ``cogflow.models.log_model`` so the CogFlow
   catalog row is created, and
5. prints ``COG_TRAINING_RESULT run_id=<id>`` as its last line.

Framework packages (``torch``, ``tensorflow``) are imported only inside
the trainer that needs them, so importing ``cogflow.trainers`` is cheap.
"""
