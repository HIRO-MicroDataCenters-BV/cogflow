"""
Tabular MLP trainer (TensorFlow / Keras).

Entrypoint for the ``tensorflow-tabular-classifier`` /
``tensorflow-tabular-regressor`` catalog algorithms::

    python -m cogflow.trainers.tensorflow.tabular \\
        --dataset-id <uuid> --output-name my-model --task regression \\
        --target-column price --hyperparams '{"epochs": 30}'

Inside a multi-replica ``TFJob`` the Training Operator sets ``TF_CONFIG``;
the trainer then builds the model under ``MultiWorkerMirroredStrategy``.
Only the chief (``task.type == "chief"``, or worker 0 when the cluster has
no chief) talks to MLflow / the catalog.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from ...utils.logging import get_logger
from .. import _common

logger = get_logger(__name__)

FRAMEWORK = "tensorflow"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "hidden_sizes": "128,64",
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "epochs": 20,
    "batch_size": 64,
    "test_split": 0.2,
    "seed": 42,
}


def tf_config() -> dict[str, Any]:
    raw = os.environ.get("TF_CONFIG", "")
    if not raw:
        return {}
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("TF_CONFIG is not valid JSON; running single-process")
        return {}
    return cfg if isinstance(cfg, dict) else {}


def is_chief(cfg: dict[str, Any] | None = None) -> bool:
    """True for the replica that owns MLflow logging and registration."""
    cfg = tf_config() if cfg is None else cfg
    task = cfg.get("task") or {}
    if not task:
        return True
    task_type = str(task.get("type", "")).lower()
    index = int(task.get("index", 0) or 0)
    if task_type == "chief":
        return True
    cluster = cfg.get("cluster") or {}
    return task_type == "worker" and index == 0 and not cluster.get("chief")


def build_model(
    n_features: int, n_outputs: int, hidden_sizes: list[int], dropout: float, task: str, learning_rate: float
):
    """Keras MLP compiled for the task (softmax + sparse CE, or linear + MSE)."""
    import tensorflow as tf

    keras = tf.keras
    layers: list[Any] = [keras.layers.Input(shape=(n_features,))]
    for hidden in hidden_sizes:
        layers.append(keras.layers.Dense(hidden, activation="relu"))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    if task == "classification":
        layers.append(keras.layers.Dense(n_outputs, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    else:
        layers.append(keras.layers.Dense(1))
        loss = "mse"
        metrics = ["mae"]
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)
    return model


def _evaluate(model, x_val, y_val, task: str, n_classes: int | None) -> dict[str, float]:
    import numpy as np

    pred = model.predict(x_val, verbose=0)
    if task == "classification":
        labels = np.argmax(pred, axis=1)
        metrics = _common.classification_metrics(y_val, labels, int(n_classes or 0))
        eps = 1e-7
        picked = np.clip(pred[np.arange(len(labels)), np.asarray(y_val).astype("int64")], eps, 1.0)
        metrics["val_loss"] = float(-np.log(picked).mean()) if len(labels) else 0.0
    else:
        metrics = _common.regression_metrics(y_val, pred.ravel())
        metrics["val_loss"] = metrics["val_rmse"] ** 2
    return metrics


def train(data: _common.PreparedData, hyperparams: dict[str, Any], *, on_epoch_end=None):
    """Train the Keras MLP and return ``(model, final_metrics)``."""
    import tensorflow as tf

    hp = hyperparams
    tf.keras.utils.set_random_seed(int(hp["seed"]))
    hidden_sizes = _common.parse_int_list(hp["hidden_sizes"])

    strategy = tf.distribute.MultiWorkerMirroredStrategy() if tf_config().get("cluster") else None
    scope = strategy.scope() if strategy is not None else _NullScope()
    with scope:
        model = build_model(
            data.n_features,
            data.n_outputs,
            hidden_sizes,
            float(hp["dropout"]),
            data.task,
            float(hp["learning_rate"]),
        )

    callbacks = []
    if on_epoch_end is not None:

        def _cb(epoch, logs):
            metrics = _evaluate(model, data.x_val, data.y_val, data.task, data.n_classes)
            metrics["train_loss"] = float((logs or {}).get("loss", 0.0))
            on_epoch_end(epoch, metrics)

        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=_cb))

    model.fit(
        data.x_train,
        data.y_train,
        epochs=int(hp["epochs"]),
        batch_size=int(hp["batch_size"]),
        validation_data=(data.x_val, data.y_val),
        verbose=2,
        callbacks=callbacks,
    )
    final = _evaluate(model, data.x_val, data.y_val, data.task, data.n_classes)
    return model, final


class _NullScope:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = _common.build_arg_parser("cogflow TensorFlow tabular trainer (Keras MLP)")
    args = parser.parse_args(argv)
    hp = _common.parse_hyperparams(args.hyperparams, DEFAULT_HYPERPARAMS)
    primary = is_chief()
    started = time.time()

    frame_path = _common.materialise_tabular_dataset(args.dataset_id, args.work_dir)
    data = _common.prepare_frame(
        _common.load_frame(frame_path),
        args.target_column,
        args.task,
        test_split=float(hp["test_split"]),
        seed=int(hp["seed"]),
    )
    logger.info(
        "Prepared dataset %s: %d train / %d val rows, %d features, task=%s",
        args.dataset_id,
        len(data.x_train),
        len(data.x_val),
        data.n_features,
        data.task,
    )

    if not primary:
        train(data, hp)
        return 0

    from ...core import models as cogflow_models

    with _common.tracked_run(
        job_id=args.job_id,
        output_name=args.output_name,
        framework=FRAMEWORK,
        algorithm=args.algorithm,
        task=args.task,
    ) as active_run:
        run_id = active_run.info.run_id
        params: dict[str, Any] = {
            **hp,
            "dataset_id": args.dataset_id,
            "target_column": args.target_column,
            "task": args.task,
            "n_features": data.n_features,
            "n_train": len(data.x_train),
            "n_val": len(data.x_val),
        }
        if data.class_names:
            params["classes"] = ",".join(data.class_names)[:500]
        cogflow_models.log_params(params)

        def _log(epoch: int, metrics: dict[str, float]) -> None:
            cogflow_models.log_metrics(metrics, step=epoch)

        model, final = train(data, hp, on_epoch_end=_log)
        final["train_seconds"] = time.time() - started
        cogflow_models.log_metrics(final)
        cogflow_models.log_model(
            model,
            artifact_path=args.artifact_path,
            registered_model_name=args.output_name,
            input_example=data.input_example,
            model_type=FRAMEWORK,
        )
        logger.info("Model '%s' logged to run %s", args.output_name, run_id)

    _common.emit_result(run_id)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in the pod
    sys.exit(main())
