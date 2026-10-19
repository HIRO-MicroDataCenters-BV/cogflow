"""
Tabular MLP trainer (TensorFlow / Keras).

Entrypoint for the ``tensorflow-tabular-classifier`` /
``tensorflow-tabular-regressor`` catalog algorithms::

    python -m cogflow.trainers.tensorflow.tabular \\
        --dataset-id <uuid> --output-name my-model --task regression \\
        --target-column price --hyperparams '{"epochs": 30}'

Single-replica only. Keras 3 (TensorFlow >= 2.16) cannot run ``model.fit``
under ``MultiWorkerMirroredStrategy`` (``ValueError: Attempt to convert a
value (PerReplica ...)`` — reproduced on TF 2.17 and 2.21), so a
multi-worker ``TF_CONFIG`` is refused up front with a clear message instead
of failing inside the training loop; the platform catalog marks the
TensorFlow algorithms as non-distributed. Only the chief (``task.type ==
"chief"``, or worker 0 when the cluster has no chief) talks to MLflow / the
catalog.
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


def _dataset(x, y=None, batch_size: int = 64, shuffle: bool = False, seed: int = 0):
    """``tf.data`` pipeline for ``fit``/``predict``.

    Under ``MultiWorkerMirroredStrategy`` Keras needs datasets, not numpy
    arrays (arrays become PerReplica values it cannot convert); sharding is
    pinned to DATA because in-memory slices have no files to shard by.
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(x if y is None else (x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10_000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(max(int(batch_size), 1))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return ds.with_options(options)


def _evaluate(model, x_val, y_val, task: str, n_classes: int | None, batch_size: int = 256) -> dict[str, float]:
    import numpy as np

    pred = model.predict(_dataset(x_val, batch_size=batch_size), verbose=0)
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

    refuse_multi_worker(tf_config())
    model = build_model(
        data.n_features,
        data.n_outputs,
        hidden_sizes,
        float(hp["dropout"]),
        data.task,
        float(hp["learning_rate"]),
    )

    def _cb(epoch, logs):
        metrics = _evaluate(model, data.x_val, data.y_val, data.task, data.n_classes)
        metrics["train_loss"] = float((logs or {}).get("loss", 0.0))
        if on_epoch_end is not None:
            _common.say(f"epoch {epoch + 1}/{int(hp['epochs'])} {_common.format_metrics(metrics)}")
            on_epoch_end(epoch, metrics)

    batch_size = int(hp["batch_size"])
    model.fit(
        _dataset(data.x_train, data.y_train, batch_size=batch_size, shuffle=True, seed=int(hp["seed"])),
        epochs=int(hp["epochs"]),
        validation_data=_dataset(data.x_val, data.y_val, batch_size=batch_size),
        shuffle=False,  # the dataset is already shuffled; silences Keras' warning
        verbose=2,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=_cb)],
    )
    final = _evaluate(model, data.x_val, data.y_val, data.task, data.n_classes)
    return model, final


MULTI_WORKER_UNSUPPORTED = (
    "multi-worker TensorFlow training is not supported: Keras 3 cannot run model.fit under "
    "MultiWorkerMirroredStrategy (PerReplica conversion error). Run this algorithm with workers=1."
)


def cluster_size(cfg: dict[str, Any]) -> int:
    """Number of tasks declared in a ``TF_CONFIG`` cluster (0 when absent)."""
    cluster = cfg.get("cluster") or {}
    return sum(len(v) for v in cluster.values() if isinstance(v, list))


def refuse_multi_worker(cfg: dict[str, Any]) -> None:
    """Fail fast, before any data or model work, on a multi-task ``TF_CONFIG``."""
    if cluster_size(cfg) > 1:
        raise RuntimeError(MULTI_WORKER_UNSUPPORTED)


def main(argv: list[str] | None = None) -> int:
    parser = _common.build_arg_parser("cogflow TensorFlow tabular trainer (Keras MLP)")
    args = parser.parse_args(argv)
    hp = _common.parse_hyperparams(args.hyperparams, DEFAULT_HYPERPARAMS)
    refuse_multi_worker(tf_config())
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
    _common.say(
        f"dataset {args.dataset_id}: {len(data.x_train)} train / {len(data.x_val)} val rows, "
        f"{data.n_features} features, task={data.task}"
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
        _common.log_preprocessing(data.preprocessing, args.work_dir)
        cogflow_models.log_model(
            model,
            artifact_path=args.artifact_path,
            registered_model_name=args.output_name,
            input_example=data.input_example,
            model_type=FRAMEWORK,
            metadata={_common.PREPROCESSING_METADATA_KEY: data.preprocessing},
        )
        _common.say(f"model '{args.output_name}' logged to run {run_id}")

    _common.emit_result(run_id)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in the pod
    sys.exit(main())
