"""
Tabular MLP trainer (PyTorch).

Entrypoint for the ``pytorch-tabular-classifier`` / ``pytorch-tabular-
regressor`` catalog algorithms::

    python -m cogflow.trainers.pytorch.tabular \\
        --dataset-id <uuid> --output-name my-model --task classification \\
        --target-column species --hyperparams '{"epochs": 30}'

Runs single-process by default. Inside a multi-replica ``PyTorchJob`` the
Training Operator sets ``WORLD_SIZE``/``RANK``/``MASTER_ADDR``/
``MASTER_PORT``; the trainer then initialises ``torch.distributed`` and
wraps the model in DDP. Only rank 0 talks to MLflow / the catalog.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from ...utils.logging import get_logger
from .. import _common

logger = get_logger(__name__)

FRAMEWORK = "pytorch"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "hidden_sizes": "128,64",
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "epochs": 20,
    "batch_size": 64,
    "test_split": 0.2,
    "seed": 42,
}


def build_model(n_features: int, n_outputs: int, hidden_sizes: list[int], dropout: float):
    """Plain MLP: ``Linear → ReLU → Dropout`` per hidden layer, linear head."""
    from torch import nn

    layers: list[Any] = []
    width = n_features
    for hidden in hidden_sizes:
        layers.append(nn.Linear(width, hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = hidden
    layers.append(nn.Linear(width, n_outputs))
    return nn.Sequential(*layers)


def _distributed_context() -> tuple[int, int]:
    """Initialise ``torch.distributed`` when the operator scheduled >1 replica."""
    import torch

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        logger.info("Initialising torch.distributed backend=%s rank=%d world_size=%d", backend, rank, world_size)
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return world_size, rank


def _evaluate(
    model, x_val, y_val, task: str, n_classes: int | None, device, batch_size: int = 1024
) -> dict[str, float]:
    """Validation metrics, computed in batches so a large split cannot exhaust device memory."""
    import numpy as np
    import torch

    model.eval()
    outputs = []
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(x_val), max(int(batch_size), 1)):
            xb = torch.as_tensor(x_val[start : start + batch_size], device=device)
            logits = model(xb)
            if task == "classification":
                yb = torch.as_tensor(y_val[start : start + batch_size], device=device)
                loss_sum += torch.nn.functional.cross_entropy(logits, yb, reduction="sum").item()
                outputs.append(logits.argmax(dim=1).cpu().numpy())
            else:
                yb = torch.as_tensor(y_val[start : start + batch_size], device=device).view(-1, 1)
                loss_sum += torch.nn.functional.mse_loss(logits, yb, reduction="sum").item()
                outputs.append(logits.view(-1).cpu().numpy())
    pred = np.concatenate(outputs) if outputs else np.zeros(0)
    if task == "classification":
        metrics = _common.classification_metrics(y_val, pred, int(n_classes or 0))
    else:
        metrics = _common.regression_metrics(y_val, pred)
    metrics["val_loss"] = float(loss_sum / max(len(x_val), 1))
    return metrics


def train(
    data: _common.PreparedData,
    hyperparams: dict[str, Any],
    *,
    world_size: int = 1,
    rank: int = 0,
    on_epoch_end=None,
):
    """Train the MLP and return ``(model, final_metrics)``; the model is unwrapped and on CPU."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    hp = hyperparams
    torch.manual_seed(int(hp["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # One process per replica → one GPU per replica. Extra GPUs on the
        # pod stay idle; the platform catalog caps this trainer at one.
        logger.warning(
            "%d GPUs visible but this trainer uses one GPU per replica (cuda:0); request gpus=1.",
            torch.cuda.device_count(),
        )

    hidden_sizes = _common.parse_int_list(hp["hidden_sizes"])
    model = build_model(data.n_features, data.n_outputs, hidden_sizes, float(hp["dropout"])).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)

    x_train = torch.as_tensor(data.x_train)
    if data.task == "classification":
        y_train = torch.as_tensor(data.y_train, dtype=torch.long)
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        y_train = torch.as_tensor(data.y_train, dtype=torch.float32).view(-1, 1)
        loss_fn = torch.nn.MSELoss()

    dataset = TensorDataset(x_train, y_train)
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=int(hp["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(hp["learning_rate"]), weight_decay=float(hp["weight_decay"])
    )

    final: dict[str, float] = {}
    for epoch in range(int(hp["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        metrics = _evaluate(model, data.x_val, data.y_val, data.task, data.n_classes, device, int(hp["batch_size"]))
        metrics["train_loss"] = running / max(seen, 1)
        final = metrics
        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics)
        if rank == 0:
            _common.say(f"epoch {epoch + 1}/{int(hp['epochs'])} {_common.format_metrics(metrics)}")

    unwrapped = model.module if hasattr(model, "module") else model
    return unwrapped.cpu().eval(), final


def main(argv: list[str] | None = None) -> int:
    parser = _common.build_arg_parser("cogflow PyTorch tabular trainer (MLP)")
    args = parser.parse_args(argv)
    hp = _common.parse_hyperparams(args.hyperparams, DEFAULT_HYPERPARAMS)

    world_size, rank = _distributed_context()
    is_primary = rank == 0
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

    if not is_primary:
        train(data, hp, world_size=world_size, rank=rank)
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
            "world_size": world_size,
        }
        if data.class_names:
            params["classes"] = ",".join(data.class_names)[:500]
        cogflow_models.log_params(params)

        def _log(epoch: int, metrics: dict[str, float]) -> None:
            cogflow_models.log_metrics(metrics, step=epoch)

        model, final = train(data, hp, world_size=world_size, rank=rank, on_epoch_end=_log)
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
