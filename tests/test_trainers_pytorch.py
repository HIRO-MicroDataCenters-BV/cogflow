"""End-to-end test of the PyTorch tabular trainer on a synthetic CSV (no cluster, no MLflow)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cogflow.trainers import _common  # noqa: E402
from cogflow.trainers.pytorch import tabular  # noqa: E402


class _FakeRun:
    def __init__(self, run_id="run-pt"):
        self.info = type("I", (), {"run_id": run_id})()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def tracking(monkeypatch):
    import cogflow.core.models as models_mod

    calls = {"params": [], "metrics": [], "models": [], "start": []}

    def start_run(**kwargs):
        calls["start"].append(kwargs)
        return _FakeRun()

    monkeypatch.setattr(models_mod, "start_run", start_run)
    monkeypatch.setattr(models_mod, "log_params", lambda p: calls["params"].append(p))
    monkeypatch.setattr(models_mod, "log_metrics", lambda m, step=None: calls["metrics"].append((step, m)))
    monkeypatch.setattr(models_mod, "log_model", lambda model, **kw: calls["models"].append((model, kw)))
    calls["artifacts"] = []
    monkeypatch.setattr(
        models_mod, "log_artifact", lambda path, artifact_path=None: calls["artifacts"].append((path, artifact_path))
    )
    return calls


def _write_csv(tmp_path, task: str):
    rng = np.random.default_rng(0)
    n = 90
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": rng.choice(["p", "q"], size=n)})
    if task == "classification":
        df["label"] = np.where(x1 + x2 > 0, "pos", "neg")
    else:
        df["label"] = 3 * x1 - 2 * x2 + rng.normal(scale=0.05, size=n)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_main_trains_logs_and_registers(tmp_path, monkeypatch, tracking, capsys, task):
    csv_path = _write_csv(tmp_path, task)
    # Regression targets are left on their raw scale (no inverse transform
    # needed at serving time), so it needs a larger step budget than the
    # separable classification case to fit within the test.
    epochs = 8 if task == "classification" else 40
    hyperparams = f'{{"epochs": {epochs}, "hidden_sizes": "16", "batch_size": 16, "learning_rate": 0.02}}'
    monkeypatch.setattr(_common, "materialise_tabular_dataset", lambda ds, work: csv_path)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    rc = tabular.main(
        [
            "--job-id",
            "job-1",
            "--dataset-id",
            "ds-1",
            "--output-name",
            "my-model",
            "--task",
            task,
            "--target-column",
            "label",
            "--algorithm",
            f"pytorch-tabular-{task}",
            "--hyperparams",
            hyperparams,
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    assert rc == 0

    # Run bracket carries the job id tag and the output name.
    start = tracking["start"][0]
    assert start["run_name"] == "my-model"
    assert start["tags"][_common.TRAINING_JOB_ID_TAG] == "job-1"
    assert start["tags"][_common.ALGORITHM_TAG] == f"pytorch-tabular-{task}"

    params = tracking["params"][0]
    assert params["epochs"] == epochs and params["hidden_sizes"] == "16"
    assert params["dataset_id"] == "ds-1" and params["target_column"] == "label"
    if task == "classification":
        assert params["classes"] == "neg,pos"

    # One metrics call per epoch (stepped) plus the final summary.
    stepped = [m for step, m in tracking["metrics"] if step is not None]
    assert len(stepped) == epochs
    final = tracking["metrics"][-1][1]
    assert "train_seconds" in final and "val_loss" in final
    if task == "classification":
        assert final["val_accuracy"] > 0.8
    else:
        assert final["val_r2"] > 0.8

    # The registered model is an unwrapped nn.Module with the platform type.
    model, kw = tracking["models"][0]
    assert isinstance(model, torch.nn.Module)
    assert kw["registered_model_name"] == "my-model"
    assert kw["model_type"] == "pytorch"
    assert kw["artifact_path"] == "model"
    # Preprocessing state travels with the model (metadata) and as an artifact.
    state = kw["metadata"][_common.PREPROCESSING_METADATA_KEY]
    assert state["feature_names"] and state["target_column"] == "label"
    assert tracking["artifacts"] and tracking["artifacts"][0][1] == _common.PREPROCESSING_ARTIFACT_DIR
    assert isinstance(kw["input_example"], np.ndarray)

    assert capsys.readouterr().out.strip().splitlines()[-1] == "COG_TRAINING_RESULT run_id=run-pt"


def test_non_primary_rank_trains_without_tracking(tmp_path, monkeypatch, tracking):
    csv_path = _write_csv(tmp_path, "regression")
    monkeypatch.setattr(_common, "materialise_tabular_dataset", lambda ds, work: csv_path)
    # Pretend to be rank 1 of a job, but keep the process single (no init_process_group).
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "1")
    rc = tabular.main(
        [
            "--dataset-id",
            "d",
            "--output-name",
            "m",
            "--task",
            "regression",
            "--target-column",
            "label",
            "--hyperparams",
            '{"epochs": 1}',
        ]
    )
    assert rc == 0
    assert tracking["start"] == [] and tracking["models"] == []


def test_build_model_layers():
    model = tabular.build_model(4, 3, [8, 4], dropout=0.0)
    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(4, 8), (8, 4), (4, 3)]
    assert not any(isinstance(m, torch.nn.Dropout) for m in model)
