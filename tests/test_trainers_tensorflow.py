"""End-to-end test of the TensorFlow tabular trainer on a synthetic CSV (skipped without TensorFlow)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from cogflow.trainers import _common  # noqa: E402
from cogflow.trainers.tensorflow import tabular  # noqa: E402


class _FakeRun:
    def __init__(self, run_id="run-tf"):
        self.info = type("I", (), {"run_id": run_id})()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def tracking(monkeypatch):
    import cogflow.core.models as models_mod

    calls = {"params": [], "metrics": [], "models": [], "start": []}
    monkeypatch.setattr(models_mod, "start_run", lambda **kw: (calls["start"].append(kw), _FakeRun())[1])
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
    df = pd.DataFrame({"x1": x1, "x2": x2})
    df["label"] = np.where(x1 + x2 > 0, "pos", "neg") if task == "classification" else 3 * x1 - 2 * x2
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_main_trains_logs_and_registers(tmp_path, monkeypatch, tracking, capsys, task):
    csv_path = _write_csv(tmp_path, task)
    monkeypatch.setattr(_common, "materialise_tabular_dataset", lambda ds, work: csv_path)
    monkeypatch.delenv("TF_CONFIG", raising=False)

    rc = tabular.main(
        [
            "--job-id",
            "job-2",
            "--dataset-id",
            "ds-2",
            "--output-name",
            "tf-model",
            "--task",
            task,
            "--target-column",
            "label",
            "--algorithm",
            f"tensorflow-tabular-{task}",
            "--hyperparams",
            '{"epochs": 6, "hidden_sizes": "16", "batch_size": 16}',
        ]
    )
    assert rc == 0
    assert tracking["start"][0]["tags"][_common.TRAINING_JOB_ID_TAG] == "job-2"
    stepped = [m for step, m in tracking["metrics"] if step is not None]
    assert len(stepped) == 6
    final = tracking["metrics"][-1][1]
    assert "val_loss" in final and "train_seconds" in final
    model, kw = tracking["models"][0]
    assert isinstance(model, tf.keras.Model)
    assert kw["model_type"] == "tensorflow"
    assert kw["registered_model_name"] == "tf-model"
    state = kw["metadata"][_common.PREPROCESSING_METADATA_KEY]
    assert state["feature_names"] and state["target_column"] == "label"
    assert tracking["artifacts"] and tracking["artifacts"][0][1] == _common.PREPROCESSING_ARTIFACT_DIR
    assert capsys.readouterr().out.strip().splitlines()[-1] == "COG_TRAINING_RESULT run_id=run-tf"


def test_is_chief_rules():
    assert tabular.is_chief({}) is True
    assert tabular.is_chief({"cluster": {"chief": ["a:1"], "worker": ["b:1"]}, "task": {"type": "chief", "index": 0}})
    assert not tabular.is_chief(
        {"cluster": {"chief": ["a:1"], "worker": ["b:1"]}, "task": {"type": "worker", "index": 0}}
    )
    assert tabular.is_chief({"cluster": {"worker": ["b:1", "c:1"]}, "task": {"type": "worker", "index": 0}})
    assert not tabular.is_chief({"cluster": {"worker": ["b:1", "c:1"]}, "task": {"type": "worker", "index": 1}})


def test_non_chief_trains_without_tracking(tmp_path, monkeypatch, tracking):
    csv_path = _write_csv(tmp_path, "regression")
    monkeypatch.setattr(_common, "materialise_tabular_dataset", lambda ds, work: csv_path)
    monkeypatch.setattr(tabular, "is_chief", lambda cfg=None: False)
    monkeypatch.delenv("TF_CONFIG", raising=False)
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


def test_detach_from_strategy_copies_weights(tmp_path):
    df = pd.DataFrame({"x1": np.arange(20.0), "x2": np.arange(20.0) * 2, "label": np.arange(20.0)})
    data = _common.prepare_frame(df, "label", "regression", test_split=0.2)
    hp = dict(tabular.DEFAULT_HYPERPARAMS, hidden_sizes="8")
    model = tabular.build_model(data.n_features, 1, [8], 0.0, "regression", 1e-3)
    plain = tabular.detach_from_strategy(model, data, hp)
    assert plain is not model
    for a, b in zip(model.get_weights(), plain.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(model.predict(data.x_val, verbose=0), plain.predict(data.x_val, verbose=0))
