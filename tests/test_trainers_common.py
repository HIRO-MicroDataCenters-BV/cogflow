"""Tests for the framework-agnostic trainer plumbing (cogflow.trainers._common)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cogflow.trainers import _common

# ---------------------------------------------------------------------
# CLI / hyperparams
# ---------------------------------------------------------------------


def test_arg_parser_contract(monkeypatch):
    monkeypatch.setenv(_common.JOB_ID_ENV, "env-job")
    parser = _common.build_arg_parser("t")
    args = parser.parse_args(
        ["--dataset-id", "ds", "--output-name", "m", "--task", "regression", "--target-column", "y"]
    )
    assert args.job_id == "env-job"
    assert args.hyperparams == "{}"
    assert args.artifact_path == "model"
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset-id", "ds", "--output-name", "m", "--task", "ranking", "--target-column", "y"])


def test_parse_hyperparams_merges_coerces_and_drops_unknown():
    defaults = {"epochs": 20, "learning_rate": 1e-3, "hidden_sizes": "128,64", "shuffle": True}
    merged = _common.parse_hyperparams(
        '{"epochs": "5", "learning_rate": 0.01, "shuffle": "false", "bogus": 1}', defaults
    )
    assert merged == {"epochs": 5, "learning_rate": 0.01, "hidden_sizes": "128,64", "shuffle": False}


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"epochs": "many"}'])
def test_parse_hyperparams_rejects_bad_input(raw):
    with pytest.raises(ValueError):
        _common.parse_hyperparams(raw, {"epochs": 1})


def test_parse_int_list():
    assert _common.parse_int_list("128, 64,") == [128, 64]
    assert _common.parse_int_list([32, 16]) == [32, 16]
    assert _common.parse_int_list("") == []
    assert _common.parse_int_list(None) == []


# ---------------------------------------------------------------------
# Dataset materialisation
# ---------------------------------------------------------------------


def _zip_with(tmp_path: Path, members: dict[str, bytes], name="dataset.zip") -> Path:
    archive = tmp_path / name
    with zipfile.ZipFile(archive, "w") as zf:
        for member, payload in members.items():
            zf.writestr(member, payload)
    return archive


def _patch_download(monkeypatch, path: Path):
    import cogflow.core.datasets as datasets_mod

    monkeypatch.setattr(datasets_mod, "download_dataset", lambda ds_id, output_path=None: str(path))


def test_materialise_unwraps_zip_with_one_csv(tmp_path, monkeypatch):
    archive = _zip_with(tmp_path, {"sub/data.CSV": b"a,b\n1,2\n"})
    _patch_download(monkeypatch, archive)
    result = _common.materialise_tabular_dataset("ds", tmp_path / "work")
    assert result.name == "data.CSV"
    assert result.read_text() == "a,b\n1,2\n"


def test_materialise_rejects_zip_slip(tmp_path, monkeypatch):
    archive = _zip_with(tmp_path, {"../evil.csv": b"a\n1\n"})
    _patch_download(monkeypatch, archive)
    with pytest.raises(RuntimeError, match="path-traversal"):
        _common.materialise_tabular_dataset("ds", tmp_path / "work")


def test_materialise_rejects_zero_or_many_tabular_files(tmp_path, monkeypatch):
    _patch_download(monkeypatch, _zip_with(tmp_path, {"readme.txt": b"x"}, name="none.zip"))
    with pytest.raises(RuntimeError, match="No .csv/.parquet file"):
        _common.materialise_tabular_dataset("ds", tmp_path / "w1")
    _patch_download(monkeypatch, _zip_with(tmp_path, {"a.csv": b"a\n1\n", "b.parquet": b"x"}, name="many.zip"))
    with pytest.raises(RuntimeError, match="multiple tabular files"):
        _common.materialise_tabular_dataset("ds", tmp_path / "w2")


def test_materialise_uses_fresh_directories_per_call(tmp_path, monkeypatch):
    """A shared work_dir must not leak an earlier dataset's files into the next call."""
    work = tmp_path / "work"
    _patch_download(monkeypatch, _zip_with(tmp_path, {"first.csv": b"a\n1\n"}, name="one.zip"))
    first = _common.materialise_tabular_dataset("ds-1", work)
    assert first.name == "first.csv"
    _patch_download(monkeypatch, _zip_with(tmp_path, {"second.csv": b"b\n2\n"}, name="two.zip"))
    second = _common.materialise_tabular_dataset("ds-2", work)
    assert second.name == "second.csv" and second.parent != first.parent
    # And a tabular-less archive must not fall back to the previous CSV.
    _patch_download(monkeypatch, _zip_with(tmp_path, {"readme.txt": b"x"}, name="three.zip"))
    with pytest.raises(RuntimeError, match="No .csv/.parquet file"):
        _common.materialise_tabular_dataset("ds-3", work)


def test_materialise_accepts_raw_csv_download(tmp_path, monkeypatch):
    raw = tmp_path / "plain.csv"
    raw.write_text("a,b\n1,2\n")
    _patch_download(monkeypatch, raw)
    assert _common.materialise_tabular_dataset("ds", tmp_path / "work") == raw


def test_load_frame_csv_and_parquet(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n")
    assert list(_common.load_frame(csv).columns) == ["a", "b"]
    pq = tmp_path / "d.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(pq)
    assert list(_common.load_frame(pq)["x"]) == [1, 2]


# ---------------------------------------------------------------------
# prepare_frame
# ---------------------------------------------------------------------


def _iris_like(n_per_class=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cls, centre in (("a", 0.0), ("b", 3.0), ("c", 6.0)):
        for _ in range(n_per_class):
            rows.append(
                {
                    "f1": centre + rng.normal(),
                    "f2": centre * 2 + rng.normal(),
                    "colour": rng.choice(["red", "blue"]),
                    "flag": bool(rng.integers(0, 2)),
                    "species": cls,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_frame_classification_shapes_and_encoding():
    df = _iris_like()
    df.loc[0, "f1"] = np.nan  # imputed
    df["empty"] = np.nan  # dropped
    data = _common.prepare_frame(df, "species", "classification", test_split=0.2, seed=1)
    assert data.task == "classification"
    assert data.n_classes == 3
    assert data.class_names == ["a", "b", "c"]
    assert data.n_outputs == 3
    # f1, f2 numeric + bool flag (numeric dtype, kept as-is) + one-hot colour (2) = 5; 'empty' dropped
    assert data.n_features == 5
    assert sorted(n for n in data.feature_names if n.startswith("colour")) == ["colour_blue", "colour_red"]
    assert all(not name.startswith("empty") for name in data.feature_names)
    assert data.x_train.dtype == np.float32 and data.x_val.dtype == np.float32
    assert data.y_train.dtype == np.int64
    assert len(data.x_train) + len(data.x_val) == 30
    # stratified: every class present in both halves
    assert set(np.unique(data.y_val)) == {0, 1, 2}
    assert set(np.unique(data.y_train)) == {0, 1, 2}
    assert not np.isnan(data.x_train).any() and not np.isnan(data.x_val).any()
    # z-scored numeric columns on the train split
    assert abs(float(data.x_train[:, 0].mean())) < 1e-5
    assert data.input_example.shape == (5, 5)


def test_prepare_frame_regression_split_and_dtypes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=50), "x2": rng.normal(size=50)})
    df["y"] = 2 * df["x1"] - df["x2"] + 0.1
    data = _common.prepare_frame(df, "y", "regression", test_split=0.25, seed=3)
    assert data.n_classes is None
    assert data.n_outputs == 1
    assert len(data.x_val) == 12 or len(data.x_val) == 13
    assert data.y_train.dtype == np.float32


@pytest.mark.parametrize(
    "df,target,task,match",
    [
        (pd.DataFrame({"a": [1, 2, 3, 4]}), "y", "regression", "not in dataset columns"),
        (pd.DataFrame({"a": [1, 2], "y": [1, 2]}), "y", "regression", "at least 4"),
        (pd.DataFrame({"y": [1, 2, 3, 4]}), "y", "regression", "no feature columns"),
        (pd.DataFrame({"a": [1, 2, 3, 4], "y": ["k", "k", "k", "k"]}), "y", "classification", "two classes"),
        (pd.DataFrame({"a": [1, 2, 3, 4], "y": [1, 2, 3, 4]}), "y", "ranking", "task must be"),
    ],
)
def test_prepare_frame_validation(df, target, task, match):
    with pytest.raises(ValueError, match=match):
        _common.prepare_frame(df, target, task)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def test_classification_metrics_perfect_and_partial():
    perfect = _common.classification_metrics([0, 1, 2, 1], [0, 1, 2, 1], 3)
    assert perfect == {"val_accuracy": 1.0, "val_f1_macro": 1.0}
    partial = _common.classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert partial["val_accuracy"] == 0.75
    assert 0.7 < partial["val_f1_macro"] < 0.8


def test_regression_metrics():
    m = _common.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m == {"val_rmse": 0.0, "val_mae": 0.0, "val_r2": 1.0}
    m = _common.regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert m["val_mae"] == pytest.approx(2 / 3)
    assert m["val_r2"] == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Tracking bracket
# ---------------------------------------------------------------------


def test_run_tags_and_tracked_run(monkeypatch):
    import cogflow.core.models as models_mod

    captured = {}

    class _Run:
        info = type("I", (), {"run_id": "run-1"})()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_start_run(**kwargs):
        captured.update(kwargs)
        return _Run()

    monkeypatch.setattr(models_mod, "start_run", fake_start_run)
    with _common.tracked_run(
        job_id="job-9", output_name="mdl", framework="pytorch", algorithm="alg", task="regression"
    ) as r:
        assert r.info.run_id == "run-1"
    assert captured["run_name"] == "mdl"
    assert captured["tags"][_common.TRAINING_JOB_ID_TAG] == "job-9"
    assert captured["tags"][_common.FRAMEWORK_TAG] == "pytorch"
    assert captured["tags"][_common.ALGORITHM_TAG] == "alg"
    assert captured["tags"][_common.TASK_TAG] == "regression"


def test_run_tags_drop_empty_values():
    tags = _common.run_tags(job_id="", output_name="m", framework="tensorflow", algorithm="", task="classification")
    assert _common.TRAINING_JOB_ID_TAG not in tags
    assert _common.ALGORITHM_TAG not in tags
    assert tags[_common.OUTPUT_NAME_TAG] == "m"


def test_emit_result_line(capsys):
    _common.emit_result("abc")
    assert capsys.readouterr().out.strip() == "COG_TRAINING_RESULT run_id=abc"


# ---------------------------------------------------------------------
# Preprocessing state (Copilot review on #124)
# ---------------------------------------------------------------------


def test_prepare_frame_rejects_empty_validation_split():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "y": ["p", "q", "r", "s"]})
    with pytest.raises(ValueError, match="validation split is empty"):
        _common.prepare_frame(df, "y", "classification", test_split=0.2)


def test_prepare_frame_exposes_fitted_preprocessing_state():
    df = _iris_like()
    data = _common.prepare_frame(df, "species", "classification", test_split=0.2, seed=1)
    state = data.preprocessing
    assert state["version"] == _common.PREPROCESSING_VERSION
    assert state["task"] == "classification" and state["target_column"] == "species"
    assert state["feature_names"] == data.feature_names
    assert state["numeric_columns"] == ["f1", "f2", "flag"]
    assert len(state["numeric_means"]) == 3 and len(state["numeric_stds"]) == 3
    assert state["categorical_columns"] == ["colour"]
    assert state["class_names"] == ["a", "b", "c"]
    import json

    json.dumps(state)  # must be serialisable as-is


def test_transform_frame_reproduces_training_features():
    df = _iris_like()
    df.loc[0, "f1"] = np.nan
    data = _common.prepare_frame(df, "species", "classification", test_split=0.25, seed=2)
    transformed = _common.transform_frame(data.preprocessing, df.drop(columns=["species"]))
    assert transformed.dtype == np.float32
    assert transformed.shape == (len(df), data.n_features)
    trained = np.concatenate([data.x_train, data.x_val])
    # Same multiset of rows (the split only reorders them).
    a = np.sort(np.round(trained, 5), axis=0)
    b = np.sort(np.round(transformed, 5), axis=0)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_transform_frame_aligns_unseen_and_missing_categories():
    df = _iris_like()
    data = _common.prepare_frame(df, "species", "classification")
    new = pd.DataFrame({"f1": [0.5], "f2": [1.0], "colour": ["green"], "flag": [True]})
    out = _common.transform_frame(data.preprocessing, new)
    idx = {n: i for i, n in enumerate(data.feature_names)}
    assert out[0, idx["colour_red"]] == 0.0 and out[0, idx["colour_blue"]] == 0.0
    with pytest.raises(ValueError, match="missing trained feature columns"):
        _common.transform_frame(data.preprocessing, new.drop(columns=["f2"]))


def test_log_preprocessing_writes_json_and_logs_artifact(tmp_path, monkeypatch):
    import json

    import cogflow.core.models as models_mod

    calls = []
    monkeypatch.setattr(
        models_mod, "log_artifact", lambda path, artifact_path=None: calls.append((path, artifact_path))
    )
    state = {"version": 1, "feature_names": ["a"]}
    path = _common.log_preprocessing(state, tmp_path / "work")
    assert path.name == _common.PREPROCESSING_FILENAME
    assert json.loads(path.read_text()) == state
    assert calls == [(str(path), _common.PREPROCESSING_ARTIFACT_DIR)]
