"""
Shared plumbing for cogflow trainers.

Everything here is framework-agnostic: the CLI contract, dataset
materialisation (CogAPI serves dataset downloads as a ZIP of exactly one
file), tabular preprocessing with pandas/numpy only, metric helpers and
the MLflow run bracket that tags the run with the platform job id.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)

# MLflow run tags the platform uses to find the run a job produced.
TRAINING_JOB_ID_TAG = "cog.training_job_id"
OUTPUT_NAME_TAG = "cog.output_name"
FRAMEWORK_TAG = "cog.framework"
ALGORITHM_TAG = "cog.algorithm"
TASK_TAG = "cog.task"

# Last stdout line of a successful trainer; the platform greps for it.
RESULT_LINE_PREFIX = "COG_TRAINING_RESULT"

JOB_ID_ENV = "COG_TRAINING_JOB_ID"

TASKS = ("classification", "regression")

# Preprocessing state: JSON artifact beside the model + ``metadata`` on it.
PREPROCESSING_VERSION = 1
PREPROCESSING_FILENAME = "preprocessing.json"
PREPROCESSING_ARTIFACT_DIR = "preprocessing"
PREPROCESSING_METADATA_KEY = "cog_preprocessing"
TABULAR_EXTENSIONS = (".csv", ".parquet")


# ---------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser every tabular trainer shares.

    The platform (CogAPI) builds these flags from the algorithm catalog;
    changing a flag name here is an API change on that side too.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--job-id",
        default=os.environ.get(JOB_ID_ENV, ""),
        help=f"Platform training-job id (default: ${JOB_ID_ENV}). Stamped as an MLflow tag.",
    )
    parser.add_argument("--dataset-id", required=True, help="CogFlow dataset id to train on.")
    parser.add_argument("--output-name", required=True, help="Name of the registered model.")
    parser.add_argument("--task", required=True, choices=TASKS, help="Learning task.")
    parser.add_argument("--target-column", required=True, help="Column of the dataset to predict.")
    parser.add_argument(
        "--hyperparams",
        default="{}",
        help="JSON object of hyperparameters; unknown keys are ignored with a warning.",
    )
    parser.add_argument("--algorithm", default="", help="Catalog algorithm id (tag only).")
    parser.add_argument("--artifact-path", default="model", help="MLflow artifact path for the model.")
    parser.add_argument(
        "--work-dir",
        default=os.environ.get("COG_TRAINING_WORK_DIR", "/tmp/cog-training"),
        help="Scratch directory for the downloaded dataset.",
    )
    return parser


def parse_hyperparams(raw: str | None, defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge the ``--hyperparams`` JSON over ``defaults``.

    Values are coerced to the default's type (``"20"`` → ``20`` for an
    ``int`` default) so a stringly-typed caller still works. Unknown keys
    are dropped with a warning rather than failing a job that already
    scheduled a pod.
    """
    try:
        given = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"--hyperparams is not valid JSON: {exc}") from exc
    if not isinstance(given, dict):
        raise ValueError("--hyperparams must be a JSON object")

    merged = dict(defaults)
    for key, value in given.items():
        if key not in defaults:
            logger.warning("Ignoring unknown hyperparameter %r", key)
            continue
        default = defaults[key]
        if value is None or default is None:
            merged[key] = value
            continue
        try:
            if isinstance(default, bool):
                merged[key] = value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes")
            elif isinstance(default, int):
                merged[key] = int(value)
            elif isinstance(default, float):
                merged[key] = float(value)
            elif isinstance(default, str) and isinstance(value, (list, tuple)):
                # e.g. hidden_sizes [16, 8] → "16,8" (the format parse_int_list reads)
                merged[key] = ",".join(str(v) for v in value)
            else:
                merged[key] = type(default)(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"hyperparameter {key!r}={value!r} is not a valid {type(default).__name__}") from exc
    return merged


def parse_int_list(value: Any) -> list[int]:
    """``"128,64"`` or ``[128, 64]`` → ``[128, 64]``; empty → ``[]``."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(part) for part in str(value).split(",") if part.strip()]


# ---------------------------------------------------------------------
# Dataset materialisation
# ---------------------------------------------------------------------


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract ``archive`` into ``destination`` refusing path-traversal members.

    Dataset uploads are user-controlled, so archive member names are too.
    """
    destination.mkdir(parents=True, exist_ok=True)
    safe_root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if safe_root != target and safe_root not in target.parents:
                raise RuntimeError(
                    f"Refusing to extract path-traversal member {member.filename!r} from {archive.name!r}"
                )
            zf.extract(member, destination)


def _pick_single_tabular_file(root: Path, archive_name: str) -> Path:
    files = [p for p in root.rglob("*") if p.is_file()]
    tabular = sorted(p for p in files if p.suffix.lower() in TABULAR_EXTENSIONS)
    if not tabular:
        contents = sorted(str(p.relative_to(root)) for p in files)
        raise RuntimeError(
            f"No {'/'.join(TABULAR_EXTENSIONS)} file found inside dataset archive {archive_name!r}; contents: {contents}"
        )
    if len(tabular) > 1:
        rel = sorted(str(p.relative_to(root)) for p in tabular)
        raise RuntimeError(
            f"Dataset archive {archive_name!r} contains multiple tabular files ({rel}); exactly one is expected."
        )
    return tabular[0]


def materialise_tabular_dataset(dataset_id: str, work_dir: str | Path) -> Path:
    """Download dataset ``dataset_id`` and return the path of its one CSV/parquet file.

    CogAPI's ``GET /datasets/{id}/file/download`` wraps the object in a ZIP
    even for a single file; the archive is unwrapped here. A raw CSV /
    parquet download (a future API shape) is accepted as-is.
    """
    from ..core import datasets as cogflow_datasets

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Fresh directories per call: a shared work_dir must never let an earlier
    # dataset's files be picked up (or trip the single-file rule) for this one.
    download_dir = Path(tempfile.mkdtemp(prefix=f"download-{dataset_id}-", dir=work_dir))
    downloaded = Path(cogflow_datasets.download_dataset(dataset_id, output_path=str(download_dir)))
    logger.info("Dataset %s downloaded to %s", dataset_id, downloaded)

    if downloaded.suffix.lower() in TABULAR_EXTENSIONS and not zipfile.is_zipfile(downloaded):
        return downloaded
    if not zipfile.is_zipfile(downloaded):
        raise RuntimeError(f"Downloaded dataset {downloaded.name!r} is neither a ZIP nor a tabular file.")

    extracted = Path(tempfile.mkdtemp(prefix=f"extracted-{dataset_id}-", dir=work_dir))
    _safe_extract(downloaded, extracted)
    return _pick_single_tabular_file(extracted, downloaded.name)


def load_frame(path: str | Path):
    """Read a CSV or parquet file into a pandas DataFrame."""
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Tabular preprocessing (pandas + numpy only)
# ---------------------------------------------------------------------


# One-hot column names are ``<column>=<category>``: ``=`` cannot appear in a
# pandas-friendly numeric column name the way ``_`` can, so a numeric column
# ``colour_red`` and category ``red`` of ``colour`` no longer collide.
ONE_HOT_SEP = "="
MISSING_CATEGORY = "<missing>"


def _reject_duplicate_feature_names(names: list[str]) -> None:
    seen: set[str] = set()
    dupes = sorted({n for n in names if n in seen or seen.add(n)})  # type: ignore[func-returns-value]
    if dupes:
        raise ValueError(
            f"generated feature names collide: {dupes}; rename the conflicting columns so every "
            "numeric column and every <column>=<category> indicator has a distinct name"
        )


@dataclass
class PreparedData:
    """Train/validation split ready for a framework."""

    x_train: Any
    y_train: Any
    x_val: Any
    y_val: Any
    feature_names: list[str]
    task: str
    n_classes: int | None = None
    class_names: list[str] = field(default_factory=list)
    input_example: Any = None
    # Fitted preprocessing state (see :func:`transform_frame`); shipped with
    # the registered model so inference reproduces the training features.
    preprocessing: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def n_outputs(self) -> int:
        """Model output width: classes for classification, 1 for regression."""
        return int(self.n_classes) if self.task == "classification" else 1


def prepare_frame(
    df,
    target_column: str,
    task: str,
    *,
    test_split: float = 0.2,
    seed: int = 42,
) -> PreparedData:
    """Turn a raw DataFrame into standardised float32 arrays.

    - rows without a target are dropped; all-empty columns are dropped;
    - categorical/boolean/text features are one-hot encoded;
    - numeric features are mean-imputed and z-scored with *training* stats;
    - classification targets are label-encoded over the sorted class names;
    - the split is a seeded shuffle, per-class for classification so every
      class with at least two rows appears in both halves.
    """
    import numpy as np
    import pandas as pd

    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if target_column not in df.columns:
        raise ValueError(f"target column {target_column!r} not in dataset columns {list(df.columns)}")
    if not 0.0 < test_split < 1.0:
        raise ValueError(f"test_split must be in (0, 1), got {test_split}")

    frame = df.dropna(subset=[target_column]).copy()
    frame = frame.dropna(axis=1, how="all")
    if len(frame) < 4:
        raise ValueError(f"dataset has only {len(frame)} usable rows; need at least 4")

    y_raw = frame[target_column]
    x_raw = frame.drop(columns=[target_column])
    if x_raw.shape[1] == 0:
        raise ValueError("dataset has no feature columns besides the target")

    numeric_cols = [c for c in x_raw.columns if pd.api.types.is_numeric_dtype(x_raw[c])]
    other_cols = [c for c in x_raw.columns if c not in numeric_cols]
    numeric = x_raw[numeric_cols].astype("float64") if numeric_cols else pd.DataFrame(index=x_raw.index)
    # Per-column category vocabulary, fitted on the whole frame: indicators
    # are built from it positionally (see ``_one_hot``), so names are only
    # labels and no alias between columns/categories can mix them up.
    categories: dict[str, list[str]] = {
        str(c): sorted(x_raw[c].astype("string").fillna(MISSING_CATEGORY).unique().tolist()) for c in other_cols
    }
    encoded = _one_hot(x_raw, categories) if other_cols else pd.DataFrame(index=x_raw.index)
    features = pd.concat([numeric, encoded], axis=1)
    feature_names = [str(c) for c in features.columns]
    _reject_duplicate_feature_names(feature_names)

    rng = np.random.default_rng(seed)
    n = len(frame)
    if task == "classification":
        classes = sorted(y_raw.astype(str).unique().tolist())
        class_index = {c: i for i, c in enumerate(classes)}
        y_all = y_raw.astype(str).map(class_index).to_numpy(dtype="int64")
        if len(classes) < 2:
            raise ValueError("classification needs at least two classes in the target column")
        val_mask = np.zeros(n, dtype=bool)
        for cls_idx in range(len(classes)):
            rows = np.flatnonzero(y_all == cls_idx)
            rng.shuffle(rows)
            k = int(round(len(rows) * test_split))
            if len(rows) >= 2:
                k = min(max(k, 1), len(rows) - 1)
            else:
                k = 0
            val_mask[rows[:k]] = True
        n_classes: int | None = len(classes)
        class_names = classes
    else:
        y_all = pd.to_numeric(y_raw, errors="raise").to_numpy(dtype="float64")
        perm = rng.permutation(n)
        k = min(max(int(round(n * test_split)), 1), n - 1)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[perm[:k]] = True
        n_classes = None
        class_names = []

    if int(val_mask.sum()) == 0 or int((~val_mask).sum()) == 0:
        raise ValueError(
            f"the validation split is empty ({n} rows, test_split={test_split}"
            + (f", {len(class_names)} classes" if task == "classification" else "")
            + "); add rows or adjust test_split so both splits are non-empty"
        )

    x_all = features.to_numpy(dtype="float64")
    x_train, x_val = x_all[~val_mask], x_all[val_mask]
    y_train, y_val = y_all[~val_mask], y_all[val_mask]

    col_mean = np.zeros(0)
    col_std = np.zeros(0)
    if numeric_cols:
        idx = [feature_names.index(str(c)) for c in numeric_cols]
        col_mean = np.nanmean(x_train[:, idx], axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        for arr in (x_train, x_val):
            block = arr[:, idx]
            nan_pos = np.isnan(block)
            block[nan_pos] = np.take(col_mean, np.nonzero(nan_pos)[1])
            arr[:, idx] = block
        col_std = x_train[:, idx].std(axis=0)
        col_std = np.where(col_std < 1e-12, 1.0, col_std)
        x_train[:, idx] = (x_train[:, idx] - col_mean) / col_std
        x_val[:, idx] = (x_val[:, idx] - col_mean) / col_std

    x_train = np.ascontiguousarray(x_train, dtype="float32")
    x_val = np.ascontiguousarray(x_val, dtype="float32")
    if task == "regression":
        y_train = y_train.astype("float32")
        y_val = y_val.astype("float32")

    preprocessing = {
        "version": PREPROCESSING_VERSION,
        "task": task,
        "target_column": str(target_column),
        "feature_names": feature_names,
        "numeric_columns": [str(c) for c in numeric_cols],
        "numeric_means": [float(v) for v in col_mean],
        "numeric_stds": [float(v) for v in col_std],
        "categorical_columns": [str(c) for c in other_cols],
        "categories": categories,
        "class_names": list(class_names),
    }

    return PreparedData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        feature_names=feature_names,
        task=task,
        n_classes=n_classes,
        class_names=class_names,
        input_example=x_train[: min(5, len(x_train))].copy(),
        preprocessing=preprocessing,
    )


def transform_frame(preprocessing: dict[str, Any], df) -> Any:
    """Apply a fitted :func:`prepare_frame` state to raw rows (inference side).

    Reproduces the training features exactly: numeric columns are
    mean-imputed and z-scored with the *training* statistics, categorical
    columns are one-hot encoded and aligned to the trained column order
    (unseen categories contribute no column, missing columns are zero).

    Returns:
        ``float32`` array of shape ``(len(df), len(feature_names))``.
    """
    import numpy as np
    import pandas as pd

    feature_names = list(preprocessing["feature_names"])
    numeric_cols = list(preprocessing.get("numeric_columns") or [])
    other_cols = list(preprocessing.get("categorical_columns") or [])
    means = np.asarray(preprocessing.get("numeric_means") or [], dtype="float64")
    stds = np.asarray(preprocessing.get("numeric_stds") or [], dtype="float64")

    missing = [c for c in numeric_cols + other_cols if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing trained feature columns: {missing}")

    parts = []
    if numeric_cols:
        numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        values = numeric.to_numpy(dtype="float64")
        nan_pos = np.isnan(values)
        values[nan_pos] = np.take(means, np.nonzero(nan_pos)[1])
        values = (values - means) / stds
        parts.append(pd.DataFrame(values, columns=numeric_cols, index=df.index))
    if other_cols:
        categories = preprocessing.get("categories")
        if not isinstance(categories, dict):
            raise ValueError("preprocessing state has no per-column category vocabulary")
        # Indicators come from the trained vocabulary per source column —
        # an unseen category yields all-zero indicators for THAT column only,
        # whatever its spelling looks like next to another column's name.
        parts.append(_one_hot(df, {c: list(categories[c]) for c in other_cols}))
    features = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    if list(features.columns) != feature_names:
        features = features.reindex(columns=feature_names, fill_value=0.0)
    return np.ascontiguousarray(features.to_numpy(dtype="float64"), dtype="float32")


def _one_hot(df, categories: dict[str, list[str]]):
    """Indicator columns ``<column>=<category>`` for every trained (column, category) pair.

    Built per source column from the vocabulary, not from the values present
    in ``df``: the same pairs always come out in the same order, missing
    categories are zero, unseen values contribute nothing.
    """
    import numpy as np
    import pandas as pd

    columns: list[str] = []
    blocks = []
    for col, cats in categories.items():
        values = df[col].astype("string").fillna(MISSING_CATEGORY).to_numpy(dtype=object)
        block = np.zeros((len(df), len(cats)), dtype="float64")
        for j, cat in enumerate(cats):
            block[:, j] = values == cat
            columns.append(f"{col}{ONE_HOT_SEP}{cat}")
        blocks.append(block)
    data = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0))
    return pd.DataFrame(data, columns=columns, index=df.index)


def log_preprocessing(preprocessing: dict[str, Any], work_dir: str | Path) -> Path:
    """Write the preprocessing state as JSON and log it as a run artifact.

    The same state also travels as ``metadata`` on the logged model, so a
    consumer holding only the model URI can rebuild the features.
    """
    from ..core import models as cogflow_models

    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PREPROCESSING_FILENAME
    path.write_text(json.dumps(preprocessing, indent=2))
    cogflow_models.log_artifact(str(path), artifact_path=PREPROCESSING_ARTIFACT_DIR)
    return path


# ---------------------------------------------------------------------
# Metrics (numpy only)
# ---------------------------------------------------------------------


def classification_metrics(y_true, y_pred, n_classes: int) -> dict[str, float]:
    """Accuracy and macro-F1 over integer labels."""
    import numpy as np

    y_true = np.asarray(y_true).astype("int64")
    y_pred = np.asarray(y_pred).astype("int64")
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1s = []
    for c in range(n_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return {"val_accuracy": accuracy, "val_f1_macro": float(np.mean(f1s)) if f1s else 0.0}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE and R² over float targets."""
    import numpy as np

    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0
    mae = float(np.mean(np.abs(err))) if len(err) else 0.0
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if len(y_true) else 0.0
    ss_res = float(np.sum(err**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"val_rmse": rmse, "val_mae": mae, "val_r2": float(r2)}


# ---------------------------------------------------------------------
# Tracking bracket
# ---------------------------------------------------------------------


def run_tags(
    *,
    job_id: str,
    output_name: str,
    framework: str,
    algorithm: str,
    task: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    tags = {
        TRAINING_JOB_ID_TAG: job_id,
        OUTPUT_NAME_TAG: output_name,
        FRAMEWORK_TAG: framework,
        ALGORITHM_TAG: algorithm,
        TASK_TAG: task,
    }
    tags.update(extra or {})
    return {k: str(v) for k, v in tags.items() if v not in (None, "")}


@contextmanager
def tracked_run(
    *,
    job_id: str,
    output_name: str,
    framework: str,
    algorithm: str,
    task: str,
    extra_tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Open the MLflow run a trainer logs into, tagged with the job id.

    The run name is the output model name so the catalog row (whose name
    falls back to ``mlflow.runName``) reads well; the job id lives in the
    ``cog.training_job_id`` tag the platform searches by.
    """
    from ..core import models as cogflow_models

    tags = run_tags(job_id=job_id, output_name=output_name, framework=framework, algorithm=algorithm, task=task)
    tags.update(extra_tags or {})
    with cogflow_models.start_run(run_name=output_name, tags=tags) as active_run:
        yield active_run


def emit_result(run_id: str) -> None:
    """Print the machine-readable last line of a successful trainer."""
    print(f"{RESULT_LINE_PREFIX} run_id={run_id}", flush=True)


def resolve_seed(hyperparams: dict[str, Any], default: int = 42) -> int:
    try:
        return int(hyperparams.get("seed", default))
    except (TypeError, ValueError):
        return default
