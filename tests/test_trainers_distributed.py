"""Two-process trainer runs: PyTorch DDP over gloo and TensorFlow MultiWorkerMirroredStrategy.

Each test launches the real ``python -m cogflow.trainers.<framework>.tabular``
entrypoint twice (rank 0 / rank 1, chief / worker) against a local CSV
(``COG_TRAINING_LOCAL_DATASET``) and a file-backed MLflow store, with a
timeout, and checks that both processes finish, that only the primary
registers a model, and that the model artifact landed in the store.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TIMEOUT_S = 420


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    n = 120
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "label": np.where(x1 + x2 > 0, "pos", "neg")})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def _base_env(tmp_path: Path, csv: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "COG_TRAINING_LOCAL_DATASET": str(csv),
            "MLFLOW_TRACKING_URI": (tmp_path / "mlruns").as_uri(),
            # Unreachable on purpose: the catalog POST is best-effort and must only warn.
            "API_PATH": "http://127.0.0.1:1",
            "PYTHONUNBUFFERED": "1",
            "GIT_PYTHON_REFRESH": "quiet",
            "COG_TRAINING_WORK_DIR": str(tmp_path / "work"),
        }
    )
    return env


def _run_all(procs: dict[str, subprocess.Popen]) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    for name, proc in procs.items():
        try:
            stdout, _ = proc.communicate(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            for p in procs.values():
                p.kill()
            pytest.fail(f"{name} did not finish within {TIMEOUT_S}s")
        out[name] = (proc.returncode, stdout.decode(errors="replace"))
    return out


def _mlmodel_files(tmp_path: Path) -> list[Path]:
    return list((tmp_path / "mlruns").rglob("MLmodel"))


def _args(module: str, extra_hp: dict) -> list[str]:
    hp = {"epochs": 3, "hidden_sizes": "8", "batch_size": 16, **extra_hp}
    return [
        sys.executable,
        "-m",
        module,
        "--job-id",
        "dist-job",
        "--dataset-id",
        "local",
        "--output-name",
        "dist-model",
        "--task",
        "classification",
        "--target-column",
        "label",
        "--hyperparams",
        json.dumps(hp),
    ]


def test_pytorch_two_process_ddp_trains_and_only_rank0_registers(tmp_path):
    pytest.importorskip("torch")
    csv = _csv(tmp_path)
    port = _free_port()
    procs = {}
    for rank in (0, 1):
        env = _base_env(tmp_path, csv)
        env.update(
            {
                "WORLD_SIZE": "2",
                "RANK": str(rank),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                # Threads per process: keep two CPU workers from fighting.
                "OMP_NUM_THREADS": "1",
            }
        )
        procs[f"rank{rank}"] = subprocess.Popen(
            _args("cogflow.trainers.pytorch.tabular", {}),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    out = _run_all(procs)
    for name, (rc, text) in out.items():
        assert rc == 0, f"{name} failed (rc={rc}):\n{text[-3000:]}"
    assert "COG_TRAINING_RESULT run_id=" in out["rank0"][1]
    assert "COG_TRAINING_RESULT" not in out["rank1"][1]
    assert "[trainer] epoch 3/3" in out["rank0"][1]
    assert "[trainer] epoch" not in out["rank1"][1], "only rank 0 reports progress"
    files = _mlmodel_files(tmp_path)
    assert len(files) == 1, f"expected one logged model, found {files}"
    assert (files[0].parent / "data" / "model.pth").is_file()


def test_tensorflow_chief_and_worker_are_refused_up_front(tmp_path):
    """Keras 3 cannot fit under MultiWorkerMirroredStrategy (verified on TF 2.17 and 2.21):
    both replicas must exit non-zero immediately with the documented message, before any
    data or model work, and nothing may be logged."""
    pytest.importorskip("tensorflow")
    csv = _csv(tmp_path)
    chief_port, worker_port = _free_port(), _free_port()
    cluster = {"chief": [f"127.0.0.1:{chief_port}"], "worker": [f"127.0.0.1:{worker_port}"]}
    procs = {}
    for task_type in ("chief", "worker"):
        env = _base_env(tmp_path, csv)
        env.update(
            {
                "TF_CONFIG": json.dumps({"cluster": cluster, "task": {"type": task_type, "index": 0}}),
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        procs[task_type] = subprocess.Popen(
            _args("cogflow.trainers.tensorflow.tabular", {}),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    out = _run_all(procs)
    for name, (rc, text) in out.items():
        assert rc != 0, f"{name} must refuse multi-worker training"
        assert "workers=1" in text, f"{name} did not print the documented refusal:\n{text[-1500:]}"
        assert "COG_TRAINING_RESULT" not in text
    assert _mlmodel_files(tmp_path) == []
