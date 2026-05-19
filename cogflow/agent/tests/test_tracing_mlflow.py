"""MLflow tracing adapter: autolog enabled without raising."""

from __future__ import annotations

import importlib

import pytest


def test_configure_tracing_mlflow_enables_autolog(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}")

    from cogflow.agent.tracing import configure_tracing

    adapter = configure_tracing(backend="mlflow", experiment="cogflow-agent-test", tracking_uri=f"file://{tmp_path}")
    assert adapter is not None
    # Disable cleanly.
    configure_tracing(backend=None)


def test_invalid_backend_raises():
    from cogflow.agent.tracing import configure_tracing

    with pytest.raises(ValueError):
        configure_tracing(backend="nope")
