"""MLflow Tracing adapter — the default backend.

Calls ``mlflow.langchain.autolog()`` which registers a LangChain callback that
emits a span for every LLM call, tool call, and chain step. Tracking URI is
read from ``cogflow.config`` so traces land on the org's existing MLflow.
"""

from __future__ import annotations

from typing import Any

from .base import TracingAdapter


class MLflowAdapter(TracingAdapter):
    name = "mlflow"

    def __init__(self) -> None:
        self._enabled = False

    def enable(
        self,
        *,
        experiment: str | None = "cogflow-agent",
        tracking_uri: str | None = None,
        **kwargs: Any,
    ) -> None:
        import mlflow
        import mlflow.langchain

        uri = tracking_uri
        if uri is None:
            try:
                from cogflow import config as _cfg  # type: ignore  # noqa: WPS433

                uri = getattr(_cfg, "MLFLOW_TRACKING_URI", None)
            except Exception:
                uri = None

        if uri:
            mlflow.set_tracking_uri(uri)
        if experiment:
            mlflow.set_experiment(experiment)

        mlflow.langchain.autolog(**kwargs)
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        import mlflow.langchain

        mlflow.langchain.autolog(disable=True)
        self._enabled = False
