"""Tracing adapters.

Usage:

    from cogflow.agent.tracing import configure_tracing
    configure_tracing(backend="mlflow", experiment="agentflow")  # default
    configure_tracing(backend="langsmith", project="agentflow")  # opt-in
    configure_tracing(backend="otel", endpoint="http://collector:4318")
    configure_tracing(backend=None)                              # disable all
"""

from __future__ import annotations

from typing import Any

from .base import TracingAdapter
from .langsmith import LangSmithAdapter
from .mlflow import MLflowAdapter
from .otel import OTelAdapter

_ADAPTERS: dict[str, TracingAdapter] = {
    "mlflow": MLflowAdapter(),
    "langsmith": LangSmithAdapter(),
    "otel": OTelAdapter(),
}


def configure_tracing(backend: str | None = "mlflow", **kwargs: Any) -> TracingAdapter | None:
    """Enable (or disable) a tracing backend.

    Successive calls with *different* backends compose (mlflow + otel can both
    be on). Successive calls with the *same* backend are idempotent: the
    adapter is first disabled (restoring any saved state) then re-enabled with
    the new kwargs, so callbacks/instrumentors never stack.
    """
    if backend is None:
        for adapter in _ADAPTERS.values():
            adapter.disable()
        return None
    if backend not in _ADAPTERS:
        raise ValueError(f"unknown tracing backend {backend!r}; one of {sorted(_ADAPTERS)}")
    adapter = _ADAPTERS[backend]
    adapter.disable()  # idempotent — no-op when not currently enabled
    adapter.enable(**kwargs)
    return adapter


__all__ = [
    "LangSmithAdapter",
    "MLflowAdapter",
    "OTelAdapter",
    "TracingAdapter",
    "configure_tracing",
]
