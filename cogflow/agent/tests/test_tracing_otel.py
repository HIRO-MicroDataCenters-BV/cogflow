"""OTel tracing adapter: install-hint paths and idempotency.

Covers the error paths that are easy to miss in practice — silently dropped
endpoints have historically been a footgun for tracing backends.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from cogflow.agent.tracing.otel import OTelAdapter


def _has_openinference() -> bool:
    """find_spec raises on missing parent package — guard against that."""
    try:
        return importlib.util.find_spec("openinference.instrumentation.langchain") is not None
    except (ModuleNotFoundError, ImportError, ValueError):
        return False


_SKIP_NO_OPENINFERENCE = pytest.mark.skipif(
    not _has_openinference(),
    reason="openinference-instrumentation-langchain not installed",
)


def test_missing_openinference_raises_with_install_hint(monkeypatch):
    """Without openinference installed, enable() must raise a clear hint.

    We shadow the top-level ``openinference`` package with a plain (non-
    package) module so any ``from openinference.instrumentation.langchain ...``
    deterministically fails with ModuleNotFoundError — independent of whether
    the real package is installed in this environment.
    """
    fake_root = types.ModuleType("openinference")  # no __path__ → not a package
    monkeypatch.setitem(sys.modules, "openinference", fake_root)
    # Also drop any cached subpackages so import resolution doesn't reuse them.
    for name in list(sys.modules):
        if name == "openinference" or name.startswith("openinference."):
            if name != "openinference":
                monkeypatch.delitem(sys.modules, name, raising=False)

    adapter = OTelAdapter()
    with pytest.raises(RuntimeError, match=r"cogflow\[agent-otel\]"):
        adapter.enable()


@_SKIP_NO_OPENINFERENCE
def test_enable_disable_are_idempotent_when_deps_present():
    adapter = OTelAdapter()
    adapter.enable()
    assert adapter._enabled is True
    first_instrumentor = adapter._instrumentor

    # Repeated enable() short-circuits — no re-instrumentation.
    adapter.enable()
    assert adapter._instrumentor is first_instrumentor

    adapter.disable()
    assert adapter._enabled is False
    assert adapter._instrumentor is None

    # disable() on an already-disabled adapter is a safe no-op.
    adapter.disable()
    assert adapter._enabled is False


@_SKIP_NO_OPENINFERENCE
def test_endpoint_without_otel_sdk_raises(monkeypatch):
    """When endpoint is set but the OTel SDK/exporter is missing, raise."""
    # Block the OTel SDK imports.
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)

    adapter = OTelAdapter()
    with pytest.raises(RuntimeError, match=r"OTel endpoint=.* was provided"):
        adapter.enable(endpoint="http://collector:4318")
