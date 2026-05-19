"""LangSmith tracing adapter: env-var snapshot + restoration.

These guard the security/ops-sensitive parts of ``LangSmithAdapter`` so a
regression that, say, leaks an API key into the process environment after
``disable()`` is caught here.
"""

from __future__ import annotations

import os

import pytest

from cogflow.agent.tracing import configure_tracing
from cogflow.agent.tracing.langsmith import LangSmithAdapter, _MANAGED_ENV_VARS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure every test starts with no LANGCHAIN_* env state."""
    for var in _MANAGED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_enable_sets_expected_env_vars():
    adapter = LangSmithAdapter()
    try:
        adapter.enable(project="proj-a", api_key="sk-test")
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_PROJECT"] == "proj-a"
        assert os.environ["LANGCHAIN_API_KEY"] == "sk-test"
    finally:
        adapter.disable()


def test_disable_restores_previously_unset():
    adapter = LangSmithAdapter()
    adapter.enable(project="proj-a", api_key="sk-test")
    adapter.disable()
    # Vars that did not exist before enable() must be popped, not left set.
    for var in _MANAGED_ENV_VARS:
        assert var not in os.environ, f"{var} leaked past disable()"


def test_disable_restores_previously_set_values(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "preserve-me")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "preexisting-project")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "sk-original")

    adapter = LangSmithAdapter()
    adapter.enable(project="overridden", api_key="sk-test")
    assert os.environ["LANGCHAIN_PROJECT"] == "overridden"

    adapter.disable()
    assert os.environ["LANGCHAIN_TRACING_V2"] == "preserve-me"
    assert os.environ["LANGCHAIN_PROJECT"] == "preexisting-project"
    assert os.environ["LANGCHAIN_API_KEY"] == "sk-original"


def test_repeated_configure_tracing_does_not_corrupt_restore_state(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "sk-original")

    # First enable through the public entry point.
    configure_tracing(backend="langsmith", project="proj-a", api_key="sk-first")
    assert os.environ["LANGCHAIN_API_KEY"] == "sk-first"

    # Second enable must NOT snapshot the already-mutated state; the original
    # value still survives the eventual disable.
    configure_tracing(backend="langsmith", project="proj-b", api_key="sk-second")
    assert os.environ["LANGCHAIN_PROJECT"] == "proj-b"
    assert os.environ["LANGCHAIN_API_KEY"] == "sk-second"

    configure_tracing(backend=None)
    assert os.environ["LANGCHAIN_API_KEY"] == "sk-original"
    assert "LANGCHAIN_TRACING_V2" not in os.environ
