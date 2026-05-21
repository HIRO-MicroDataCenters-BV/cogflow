"""Provider-gate tests for ``cogflow.agent.chat_models.openai``.

Two cases:
 1. ``langchain-openai`` not installed: importing the module raises a
    ``ModuleNotFoundError`` whose message contains the actionable hint
    ``pip install cogflow[openai]``.
 2. ``langchain-openai`` is installed: the re-export resolves to the
    real ``ChatOpenAI`` class.

The "uninstalled" case is the same shape as the OTel adapter gate test
(``test_tracing_otel.py``). ``sys.modules[X] = None`` is not reliable
when X is actually present on the host — Python may re-import it. We
shadow the top-level ``langchain_openai`` package with a plain
``types.ModuleType`` so the ``from langchain_openai import ChatOpenAI``
in the SUT deterministically fails regardless of the test environment.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import pytest


def _has_langchain_openai() -> bool:
    try:
        return importlib.util.find_spec("langchain_openai") is not None
    except (ModuleNotFoundError, ImportError, ValueError):
        return False


def test_openai_reexport_raises_friendly_error_when_uninstalled(monkeypatch):
    # Shadow the top-level package with a plain (non-package) module so
    # ``from langchain_openai import ChatOpenAI`` deterministically fails
    # regardless of what's installed locally. The immediate exception is
    # an ``ImportError`` (Python finds the shadowed module but the symbol
    # ``ChatOpenAI`` is missing from it); the SUT catches the broader
    # ``ImportError`` and re-raises as ``ModuleNotFoundError`` with the
    # install hint, which is the user-facing shape we assert on below.
    fake_root = types.ModuleType("langchain_openai")
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_root)

    # Drop any cached cogflow.agent.chat_models.openai so the next import
    # re-runs the module body against the shadowed langchain_openai.
    monkeypatch.delitem(sys.modules, "cogflow.agent.chat_models.openai", raising=False)

    with pytest.raises(ModuleNotFoundError, match=r"cogflow\[openai\]"):
        importlib.import_module("cogflow.agent.chat_models.openai")


@pytest.mark.skipif(
    not _has_langchain_openai(),
    reason="langchain-openai not installed; gated path is covered by the other test",
)
def test_openai_reexport_resolves_to_real_chatopenai():
    # Defensive: clear any prior shadow before the real import.
    sys.modules.pop("cogflow.agent.chat_models.openai", None)

    from cogflow.agent.chat_models.openai import ChatOpenAI
    from langchain_openai import ChatOpenAI as _Upstream

    assert ChatOpenAI is _Upstream


def test_openai_reexport_propagates_unrelated_import_errors(monkeypatch):
    """A failure inside langchain_openai (e.g. its transitive ``openai`` dep
    missing) must NOT be re-raised as a misleading "install langchain-openai"
    hint. The SUT only converts ImportErrors whose ``name`` points at
    ``langchain_openai`` itself; everything else propagates unchanged so the
    user sees the real cause.
    """

    # Build a shadow ``langchain_openai`` whose ``__getattr__`` raises
    # ModuleNotFoundError(name="openai") — simulating a broken transitive.
    # ``from langchain_openai import ChatOpenAI`` invokes module __getattr__
    # under PEP 562 when the attribute isn't in the module dict.
    fake_root = types.ModuleType("langchain_openai")

    def _broken_getattr(name: str):
        raise ModuleNotFoundError("No module named 'openai'", name="openai")

    fake_root.__getattr__ = _broken_getattr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_root)
    monkeypatch.delitem(sys.modules, "cogflow.agent.chat_models.openai", raising=False)

    # The transitive failure surfaces with its real ``name="openai"`` — not
    # masked as a langchain-openai install hint.
    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module("cogflow.agent.chat_models.openai")
    assert exc_info.value.name == "openai"
    assert "cogflow[openai]" not in str(exc_info.value)
