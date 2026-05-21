"""Cognitive Framework Agent SDK.

A LangGraph-shaped Python API whose graphs round-trip to Flowise V2 AgentFlow
JSON. See ``cogflow/agent/__init__.py`` callers for the full mapping table in
the architecture plan.

Drop-in import migration (see plan §3a):

    from cogflow.agent import StateGraph, START, END, MessagesState
    from cogflow.agent import create_react_agent, interrupt, Command
    from cogflow.agent.tools import tool, ToolNode
"""

from __future__ import annotations

# Defer the heavy imports behind a single helpful message so users who pip-
# installed bare ``cogflow`` see actionable text instead of a raw
# ``ModuleNotFoundError`` from langgraph OR langchain_core (both are required
# at import time because submodules below depend on each).
try:
    import langchain_core  # noqa: F401 — verified-then-used by submodules
    from langgraph.graph import END, START
except ModuleNotFoundError as _exc:  # pragma: no cover - install-time guard
    raise ModuleNotFoundError(
        f"cogflow.agent requires `{_exc.name}`. "
        "Install with `pip install cogflow[agent]`."
    ) from _exc

try:
    from langgraph.types import Command, interrupt  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - older langgraph
    interrupt = None  # type: ignore[assignment]
    Command = None  # type: ignore[assignment]

from . import (
    chat_models,
    compile,
    fakes,
    messages,
    nodes,
    parse,
    prompts,
    runnables,
    tracing,
)
from .graph import StateGraph
from .prebuilt import create_react_agent
from .state import MessagesState, add_messages
from .tools import ToolNode, tool

__all__ = [
    "Command",
    "END",
    "MessagesState",
    "START",
    "StateGraph",
    "ToolNode",
    "add_messages",
    "chat_models",
    "compile",
    "create_react_agent",
    "fakes",
    "interrupt",
    "messages",
    "nodes",
    "parse",
    "prompts",
    "runnables",
    "tool",
    "tracing",
]
