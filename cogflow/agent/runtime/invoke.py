"""Thin pass-through wrappers for the LangGraph runnable interface.

A compiled cogflow.agent graph is just a ``langgraph.graph.CompiledStateGraph``;
nothing here needs to do real work. The wrappers exist so users can import the
helpers from ``cogflow.agent.runtime.invoke`` in scripts that don't already hold
a reference to the compiled graph.
"""

from __future__ import annotations

from typing import Any


def invoke(app: Any, inputs: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    return app.invoke(inputs, config=config)


def stream(app: Any, inputs: dict[str, Any], *, config: dict[str, Any] | None = None):
    yield from app.stream(inputs, config=config)


async def ainvoke(
    app: Any, inputs: dict[str, Any], *, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    return await app.ainvoke(inputs, config=config)


async def astream(app: Any, inputs: dict[str, Any], *, config: dict[str, Any] | None = None):
    async for step in app.astream(inputs, config=config):
        yield step


__all__ = ["ainvoke", "astream", "invoke", "stream"]
