"""Tool node — exposes a callable that an Agent node may invoke.

Two runtime shapes are supported. If the user passed a ``@tool``-decorated
``BaseTool`` (the LangChain idiom), we delegate to ``langgraph.prebuilt.ToolNode``
so the node honours the standard contract: read ``tool_calls`` off the last
AIMessage in state, execute each call, emit one ``ToolMessage`` per call. If
the user passed a plain Python callable, we keep the simpler historic
contract: read ``state["tool_input"]``, call ``fn``, stash the return value
under ``tool_output``. Both contracts coexist so Flowise's ``toolAgentflow``
(a plain callable) and a Python-first ``@tool`` continue to drop in.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from ..ir.model import IRNode
from .base import NodeFactory


class ToolFactory(NodeFactory):
    ir_type = "tool"

    def __init__(
        self,
        *,
        name: str | None = None,
        fn: Callable[..., Any] | None = None,
        description: str | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            toolName=name or (getattr(fn, "__name__", "tool") if fn else "tool"),
            toolDescription=description or (getattr(fn, "__doc__", "") if fn else ""),
            **extra,
        )
        self._fn = fn

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        fn = getattr(self, "_fn", None)

        # If ``fn`` is a ``@tool``-decorated BaseTool, delegate to LangGraph's
        # ToolNode so the standard tool_calls contract is honoured (read
        # last-AIMessage tool_calls, execute each, emit ToolMessages). Falls
        # back to the plain-callable path on any import or type mismatch so
        # JSON-imported nodes (which carry no live ``fn``) keep working.
        if fn is not None:
            try:
                from langchain_core.tools import BaseTool  # type: ignore[import-not-found]
                from langgraph.prebuilt import ToolNode as _LGToolNode  # type: ignore[attr-defined]

                if isinstance(fn, BaseTool):
                    return _LGToolNode([fn])
            except ImportError:
                pass

        def tool_node(state: dict[str, Any]) -> dict[str, Any]:
            if fn is None:
                return {}
            args = state.get("tool_input")
            # Pick the call shape from the callable's signature instead of
            # try/except'ing TypeError — that earlier approach swallowed real
            # TypeErrors thrown inside the tool body and could call ``fn``
            # twice when the first call partially succeeded.
            try:
                sig = inspect.signature(fn)
                params = sig.parameters
            except (TypeError, ValueError):
                # Builtins / C-implemented callables: best-effort fallback.
                result = fn(args) if args is not None else fn()
                return {"tool_output": result}

            if not params:
                result = fn()
            elif isinstance(args, Mapping):
                # Match keys to named parameters; pass everything if **kwargs
                # is accepted, else only matching keys.
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                if accepts_var_kw:
                    result = fn(**args)
                else:
                    result = fn(**{k: v for k, v in args.items() if k in params})
            elif args is None:
                # No tool_input supplied; only call zero-arg form if signature allows.
                positional_required = [
                    p
                    for p in params.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    and p.default is inspect.Parameter.empty
                ]
                result = fn() if not positional_required else fn(None)
            else:
                result = fn(args)
            return {"tool_output": result}

        return tool_node


def tool_node(**kwargs: Any) -> ToolFactory:
    return ToolFactory(**kwargs)
