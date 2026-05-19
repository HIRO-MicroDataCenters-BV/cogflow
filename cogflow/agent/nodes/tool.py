"""Tool node — exposes a callable that an Agent node may invoke."""

from __future__ import annotations

from typing import Any, Callable, Mapping

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
        fn = self._fn

        def tool_node(state: dict[str, Any]) -> dict[str, Any]:
            if fn is None:
                return {}
            args = state.get("tool_input") or {}
            try:
                result = fn(**args) if isinstance(args, Mapping) else fn(args)
            except TypeError:
                result = fn()
            return {"tool_output": result}

        return tool_node


def tool_node(**kwargs: Any) -> ToolFactory:
    return ToolFactory(**kwargs)
