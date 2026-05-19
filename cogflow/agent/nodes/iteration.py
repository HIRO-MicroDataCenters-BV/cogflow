"""Iteration node — fan-out over a collection.

Maps Flowise's ``iterationAgentflow`` to LangGraph's ``Send`` API: the node
reads ``over`` from state (a list or callable) and returns a list of
``Send(child_node, item_state)`` so each item is processed in parallel.

Reduction of per-item results is governed by the state schema's reducers
(typically ``add_messages`` or ``operator.add``). If the child node isn't
present in the graph this node degrades to a no-op so JSON imports of
malformed flows don't crash.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from ..ir.model import IRNode
from .base import NodeFactory


class IterationFactory(NodeFactory):
    ir_type = "iteration"

    def __init__(
        self,
        *,
        over: str = "items",
        body: str | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            iterationOver=over,
            iterationBody=body or "",
            **extra,
        )

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        # Resolve ``Send`` lazily so the module can be imported on older
        # langgraph builds that pre-date the API; iteration just won't fan
        # out there (we degrade to a passthrough).
        try:
            from langgraph.types import Send  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover
            Send = None  # type: ignore[assignment]

        over_key = self.config.get("iterationOver", "items") or "items"
        body_node_id = self.config.get("iterationBody") or None

        def iteration_node(state: dict[str, Any]) -> Any:
            if Send is None or not body_node_id:
                return {}
            raw = state.get(over_key)
            items: Iterable[Any]
            if callable(raw):
                items = raw(state)
            elif isinstance(raw, (list, tuple)):
                items = raw
            else:
                items = []
            return [Send(body_node_id, {**state, "_iteration_item": item}) for item in items]

        return iteration_node


def iteration(**kwargs: Any) -> IterationFactory:
    return IterationFactory(**kwargs)
