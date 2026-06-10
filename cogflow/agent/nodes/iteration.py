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

import json
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..ir.model import IRNode
from .base import NodeFactory

_HTML_TAG = re.compile(r"<[^>]+>")


def _coerce_literal_source(source: Any) -> list[Any]:
    """Best-effort coerce a Flowise ``iterationInput`` value to a Python list.

    Flowise stores the field as rich-text HTML around a JSON string. We strip
    the tags then try JSON; on any failure fall back to a single-item list.
    """
    if isinstance(source, (list, tuple)):
        return list(source)
    if not isinstance(source, str):
        return [source]
    stripped = _HTML_TAG.sub("", source).strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return [stripped]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


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
        # Resolve the body node id: explicit Python-first ``iterationBody``
        # config wins; otherwise fall back to the value the parser stashed
        # in ``flowise_provenance.iteration_body_id`` (derived from
        # ``parentNode`` references at import time).
        body_node_id = (
            self.config.get("iterationBody") or (node.flowise_provenance or {}).get("iteration_body_id") or None
        )
        # Validate the body id against the set of nodes the compiler is
        # actually registering. Without this guard, a malformed import where
        # ``parentNode`` references a node we skip at compile (e.g., sticky
        # note, or a node type we don't yet support) would emit
        # ``Send(missing_id, ...)`` and LangGraph would raise at runtime.
        runtime_ids = (ctx or {}).get("runtime_node_ids") if ctx else None
        if body_node_id and runtime_ids is not None and body_node_id not in runtime_ids:
            body_node_id = None
        # ``iterationInput`` is Flowise's literal-source field — a stringified
        # JSON array (or rich-text). Used when no state key is supplied;
        # parsed best-effort as JSON, otherwise treated as a single-item list.
        literal_source = self.config.get("iterationInput")

        def _items_from_state(state: dict[str, Any]) -> Iterable[Any]:
            raw = state.get(over_key)
            if callable(raw):
                return raw(state)
            if isinstance(raw, (list, tuple)):
                return raw
            if raw is None and literal_source:
                return _coerce_literal_source(literal_source)
            return []

        def iteration_node(state: dict[str, Any]) -> Any:
            if Send is None or not body_node_id:
                return {}
            items = _items_from_state(state)
            return [Send(body_node_id, {**state, "_iteration_item": item}) for item in items]

        return iteration_node


def iteration(**kwargs: Any) -> IterationFactory:
    return IterationFactory(**kwargs)
