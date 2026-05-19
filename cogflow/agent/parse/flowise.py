"""Flowise V2 AgentFlow JSON -> IR.

Strategy: extract the runtime-relevant fields (id, position, type, label,
inputs) into first-class IR fields; store the full original node/edge dicts in
``flowise_provenance`` so emitter can reconstruct byte-for-byte JSON later.

The reverse direction is in ``cogflow.agent.compile.to_flowise``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..constants import FLOWISE_NAME_TO_IR_TYPE
from ..ir.model import (
    IREdge,
    IRGraph,
    IRNode,
    IRPort,
    IRPosition,
    IRStateField,
)


def _ir_type_for(name: str) -> str:
    if name not in FLOWISE_NAME_TO_IR_TYPE:
        # Unknown node names map to ``unknown`` so the compiler treats them as
        # passthrough runtime nodes (keeps surrounding edges flowing) while
        # ``flowise_provenance`` preserves the original JSON for round-trip.
        return "unknown"
    return FLOWISE_NAME_TO_IR_TYPE[name]


def _coerce_inputs(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _ports(raw: list[Any] | None) -> list[IRPort]:
    if not raw:
        return []
    out: list[IRPort] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        out.append(
            IRPort(
                id=entry.get("id", ""),
                name=entry.get("name", ""),
                label=entry.get("label"),
                type=entry.get("type", "default"),
            )
        )
    return out


def _state_from_start(inputs: dict[str, Any]) -> list[IRStateField]:
    raw = inputs.get("startState")
    out: list[IRStateField] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key")
            if not key:
                continue
            out.append(IRStateField(key=key, default=entry.get("value")))

    # Always normalize the ``messages`` field to type=messages + add_messages.
    # Whether the user declared it explicitly or not, LangGraph needs the
    # reducer to preserve chat semantics across nodes.
    existing = next((f for f in out if f.key == "messages"), None)
    if existing is None:
        out.append(IRStateField(key="messages", type="messages", reducer="add_messages"))
    else:
        existing.type_ = "messages"
        existing.reducer = "add_messages"
    return out


def from_dict(raw: dict[str, Any]) -> IRGraph:
    """Parse a Flowise V2 AgentFlow JSON dict into an IRGraph."""
    graph = IRGraph(
        description=raw.get("description"),
        viewport=raw.get("viewport"),
        flowise_provenance={
            "usecases": raw.get("usecases"),
            # keep any unknown top-level keys
            "extra": {k: v for k, v in raw.items() if k not in {"description", "usecases", "nodes", "edges", "viewport"}},
        },
    )

    state_set = False
    for raw_node in raw.get("nodes") or []:
        data = raw_node.get("data") or {}
        name = data.get("name") or ""
        ir_type = _ir_type_for(name)
        inputs = _coerce_inputs(data.get("inputs"))
        position = raw_node.get("position") or {}

        if ir_type == "start" and not state_set:
            graph.state = _state_from_start(inputs)
            state_set = True

        node = IRNode(
            id=raw_node.get("id") or data.get("id") or "",
            type=ir_type,
            label=data.get("label") or raw_node.get("id") or "",
            config=dict(inputs),
            inputs=_ports(data.get("inputAnchors")),
            outputs=_ports(data.get("outputAnchors")),
            position=IRPosition(x=position.get("x", 0.0), y=position.get("y", 0.0)),
            flowise_provenance={"node": raw_node},
        )
        graph.nodes.append(node)
        if ir_type == "start" and graph.entry is None:
            graph.entry = node.id

    for raw_edge in raw.get("edges") or []:
        data = raw_edge.get("data") or {}
        graph.edges.append(
            IREdge(
                id=raw_edge.get("id", ""),
                source=raw_edge.get("source", ""),
                target=raw_edge.get("target", ""),
                source_handle=raw_edge.get("sourceHandle"),
                target_handle=raw_edge.get("targetHandle"),
                # ``edgeLabel`` carries the Condition / ConditionAgent branch
                # name. ``compile.to_langgraph`` reads ``edge.label`` to build
                # ``add_conditional_edges`` routing, so dropping this would
                # silently mis-route imported conditional flows.
                label=data.get("edgeLabel"),
                is_human_input=bool(data.get("isHumanInput")),
                flowise_provenance={"edge": raw_edge},
            )
        )

    # If a chat-style graph has no explicit ``messages`` reducer in the Start
    # node, still ensure runtime state has one (mirrors LangGraph defaults).
    if not state_set and not any(f.key == "messages" for f in graph.state):
        graph.state.append(IRStateField(key="messages", type="messages", reducer="add_messages"))

    return graph


def from_file(path: str | Path) -> IRGraph:
    """Read a Flowise V2 JSON file and parse it into an IRGraph."""
    p = Path(path)
    return from_dict(json.loads(p.read_text()))


__all__ = ["from_dict", "from_file"]
