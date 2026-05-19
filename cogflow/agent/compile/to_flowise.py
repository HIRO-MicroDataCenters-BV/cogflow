"""IR -> Flowise V2 AgentFlow JSON.

When an IR node carries ``flowise_provenance['node']`` (set by the importer),
we re-emit that dict verbatim, overlaying any current ``config`` so user
mutations in IR survive the round-trip. For Python-built nodes without
provenance we synthesize a minimal-but-valid Flowise node block from the IR.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..constants import (
    IR_TYPE_TO_FLOWISE_CATEGORY,
    IR_TYPE_TO_FLOWISE_NAME,
)
from ..ir.model import IREdge, IRGraph, IRNode


def _default_output_handle(graph: IRGraph, node_id: str) -> str:
    """The canonical sourceHandle id Flowise expects on the source node."""
    node = graph.node_by_id(node_id)
    if node is not None and node.outputs:
        return node.outputs[0].id
    flowise_name = IR_TYPE_TO_FLOWISE_NAME.get(node.type, "output") if node else "output"
    return f"{node_id}-output-{flowise_name}"


def _default_target_handle(node_id: str) -> str:
    """Flowise targetHandle defaults to the bare node id."""
    return node_id


def _node_to_dict(node: IRNode) -> dict[str, Any]:
    raw = (node.flowise_provenance or {}).get("node") if node.flowise_provenance else None
    if isinstance(raw, dict):
        result = copy.deepcopy(raw)
        result["id"] = node.id
        pos = {"x": node.position.x, "y": node.position.y}
        result["position"] = pos
        # Flowise reads both ``position`` and ``positionAbsolute`` — keep them
        # in sync so a mutated IR position doesn't leave the canvas inconsistent.
        result["positionAbsolute"] = dict(pos)
        data = result.setdefault("data", {})
        data["id"] = node.id
        data["label"] = node.label
        # Always overlay node.config onto data.inputs (including ``{}``) so
        # IR-side mutations — including intentional clears — survive export.
        data["inputs"] = dict(node.config)
        return result

    flowise_name = IR_TYPE_TO_FLOWISE_NAME[node.type]
    category = IR_TYPE_TO_FLOWISE_CATEGORY[node.type]
    return {
        "id": node.id,
        "type": "stickyNote" if node.type == "sticky_note" else "agentFlow",
        "position": {"x": node.position.x, "y": node.position.y},
        "positionAbsolute": {"x": node.position.x, "y": node.position.y},
        "data": {
            "id": node.id,
            "label": node.label,
            "version": 1,
            "name": flowise_name,
            "type": category,
            "baseClasses": [category],
            "category": "Agent Flows",
            "description": "",
            "inputParams": [],
            "inputAnchors": [
                {"id": p.id, "name": p.name, "label": p.label, "type": p.type_} for p in node.inputs
            ],
            "inputs": node.config or {},
            "outputAnchors": [
                {"id": p.id, "name": p.name, "label": p.label} for p in node.outputs
            ],
            "outputs": {},
            "selected": False,
        },
        "selected": False,
        "dragging": False,
    }


def _edge_to_dict(edge: IREdge, graph: IRGraph) -> dict[str, Any]:
    raw = (edge.flowise_provenance or {}).get("edge") if edge.flowise_provenance else None
    if isinstance(raw, dict):
        result = copy.deepcopy(raw)
        result["id"] = edge.id
        result["source"] = edge.source
        result["target"] = edge.target
        if edge.source_handle is not None:
            result["sourceHandle"] = edge.source_handle
        if edge.target_handle is not None:
            result["targetHandle"] = edge.target_handle
        # Overlay is_human_input onto data.isHumanInput so IR-side mutations
        # survive re-emission of provenance-carrying edges.
        data = result.setdefault("data", {})
        data["isHumanInput"] = edge.is_human_input
        return result

    return {
        "id": edge.id,
        "source": edge.source,
        # Default handles derived from actual node anchors so synthesized edges
        # round-trip cleanly through Flowise (the canvas matches by handle id).
        "sourceHandle": edge.source_handle or _default_output_handle(graph, edge.source),
        "target": edge.target,
        "targetHandle": edge.target_handle or _default_target_handle(edge.target),
        "type": "agentFlow",
        "data": {"isHumanInput": edge.is_human_input},
    }


def _start_state_payload(graph: IRGraph) -> list[dict[str, Any]]:
    """``data.inputs.startState`` shape Flowise expects, derived from IR state."""
    return [
        {"key": f.key, "value": "" if f.default is None else f.default} for f in graph.state
    ]


def to_dict(graph: IRGraph, *, analytics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialize an IRGraph to a Flowise V2 AgentFlow JSON dict."""
    provenance = graph.flowise_provenance or {}
    out: dict[str, Any] = {}
    if graph.description is not None:
        out["description"] = graph.description
    usecases = provenance.get("usecases") if isinstance(provenance, dict) else None
    if usecases is not None:
        out["usecases"] = usecases

    out["nodes"] = [_node_to_dict(n) for n in graph.nodes]
    out["edges"] = [_edge_to_dict(e, graph) for e in graph.edges]

    # Direction B: when the Start node has no Flowise provenance (i.e. the
    # graph was authored in Python), populate its ``data.inputs.startState``
    # from ``graph.state`` so the exported canvas reflects the declared
    # schema. Provenance-backed Start nodes are left untouched to preserve
    # lossless round-trip on JSON-imported graphs.
    if graph.state:
        for ir_node, emitted in zip(graph.nodes, out["nodes"]):
            if ir_node.type != "start":
                continue
            if ir_node.flowise_provenance and "node" in ir_node.flowise_provenance:
                continue
            data = emitted.setdefault("data", {})
            inputs = data.get("inputs")
            if not isinstance(inputs, dict):
                inputs = {}
                data["inputs"] = inputs
            inputs["startState"] = _start_state_payload(graph)
            break

    if graph.viewport is not None:
        out["viewport"] = graph.viewport

    extra = provenance.get("extra") if isinstance(provenance, dict) else None
    if isinstance(extra, dict):
        for k, v in extra.items():
            out.setdefault(k, v)

    if analytics is not None:
        out["analytic"] = analytics

    return out


def to_file(graph: IRGraph, path: str | Path, *, indent: int = 4, **kwargs: Any) -> Path:
    p = Path(path)
    p.write_text(json.dumps(to_dict(graph, **kwargs), indent=indent))
    return p


__all__ = ["to_dict", "to_file"]
