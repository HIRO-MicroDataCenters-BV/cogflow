"""Topology validation for IR graphs."""

from __future__ import annotations

from .model import IRGraph

# Sentinel target for edges that terminate the graph. Matches LangGraph's
# ``END`` (``"__end__"``) so the compile path can translate without a lookup.
END_SENTINEL = "__end__"


class IRValidationError(ValueError):
    """Raised when an IRGraph fails topology validation."""


def validate(graph: IRGraph) -> None:
    """Raise IRValidationError if the graph is structurally invalid."""

    node_ids = {n.id for n in graph.nodes}

    if len(node_ids) != len(graph.nodes):
        raise IRValidationError("duplicate node ids in graph")

    starts = [n for n in graph.nodes if n.type == "start"]
    if len(starts) > 1:
        raise IRValidationError(f"expected at most one Start node, found {len(starts)}")
    if graph.nodes and not starts and graph.entry is None:
        raise IRValidationError("graph has nodes but no Start node and no entry override")

    if graph.entry is not None and graph.entry not in node_ids:
        raise IRValidationError(f"entry node id {graph.entry!r} not in graph")

    for fid in graph.finish:
        if fid not in node_ids:
            raise IRValidationError(f"finish node id {fid!r} not in graph")

    edge_ids: set[str] = set()
    for e in graph.edges:
        if e.id in edge_ids:
            raise IRValidationError(f"duplicate edge id {e.id!r}")
        edge_ids.add(e.id)
        if e.source not in node_ids:
            raise IRValidationError(f"edge {e.id!r} source {e.source!r} missing")
        # The END sentinel is a legitimate edge target for conditional branches
        # that terminate; it doesn't correspond to a node.
        if e.target != END_SENTINEL and e.target not in node_ids:
            raise IRValidationError(f"edge {e.id!r} target {e.target!r} missing")
