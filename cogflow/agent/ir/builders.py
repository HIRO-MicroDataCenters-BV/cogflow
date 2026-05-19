"""Helpers for assembling an IRGraph from the StateGraph facade."""

from __future__ import annotations

from .model import IREdge, IRGraph, IRNode


def fresh_edge_id(graph: IRGraph, source: str, target: str, source_handle: str | None) -> str:
    """Generate an edge id that's deterministic *within* one graph.

    The suffix advances a per-graph counter on ``IRGraph`` so two processes
    building the same graph in the same order produce identical ids — which
    matters for diffing exported Flowise JSON.
    """
    suffix = graph.next_edge_seq()
    sh = source_handle or f"{source}-output"
    return f"{source}-{sh}-{target}-{suffix}"


def add_node(graph: IRGraph, node: IRNode) -> IRNode:
    graph.nodes.append(node)
    if node.type == "start" and graph.entry is None:
        graph.entry = node.id
    return node


def add_edge(
    graph: IRGraph,
    source: str,
    target: str,
    *,
    source_handle: str | None = None,
    target_handle: str | None = None,
    label: str | None = None,
    is_human_input: bool = False,
) -> IREdge:
    edge = IREdge(
        id=fresh_edge_id(graph, source, target, source_handle),
        source=source,
        target=target,
        source_handle=source_handle,
        target_handle=target_handle,
        label=label,
        is_human_input=is_human_input,
    )
    graph.edges.append(edge)
    return edge
