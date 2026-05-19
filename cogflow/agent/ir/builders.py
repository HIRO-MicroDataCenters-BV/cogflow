"""Helpers for assembling an IRGraph from the StateGraph facade."""

from __future__ import annotations

import itertools

from .model import IREdge, IRGraph, IRNode


_counter = itertools.count()


def fresh_edge_id(source: str, target: str, source_handle: str | None) -> str:
    suffix = next(_counter)
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
        id=fresh_edge_id(source, target, source_handle),
        source=source,
        target=target,
        source_handle=source_handle,
        target_handle=target_handle,
        label=label,
        is_human_input=is_human_input,
    )
    graph.edges.append(edge)
    return edge
