"""Topology helpers for compilation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ..ir.model import IREdge, IRGraph


def edges_from(graph: IRGraph, source: str) -> list[IREdge]:
    return [e for e in graph.edges if e.source == source]


def edges_to(graph: IRGraph, target: str) -> list[IREdge]:
    return [e for e in graph.edges if e.target == target]


def adjacency(graph: IRGraph) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges:
        adj[e.source].append(e.target)
    return adj


def reachable(graph: IRGraph, start: str) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    adj = adjacency(graph)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adj.get(node, []))
    return seen


def terminal_node_ids(graph: IRGraph, runtime_node_ids: Iterable[str]) -> set[str]:
    """Nodes with no outgoing runtime edges — they should route to END."""
    runtime_set = set(runtime_node_ids)
    has_out: set[str] = set()
    for e in graph.edges:
        if e.source in runtime_set and e.target in runtime_set:
            has_out.add(e.source)
    return runtime_set - has_out


__all__ = ["adjacency", "edges_from", "edges_to", "reachable", "terminal_node_ids"]
