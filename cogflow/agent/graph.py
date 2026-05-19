"""StateGraph facade — LangGraph-shaped API that also captures IR.

Users get a drop-in subclass of ``langgraph.graph.StateGraph``: every method
that mutates the graph (``add_node``, ``add_edge``, ``add_conditional_edges``,
``set_entry_point``, ``set_finish_point``) also updates an in-memory
``IRGraph`` accessible via the ``.ir`` attribute. The compiled output is a
real ``langgraph.graph.CompiledStateGraph``.

Phase 1 behavior:
 - Passing a ``NodeFactory`` instance to ``add_node`` captures its IR type and
   config faithfully (so the graph round-trips to Flowise JSON).
 - Passing a raw callable is supported and registered with LangGraph
   unchanged; the IR row is recorded as a ``custom_function`` node, which
   round-trips structurally but won't be runtime-equivalent in Flowise.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Mapping

from langgraph.graph import END, START
from langgraph.graph import StateGraph as _LangGraphStateGraph

from .ir import (
    IREdge,
    IRGraph,
    IRNode,
    IRPort,
    add_edge as _ir_add_edge,
    add_node as _ir_add_node,
)
from .nodes.base import NodeFactory
from .state import introspect_state


def _synthesize_start_node(graph: IRGraph) -> IRNode:
    for n in graph.nodes:
        if n.type == "start":
            return n
    node = IRNode(
        id="__start__",
        type="start",
        label="Start",
        outputs=[
            IRPort(
                id="__start__-output-startAgentflow",
                name="startAgentflow",
                label="Start",
                type="start",
            )
        ],
    )
    graph.nodes.insert(0, node)
    graph.entry = node.id
    return node


class StateGraph(_LangGraphStateGraph):
    """LangGraph-shaped StateGraph that also captures an IRGraph."""

    def __init__(self, state_schema: type | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(state_schema, *args, **kwargs)
        self.ir = IRGraph(state=introspect_state(state_schema))
        self._factories: dict[str, NodeFactory] = {}
        self._node_counter = itertools.count()

    # ------------------------------------------------------------------
    # add_node
    # ------------------------------------------------------------------
    def add_node(self, name: str, node: Any = None, **kwargs: Any) -> "StateGraph":  # type: ignore[override]
        if node is None:
            # langgraph's add_node(fn) form — name omitted
            return super().add_node(name, **kwargs)  # type: ignore[no-any-return]

        if isinstance(node, NodeFactory):
            ir_node = node.emit_ir(name, label=kwargs.pop("label", name))
            self._factories[name] = node
            _ir_add_node(self.ir, ir_node)
            super().add_node(name, node.to_callable(ir_node), **kwargs)
            return self

        # Raw callable — record as custom_function for IR provenance only.
        idx = next(self._node_counter)
        ir_node = IRNode(
            id=name,
            type="custom_function",
            label=kwargs.pop("label", name),
            config={"customFunctionPython": getattr(node, "__name__", f"node_{idx}")},
        )
        _ir_add_node(self.ir, ir_node)
        super().add_node(name, node, **kwargs)
        return self

    # ------------------------------------------------------------------
    # edges
    # ------------------------------------------------------------------
    def add_edge(self, start_key: str, end_key: str) -> "StateGraph":  # type: ignore[override]
        if start_key == START:
            start_node = _synthesize_start_node(self.ir)
            _ir_add_edge(
                self.ir,
                source=start_node.id,
                target=end_key,
                source_handle=f"{start_node.id}-output-startAgentflow",
            )
        elif end_key == END:
            self.ir.finish.append(start_key)
        else:
            _ir_add_edge(self.ir, source=start_key, target=end_key)
        return super().add_edge(start_key, end_key)  # type: ignore[no-any-return]

    def add_conditional_edges(  # type: ignore[override]
        self,
        source: str,
        path: Callable[..., Any],
        path_map: Mapping[str, str] | list[str] | None = None,
        then: str | None = None,
    ) -> "StateGraph":
        mapping = path_map or {}
        if isinstance(mapping, list):
            mapping = {k: k for k in mapping}
        for branch, target in mapping.items():
            if target == END:
                self.ir.finish.append(source)
                continue
            edge = _ir_add_edge(self.ir, source=source, target=target, label=str(branch))
            edge.target_handle = target
        return super().add_conditional_edges(source, path, path_map, then)  # type: ignore[no-any-return]

    def set_entry_point(self, key: str) -> "StateGraph":  # type: ignore[override]
        start_node = _synthesize_start_node(self.ir)
        _ir_add_edge(
            self.ir,
            source=start_node.id,
            target=key,
            source_handle=f"{start_node.id}-output-startAgentflow",
        )
        return super().set_entry_point(key)  # type: ignore[no-any-return]

    def set_finish_point(self, key: str) -> "StateGraph":  # type: ignore[override]
        self.ir.finish.append(key)
        return super().set_finish_point(key)  # type: ignore[no-any-return]
