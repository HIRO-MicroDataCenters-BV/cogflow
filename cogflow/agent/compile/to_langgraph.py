"""IR -> langgraph.graph.CompiledStateGraph.

Sticky-note nodes are skipped at compile time. Condition / Condition-Agent
nodes are folded into a ``add_conditional_edges`` registration; their runtime
callable still runs (so users can read ``_condition_branch`` from state),
but routing comes from the IR edge fan-out.

For nodes loaded from JSON we don't have a factory instance carrying live
references to a model or tool callable. Such nodes get a passthrough body —
useful for structure / round-trip tests. Pass ``factories={node_id: factory}``
to ``to_langgraph`` to inject live runtime objects.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from langgraph.graph import END, START, StateGraph

from ..ir.model import IRGraph, IRNode
from ..ir.validate import validate
from ..nodes import FACTORY_BY_IR_TYPE, NodeFactory
from ..state import introspect_state, synthesize_typeddict
from .topo import edges_from


_SKIP_TYPES = {"sticky_note"}
_CONDITION_TYPES = {"condition", "condition_agent"}


def _passthrough(_state: Any) -> dict[str, Any]:
    return {}


def _callable_for(node: IRNode, factories: Mapping[str, NodeFactory] | None) -> Callable[..., Any]:
    if factories and node.id in factories:
        return factories[node.id].to_callable(node)
    factory_cls = FACTORY_BY_IR_TYPE.get(node.type)
    if factory_cls is None:
        return _passthrough
    # Hydrate via the stable factory API rather than reaching into private
    # attributes. Each factory's ``to_callable`` reads its runtime state
    # through ``getattr(self, "_attr", default)`` so a hydrated-from-IR
    # instance behaves as a safe passthrough until a live model/tool is
    # injected via the ``factories=`` parameter.
    instance = factory_cls.from_ir(node)
    return instance.to_callable(node)


def _state_schema(graph: IRGraph) -> type:
    if graph.state:
        return synthesize_typeddict(graph.state)
    # Fall back to a TypedDict carrying ``messages`` only.
    from .. import state as _state_mod  # type: ignore  # noqa: F401
    return synthesize_typeddict(introspect_state(_state_mod.MessagesState))


def to_langgraph(
    graph: IRGraph,
    *,
    factories: Mapping[str, NodeFactory] | None = None,
    state_schema: type | None = None,
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
):
    """Compile an IRGraph into a ``CompiledStateGraph``."""
    validate(graph)

    schema = state_schema or _state_schema(graph)
    builder = StateGraph(schema)

    runtime_nodes = [n for n in graph.nodes if n.type not in _SKIP_TYPES and n.type != "start"]
    runtime_ids = {n.id for n in runtime_nodes}

    for node in runtime_nodes:
        builder.add_node(node.id, _callable_for(node, factories))

    # Entry edge from LangGraph's START sentinel. Two cases:
    #   - ``graph.entry`` points at a Start IR node (the common case from a
    #     Flowise import): the Start node has no runtime body, so we fan out
    #     to whatever it links to.
    #   - ``graph.entry`` points at a regular runtime node (the "entry
    #     override" case ``ir.validate`` allows when no Start node is
    #     present): wire START directly to it so its body runs.
    start_id = graph.entry
    start_wired = False
    if start_id is not None:
        entry_node = graph.node_by_id(start_id)
        if entry_node is not None and entry_node.type == "start":
            for e in edges_from(graph, start_id):
                if e.target in runtime_ids:
                    builder.add_edge(START, e.target)
                    start_wired = True
        elif start_id in runtime_ids:
            builder.add_edge(START, start_id)
            start_wired = True

    # Degenerate case: a Start node with no runtime targets (e.g., a Flowise
    # import containing only a Start node, or Start fanning into nodes we
    # skip at compile). LangGraph would refuse to compile without any START
    # edge, so wire a safe START → END so the graph still produces a valid
    # CompiledStateGraph that immediately terminates.
    if not start_wired and not runtime_nodes:
        builder.add_edge(START, END)
    elif not start_wired and runtime_nodes:
        # Fallback for malformed graphs whose entry doesn't reach any runtime
        # node: enter the first declared runtime node so something runs.
        builder.add_edge(START, runtime_nodes[0].id)

    # Runtime edges. Condition nodes use add_conditional_edges; everything else
    # is a plain add_edge.
    finish_ids = {fid for fid in graph.finish if fid in runtime_ids}
    end_edged: set[str] = set()  # nodes already wired to END (avoid duplicates)

    for node in runtime_nodes:
        # Only route to nodes actually registered with the builder. Edges that
        # target the Start (or skipped) node are dropped — looping back to
        # entry would require routing to LangGraph's ``START`` sentinel, which
        # is not what an edge to a removed Start node means.
        outs = [e for e in edges_from(graph, node.id) if e.target in runtime_ids]
        if not outs:
            builder.add_edge(node.id, END)
            end_edged.add(node.id)
            continue

        if node.type in _CONDITION_TYPES and len(outs) > 1:
            branch_to_target: dict[str, str] = {}
            for e in outs:
                key = e.label or e.target_handle or e.target
                branch_to_target[key] = e.target

            def router(state: dict[str, Any], _mapping: dict[str, str] = branch_to_target) -> str:
                branch = state.get("_condition_branch")
                if isinstance(branch, str) and branch in _mapping:
                    return _mapping[branch]
                return next(iter(_mapping.values()))

            builder.add_conditional_edges(node.id, router)
        else:
            for e in outs:
                builder.add_edge(node.id, e.target)

    # Explicit finish points (``IRGraph.finish``) — wire them to END unless we
    # already did so via the leaf-node path above. A node can be both a finish
    # point AND have outgoing edges; in that case it needs an explicit END edge.
    for fid in finish_ids - end_edged:
        builder.add_edge(fid, END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
        interrupt_after=interrupt_after or [],
    )


__all__ = ["to_langgraph"]
