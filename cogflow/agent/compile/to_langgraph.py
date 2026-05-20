"""IR -> langgraph.graph.CompiledStateGraph.

Sticky-note nodes are skipped at compile time. Condition / Condition-Agent
nodes are folded into a ``add_conditional_edges`` registration; their runtime
callable still runs (so users can read ``_condition_branch`` from state),
but routing comes from the IR edge fan-out.

For nodes loaded from JSON we don't have a factory instance carrying live
references to a model or tool callable. Two injection points cover that:

  - ``factories={node_id: factory_instance}`` — per-node, used when you want
    one specific node in an imported graph to have a live model/tool.
  - ``ctx={...}`` — shared by every node, threaded through to every
    factory's ``to_callable(node, ctx=ctx)``. Bridge-node factories use it
    for runtime objects they can't capture at construction time (httpx
    stub, retriever store, allow_custom_code gate, etc.).

Loop nodes additionally rewrite themselves into a conditional back-edge
(target → loop body → … → loop) bounded by ``loopMaxIterations``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from langgraph.graph import END, START, StateGraph

from ..ir.model import IRGraph, IRNode
from ..ir.validate import END_SENTINEL, validate
from ..nodes import FACTORY_BY_IR_TYPE, NodeFactory
from ..state import introspect_state, synthesize_typeddict
from .topo import edges_from


_SKIP_TYPES = {"sticky_note"}
_CONDITION_TYPES = {"condition", "condition_agent"}


def _passthrough(_state: Any) -> dict[str, Any]:
    return {}


def _callable_for(
    node: IRNode,
    factories: Mapping[str, NodeFactory] | None,
    ctx: Mapping[str, Any] | None = None,
) -> Callable[..., Any]:
    if factories and node.id in factories:
        return factories[node.id].to_callable(node, ctx=ctx)
    factory_cls = FACTORY_BY_IR_TYPE.get(node.type)
    if factory_cls is None:
        return _passthrough
    # Hydrate via the stable factory API rather than reaching into private
    # attributes. Each factory's ``to_callable`` reads its runtime state
    # through ``getattr(self, "_attr", default)`` so a hydrated-from-IR
    # instance behaves as a safe passthrough until a live runtime object is
    # injected via either ``factories=`` (per-node) or ``ctx=`` (shared).
    instance = factory_cls.from_ir(node)
    return instance.to_callable(node, ctx=ctx)


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
    ctx: Mapping[str, Any] | None = None,
    state_schema: type | None = None,
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
):
    """Compile an IRGraph into a ``CompiledStateGraph``.

    ``ctx`` is a shared mapping threaded through every node's ``to_callable``.
    Bridge-node factories use it to resolve runtime objects they couldn't
    capture at construction time — e.g., ``ctx={"httpx": stub_client,
    "flows": {flow_id: compiled_graph}, "retriever": store,
    "allow_custom_code": True}``.
    """
    validate(graph)

    schema = state_schema or _state_schema(graph)
    builder = StateGraph(schema)

    runtime_nodes = [n for n in graph.nodes if n.type not in _SKIP_TYPES and n.type != "start"]
    runtime_ids = {n.id for n in runtime_nodes}

    for node in runtime_nodes:
        builder.add_node(node.id, _callable_for(node, factories, ctx))

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

    # Runtime edges. Condition nodes use add_conditional_edges; loop nodes
    # use a conditional back-edge bounded by their max-iterations counter;
    # everything else is a plain add_edge.
    finish_ids = {fid for fid in graph.finish if fid in runtime_ids}
    end_edged: set[str] = set()  # nodes already wired to END (avoid duplicates)
    routed_via_router: set[str] = set()  # nodes that already have a conditional router

    for node in runtime_nodes:
        if node.type == "loop":
            # Accept both Python-first keys (loopTarget / loopMaxIterations)
            # and Flowise-native ones (loopBackToNode / maxLoopCount). The
            # Flowise value encodes ``{node_id}-{label}``; strip the label
            # suffix so the routing actually finds the runtime node.
            raw_target = (
                node.config.get("loopTarget")
                or node.config.get("loopBackToNode")
                or ""
            )
            if raw_target and "-" in raw_target and raw_target not in runtime_ids:
                raw_target = raw_target.split("-", 1)[0]
            target = raw_target
            max_iters = int(
                node.config.get("loopMaxIterations")
                or node.config.get("maxLoopCount")
                or 5
            )
            counter_key = f"_loop_count__{node.id}"
            # Follow the loop's normal outgoing edge (if any) when the cap is hit.
            exit_target: Any = END
            for e in edges_from(graph, node.id):
                if e.target in runtime_ids and e.target != target:
                    exit_target = e.target
                    break
                if e.target == END_SENTINEL:
                    exit_target = END
                    break

            def _loop_router(
                state: dict[str, Any],
                _counter_key: str = counter_key,
                _max: int = max_iters,
                _target: str = target,
                _exit: Any = exit_target,
            ) -> Any:
                if int(state.get(_counter_key, 0) or 0) < _max and _target:
                    return _target
                return _exit

            builder.add_conditional_edges(node.id, _loop_router)
            routed_via_router.add(node.id)
            if exit_target == END:
                end_edged.add(node.id)
            continue
        # Only route to nodes actually registered with the builder, plus the
        # END sentinel for branches that terminate. Edges that target the
        # Start (or skipped) node are dropped — looping back to entry would
        # require routing to LangGraph's ``START`` sentinel, which is not what
        # an edge to a removed Start node means.
        outs = [
            e
            for e in edges_from(graph, node.id)
            if e.target in runtime_ids or e.target == END_SENTINEL
        ]
        if not outs:
            builder.add_edge(node.id, END)
            end_edged.add(node.id)
            continue

        if node.type in _CONDITION_TYPES and len(outs) > 1:
            branch_to_target: dict[str, Any] = {}
            for e in outs:
                key = e.label or e.target_handle or e.target
                # Translate the IR END sentinel back to LangGraph's END object
                # so only this specific branch terminates (not the whole node).
                branch_to_target[key] = END if e.target == END_SENTINEL else e.target

            def router(state: dict[str, Any], _mapping: dict[str, Any] = branch_to_target) -> Any:
                branch = state.get("_condition_branch")
                if isinstance(branch, str) and branch in _mapping:
                    return _mapping[branch]
                return next(iter(_mapping.values()))

            builder.add_conditional_edges(node.id, router)
            routed_via_router.add(node.id)
        else:
            for e in outs:
                target = END if e.target == END_SENTINEL else e.target
                builder.add_edge(node.id, target)
                if target == END:
                    end_edged.add(node.id)

    # Explicit finish points (``IRGraph.finish``) — wire them to END unless we
    # already did so via the leaf-node path above. A node can be both a finish
    # point AND have outgoing edges; in that case it needs an explicit END edge.
    # Skip any node that already has a conditional router — LangGraph forbids
    # mixing an unconditional and conditional outgoing edge on the same node.
    for fid in finish_ids - end_edged - routed_via_router:
        builder.add_edge(fid, END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
        interrupt_after=interrupt_after or [],
    )


__all__ = ["to_langgraph"]
