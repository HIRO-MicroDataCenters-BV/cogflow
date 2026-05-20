"""IR -> standalone LangGraph Python source.

Emits a self-contained ``.py`` file that builds the same graph using only
``langgraph`` and ``langchain_core`` — no runtime dependency on
``cogflow.agent``. Useful for:

  - shipping an agent to an environment that has only the upstream
    LangGraph stack installed;
  - inspecting the generated structure during debugging;
  - serving as a starting point for hand-tuning (the output is ordinary
    Python that runs through ``black``/``ruff`` cleanly).

The contract is best-effort, not lossless: the emitted file always
produces a valid ``CompiledStateGraph`` for the MVP-7 nodes (Start, LLM,
Agent, Tool, Condition, ConditionAgent, DirectReply) plus Loop and
HumanInput, which have clean LangGraph mappings. Bridge nodes with
opaque runtime semantics (HTTP, Retriever, CustomFunction body, ExecuteFlow,
Iteration) emit a clearly-marked TODO stub so the user knows where to
wire their own implementation. StickyNote nodes are skipped entirely.
Unknown nodes compile as passthrough runtime nodes so surrounding edges
keep flowing — same contract as ``compile.to_langgraph`` and documented
on the IR.

Public entry points:

    cogflow.agent.compile.to_python.to_source(graph) -> str
    cogflow.agent.compile.to_python.to_file(graph, path) -> Path
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..ir.model import IRGraph, IRNode, IRStateField
from ..ir.validate import END_SENTINEL, validate
from .topo import edges_from


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


_PY_TYPE_FOR_STATE: dict[str, str] = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "list": "list",
    "dict": "dict",
    "messages": "list",
}


def _state_typeddict(fields: list[IRStateField]) -> str:
    """Render the IR state as a TypedDict class definition."""
    if not fields:
        return (
            "class FlowState(TypedDict, total=False):\n"
            "    messages: Annotated[list, add_messages]\n"
        )
    lines = ["class FlowState(TypedDict, total=False):"]
    for f in fields:
        py = _PY_TYPE_FOR_STATE.get(f.type_, "Any")
        if f.reducer == "add_messages" or f.type_ == "messages":
            lines.append(f"    {f.key}: Annotated[list, add_messages]")
        elif f.reducer == "operator.add":
            lines.append(f"    {f.key}: Annotated[{py}, operator.add]")
        else:
            lines.append(f"    {f.key}: {py}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Per-node-type body renderers
# ---------------------------------------------------------------------------


def _py_literal(value: Any) -> str:
    """Best-effort repr for emitting into source. Falls back to repr()."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    return repr(value)


def _render_passthrough(node: IRNode) -> str:
    return (
        f"def {_node_fn(node.id)}(state: FlowState) -> dict:\n"
        f"    \"\"\"{node.type} node — runtime is a no-op.\"\"\"\n"
        f"    return {{}}\n"
    )


def _render_direct_reply(node: IRNode) -> str:
    # Coerce to ``str`` — partially-populated IRs (or imported flows with
    # an explicit ``null``) can carry ``directReplyMessage: None``, and the
    # emitted code would otherwise call ``None.format(...)`` and crash.
    raw = node.config.get("directReplyMessage")
    message = raw if isinstance(raw, str) else ""
    # Catch ValueError too — str.format raises it for malformed templates
    # (unmatched braces, replacement-field syntax errors), which happens
    # whenever the user's reply text legitimately contains a ``{`` or ``}``.
    return (
        f"def {_node_fn(node.id)}(state: FlowState) -> dict:\n"
        f"    rendered = {message!r}\n"
        f"    try:\n"
        f"        rendered = {message!r}.format(**state)\n"
        f"    except (KeyError, IndexError, ValueError):\n"
        f"        pass\n"
        f"    return {{'messages': [AIMessage(content=rendered)] }} if rendered else {{}}\n"
    )


def _render_llm(node: IRNode) -> str:
    return (
        f"def {_node_fn(node.id)}(state: FlowState) -> dict:\n"
        f"    \"\"\"LLM node — wire your ChatModel via the ``MODEL`` symbol.\"\"\"\n"
        f"    if MODEL is None:\n"
        f"        return {{}}\n"
        f"    history = list(state.get('messages') or [])\n"
        f"    response = MODEL.invoke(history)\n"
        f"    return {{'messages': [response]}}\n"
    )


def _render_agent(node: IRNode) -> str:
    return (
        f"def {_node_fn(node.id)}(state: FlowState) -> dict:\n"
        f"    \"\"\"Agent node — bind tools via ``TOOLS`` and a model via ``MODEL``.\"\"\"\n"
        f"    if MODEL is None:\n"
        f"        return {{}}\n"
        f"    bound = MODEL.bind_tools(TOOLS) if TOOLS and hasattr(MODEL, 'bind_tools') else MODEL\n"
        f"    response = bound.invoke(list(state.get('messages') or []))\n"
        f"    return {{'messages': [response]}}\n"
    )


def _render_tool(node: IRNode) -> str:
    return (
        f"def {_node_fn(node.id)}(state: FlowState) -> dict:\n"
        f"    \"\"\"Tool node — replace with your @tool function.\"\"\"\n"
        f"    return {{}}\n"
    )


def _render_condition(node: IRNode) -> str:
    return (
        f"def {_node_fn(node.id)}(state: FlowState) -> dict:\n"
        f"    \"\"\"Condition node — populates `_condition_branch` for routing.\"\"\"\n"
        f"    # TODO: implement rule evaluation; ``add_conditional_edges`` reads\n"
        f"    # the returned ``_condition_branch`` value.\n"
        f"    return {{'_condition_branch': 'default'}}\n"
    )


def _render_loop(node: IRNode) -> str:
    counter_key = f"_loop_count__{node.id}"
    return (
        f"def {_node_fn(node.id)}(state: FlowState) -> dict:\n"
        f"    \"\"\"Loop node — increments a per-node counter; routing wired below.\"\"\"\n"
        f"    current = int(state.get({counter_key!r}, 0) or 0)\n"
        f"    return {{{counter_key!r}: current + 1}}\n"
    )


def _render_human_input(node: IRNode) -> str:
    prompt = node.config.get("humanInputPrompt") or node.config.get("humanInputDescription") or ""
    return (
        f"def {_node_fn(node.id)}(state: FlowState) -> dict:\n"
        f"    \"\"\"HumanInput node — pauses for user input via ``interrupt``.\"\"\"\n"
        f"    reply = interrupt({{'prompt': {prompt!r}, 'node': {node.id!r}}})\n"
        f"    return {{'human_input': reply}}\n"
    )


def _render_todo(node: IRNode) -> str:
    """Bridge nodes whose runtime body is too domain-specific to auto-emit."""
    return (
        f"def {_node_fn(node.id)}(state: FlowState) -> dict:\n"
        f"    # TODO: cogflow.agent bridge node `{node.type}` — implement manually.\n"
        f"    # IR config: {dict(node.config)!r}\n"
        f"    return {{}}\n"
    )


_RENDERERS: dict[str, Callable[[IRNode], str]] = {
    "llm": _render_llm,
    "agent": _render_agent,
    "tool": _render_tool,
    "condition": _render_condition,
    "condition_agent": _render_condition,  # same shape: writes _condition_branch
    "direct_reply": _render_direct_reply,
    "loop": _render_loop,
    "human_input": _render_human_input,
    "iteration": _render_todo,
    "http": _render_todo,
    "retriever": _render_todo,
    "custom_function": _render_todo,
    "execute_flow": _render_todo,
}


def _node_fn(node_id: str) -> str:
    """Convert an IR node id to a valid Python identifier.

    Naive char-replacement collides on ids like ``a-b`` vs ``a_b`` (both
    become ``node_a_b``). Append a short deterministic hash of the raw id
    whenever sanitization changes anything, so distinct ids always produce
    distinct function names.
    """
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in node_id)
    if sanitized != node_id:
        # Use blake2b for a short, stable, dependency-free suffix.
        import hashlib

        suffix = hashlib.blake2b(node_id.encode("utf-8"), digest_size=4).hexdigest()
        sanitized = f"{sanitized}__{suffix}"
    if sanitized and sanitized[0].isdigit():
        sanitized = "n_" + sanitized
    return f"node_{sanitized}"


# ---------------------------------------------------------------------------
# Edge wiring
# ---------------------------------------------------------------------------


# Mirror ``compile.to_langgraph``: only ``sticky_note`` and ``start`` are
# excluded from the runtime set. ``unknown`` nodes get a passthrough body so
# surrounding edges keep flowing (matches the IR contract documented in
# ``ir/model.py`` and the parse/import path).
_SKIP_TYPES = {"sticky_note", "start"}
_CONDITION_TYPES = {"condition", "condition_agent"}


def _render_edges(graph: IRGraph, runtime_ids: set[str]) -> list[str]:
    lines: list[str] = []

    # Entry edge from START.
    start_id = graph.entry
    start_wired = False
    if start_id is not None:
        entry_node = graph.node_by_id(start_id)
        if entry_node is not None and entry_node.type == "start":
            for e in edges_from(graph, start_id):
                if e.target in runtime_ids:
                    lines.append(f"builder.add_edge(START, {e.target!r})")
                    start_wired = True
        elif start_id in runtime_ids:
            lines.append(f"builder.add_edge(START, {start_id!r})")
            start_wired = True

    if not start_wired and not runtime_ids:
        lines.append("builder.add_edge(START, END)")
    elif not start_wired and runtime_ids:
        first = sorted(runtime_ids)[0]
        lines.append(f"builder.add_edge(START, {first!r})")

    routed: set[str] = set()
    ended: set[str] = set()

    for node in graph.nodes:
        if node.id not in runtime_ids:
            continue

        if node.type == "loop":
            target = (
                node.config.get("loopTarget")
                or node.config.get("loopBackToNode")
                or ""
            )
            if target and "-" in target and target not in runtime_ids:
                target = target.split("-", 1)[0]
            if target and target not in runtime_ids:
                target = ""
            max_iters = int(
                node.config.get("loopMaxIterations") or node.config.get("maxLoopCount") or 5
            )
            counter_key = f"_loop_count__{node.id}"
            # Walk outgoing edges in order, accepting the first eligible exit:
            # an explicit END edge (END_SENTINEL) or a non-target runtime edge.
            # Mirrors ``compile.to_langgraph``'s scan so the two emit paths
            # don't diverge based on edge ordering.
            exit_target_repr = "END"
            for e in edges_from(graph, node.id):
                if e.target == END_SENTINEL:
                    exit_target_repr = "END"
                    break
                if e.target in runtime_ids and e.target != target:
                    exit_target_repr = repr(e.target)
                    break
            lines.append(
                f"def _{_node_fn(node.id)}_router(state, "
                f"_k={counter_key!r}, _max={max_iters!r}, _t={target!r}, _exit={exit_target_repr}):\n"
                f"    return _t if (_t and int(state.get(_k, 0) or 0) < _max) else _exit"
            )
            lines.append(f"builder.add_conditional_edges({node.id!r}, _{_node_fn(node.id)}_router)")
            routed.add(node.id)
            if exit_target_repr == "END":
                ended.add(node.id)
            continue

        outs = [e for e in edges_from(graph, node.id) if e.target in runtime_ids or e.target == END_SENTINEL]
        if not outs:
            lines.append(f"builder.add_edge({node.id!r}, END)")
            ended.add(node.id)
            continue

        if node.type in _CONDITION_TYPES and len(outs) > 1:
            mapping_items = []
            for e in outs:
                key = e.label or e.target_handle or e.target
                value = "END" if e.target == END_SENTINEL else repr(e.target)
                mapping_items.append(f"    {key!r}: {value}")
            mapping_body = ",\n".join(mapping_items)
            router_name = f"_{_node_fn(node.id)}_router"
            lines.append(
                f"def {router_name}(state):\n"
                f"    _mapping = {{\n{mapping_body},\n    }}\n"
                f"    branch = state.get('_condition_branch')\n"
                f"    return _mapping[branch] if isinstance(branch, str) and branch in _mapping else next(iter(_mapping.values()))"
            )
            lines.append(f"builder.add_conditional_edges({node.id!r}, {router_name})")
            routed.add(node.id)
        else:
            for e in outs:
                target_repr = "END" if e.target == END_SENTINEL else repr(e.target)
                lines.append(f"builder.add_edge({node.id!r}, {target_repr})")
                if e.target == END_SENTINEL:
                    ended.add(node.id)

    for fid in graph.finish:
        if fid in runtime_ids and fid not in ended and fid not in routed:
            lines.append(f"builder.add_edge({fid!r}, END)")

    return lines


# ---------------------------------------------------------------------------
# Top-level emit
# ---------------------------------------------------------------------------


_HEADER = '''\
"""Auto-generated from a cogflow.agent IRGraph via compile.to_python.

Edit by hand at your own risk — re-emitting will overwrite this file.
"""

# Note: no ``from __future__ import annotations`` — LangGraph's TypedDict
# introspection (``get_type_hints``) needs the ``Annotated`` symbol bound
# in the module's globalns at evaluation time, which only works when
# annotations stay non-string at class-definition time.

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph import add_messages

try:
    from langgraph.types import interrupt
except ImportError:  # older langgraph
    def interrupt(payload):
        raise RuntimeError("langgraph.types.interrupt not available")


# Wire your model / tools here before invoking ``app``.
MODEL: Any = None
TOOLS: list[Any] = []

'''


import keyword as _kw


def to_source(graph: IRGraph, *, app_var: str = "app") -> str:
    """Render an IRGraph as a self-contained Python module string."""
    # ``app_var`` is interpolated verbatim into the emitted source; refuse
    # anything that isn't a bare identifier so the contract ("output is
    # always valid Python") holds even if the caller passes a hostile value.
    if not isinstance(app_var, str) or not app_var.isidentifier() or _kw.iskeyword(app_var):
        raise ValueError(
            f"app_var must be a valid non-keyword Python identifier; got {app_var!r}"
        )
    validate(graph)

    runtime_nodes = [n for n in graph.nodes if n.type not in _SKIP_TYPES]
    runtime_ids = {n.id for n in runtime_nodes}

    parts: list[str] = [_HEADER, _state_typeddict(graph.state), "\n"]

    for node in runtime_nodes:
        renderer = _RENDERERS.get(node.type, _render_passthrough)
        parts.append(renderer(node))
        parts.append("\n")

    parts.append("builder = StateGraph(FlowState)\n")
    for node in runtime_nodes:
        parts.append(f"builder.add_node({node.id!r}, {_node_fn(node.id)})\n")
    parts.append("\n")

    edge_lines = _render_edges(graph, runtime_ids)
    for line in edge_lines:
        parts.append(line + "\n")
    parts.append("\n")
    parts.append(f"{app_var} = builder.compile()\n")

    return "".join(parts)


def to_file(graph: IRGraph, path: str | Path, *, app_var: str = "app") -> Path:
    """Write the rendered source to ``path`` and return the Path."""
    out = Path(path)
    out.write_text(to_source(graph, app_var=app_var))
    return out


__all__ = ["to_source", "to_file"]
