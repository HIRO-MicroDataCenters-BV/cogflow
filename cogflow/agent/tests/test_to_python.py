"""Tests for cogflow.agent.compile.to_python source emit.

Each test renders the IR to a Python module string, ensures it parses
(``compile()``), then ``exec``s it in a clean namespace and asserts the
resulting ``app`` is a real ``CompiledStateGraph`` you can invoke without
cogflow.agent imported.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from cogflow.agent.compile import to_python
from cogflow.agent.ir.model import IREdge, IRGraph, IRNode, IRPort, IRStateField
from cogflow.agent.parse.flowise import from_file


_FAKE_MODULE_COUNTER = 0


def _exec_module(source: str) -> dict[str, Any]:
    """Compile + exec the rendered source in a fresh module-style namespace.

    LangGraph's ``get_type_hints`` resolves forward refs through the class's
    ``__module__``, so a bare ``exec(src, {})`` fails to find ``Annotated``
    even when it's imported in the source. We register the namespace as a
    fake module in ``sys.modules`` so name resolution works.
    """
    import sys
    import types

    global _FAKE_MODULE_COUNTER
    _FAKE_MODULE_COUNTER += 1
    mod_name = f"_cogflow_to_python_emitted_{_FAKE_MODULE_COUNTER}"
    module = types.ModuleType(mod_name)
    sys.modules[mod_name] = module
    code = compile(source, mod_name, "exec")
    exec(code, module.__dict__)
    return module.__dict__


def _basic_graph() -> IRGraph:
    """Minimal IR: Start → DirectReply (returns 'hi')."""
    start = IRNode(
        id="start_0",
        type="start",
        label="Start",
        outputs=[IRPort(id="start_0-output-start", name="startAgentflow", type="start")],
    )
    reply = IRNode(
        id="reply_0",
        type="direct_reply",
        label="Reply",
        config={"directReplyMessage": "hi"},
    )
    edge = IREdge(id="e0", source="start_0", target="reply_0")
    return IRGraph(
        state=[IRStateField(key="messages", type="messages", reducer="add_messages")],
        nodes=[start, reply],
        edges=[edge],
        entry="start_0",
    )


def test_to_source_is_syntactically_valid_python():
    src = to_python.to_source(_basic_graph())
    # Parse-only check first so any syntax issue surfaces with a clear error.
    ast.parse(src)


def test_to_source_produces_compiled_state_graph():
    src = to_python.to_source(_basic_graph())
    ns = _exec_module(src)
    assert "app" in ns
    assert "FlowState" in ns
    # The compiled object has an ``invoke`` method (LangGraph runnable).
    assert hasattr(ns["app"], "invoke")


def test_emitted_module_runs_end_to_end():
    """Invoke the exec'd app and verify the DirectReply message lands."""
    src = to_python.to_source(_basic_graph())
    ns = _exec_module(src)
    result = ns["app"].invoke({"messages": []})
    out_messages = result.get("messages", [])
    contents = [getattr(m, "content", "") for m in out_messages]
    assert "hi" in contents


def test_to_file_writes_to_disk(tmp_path: Path):
    out = to_python.to_file(_basic_graph(), tmp_path / "agent.py")
    assert out.is_file()
    text = out.read_text()
    assert "builder = StateGraph(FlowState)" in text
    assert "app = builder.compile()" in text


def test_state_synthesis_emits_add_messages_reducer():
    src = to_python.to_source(_basic_graph())
    assert "Annotated[list, add_messages]" in src


def test_simple_rag_fixture_compiles_to_python(simple_rag_path: Path):
    """A real Flowise marketplace fixture should emit valid Python."""
    ir = from_file(simple_rag_path)
    src = to_python.to_source(ir)
    ast.parse(src)  # syntactic validity
    ns = _exec_module(src)
    assert hasattr(ns["app"], "invoke")


def test_iterations_fixture_emits_todo_for_bridge_nodes(iterations_path: Path):
    """Bridge-node types must emit a TODO marker so users see what's stubbed."""
    ir = from_file(iterations_path)
    src = to_python.to_source(ir)
    assert "TODO: cogflow.agent bridge node `iteration`" in src
    # And the file is still syntactically valid Python.
    ast.parse(src)


def test_sticky_note_and_unknown_nodes_are_skipped():
    """Sticky notes and unknown types must not generate node functions or edges."""
    start = IRNode(id="s0", type="start", label="Start")
    sticky = IRNode(id="sticky_0", type="sticky_note", label="Note", config={"note": "hello"})
    reply = IRNode(id="r0", type="direct_reply", label="Reply", config={"directReplyMessage": "bye"})
    graph = IRGraph(
        nodes=[start, sticky, reply],
        edges=[IREdge(id="e0", source="s0", target="r0")],
        entry="s0",
    )
    src = to_python.to_source(graph)
    assert "sticky_0" not in src  # not registered as a node
    assert "node_r0" in src       # the real reply node IS rendered


def test_conditional_branch_to_end_emits_END_in_mapping():
    """Conditional routing with an END branch must translate to LangGraph's END."""
    from cogflow.agent.ir.validate import END_SENTINEL

    start = IRNode(id="s0", type="start", label="Start")
    cond = IRNode(id="c0", type="condition", label="Route")
    other = IRNode(id="o0", type="direct_reply", label="Other", config={"directReplyMessage": "x"})
    graph = IRGraph(
        nodes=[start, cond, other],
        edges=[
            IREdge(id="e0", source="s0", target="c0"),
            IREdge(id="e1", source="c0", target="o0", label="go"),
            IREdge(id="e2", source="c0", target=END_SENTINEL, label="stop"),
        ],
        entry="s0",
    )
    src = to_python.to_source(graph)
    assert "'go': 'o0'" in src
    assert "'stop': END" in src
    # Confirm the emitted source still parses + execs cleanly.
    ns = _exec_module(src)
    assert hasattr(ns["app"], "invoke")


def test_node_id_with_special_chars_is_sanitized():
    """Function names must be valid Python identifiers regardless of node id shape."""
    start = IRNode(id="start", type="start", label="Start")
    weird = IRNode(
        id="agent-with-dashes.and.dots",
        type="direct_reply",
        label="W",
        config={"directReplyMessage": "ok"},
    )
    graph = IRGraph(
        nodes=[start, weird],
        edges=[IREdge(id="e0", source="start", target="agent-with-dashes.and.dots")],
        entry="start",
    )
    src = to_python.to_source(graph)
    ast.parse(src)  # no SyntaxError from invalid identifiers
    # Underscored version appears as the function name.
    assert "def node_agent_with_dashes_and_dots" in src
