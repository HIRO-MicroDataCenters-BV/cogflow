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


@pytest.fixture
def exec_module():
    """Compile + exec rendered source in a fresh module; auto-clean sys.modules.

    LangGraph's ``get_type_hints`` resolves forward refs through the class's
    ``__module__``, so a bare ``exec(src, {})`` fails to find ``Annotated``.
    We register a real module so name resolution works, and tear it down in
    the fixture's finalizer so the suite doesn't leak modules.
    """
    import sys
    import types

    created: list[str] = []

    def _run(source: str) -> dict[str, Any]:
        global _FAKE_MODULE_COUNTER
        _FAKE_MODULE_COUNTER += 1
        mod_name = f"_cogflow_to_python_emitted_{_FAKE_MODULE_COUNTER}"
        module = types.ModuleType(mod_name)
        sys.modules[mod_name] = module
        created.append(mod_name)
        code = compile(source, mod_name, "exec")
        exec(code, module.__dict__)
        return module.__dict__

    yield _run

    for name in created:
        sys.modules.pop(name, None)


# Same behaviour as the ``exec_module`` fixture above, but callable directly
# from tests that don't take a fixture argument. The fake module is popped
# from ``sys.modules`` in a ``finally`` block so the suite never accumulates
# stale entries — only the returned namespace dict keeps the live refs.
def _exec_module(source: str) -> dict[str, Any]:
    import sys
    import types

    global _FAKE_MODULE_COUNTER
    _FAKE_MODULE_COUNTER += 1
    mod_name = f"_cogflow_to_python_emitted_{_FAKE_MODULE_COUNTER}"
    module = types.ModuleType(mod_name)
    sys.modules[mod_name] = module
    try:
        code = compile(source, mod_name, "exec")
        exec(code, module.__dict__)
        return module.__dict__
    finally:
        # Pop from sys.modules so the test suite doesn't accumulate fake
        # modules; the returned dict still holds the live references the
        # caller needs.
        sys.modules.pop(mod_name, None)


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


def test_state_keys_with_non_identifier_chars_emit_valid_python():
    """Flowise ``startState`` keys can be arbitrary strings (``foo bar``, ``foo-bar``).

    The class-syntax TypedDict would have produced a SyntaxError; the
    functional form (``TypedDict('FlowState', {...}, total=False)``) handles
    arbitrary string keys cleanly.
    """
    start = IRNode(id="s0", type="start", label="Start")
    reply = IRNode(id="r0", type="direct_reply", label="Reply", config={"directReplyMessage": "ok"})
    graph = IRGraph(
        state=[
            IRStateField(key="user id", type="str"),
            IRStateField(key="step-count", type="int"),
            IRStateField(key="messages", type="messages", reducer="add_messages"),
        ],
        nodes=[start, reply],
        edges=[IREdge(id="e0", source="s0", target="r0")],
        entry="s0",
    )
    src = to_python.to_source(graph)
    ast.parse(src)
    # The functional form preserves the literal keys as dict keys.
    assert "'user id'" in src
    assert "'step-count'" in src
    ns = _exec_module(src)
    assert hasattr(ns["app"], "invoke")


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


def test_sticky_note_is_skipped():
    """Sticky notes must not generate node functions or edges."""
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


def test_unknown_node_is_passthrough_keeping_topology_flowing():
    """Unknown types compile as passthrough so surrounding edges keep flowing.

    Mirrors the IR contract documented in ir/model.py and the parse path —
    ``unknown`` nodes must be reachable, not silently dropped like StickyNote.
    """
    start = IRNode(id="s0", type="start", label="Start")
    mystery = IRNode(id="m0", type="unknown", label="Mystery")
    reply = IRNode(id="r0", type="direct_reply", label="Reply", config={"directReplyMessage": "hi"})
    graph = IRGraph(
        nodes=[start, mystery, reply],
        edges=[
            IREdge(id="e0", source="s0", target="m0"),
            IREdge(id="e1", source="m0", target="r0"),
        ],
        entry="s0",
    )
    src = to_python.to_source(graph)
    # The unknown node must be present (passthrough fn + node registration).
    assert "node_m0" in src
    assert "builder.add_node('m0'" in src
    # And the edge through it must be preserved.
    assert "builder.add_edge('m0', 'r0')" in src
    # The whole module still execs cleanly.
    ns = _exec_module(src)
    assert hasattr(ns["app"], "invoke")


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


def test_human_input_emit_honours_custom_output_key():
    """HumanInput's emitted source must use the configured output key."""
    start = IRNode(id="s0", type="start", label="Start")
    hi = IRNode(
        id="hi0",
        type="human_input",
        label="Ask",
        config={"humanInputDescription": "approve?", "humanInputOutputKey": "user_reply"},
    )
    graph = IRGraph(
        nodes=[start, hi],
        edges=[IREdge(id="e0", source="s0", target="hi0")],
        entry="s0",
    )
    src = to_python.to_source(graph)
    ast.parse(src)
    # Emitted return should use the custom key, not the hardcoded "human_input".
    assert "'user_reply': reply" in src
    assert "'human_input': reply" not in src


def test_bridge_node_todo_redacts_secret_config_values():
    """Bridge-node TODO stubs must not leak API keys / tokens / passwords."""
    start = IRNode(id="s0", type="start", label="Start")
    http_node = IRNode(
        id="h0",
        type="http",
        label="Call",
        config={
            "httpUrl": "https://api.example.com/v1/thing",
            "httpHeaders": {"x-api-key": "sk-LIVE-supersecret"},
            "apiKey": "sk-also-secret",
            "bearerToken": "eyJraWQ-not-actually",
            "password": "hunter2",
            "httpMethod": "GET",
        },
    )
    graph = IRGraph(
        nodes=[start, http_node],
        edges=[IREdge(id="e0", source="s0", target="h0")],
        entry="s0",
    )
    src = to_python.to_source(graph)
    # The literal secret values must not appear in the emitted source.
    assert "sk-LIVE-supersecret" not in src
    assert "sk-also-secret" not in src
    assert "eyJraWQ-not-actually" not in src
    assert "hunter2" not in src
    # Non-secret keys still appear (so the TODO stub is informative).
    assert "httpUrl" in src
    assert "https://api.example.com/v1/thing" in src
    # The redaction marker appears.
    assert "<REDACTED>" in src


def test_direct_reply_survives_non_identifier_state_keys():
    """``state`` can contain Flowise-style keys like ``"user id"``; ``.format(**state)`` would otherwise TypeError."""
    start = IRNode(id="s0", type="start", label="Start")
    reply = IRNode(
        id="r0",
        type="direct_reply",
        label="Reply",
        # Template doesn't reference the weird key, but ``**state`` still
        # unpacks it — must not crash.
        config={"directReplyMessage": "hello world"},
    )
    graph = IRGraph(
        state=[
            IRStateField(key="user id", type="str"),
            IRStateField(key="messages", type="messages", reducer="add_messages"),
        ],
        nodes=[start, reply],
        edges=[IREdge(id="e0", source="s0", target="r0")],
        entry="s0",
    )
    src = to_python.to_source(graph)
    ast.parse(src)
    ns = _exec_module(src)
    result = ns["app"].invoke({"user id": "alice", "messages": []})
    contents = [getattr(m, "content", "") for m in result.get("messages", [])]
    assert "hello world" in contents


def test_direct_reply_with_none_message_does_not_crash():
    """Partially-populated IRs can carry ``directReplyMessage=None``."""
    start = IRNode(id="s0", type="start", label="Start")
    reply = IRNode(id="r0", type="direct_reply", label="Reply", config={"directReplyMessage": None})
    graph = IRGraph(
        nodes=[start, reply],
        edges=[IREdge(id="e0", source="s0", target="r0")],
        entry="s0",
    )
    src = to_python.to_source(graph)
    ast.parse(src)
    ns = _exec_module(src)
    # No message → no AIMessage emitted; just an empty state update.
    assert ns["app"].invoke({"messages": []}).get("messages", []) == []


def test_node_fn_avoids_collisions_on_different_special_char_ids():
    """``a-b`` and ``a_b`` must not collapse to the same emitted function."""
    start = IRNode(id="s0", type="start", label="Start")
    a = IRNode(id="a-b", type="direct_reply", label="A", config={"directReplyMessage": "A"})
    b = IRNode(id="a_b", type="direct_reply", label="B", config={"directReplyMessage": "B"})
    graph = IRGraph(
        nodes=[start, a, b],
        edges=[
            IREdge(id="e0", source="s0", target="a-b"),
            IREdge(id="e1", source="a-b", target="a_b"),
        ],
        entry="s0",
    )
    src = to_python.to_source(graph)
    ast.parse(src)
    # ``a-b`` gets a hash-suffixed identifier; ``a_b`` keeps its bare form.
    # No matter what, the two ``def node_*`` lines must be distinct.
    fn_defs = [line for line in src.splitlines() if line.startswith("def node_")]
    assert len(fn_defs) == len(set(fn_defs)), f"function name collision: {fn_defs}"


def test_to_source_rejects_invalid_app_var():
    """``app_var`` must be a Python identifier (not a keyword, not arbitrary code)."""
    g = _basic_graph()
    for bad in ("not an identifier", "1nope", "app; print('hi')", "class", ""):
        with pytest.raises(ValueError, match="app_var"):
            to_python.to_source(g, app_var=bad)


def test_to_source_accepts_custom_valid_app_var():
    g = _basic_graph()
    src = to_python.to_source(g, app_var="my_agent")
    assert "my_agent = builder.compile()" in src
    ast.parse(src)


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
