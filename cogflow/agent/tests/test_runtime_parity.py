"""Runtime parity coverage for the LLM / Agent / Tool / direct-reply factories.

What's exercised here, in order:

  - Stage A — ``.format(**state)`` wider exception tuple. Templates with
    malformed braces or bad builtin-type format specs (``ValueError``)
    and values whose custom ``__format__`` raises (``TypeError``) must
    fall back to the literal instead of crashing the graph. Extra
    unreferenced state keys (including non-identifier strings like
    ``"user id"``) are inert in CPython 3.10+. The runtime path is
    brought in line with what ``compile/to_python.py`` already emits.

  - Stage B — ``llmUpdateState`` / ``agentUpdateState`` directives.
    LLM and Agent now apply the directive list after invoking the model,
    so a Python-first declaration like ``update_state=[{"key": "summary",
    "value": "{response}"}]`` actually mutates state at runtime.

  - Stage C — ReAct loop. When tools are provided to ``AgentFactory`` and
    the model has ``bind_tools``, the inner ``create_react_agent``
    subgraph runs the loop. Verified with a fake tool-calling chat model.

  - Stage D — ToolFactory + ``@tool``. A ``BaseTool`` wrapped in the
    factory now goes through LangGraph's ``ToolNode``, honouring
    ``tool_calls`` on the last AIMessage; a plain callable keeps the
    original ``tool_input → tool_output`` contract.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from cogflow.agent import add_messages
from cogflow.agent.ir.model import IRNode
from cogflow.agent.nodes.agent import AgentFactory
from cogflow.agent.nodes.direct_reply import DirectReplyFactory
from cogflow.agent.nodes.llm import LLMFactory
from cogflow.agent.nodes.tool import ToolFactory


class _MessagesOnlyState(TypedDict, total=False):
    """Module-level TypedDict so ``langgraph.get_type_hints`` can resolve
    the ``Annotated[..., add_messages]`` reducer at compile time. Defining
    this inside a test function makes the reducer invisible to LangGraph's
    schema introspection (``NameError`` on ``Annotated``).
    """

    messages: Annotated[list, add_messages]


class _ToolCallingFakeModel(FakeMessagesListChatModel):
    """``FakeMessagesListChatModel`` + a no-op ``bind_tools``.

    ``langgraph.prebuilt.create_react_agent`` requires a proper Runnable that
    supports ``bind_tools`` (the stock fake raises NotImplementedError). We
    just return ``self`` from ``bind_tools`` so the agent factory's react
    path can construct its subgraph and the scripted ``responses`` drive the
    loop.
    """

    def bind_tools(self, _tools: Any, **_kw: Any) -> _ToolCallingFakeModel:
        return self


# ---------------------------------------------------------------------------
# Stage A — wider .format exception tuple
# ---------------------------------------------------------------------------


def test_direct_reply_tolerates_malformed_template_braces():
    """A literal ``{`` in the reply text used to raise ValueError."""
    factory = DirectReplyFactory(message="see {section 1}")
    fn = factory.to_callable(IRNode(id="n", type="direct_reply", label="n"))
    out = fn({"messages": []})
    # Literal ``{section 1}`` is not a valid format field; runtime must
    # fall back to the unrendered template, NOT propagate ValueError.
    assert out == {"messages": [AIMessage(content="see {section 1}")]}


def test_direct_reply_tolerates_extra_unreferenced_state_keys():
    """Extra state keys the template doesn't reference must not interfere.

    Originally framed as "non-identifier state keys" (Flowise's startState
    permits ``"user id"`` / ``"step-count"``). That framing was misleading
    — ``str.format(**mapping)`` only requires keys be strings, not valid
    Python identifiers, when those keys aren't referenced by the template.
    The behaviour we actually care about is: rendering with the keys the
    template DOES want, ignoring the rest, regardless of their shape.
    """
    factory = DirectReplyFactory(message="hi {name}")
    fn = factory.to_callable(IRNode(id="n", type="direct_reply", label="n"))
    out = fn({"name": "alice", "user id": "u-1", "step-count": 3})
    assert out == {"messages": [AIMessage(content="hi alice")]}


def test_direct_reply_tolerates_custom_format_typeerror():
    """A custom ``__format__`` that raises TypeError must not crash the node.

    This is the actual TypeError hot path for ``str.format(**state)``:
    a value in state whose ``__format__`` raises (custom types, certain
    numeric conversions). The widened exception tuple catches it; the
    node falls back to the unrendered template.
    """

    class _BadFormat:
        def __format__(self, spec: str) -> str:
            raise TypeError("custom __format__ refuses to render")

    factory = DirectReplyFactory(message="hi {bad}")
    fn = factory.to_callable(IRNode(id="n", type="direct_reply", label="n"))
    out = fn({"bad": _BadFormat()})
    assert out == {"messages": [AIMessage(content="hi {bad}")]}


def test_direct_reply_coerces_null_message_to_empty():
    """JSON-imported IR with ``directReplyMessage: null`` must not crash.

    ``None.format(...)`` would raise ``AttributeError`` (NOT in the widened
    tuple). The factory now coerces non-string config values to ``""``
    before the format call, matching what ``compile/to_python.py`` already
    emits.
    """
    factory = DirectReplyFactory.from_ir(
        IRNode(id="n", type="direct_reply", label="n", config={"directReplyMessage": None})
    )
    fn = factory.to_callable(IRNode(id="n", type="direct_reply", label="n"))
    out = fn({"messages": []})
    # Empty rendered → factory returns no delta rather than crashing.
    assert out == {}


# ---------------------------------------------------------------------------
# Stage B — llmUpdateState / agentUpdateState
# ---------------------------------------------------------------------------


class _FixedReplyModel:
    """Minimal chat-model double whose ``invoke`` always returns the same AIMessage."""

    def __init__(self, content: str = "the answer"):
        self._content = content

    def invoke(self, _messages: Any, **_kw: Any) -> AIMessage:
        return AIMessage(content=self._content)


def test_llm_update_state_applies_after_invoke():
    factory = LLMFactory(
        model=_FixedReplyModel("paris"),
        update_state=[
            {"key": "summary", "value": "answer was: {response}"},
            {"key": "raw", "value": "{output}"},
            {"key": "literal_int", "value": 42},  # non-string passes through
        ],
    )
    fn = factory.to_callable(IRNode(id="n", type="llm", label="n"))
    out = fn({"messages": [HumanMessage(content="capital of france?")]})
    # Response message still emitted on the messages channel.
    assert any(isinstance(m, AIMessage) and m.content == "paris" for m in out["messages"])
    # Update directives folded into the delta. ``response`` and ``output``
    # are aliases for the model reply content.
    assert out["summary"] == "answer was: paris"
    assert out["raw"] == "paris"
    assert out["literal_int"] == 42


def test_llm_update_state_unset_marker_is_noop():
    """Empty-string sentinel (Flowise "unset") must not crash or mutate state."""
    factory = LLMFactory(model=_FixedReplyModel("ok"))
    # __init__ writes "" when update_state is not provided.
    assert factory.config["llmUpdateState"] == ""
    fn = factory.to_callable(IRNode(id="n", type="llm", label="n"))
    out = fn({"messages": []})
    assert "summary" not in out
    assert out["messages"][0].content == "ok"


def test_llm_update_state_format_failure_falls_back_to_literal():
    """A bad template in ``value`` must not crash the graph."""
    factory = LLMFactory(
        model=_FixedReplyModel("hi"),
        update_state=[{"key": "note", "value": "literal {missing_key} text"}],
    )
    fn = factory.to_callable(IRNode(id="n", type="llm", label="n"))
    out = fn({"messages": []})
    # Missing-key formatter failure → literal value preserved.
    assert out["note"] == "literal {missing_key} text"


def test_llm_update_state_skips_malformed_entries():
    """Entries missing ``key`` or with a non-string key are ignored, not raised."""
    factory = LLMFactory(
        model=_FixedReplyModel("ok"),
        update_state=[
            {"value": "no key"},
            {"key": "", "value": "empty key"},
            {"key": 123, "value": "non-string key"},
            "not a dict",
            {"key": "good", "value": "{response}"},
        ],
    )
    fn = factory.to_callable(IRNode(id="n", type="llm", label="n"))
    out = fn({"messages": []})
    assert out["good"] == "ok"
    assert "" not in out
    assert 123 not in out


# ---------------------------------------------------------------------------
# Stage C — ReAct loop in AgentFactory
# ---------------------------------------------------------------------------


def test_agent_runs_react_loop_when_tools_provided():
    """Agent factory should dispatch the tool and append a ToolMessage."""
    from langchain_core.tools import tool

    calls: list[dict[str, Any]] = []

    @tool
    def search(q: str) -> str:
        """Search."""
        calls.append({"q": q})
        return f"results-for-{q}"

    # First response carries a tool_call; second is the terminal AIMessage.
    model = _ToolCallingFakeModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "ping"}, "id": "tc-1"}]),
            AIMessage(content="done"),
        ]
    )
    factory = AgentFactory(model=model, tools=[search])
    fn = factory.to_callable(IRNode(id="agent", type="agent", label="agent"))
    out = fn({"messages": [HumanMessage(content="search please")]})

    assert calls == [{"q": "ping"}], "tool should have been dispatched by the loop"
    contents = [m.content for m in out["messages"] if isinstance(m, BaseMessage)]
    # The terminal message ("done") must be present; the ToolMessage carrying
    # the search result must also be present so the trace is complete.
    assert "done" in contents
    assert any("results-for-ping" in str(c) for c in contents)


def test_agent_single_shot_when_no_tools():
    """No tools → no react app constructed; behaviour matches the prior single-shot."""
    factory = AgentFactory(model=_FixedReplyModel("hi there"))
    fn = factory.to_callable(IRNode(id="agent", type="agent", label="agent"))
    out = fn({"messages": [HumanMessage(content="howdy")]})
    assert len(out["messages"]) == 1
    assert out["messages"][0].content == "hi there"


def test_agent_update_state_applies_after_react_loop():
    """``agentUpdateState`` must fire on the loop's terminal message."""
    from langchain_core.tools import tool

    @tool
    def noop(x: str) -> str:
        """No-op."""
        return f"noop-{x}"

    model = _ToolCallingFakeModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "noop", "args": {"x": "go"}, "id": "tc-1"}]),
            AIMessage(content="done"),
        ]
    )
    factory = AgentFactory(
        model=model,
        tools=[noop],
        update_state=[{"key": "final_text", "value": "agent said: {response}"}],
    )
    fn = factory.to_callable(IRNode(id="agent", type="agent", label="agent"))
    out = fn({"messages": [HumanMessage(content="go")]})
    assert out["final_text"] == "agent said: done"


# ---------------------------------------------------------------------------
# Stage D — ToolFactory @tool / ToolNode semantics
# ---------------------------------------------------------------------------


def test_tool_factory_with_basetool_delegates_to_langgraph_toolnode():
    """A ``@tool``-decorated callable must produce a LangGraph ``ToolNode``.

    Type check only — see the next test for an end-to-end compiled-graph
    invocation. Standalone ``.invoke({...})`` on a ToolNode raises
    ``ValueError: Missing required config key 'N/A' for 'tools'`` because
    LangGraph expects ToolNodes to be invoked inside a compiled StateGraph
    (where the Runtime context is established). That's a property of the
    LangGraph version actually installed in this environment, not a tie
    to a specific major; the next test covers the integration through a
    real StateGraph regardless of which LangGraph the user resolved.
    """
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two ints."""
        return a * b

    factory = ToolFactory(fn=multiply)
    runtime = factory.to_callable(IRNode(id="t", type="tool", label="t"))
    assert isinstance(runtime, ToolNode), (
        "ToolFactory.to_callable should return a langgraph ToolNode when fn is a @tool-decorated BaseTool"
    )


def test_tool_factory_basetool_dispatches_in_compiled_stategraph():
    """End-to-end: feed an AIMessage(tool_calls=...) through a StateGraph
    whose only node is the ToolFactory-produced ToolNode; assert that
    exactly one ToolMessage carrying the tool's return value lands on the
    messages channel. This is the runtime check the previous test
    intentionally leaves out (ToolNode standalone-invoke is forbidden,
    but inside a compiled graph the Runtime context is provided and the
    standard tool_calls contract holds).
    """
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph

    @tool
    def echo(value: str) -> str:
        """Return the value unchanged."""
        return f"echoed-{value}"

    factory = ToolFactory(fn=echo)
    runtime = factory.to_callable(IRNode(id="t", type="tool", label="t"))

    g = StateGraph(_MessagesOnlyState)
    g.add_node("tools", runtime)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    app = g.compile()

    ai = AIMessage(content="", tool_calls=[{"name": "echo", "args": {"value": "hi"}, "id": "tc-1"}])
    result = app.invoke({"messages": [ai]})

    # ToolNode appends one ToolMessage to ``messages`` for the single
    # tool_call. The original AIMessage is still there too (add_messages
    # reducer accumulates), so look for the ToolMessage specifically.
    from langchain_core.messages import ToolMessage

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "echoed-hi" in str(tool_messages[0].content)


def test_tool_factory_with_plain_callable_keeps_tool_input_contract():
    """Plain Python callable path should still read ``tool_input``."""

    def upper(value: str) -> str:
        return value.upper()

    factory = ToolFactory(fn=upper)
    fn = factory.to_callable(IRNode(id="t", type="tool", label="t"))
    out = fn({"tool_input": "hello"})
    assert out == {"tool_output": "HELLO"}


def test_tool_factory_without_fn_is_passthrough():
    """JSON-loaded ToolFactory with no live ``fn`` must remain a safe no-op."""
    factory = ToolFactory.from_ir(IRNode(id="t", type="tool", label="t"))
    fn = factory.to_callable(IRNode(id="t", type="tool", label="t"))
    out = fn({"tool_input": "anything"})
    assert out == {}


# ---------------------------------------------------------------------------
# Hardening — ensure none of the above broke the integration paths
# ---------------------------------------------------------------------------


def test_llm_unwired_model_still_passthrough():
    """An LLM with no live model continues to return {} (regression guard)."""
    factory = LLMFactory(model=None, update_state=[{"key": "x", "value": "{response}"}])
    fn = factory.to_callable(IRNode(id="n", type="llm", label="n"))
    assert fn({"messages": []}) == {}


def test_llm_invoke_with_no_update_state_returns_only_messages():
    """Default ``llmUpdateState = ""`` should not introduce extra keys."""
    factory = LLMFactory(model=_FixedReplyModel("ok"))
    fn = factory.to_callable(IRNode(id="n", type="llm", label="n"))
    out = fn({"messages": []})
    assert set(out.keys()) == {"messages"}


# ---------------------------------------------------------------------------
# Copilot review follow-ups (PR #102)
# ---------------------------------------------------------------------------


def test_update_state_drops_reserved_key_messages():
    """An ``update_state`` directive targeting ``messages`` must NOT clobber
    the model reply or break the ``add_messages`` reducer contract.

    The filter lives in ``apply_update_state`` (primary defence); the
    dict-spread ordering in ``llm_node`` / ``agent_node`` is the secondary
    guarantee. Both together mean even a malicious / mistaken directive
    list can't drop the messages channel.
    """
    factory = LLMFactory(
        model=_FixedReplyModel("the-reply"),
        update_state=[
            {"key": "messages", "value": "OVERRIDE-ATTEMPT"},
            {"key": "summary", "value": "{response}"},  # legitimate, should land
        ],
    )
    fn = factory.to_callable(IRNode(id="n", type="llm", label="n"))
    out = fn({"messages": []})
    # Messages channel carries the actual model reply, not the override.
    assert len(out["messages"]) == 1
    assert out["messages"][0].content == "the-reply"
    # Legitimate directive landed.
    assert out["summary"] == "the-reply"
    # ``messages`` did NOT get a string value.
    assert not isinstance(out["messages"], str)


def test_update_state_reserved_placeholders_not_shadowed_by_state():
    """``{response}`` / ``{output}`` must always resolve to the model reply.

    A prior node could have stashed something under ``state["response"]``;
    that must NOT bleed through into the template — the helper splats
    state first and sets the reserved keys last, so the reply wins.
    """
    factory = LLMFactory(
        model=_FixedReplyModel("the-real-reply"),
        update_state=[{"key": "summary", "value": "{response}"}],
    )
    fn = factory.to_callable(IRNode(id="n", type="llm", label="n"))
    out = fn({"messages": [], "response": "STALE-FROM-EARLIER", "output": "ALSO-STALE"})
    assert out["summary"] == "the-real-reply"


def test_tool_factory_reads_basetool_name_and_description():
    """``@tool(name="custom_name")`` must persist into config — not __name__."""
    from langchain_core.tools import tool

    @tool("custom_name", description="custom description")
    def underlying(x: str) -> str:
        """Docstring that should NOT appear in the config."""
        return x

    factory = ToolFactory(fn=underlying)
    assert factory.config["toolName"] == "custom_name"
    assert factory.config["toolDescription"] == "custom description"


def test_tool_factory_plain_callable_still_uses_dunder_name():
    """Regression: plain callables (no BaseTool) keep the historic fallback."""

    def my_helper(x: int) -> int:
        """Plain helper."""
        return x + 1

    factory = ToolFactory(fn=my_helper)
    assert factory.config["toolName"] == "my_helper"
    assert "Plain helper" in factory.config["toolDescription"]


class _BindToolsRaisesModel(FakeMessagesListChatModel):
    """Fake whose ``bind_tools`` raises NotImplementedError (BaseChatModel default).

    Used to verify the single-shot fallback in ``AgentFactory`` swallows
    the NotImplementedError and continues with the unbound model rather
    than crashing.
    """


def test_agent_single_shot_tolerates_bind_tools_not_implemented():
    """``bind_tools`` raising NotImplementedError must not crash the node."""

    def trivial_tool(x: str) -> str:
        return x  # plain callable, NOT a BaseTool — keeps the single-shot path

    # FakeMessagesListChatModel has hasattr(bind_tools) True but the base
    # implementation raises NotImplementedError. ``create_react_agent`` would
    # also reject this model (no proper bind_tools), so the agent factory
    # falls through to the single-shot path — which used to crash here.
    model = _BindToolsRaisesModel(responses=[AIMessage(content="ok")])
    factory = AgentFactory(model=model, tools=[trivial_tool])
    fn = factory.to_callable(IRNode(id="agent", type="agent", label="agent"))
    out = fn({"messages": [HumanMessage(content="hi")]})
    assert out["messages"][0].content == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
