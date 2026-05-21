"""Runtime parity coverage for the LLM / Agent / Tool / direct-reply factories.

What's exercised here, in order:

  - Stage A — ``.format(**state)`` wider exception tuple. Templates with
    malformed braces (``ValueError``) and non-identifier state keys
    (``TypeError``) must fall back to the literal instead of crashing
    the graph. The runtime path is brought in line with what
    ``compile/to_python.py`` already emits.

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

from typing import Any

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from cogflow.agent.ir.model import IRNode
from cogflow.agent.nodes.agent import AgentFactory
from cogflow.agent.nodes.direct_reply import DirectReplyFactory
from cogflow.agent.nodes.llm import LLMFactory
from cogflow.agent.nodes.tool import ToolFactory


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


def test_direct_reply_tolerates_non_identifier_state_key():
    """Flowise startState keys like ``"user id"`` used to raise TypeError."""
    factory = DirectReplyFactory(message="hi {name}")
    fn = factory.to_callable(IRNode(id="n", type="direct_reply", label="n"))
    out = fn({"name": "alice", "user id": "u-1"})
    assert out == {"messages": [AIMessage(content="hi alice")]}


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

    We verify by type rather than by invoking standalone — ``ToolNode`` in
    LangGraph 1.x requires a Runtime context (it expects to be invoked as
    part of a compiled StateGraph), so a bare ``.invoke({...})`` call raises
    ``ValueError: Missing required config key 'N/A' for 'tools'``. The
    type check is sufficient: ToolNode's own test suite covers its runtime
    behaviour, and ``test_agent_runs_react_loop_when_tools_provided`` above
    exercises the full path end-to-end through a compiled subgraph.
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
