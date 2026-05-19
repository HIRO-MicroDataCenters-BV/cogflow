"""Build a tiny graph with the Python facade and invoke it via LangGraph."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage

from cogflow.agent import END, START, StateGraph, add_messages


class _State(TypedDict, total=False):
    messages: Annotated[list, add_messages]


def _greet(state: _State) -> dict:
    return {"messages": [AIMessage(content="hello")]}


def test_simple_graph_invokes():
    g = StateGraph(_State)
    g.add_node("greet", _greet)
    g.add_edge(START, "greet")
    g.add_edge("greet", END)

    app = g.compile()
    result = app.invoke({"messages": [HumanMessage(content="hi")]})

    assert isinstance(result, dict)
    assert any(getattr(m, "content", "") == "hello" for m in result.get("messages", []))


def test_state_graph_captures_ir():
    g = StateGraph(_State)
    g.add_node("greet", _greet)
    g.add_edge(START, "greet")
    g.add_edge("greet", END)

    assert g.ir is not None
    # Start node was synthesized + greet was recorded as a custom_function node.
    types = sorted(n.type for n in g.ir.nodes)
    assert "start" in types
    assert "custom_function" in types
    # Edge from start node to greet exists.
    assert any(e.target == "greet" for e in g.ir.edges)
