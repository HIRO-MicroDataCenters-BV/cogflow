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


def test_conditional_branch_to_end_does_not_force_other_branches():
    """Regression: one branch routing to END must NOT terminate the whole node."""
    from cogflow.agent.ir import END_SENTINEL

    def _route(state: _State) -> str:
        return "go"

    def _other(state: _State) -> dict:
        return {"messages": [AIMessage(content="other")]}

    g = StateGraph(_State)
    g.add_node("router", _greet)
    g.add_node("other", _other)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route, {"stop": END, "go": "other"})
    g.add_edge("other", END)

    # Router source must NOT be flagged as finish — only the "stop" branch
    # should terminate. Branch "go" should still route to "other".
    assert "router" not in g.ir.finish
    targets_from_router = {e.target for e in g.ir.edges if e.source == "router"}
    assert "other" in targets_from_router
    assert END_SENTINEL in targets_from_router

    # Compile must succeed (LangGraph would reject a node with both an
    # unconditional END edge and a conditional router).
    app = g.compile()
    assert app is not None
