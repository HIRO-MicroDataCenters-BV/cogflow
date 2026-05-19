"""Direction B smoke: Python-first graph -> Flowise V2 JSON dict shape."""

from __future__ import annotations

from typing import Annotated, TypedDict

from cogflow.agent import END, START, StateGraph, add_messages
from cogflow.agent.compile.to_flowise import to_dict as ir_to_flowise
from cogflow.agent.nodes import agent


class _State(TypedDict, total=False):
    messages: Annotated[list, add_messages]


def test_python_first_emits_valid_flowise_shape():
    g = StateGraph(_State)
    g.add_node("qna", agent(model="openai:gpt-4o-mini"))
    g.add_edge(START, "qna")
    g.add_edge("qna", END)

    out = ir_to_flowise(g.ir)
    names = [n["data"]["name"] for n in out["nodes"]]
    assert "startAgentflow" in names
    assert "agentAgentflow" in names
    # At least one edge from the synthesized start.
    sources = [e["source"] for e in out["edges"]]
    assert any(s.startswith("__start__") or s == "__start__" for s in sources)
