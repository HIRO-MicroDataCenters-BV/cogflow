"""Integration: load Simple RAG.json, compile to LangGraph, invoke."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage

from cogflow.agent.compile.to_langgraph import to_langgraph
from cogflow.agent.parse.flowise import from_file


def test_simple_rag_compiles_and_invokes(simple_rag_path: Path):
    ir = from_file(simple_rag_path)
    assert ir.entry is not None
    assert any(n.type == "agent" for n in ir.nodes)

    app = to_langgraph(ir)
    out = app.invoke({"messages": [HumanMessage(content="what is RAG?")]})
    # Without a live model the agent node is a passthrough; we just verify the
    # graph runs to completion and returns the state dict it started with.
    assert isinstance(out, dict)
    assert "messages" in out
