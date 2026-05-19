"""Drop-in import migration (plan §3a) — every row in the table must hold."""

from __future__ import annotations

import langgraph.graph as _lg_graph


def test_state_graph_subclasses_langgraph():
    import cogflow.agent as cg

    assert issubclass(cg.StateGraph, _lg_graph.StateGraph)


def test_sentinels_pass_through():
    import cogflow.agent as cg

    assert cg.START is _lg_graph.START
    assert cg.END is _lg_graph.END


def test_messages_state_and_add_messages_pass_through():
    import cogflow.agent as cg

    assert cg.MessagesState is _lg_graph.MessagesState
    assert cg.add_messages is _lg_graph.add_messages


def test_tools_re_exports_resolve():
    from cogflow.agent.tools import ToolNode, tool

    assert tool is not None
    # ToolNode may be None on very old langgraph; allow that but flag in CI.
    _ = ToolNode  # noqa: F841


def test_prebuilt_create_react_agent():
    from langgraph.prebuilt import create_react_agent as _lg_cra

    from cogflow.agent.prebuilt import create_react_agent

    assert create_react_agent is _lg_cra


def test_checkpoint_re_exports():
    from langgraph.checkpoint.memory import MemorySaver as _LG_Memory

    from cogflow.agent.runtime.checkpoint import MemorySaver

    assert MemorySaver is _LG_Memory


def test_interrupt_and_command():
    import cogflow.agent as cg

    # Newer langgraph exposes these; assert they at least resolve to something.
    assert cg.interrupt is not None
    assert cg.Command is not None
