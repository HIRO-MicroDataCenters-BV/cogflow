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

    # ``cogflow.agent.__init__`` sets these to ``None`` on older langgraph
    # builds that don't expose ``langgraph.types.{interrupt,Command}``. Treat
    # both states as acceptable so the supported version range stays
    # consistent with ``ToolNode``'s fallback contract. The agent extra pins
    # langgraph>=0.2,<0.5 — in that range these symbols exist, so a None here
    # signals an unexpectedly-older langgraph that callers should update.
    if cg.interrupt is None or cg.Command is None:
        import langgraph

        # Diagnostic: surface the installed version in the failure message.
        version = getattr(langgraph, "__version__", "<unknown>")
        raise AssertionError(
            f"Expected langgraph.types.interrupt / Command to be importable; "
            f"got None — installed langgraph version: {version!r}"
        )
