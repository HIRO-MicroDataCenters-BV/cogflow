"""Direct Reply node — writes a terminal assistant message and routes to END."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import AIMessage

from ..ir.model import IRNode
from .base import NodeFactory


class DirectReplyFactory(NodeFactory):
    ir_type = "direct_reply"

    def __init__(self, *, message: str = "", **extra: Any) -> None:
        super().__init__(directReplyMessage=message, **extra)

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        # Coerce non-string values to ``""`` — partially-populated IR nodes
        # (or imported flows whose ``directReplyMessage`` came across as
        # ``null``) can carry ``None`` here, and ``None.format(...)`` would
        # raise ``AttributeError`` which is not in the widened tuple below.
        # ``compile/to_python.py:_render_direct_reply`` guards the same way.
        raw_message = self.config.get("directReplyMessage", "")
        message = raw_message if isinstance(raw_message, str) else ""

        def direct_reply_node(state: dict[str, Any]) -> dict[str, Any]:
            # ``.format(**state)`` can raise four ways: ``KeyError`` /
            # ``IndexError`` for a missing template slot, ``ValueError`` for
            # an unmatched brace (the reply text literally contains ``{`` or
            # ``}``), and ``TypeError`` when ``state`` carries keys that
            # aren't valid Python identifiers (Flowise's startState allows
            # ``"user id"`` / ``"step-count"``). In all four cases fall back
            # to the unrendered template so the node still produces output.
            # ``compile/to_python.py:_render_direct_reply`` emits the same
            # tuple — keep them in lockstep.
            rendered = message
            try:
                rendered = message.format(**state) if message else ""
            except (KeyError, IndexError, ValueError, TypeError):
                rendered = message
            return {"messages": [AIMessage(content=rendered)]} if rendered else {}

        return direct_reply_node


def direct_reply(**kwargs: Any) -> DirectReplyFactory:
    return DirectReplyFactory(**kwargs)
