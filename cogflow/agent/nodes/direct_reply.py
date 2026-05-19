"""Direct Reply node — writes a terminal assistant message and routes to END."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from langchain_core.messages import AIMessage

from ..ir.model import IRNode
from .base import NodeFactory


class DirectReplyFactory(NodeFactory):
    ir_type = "direct_reply"

    def __init__(self, *, message: str = "", **extra: Any) -> None:
        super().__init__(directReplyMessage=message, **extra)

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        message = self.config.get("directReplyMessage", "")

        def direct_reply_node(state: dict[str, Any]) -> dict[str, Any]:
            rendered = message
            try:
                rendered = message.format(**state) if message else ""
            except (KeyError, IndexError):
                rendered = message
            return {"messages": [AIMessage(content=rendered)]} if rendered else {}

        return direct_reply_node


def direct_reply(**kwargs: Any) -> DirectReplyFactory:
    return DirectReplyFactory(**kwargs)
