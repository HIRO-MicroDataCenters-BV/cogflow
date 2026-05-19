"""Start node — entry point of an agentflow."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..ir.model import IRNode
from .base import NodeFactory, passthrough


class StartFactory(NodeFactory):
    ir_type = "start"

    def __init__(
        self,
        *,
        input_type: str = "chatInput",
        ephemeral_memory: bool = False,
        persist_state: bool = False,
        state: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            startInputType=input_type,
            startEphemeralMemory=ephemeral_memory,
            startPersistState=persist_state,
            startState=state or "",
            **extra,
        )

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        return passthrough


def start(**kwargs: Any) -> StartFactory:
    return StartFactory(**kwargs)
