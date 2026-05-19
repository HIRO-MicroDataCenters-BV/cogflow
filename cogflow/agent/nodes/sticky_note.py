"""Sticky Note node — IR-only documentation; skipped at LangGraph compile."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..ir.model import IRNode
from .base import NodeFactory, passthrough


class StickyNoteFactory(NodeFactory):
    ir_type = "sticky_note"

    def __init__(self, *, text: str = "", **extra: Any) -> None:
        super().__init__(note=text, **extra)

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        return passthrough


def sticky_note(**kwargs: Any) -> StickyNoteFactory:
    return StickyNoteFactory(**kwargs)
