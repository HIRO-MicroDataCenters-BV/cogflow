"""HumanInput node — pauses execution via langgraph.interrupt().

Flowise's ``humanInputAgentflow`` shows a prompt to the user and waits for
their reply before the flow continues. We bridge to LangGraph's
``interrupt(payload)``: the call raises ``GraphInterrupt`` so the runtime
halts, the client resumes with ``Command(resume=...)``, and the resume
value is returned from ``interrupt`` on the next pass.

Requires a checkpointer on ``compile()`` so the suspended state can be
revived.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..ir.model import IRNode
from .base import NodeFactory


class HumanInputFactory(NodeFactory):
    ir_type = "human_input"

    def __init__(
        self,
        *,
        prompt: str = "",
        output_key: str = "human_input",
        **extra: Any,
    ) -> None:
        super().__init__(
            humanInputPrompt=prompt,
            humanInputOutputKey=output_key,
            **extra,
        )

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        # Accept either our Python-first ``humanInputPrompt`` key or Flowise's
        # native ``humanInputDescription`` so JSON-loaded nodes work too.
        prompt = self.config.get("humanInputPrompt") or self.config.get("humanInputDescription") or ""
        output_key = self.config.get("humanInputOutputKey") or "human_input"

        def human_input_node(state: dict[str, Any]) -> dict[str, Any]:
            try:
                from langgraph.types import interrupt  # type: ignore[attr-defined]
            except ImportError:
                # Older langgraph without interrupts: degrade to passthrough.
                return {}
            # Same widened tuple as direct_reply / compile.to_python — see
            # ``direct_reply.py`` for the full failure-mode breakdown.
            # Covers missing template slots (KeyError/IndexError), malformed
            # templates and builtin-type format-spec errors (ValueError),
            # and defensive TypeError catch for custom ``__format__``
            # implementations that raise.
            rendered = prompt
            if prompt:
                try:
                    rendered = prompt.format(**state)
                except (KeyError, IndexError, ValueError, TypeError):
                    rendered = prompt
            reply = interrupt({"prompt": rendered, "node": node.id})
            return {output_key: reply}

        return human_input_node


def human_input(**kwargs: Any) -> HumanInputFactory:
    return HumanInputFactory(**kwargs)
