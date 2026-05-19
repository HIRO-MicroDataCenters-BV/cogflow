"""Loop node — conditional back-edge with a per-graph iteration counter.

LangGraph has no native ``loop`` primitive. We bridge by emitting a router
that routes back to a target node while the counter is below
``max_iterations``, then routes to ``__end__`` when the limit is reached.

The counter is stored in state under a synthesized key
``_loop_count__<node_id>`` so multiple loops in one graph don't collide.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..ir.model import IRNode
from .base import NodeFactory


class LoopFactory(NodeFactory):
    ir_type = "loop"

    def __init__(
        self,
        *,
        target: str | None = None,
        max_iterations: int = 5,
        **extra: Any,
    ) -> None:
        super().__init__(
            loopTarget=target or "",
            loopMaxIterations=int(max_iterations),
            **extra,
        )

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        max_iters = int(self.config.get("loopMaxIterations", 5) or 5)
        counter_key = f"_loop_count__{node.id}"

        def loop_node(state: dict[str, Any]) -> dict[str, Any]:
            current = int(state.get(counter_key, 0) or 0)
            return {counter_key: current + 1}

        # Expose the counter key + limit so the compiler can wire a conditional
        # router back-edge without re-deriving them.
        loop_node._cogflow_loop_counter_key = counter_key  # type: ignore[attr-defined]
        loop_node._cogflow_loop_max = max_iters  # type: ignore[attr-defined]
        return loop_node


def loop(**kwargs: Any) -> LoopFactory:
    return LoopFactory(**kwargs)
