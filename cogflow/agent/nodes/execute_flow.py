"""ExecuteFlow node — invokes a nested compiled graph.

Two ways to supply the child graph:
 - ``graph=`` constructor arg (Python-first; pass a compiled state graph
   directly).
 - ``ctx={"flows": {flow_id: compiled_graph}}`` at compile time (JSON
   imports: the IR carries a ``executeFlowId`` config string and the
   runtime registry resolves it).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..ir.model import IRNode
from .base import NodeFactory


class ExecuteFlowFactory(NodeFactory):
    ir_type = "execute_flow"

    def __init__(
        self,
        *,
        flow_id: str = "",
        graph: Any = None,
        input_keys: list[str] | None = None,
        output_key: str = "subflow_result",
        **extra: Any,
    ) -> None:
        super().__init__(
            executeFlowId=flow_id,
            executeFlowInputKeys=list(input_keys) if input_keys is not None else "",
            executeFlowOutputKey=output_key,
            **extra,
        )
        self._graph = graph

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        live = getattr(self, "_graph", None)
        flow_id = self.config.get("executeFlowId") or ""
        if live is None and ctx:
            live = (ctx.get("flows") or {}).get(flow_id)
        input_keys = self.config.get("executeFlowInputKeys") or []
        output_key = self.config.get("executeFlowOutputKey") or "subflow_result"

        def execute_flow_node(state: dict[str, Any]) -> dict[str, Any]:
            if live is None:
                return {}
            child_input = (
                {k: state.get(k) for k in input_keys} if isinstance(input_keys, list) and input_keys else dict(state)
            )
            result = live.invoke(child_input)
            return {output_key: result}

        return execute_flow_node


def execute_flow(**kwargs: Any) -> ExecuteFlowFactory:
    return ExecuteFlowFactory(**kwargs)
