"""Base classes for node factories.

A NodeFactory carries the user-provided config for one node. The runtime
contract is intentionally minimal: ``emit_ir`` produces the IR row, and
``to_callable`` produces the Python function LangGraph will invoke. Round-trip
to Flowise JSON is delegated to ``compile.to_flowise``, which reads
``IRNode.flowise_provenance`` when present (lossless) and falls back to
``flowise_template`` (Python-first nodes).

For Phase 1 the runtime callables for nodes that wrap chat models return their
state unchanged when ``model`` is None. Real LLM wiring is exercised in tests
via ``langchain_core.language_models.fake.FakeListChatModel``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

from ..ir.model import IRNode, IRNodeType, IRPort
from ..constants import IR_TYPE_TO_FLOWISE_CATEGORY, IR_TYPE_TO_FLOWISE_NAME


class NodeFactory(ABC):
    """One factory instance per ``g.add_node(name, factory)`` call."""

    ir_type: IRNodeType

    def __init__(self, **config: Any) -> None:
        self.config: dict[str, Any] = dict(config)

    def default_outputs(self, node_id: str) -> list[IRPort]:
        flowise_name = IR_TYPE_TO_FLOWISE_NAME[self.ir_type]
        return [
            IRPort(
                id=f"{node_id}-output-{flowise_name}",
                name=flowise_name,
                label=IR_TYPE_TO_FLOWISE_CATEGORY[self.ir_type],
                type=self.ir_type,
            )
        ]

    def emit_ir(self, node_id: str, label: str | None = None) -> IRNode:
        return IRNode(
            id=node_id,
            type=self.ir_type,
            label=label or node_id,
            config=dict(self.config),
            inputs=[],
            outputs=self.default_outputs(node_id),
        )

    @abstractmethod
    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        """Return a Python function suitable for ``StateGraph.add_node``."""

    def flowise_inputs(self, node: IRNode) -> dict[str, Any]:
        """The ``data.inputs`` block emitted when there's no provenance to copy."""
        return dict(node.config)


def passthrough(state: Any) -> dict[str, Any]:
    """Returns an empty update; useful for nodes whose runtime is stubbed."""
    return {}
