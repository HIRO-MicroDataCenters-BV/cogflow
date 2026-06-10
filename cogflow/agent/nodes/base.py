"""Base classes for node factories.

A NodeFactory carries the user-provided config for one node. The runtime
contract is intentionally minimal: ``emit_ir`` produces the IR row, and
``to_callable`` produces the Python function LangGraph will invoke. Round-trip
to Flowise JSON is delegated to ``compile.to_flowise``, which reads
``IRNode.flowise_provenance`` when present (lossless) and falls back to
``flowise_template`` (Python-first nodes).

For Phase 1 the runtime callables for nodes that wrap chat models return an
empty state update when ``model`` is None. Live-LLM behaviour is verified by
``test_compile_langgraph.test_simple_graph_invokes`` (plain Python callable
node) and by ``test_marketplace_simple_rag`` (JSON-loaded Agent node with
the model left unwired). Wiring against a real ``FakeListChatModel`` /
``FakeMessagesListChatModel`` for the model-bearing factories is deferred
to Phase 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.runnables import Runnable

from ..constants import IR_TYPE_TO_FLOWISE_CATEGORY, IR_TYPE_TO_FLOWISE_NAME
from ..ir.model import IRNode, IRNodeType, IRPort

# ``StateGraph.add_node`` accepts either a plain Python callable or any
# LangChain ``Runnable`` (which is the supertype of ``ToolNode``,
# ``CompiledStateGraph``, prebuilt agents, etc.). ``ToolFactory`` started
# returning a ``ToolNode`` for ``@tool``-decorated inputs; ``ToolNode`` is
# a Runnable but has no instance ``__call__``, so the older narrower
# ``Callable[..., Any]`` annotation was technically a lie. The alias keeps
# every other factory's plain-callable return shape unchanged.
NodeRuntime = Callable[..., Any] | Runnable


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
    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> NodeRuntime:
        """Return something ``StateGraph.add_node`` accepts.

        Most factories return a plain Python callable. ``ToolFactory`` may
        return a ``langgraph.prebuilt.ToolNode`` when the input is a
        ``BaseTool`` (a Runnable, not a Python callable). Both shapes are
        valid for ``add_node``; see ``NodeRuntime``.
        """

    def flowise_inputs(self, node: IRNode) -> dict[str, Any]:
        """The ``data.inputs`` block emitted when there's no provenance to copy."""
        return dict(node.config)

    # ------------------------------------------------------------------
    # JSON-loaded hydration
    # ------------------------------------------------------------------
    @classmethod
    def from_ir(cls, node: IRNode) -> NodeFactory:
        """Construct a factory instance from an IR row.

        Used by ``compile.to_langgraph`` when no live factory was registered
        for the node (i.e. the graph came from a Flowise JSON import). The
        default implementation skips ``__init__`` so subclasses with strict
        keyword args don't break, and seeds only the common attributes that
        every factory's ``to_callable`` consults. Subclasses with additional
        runtime state should override this method.
        """
        instance = cls.__new__(cls)
        instance.config = dict(node.config)
        return instance


def passthrough(state: Any) -> dict[str, Any]:
    """Returns an empty update; useful for nodes whose runtime is stubbed."""
    return {}
