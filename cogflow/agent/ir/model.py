"""Intermediate Representation for cogflow.agent graphs.

The IR is the single source of truth shared by:
    - parse.flowise        (Flowise V2 AgentFlow JSON -> IR)
    - compile.to_flowise   (IR -> Flowise V2 AgentFlow JSON)
    - compile.to_langgraph (IR -> langgraph CompiledStateGraph)
    - graph.StateGraph     (Python builder API mutates an IR alongside langgraph)

`flowise_provenance` on every node/edge/graph captures everything we don't
explicitly model so round-trip is lossless even for nodes whose runtime
semantics we only partially translate.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

IRNodeType = Literal[
    "start",
    "llm",
    "agent",
    "tool",
    "condition",
    "condition_agent",
    "direct_reply",
    "loop",
    "iteration",
    "http",
    "retriever",
    "custom_function",
    "human_input",
    "execute_flow",
    "sticky_note",
]

IRStateType = Literal["str", "int", "float", "bool", "list", "dict", "messages"]


class IRPort(BaseModel):
    """An input or output anchor on an IR node."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    label: str | None = None
    type_: str = Field("default", alias="type")


class IRPosition(BaseModel):
    x: float = 0.0
    y: float = 0.0


class IRStateField(BaseModel):
    """One field of the shared graph state."""

    model_config = ConfigDict(extra="allow")

    key: str
    type_: IRStateType = Field("str", alias="type")
    default: Any = None
    reducer: str | None = None  # "add_messages" | "operator.add" | None


class IRNode(BaseModel):
    """An IR node."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: IRNodeType
    label: str
    config: dict[str, Any] = Field(default_factory=dict)
    inputs: list[IRPort] = Field(default_factory=list)
    outputs: list[IRPort] = Field(default_factory=list)
    position: IRPosition = Field(default_factory=IRPosition)
    flowise_provenance: dict[str, Any] | None = None


class IREdge(BaseModel):
    """An IR edge."""

    model_config = ConfigDict(extra="allow")

    id: str
    source: str
    target: str
    source_handle: str | None = None
    target_handle: str | None = None
    label: str | None = None
    is_human_input: bool = False
    flowise_provenance: dict[str, Any] | None = None


class IRGraph(BaseModel):
    """A complete graph in IR form."""

    model_config = ConfigDict(extra="allow")

    schema_version: Literal["1.0"] = "1.0"
    name: str = "agentflow"
    description: str | None = None
    state: list[IRStateField] = Field(default_factory=list)
    nodes: list[IRNode] = Field(default_factory=list)
    edges: list[IREdge] = Field(default_factory=list)
    entry: str | None = None
    finish: list[str] = Field(default_factory=list)
    viewport: dict[str, float] | None = None
    flowise_provenance: dict[str, Any] | None = None

    def node_by_id(self, node_id: str) -> IRNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def outgoing(self, node_id: str) -> list[IREdge]:
        return [e for e in self.edges if e.source == node_id]

    def incoming(self, node_id: str) -> list[IREdge]:
        return [e for e in self.edges if e.target == node_id]
