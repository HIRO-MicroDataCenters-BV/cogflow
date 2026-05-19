"""Pydantic mirror of the Flowise V2 AgentFlow JSON shape.

Mirrors ``packages/agentflow/src/core/types/index.ts`` in the Flowise source.
We use ``extra='allow'`` so we never reject a JSON field — anything we don't
model passes through into IR provenance and is re-emitted on export.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlowisePosition(BaseModel):
    model_config = ConfigDict(extra="allow")
    x: float = 0.0
    y: float = 0.0


class FlowiseAnchor(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str
    label: str | None = None
    type: str | None = None


class FlowiseNodeData(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str  # discriminator: startAgentflow, llmAgentflow, ...
    label: str = ""
    type: str | None = None
    inputs: dict[str, Any] | str = Field(default_factory=dict)
    inputAnchors: list[FlowiseAnchor] = Field(default_factory=list)
    outputAnchors: list[FlowiseAnchor] = Field(default_factory=list)


class FlowiseNode(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: str | None = "agentFlow"
    position: FlowisePosition = Field(default_factory=FlowisePosition)
    data: FlowiseNodeData
    width: float | None = None
    height: float | None = None


class FlowiseEdgeData(BaseModel):
    model_config = ConfigDict(extra="allow")
    sourceColor: str | None = None
    targetColor: str | None = None
    edgeLabel: str | None = None
    isHumanInput: bool | None = None


class FlowiseEdge(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None
    type: str | None = "agentFlow"
    data: FlowiseEdgeData | None = None


class FlowiseFlow(BaseModel):
    model_config = ConfigDict(extra="allow")
    description: str | None = None
    usecases: list[str] | None = None
    nodes: list[FlowiseNode] = Field(default_factory=list)
    edges: list[FlowiseEdge] = Field(default_factory=list)
    viewport: dict[str, float] | None = None
