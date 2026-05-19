"""IR (intermediate representation) for cogflow.agent graphs."""

from .builders import add_edge, add_node, fresh_edge_id
from .model import (
    IREdge,
    IRGraph,
    IRNode,
    IRNodeType,
    IRPort,
    IRPosition,
    IRStateField,
    IRStateType,
)
from .validate import IRValidationError, validate

__all__ = [
    "IREdge",
    "IRGraph",
    "IRNode",
    "IRNodeType",
    "IRPort",
    "IRPosition",
    "IRStateField",
    "IRStateType",
    "IRValidationError",
    "add_edge",
    "add_node",
    "fresh_edge_id",
    "validate",
]
