"""State / channel / reducer plumbing.

LangGraph's runtime state is a TypedDict. Fields may carry a reducer via
``Annotated[T, reducer]`` (e.g. ``add_messages`` for chat history). Flowise V2
declares state instead as a list of ``{key, value}`` pairs on the Start node's
``startState`` input. This module bridges both forms through ``IRStateField``.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from langgraph.graph import MessagesState as _LangGraphMessagesState
from langgraph.graph import add_messages as _add_messages

from .ir.model import IRStateField, IRStateType

# Re-exports so users can ``from cogflow.agent import MessagesState, add_messages``.
MessagesState = _LangGraphMessagesState
add_messages = _add_messages


_REDUCER_REGISTRY: dict[str, Any] = {
    "add_messages": add_messages,
    "operator.add": operator.add,
}


def register_reducer(name: str, fn: Any) -> None:
    _REDUCER_REGISTRY[name] = fn


def reducer_by_name(name: str | None) -> Any:
    if name is None:
        return None
    if name not in _REDUCER_REGISTRY:
        raise KeyError(f"unknown reducer {name!r}; register with register_reducer()")
    return _REDUCER_REGISTRY[name]


def _reducer_name(fn: Any) -> str | None:
    """Best-effort identification of a known reducer callable."""
    for name, registered in _REDUCER_REGISTRY.items():
        if fn is registered:
            return name
    mod = getattr(fn, "__module__", "")
    qual = getattr(fn, "__qualname__", "")
    if mod == "operator" and qual == "add":
        return "operator.add"
    return None


_TYPE_TO_IR: dict[type, IRStateType] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    list: "list",
    dict: "dict",
}


def _python_type_to_ir(tp: Any) -> IRStateType:
    origin = get_origin(tp) or tp
    if origin in _TYPE_TO_IR:
        return _TYPE_TO_IR[origin]
    return "str"


def introspect_state(state_schema: type | None) -> list[IRStateField]:
    """Convert a TypedDict (optionally with Annotated reducers) into IRStateFields."""
    if state_schema is None:
        return []
    try:
        hints = get_type_hints(state_schema, include_extras=True)
    except Exception:
        return []

    out: list[IRStateField] = []
    for key, hint in hints.items():
        reducer_name: str | None = None
        type_ir: IRStateType = "str"
        if get_origin(hint) is Annotated:
            base, *meta = get_args(hint)
            type_ir = _python_type_to_ir(base)
            for m in meta:
                name = _reducer_name(m)
                if name is not None:
                    reducer_name = name
                    break
        else:
            type_ir = _python_type_to_ir(hint)

        # Always normalize the ``messages`` channel: type=messages and the
        # ``add_messages`` reducer. Mirrors ``parse.flowise._state_from_start``
        # so introspect(TypedDict) ↔ Flowise startState round-trips through
        # IR with a single canonical shape.
        if key == "messages":
            type_ir = "messages"
            if reducer_name is None:
                reducer_name = "add_messages"
        out.append(IRStateField(key=key, type=type_ir, reducer=reducer_name))
    return out


_IR_TYPE_TO_PY: dict[IRStateType, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "messages": list,
}


def synthesize_typeddict(fields: Iterable[IRStateField], name: str = "FlowState") -> type:
    """Build a TypedDict class from IRStateField list at runtime.

    Annotated reducers are reattached so LangGraph's channel system works
    unchanged (``add_messages``, ``operator.add``, …).
    """
    from typing import TypedDict  # local import to keep top-level clean

    annotations: dict[str, Any] = {}
    for f in fields:
        base = _IR_TYPE_TO_PY.get(f.type_, str)
        if f.reducer is not None:
            annotations[f.key] = Annotated[base, reducer_by_name(f.reducer)]
        else:
            annotations[f.key] = base

    return TypedDict(name, annotations, total=False)  # type: ignore[operator]
