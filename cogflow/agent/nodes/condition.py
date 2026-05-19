"""Condition node — deterministic rule-based router."""

from __future__ import annotations

import operator as _op
import re
from typing import Any, Callable, Mapping

from ..ir.model import IRNode
from .base import NodeFactory


def _as_str(value: Any) -> str:
    """Coerce any value to a string for string-shaped operators."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _regex_match(a: Any, b: Any) -> bool:
    try:
        return bool(re.search(_as_str(b), _as_str(a)))
    except re.error:
        return False


def _safe_cmp(op: Callable[[Any, Any], bool]) -> Callable[[Any, Any], bool]:
    def cmp(a: Any, b: Any) -> bool:
        try:
            return op(a, b)
        except TypeError:
            return False

    return cmp


_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "Equal": _op.eq,
    "NotEqual": _op.ne,
    "Contains": lambda a, b: _as_str(b) in _as_str(a),
    "NotContains": lambda a, b: _as_str(b) not in _as_str(a),
    "StartsWith": lambda a, b: _as_str(a).startswith(_as_str(b)),
    "EndsWith": lambda a, b: _as_str(a).endswith(_as_str(b)),
    "RegexMatch": _regex_match,
    "Greater": _safe_cmp(_op.gt),
    "Less": _safe_cmp(_op.lt),
}


class ConditionFactory(NodeFactory):
    ir_type = "condition"

    def __init__(
        self,
        *,
        rules: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(conditionItems=rules or [], **extra)
        self._rules = rules or []

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        rules = list(self._rules)

        def condition_node(state: dict[str, Any]) -> dict[str, Any]:
            for rule in rules:
                op = _OPS.get(rule.get("operation", "Equal"), _op.eq)
                left = state.get(rule.get("variable") or "")
                right = rule.get("value")
                if op(left, right):
                    return {"_condition_branch": rule.get("output", "match")}
            return {"_condition_branch": "default"}

        return condition_node


def condition(**kwargs: Any) -> ConditionFactory:
    return ConditionFactory(**kwargs)
