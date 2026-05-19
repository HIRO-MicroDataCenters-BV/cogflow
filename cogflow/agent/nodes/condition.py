"""Condition node — deterministic rule-based router."""

from __future__ import annotations

import operator as _op
import re
from typing import Any, Callable, Mapping

from ..ir.model import IRNode
from .base import NodeFactory

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "Equal": _op.eq,
    "NotEqual": _op.ne,
    "Contains": lambda a, b: b in (a or ""),
    "NotContains": lambda a, b: b not in (a or ""),
    "StartsWith": lambda a, b: isinstance(a, str) and a.startswith(b),
    "EndsWith": lambda a, b: isinstance(a, str) and a.endswith(b),
    "RegexMatch": lambda a, b: bool(re.search(b, a or "")),
    "Greater": _op.gt,
    "Less": _op.lt,
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
