"""CustomFunction node — Python-only, gated.

Flowise's ``customFunctionAgentflow`` lets users embed code in a flow. We
honour that for Python bodies only — JavaScript bodies are rejected with
``UnsupportedNodeError``. Python execution is gated on
``allow_custom_code=True`` at the parse/importer level (see the parser's
upcoming gate) and the namespace is restricted, but it is **not** a
security sandbox. Don't execute untrusted flows.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..ir.model import IRNode
from .base import NodeFactory


class UnsupportedNodeError(RuntimeError):
    """Raised when a node can't be safely materialized (e.g., JS body)."""


_RESTRICTED_BUILTINS: dict[str, Any] = {
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "any": any,
    "all": all,
    "abs": abs,
    "round": round,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


class CustomFunctionFactory(NodeFactory):
    ir_type = "custom_function"

    @classmethod
    def from_ir(cls, node: IRNode) -> "NodeFactory":
        # Mirror ``__init__`` validation: JSON-loaded nodes carrying a JS body
        # must be rejected here, otherwise the JS would slip through and try
        # to execute under ``allow_custom_code=True``.
        language = (node.config.get("customFunctionLanguage") or "python").lower()
        if language not in ("python", "py", ""):
            raise UnsupportedNodeError(
                f"CustomFunction node {node.id!r} carries an unsupported "
                f"language={language!r}; only Python bodies are executable."
            )
        instance = cls.__new__(cls)
        instance.config = dict(node.config)
        instance._fn = None
        instance._body = node.config.get("customFunctionPython")
        return instance

    def __init__(
        self,
        *,
        fn: Callable[..., Any] | None = None,
        body: str | None = None,
        language: str = "python",
        output_key: str = "custom_output",
        **extra: Any,
    ) -> None:
        if language.lower() not in ("python", "py"):
            raise UnsupportedNodeError(
                f"CustomFunction node only supports Python bodies; got language={language!r}"
            )
        super().__init__(
            customFunctionPython=body or (getattr(fn, "__name__", "") if fn else ""),
            customFunctionOutputKey=output_key,
            **extra,
        )
        self._fn = fn
        self._body = body

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        fn = getattr(self, "_fn", None)
        body = getattr(self, "_body", None) or (self.config.get("customFunctionPython") or "")
        output_key = self.config.get("customFunctionOutputKey") or "custom_output"
        allow_exec = bool((ctx or {}).get("allow_custom_code", False)) if ctx else False

        def custom_fn_node(state: dict[str, Any]) -> dict[str, Any]:
            if fn is not None:
                try:
                    result = fn(state)
                except TypeError:
                    result = fn()
                return {output_key: result}

            # String body path is gated — refuse to exec without explicit opt-in.
            if not allow_exec or not body:
                return {}
            namespace: dict[str, Any] = {
                "__builtins__": _RESTRICTED_BUILTINS,
                "state": dict(state),
                "result": None,
            }
            exec(compile(body, f"<custom_function:{node.id}>", "exec"), namespace)  # noqa: S102
            return {output_key: namespace.get("result")}

        return custom_fn_node


def custom_function(**kwargs: Any) -> CustomFunctionFactory:
    return CustomFunctionFactory(**kwargs)
