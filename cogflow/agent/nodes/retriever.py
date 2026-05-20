"""Retriever node — wraps a LangChain Retriever.

The retriever object can be injected via ``ctx={"retriever": ...}`` for
JSON-loaded graphs (where we don't have a live reference) or supplied at
construction time for Python-first graphs.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..ir.model import IRNode
from .base import NodeFactory


class RetrieverFactory(NodeFactory):
    ir_type = "retriever"

    def __init__(
        self,
        *,
        retriever: Any = None,
        store: str = "",
        k: int = 4,
        query_key: str = "query",
        output_key: str = "documents",
        **extra: Any,
    ) -> None:
        super().__init__(
            retrieverStore=store,
            retrieverK=int(k),
            retrieverQueryKey=query_key,
            retrieverOutputKey=output_key,
            **extra,
        )
        self._retriever = retriever

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        live = getattr(self, "_retriever", None) or ((ctx or {}).get("retriever") if ctx else None)
        query_key = self.config.get("retrieverQueryKey") or "query"
        output_key = self.config.get("retrieverOutputKey") or "documents"
        k = int(self.config.get("retrieverK") or 4)

        def retriever_node(state: dict[str, Any]) -> dict[str, Any]:
            if live is None:
                return {}
            query = state.get(query_key) or ""
            if hasattr(live, "invoke"):
                docs = live.invoke(query)
            elif hasattr(live, "get_relevant_documents"):
                docs = live.get_relevant_documents(query)
            elif callable(live):
                docs = live(query)
            else:
                return {}
            docs = list(docs)[:k] if hasattr(docs, "__iter__") and not isinstance(docs, str) else [docs]
            return {output_key: docs}

        return retriever_node


def retriever(**kwargs: Any) -> RetrieverFactory:
    return RetrieverFactory(**kwargs)
