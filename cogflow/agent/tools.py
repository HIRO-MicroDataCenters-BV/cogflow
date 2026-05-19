"""Tools: re-exports of LangChain's ``@tool`` decorator and LangGraph's ``ToolNode``.

These pass through verbatim; users get drop-in behavior.
"""

from langchain_core.tools import tool

try:
    from langgraph.prebuilt import ToolNode
except ImportError:  # pragma: no cover - older langgraph
    ToolNode = None  # type: ignore[assignment]

__all__ = ["ToolNode", "tool"]
