"""Tools: re-exports of LangChain's ``@tool`` decorator and LangGraph's ``ToolNode``.

These pass through verbatim; users get drop-in behavior. ``cogflow.agent``'s
``__init__`` already enforces both ``langgraph`` and ``langchain_core`` are
installed, but we still guard ``ToolNode`` for older langgraph builds that
ship without the prebuilt module.
"""

try:
    from langchain_core.tools import tool
except ModuleNotFoundError as _exc:  # pragma: no cover - belt-and-braces
    raise ModuleNotFoundError(
        "cogflow.agent.tools requires `langchain-core`. "
        "Install with `pip install cogflow[agent]`."
    ) from _exc

try:
    from langgraph.prebuilt import ToolNode
except ModuleNotFoundError as _exc:  # pragma: no cover
    # Only swallow when the missing module is langgraph-related; anything else
    # (e.g. a missing transitive dep, a broken installation) should bubble up
    # with its original message so users can act on it.
    _name = _exc.name or ""
    if _name == "langgraph":
        raise ModuleNotFoundError(
            "cogflow.agent.tools requires `langgraph`. "
            "Install with `pip install cogflow[agent]`."
        ) from _exc
    if not _name.startswith("langgraph."):
        raise
    # Older langgraph without the ``prebuilt`` submodule — graceful fallback.
    ToolNode = None  # type: ignore[assignment]

__all__ = ["ToolNode", "tool"]
