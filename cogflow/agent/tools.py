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
    # Distinguish "langgraph not installed at all" from "langgraph installed
    # without the prebuilt submodule". The former is a user-actionable error;
    # the latter (older langgraph) gracefully falls back to ToolNode=None.
    if _exc.name == "langgraph" or (_exc.name or "").startswith("langgraph."):
        if _exc.name == "langgraph":
            raise ModuleNotFoundError(
                "cogflow.agent.tools requires `langgraph`. "
                "Install with `pip install cogflow[agent]`."
            ) from _exc
    ToolNode = None  # type: ignore[assignment]

__all__ = ["ToolNode", "tool"]
