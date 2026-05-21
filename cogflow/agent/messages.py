"""LangChain message-class re-exports.

Pure pass-through of the message classes downstream agent code uses to
type chat history, construct records, and filter streamed chunks. No
wrapping — these are the same objects ``langchain_core.messages``
exports, accessible via the ``cogflow.agent.*`` namespace.

This module deliberately reverses the Phase-1 architecture decision to
keep message classes as direct LangChain imports (see
``docs/agent/proposals/langchain-reexports.md``). Evidence from a real
downstream port showed every agent needs ``BaseMessage`` /
``AIMessage`` / ``HumanMessage`` for typing and construction, so making
them reachable from the SDK's own namespace is the right ergonomics.
"""

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

__all__ = [
    "AIMessage",
    "AIMessageChunk",
    "BaseMessage",
    "HumanMessage",
    "SystemMessage",
    "ToolMessage",
]
