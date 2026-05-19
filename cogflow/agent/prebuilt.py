"""Re-exports of LangGraph prebuilt helpers.

A future Phase-3 wrapper will replace ``create_react_agent`` with a version that
also emits proper IR. For Phase 1 we expose it as-is so the migration table
holds: ``from cogflow.agent.prebuilt import create_react_agent``.
"""

from langgraph.prebuilt import create_react_agent

__all__ = ["create_react_agent"]
