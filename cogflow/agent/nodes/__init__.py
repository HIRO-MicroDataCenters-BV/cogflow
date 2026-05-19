"""Node factories — one per Flowise V2 AgentFlow node type.

MVP-7 nodes ship in Phase 1; the remaining 8 bridge nodes are added in Phase 2.
"""

from .agent import AgentFactory, agent
from .base import NodeFactory
from .condition import ConditionFactory, condition
from .condition_agent import ConditionAgentFactory, condition_agent
from .direct_reply import DirectReplyFactory, direct_reply
from .llm import LLMFactory, llm
from .start import StartFactory, start
from .sticky_note import StickyNoteFactory, sticky_note
from .tool import ToolFactory, tool_node

FACTORY_BY_IR_TYPE: dict[str, type[NodeFactory]] = {
    "start": StartFactory,
    "llm": LLMFactory,
    "agent": AgentFactory,
    "tool": ToolFactory,
    "condition": ConditionFactory,
    "condition_agent": ConditionAgentFactory,
    "direct_reply": DirectReplyFactory,
    "sticky_note": StickyNoteFactory,
}

__all__ = [
    "AgentFactory",
    "ConditionAgentFactory",
    "ConditionFactory",
    "DirectReplyFactory",
    "FACTORY_BY_IR_TYPE",
    "LLMFactory",
    "NodeFactory",
    "StartFactory",
    "StickyNoteFactory",
    "ToolFactory",
    "agent",
    "condition",
    "condition_agent",
    "direct_reply",
    "llm",
    "start",
    "sticky_note",
    "tool_node",
]
