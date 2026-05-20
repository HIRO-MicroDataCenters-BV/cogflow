"""Node factories — one per Flowise V2 AgentFlow node type.

Phase 1: MVP-7 (Start, LLM, Agent, Tool, Condition, ConditionAgent,
DirectReply) plus StickyNote.

Phase 2: bridge nodes (Loop, Iteration, HTTP, Retriever, CustomFunction,
HumanInput, ExecuteFlow). These map to LangGraph patterns rather than
native primitives — runtime behaviour is documented per file.
"""

from .agent import AgentFactory, agent
from .base import NodeFactory
from .condition import ConditionFactory, condition
from .condition_agent import ConditionAgentFactory, condition_agent
from .custom_function import CustomFunctionFactory, UnsupportedNodeError, custom_function
from .direct_reply import DirectReplyFactory, direct_reply
from .execute_flow import ExecuteFlowFactory, execute_flow
from .http import HTTPFactory, http
from .human_input import HumanInputFactory, human_input
from .iteration import IterationFactory, iteration
from .llm import LLMFactory, llm
from .loop import LoopFactory, loop
from .retriever import RetrieverFactory, retriever
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
    "loop": LoopFactory,
    "iteration": IterationFactory,
    "http": HTTPFactory,
    "retriever": RetrieverFactory,
    "custom_function": CustomFunctionFactory,
    "human_input": HumanInputFactory,
    "execute_flow": ExecuteFlowFactory,
    "sticky_note": StickyNoteFactory,
}

__all__ = [
    "AgentFactory",
    "ConditionAgentFactory",
    "ConditionFactory",
    "CustomFunctionFactory",
    "DirectReplyFactory",
    "ExecuteFlowFactory",
    "FACTORY_BY_IR_TYPE",
    "HTTPFactory",
    "HumanInputFactory",
    "IterationFactory",
    "LLMFactory",
    "LoopFactory",
    "NodeFactory",
    "RetrieverFactory",
    "StartFactory",
    "StickyNoteFactory",
    "ToolFactory",
    "UnsupportedNodeError",
    "agent",
    "condition",
    "condition_agent",
    "custom_function",
    "direct_reply",
    "execute_flow",
    "http",
    "human_input",
    "iteration",
    "llm",
    "loop",
    "retriever",
    "start",
    "sticky_note",
    "tool_node",
]
