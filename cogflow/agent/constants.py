"""Constants for cogflow.agent.

START/END are imported from langgraph so object identity holds for users who
mix and match ``cogflow.agent.END`` with ``langgraph.graph.END``.
"""

from langgraph.graph import END, START

# Mapping from our IR node type to the Flowise V2 ``data.name`` discriminator.
IR_TYPE_TO_FLOWISE_NAME: dict[str, str] = {
    "start": "startAgentflow",
    "llm": "llmAgentflow",
    "agent": "agentAgentflow",
    "tool": "toolAgentflow",
    "condition": "conditionAgentflow",
    "condition_agent": "conditionAgentAgentflow",
    "direct_reply": "directReplyAgentflow",
    "loop": "loopAgentflow",
    "iteration": "iterationAgentflow",
    "http": "httpAgentflow",
    "retriever": "retrieverAgentflow",
    "custom_function": "customFunctionAgentflow",
    "human_input": "humanInputAgentflow",
    "execute_flow": "executeFlowAgentflow",
    "sticky_note": "stickyNoteAgentflow",
}

FLOWISE_NAME_TO_IR_TYPE: dict[str, str] = {v: k for k, v in IR_TYPE_TO_FLOWISE_NAME.items()}

# Flowise capitalises ``data.type`` (the category type, distinct from ``data.name``).
IR_TYPE_TO_FLOWISE_CATEGORY: dict[str, str] = {
    "start": "Start",
    "llm": "LLM",
    "agent": "Agent",
    "tool": "Tool",
    "condition": "Condition",
    "condition_agent": "ConditionAgent",
    "direct_reply": "DirectReply",
    "loop": "Loop",
    "iteration": "Iteration",
    "http": "HTTP",
    "retriever": "Retriever",
    "custom_function": "CustomFunction",
    "human_input": "HumanInput",
    "execute_flow": "ExecuteFlow",
    "sticky_note": "StickyNote",
}

__all__ = [
    "END",
    "FLOWISE_NAME_TO_IR_TYPE",
    "IR_TYPE_TO_FLOWISE_CATEGORY",
    "IR_TYPE_TO_FLOWISE_NAME",
    "START",
]
