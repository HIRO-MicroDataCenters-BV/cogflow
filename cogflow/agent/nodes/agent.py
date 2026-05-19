"""Agent node — ReAct-style agent loop wrapped as a single node."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from langchain_core.messages import AIMessage, BaseMessage

from ..ir.model import IRNode
from .base import NodeFactory
from .llm import _coerce_messages


class AgentFactory(NodeFactory):
    ir_type = "agent"

    def __init__(
        self,
        *,
        model: Any = None,
        tools: list[Any] | None = None,
        messages: list[dict[str, str]] | None = None,
        enable_memory: bool = True,
        memory_type: str = "allMessages",
        return_response_as: str = "userMessage",
        update_state: list[dict[str, str]] | None = None,
        **extra: Any,
    ) -> None:
        # Preserve explicit empty lists. Flowise treats "" (unset) and [] (declared
        # empty) as different shapes; coercing one to the other breaks round-trip.
        super().__init__(
            agentModel=model if isinstance(model, str) else "chatOpenAI",
            agentMessages=messages if messages is not None else "",
            agentTools="",
            agentEnableMemory=enable_memory,
            agentMemoryType=memory_type,
            agentReturnResponseAs=return_response_as,
            agentUpdateState=update_state if update_state is not None else "",
            **extra,
        )
        self._model = model
        self._tools = tools or []

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        raw_model = getattr(self, "_model", None)
        model = raw_model if not isinstance(raw_model, str) else None
        tools = list(getattr(self, "_tools", []) or [])
        # Mirror LLMFactory: surface the configured prompt (Flowise
        # ``agentMessages``) as a prefix on every invocation so system/dev
        # prompts apply to both Python-first and JSON-loaded graphs.
        prompts = _coerce_messages(self.config.get("agentMessages"))

        def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            if model is None:
                return {}
            history = list(state.get("messages") or [])
            bound = model.bind_tools(tools) if tools and hasattr(model, "bind_tools") else model
            response = bound.invoke(prompts + history)
            if not isinstance(response, BaseMessage):
                response = AIMessage(content=str(response))
            return {"messages": [response]}

        return agent_node


def agent(**kwargs: Any) -> AgentFactory:
    return AgentFactory(**kwargs)
