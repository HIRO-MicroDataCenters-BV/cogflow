"""LLM node — single chat-model call, no tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..ir.model import IRNode
from ._update_state import apply_update_state
from .base import NodeFactory


def _coerce_messages(items: Any) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    if not items:
        return out
    if isinstance(items, str):
        return [SystemMessage(content=items)]
    for entry in items:
        if isinstance(entry, BaseMessage):
            out.append(entry)
            continue
        if isinstance(entry, Mapping):
            role = entry.get("role", "user")
            content = entry.get("content", "")
            # Flowise's message UI exposes a ``developer`` role alongside
            # system/assistant/user. Map it to SystemMessage so prompt intent
            # survives import (LangChain has no first-class developer message).
            if role in ("system", "developer"):
                out.append(SystemMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
    return out


class LLMFactory(NodeFactory):
    ir_type = "llm"

    def __init__(
        self,
        *,
        model: Any = None,
        messages: list[dict[str, str]] | None = None,
        update_state: list[dict[str, str]] | None = None,
        return_response_as: str = "assistantMessage",
        **extra: Any,
    ) -> None:
        # Preserve explicit empty lists. Flowise treats "" (unset) and [] (declared
        # empty) as different shapes; coercing one to the other breaks round-trip.
        super().__init__(
            llmModel=model if isinstance(model, str) else "chatOpenAI",
            llmMessages=messages if messages is not None else "",
            llmUpdateState=update_state if update_state is not None else "",
            llmReturnResponseAs=return_response_as,
            **extra,
        )
        self._model = model

    def to_callable(self, node: IRNode, ctx: Mapping[str, Any] | None = None) -> Callable[..., Any]:
        # ``_model`` may be absent on instances hydrated via ``from_ir``;
        # fall back to None so JSON-loaded nodes become passthroughs.
        raw_model = getattr(self, "_model", None)
        model = raw_model if not isinstance(raw_model, str) else None
        prompts = _coerce_messages(self.config.get("llmMessages"))

        # Honour Flowise's ``llmUpdateState`` directive list — evaluated
        # against the response and folded into the returned delta. See
        # ``_update_state.apply_update_state`` for the templating rules.
        update_directives = self.config.get("llmUpdateState")

        def llm_node(state: dict[str, Any]) -> dict[str, Any]:
            history = list(state.get("messages") or [])
            if model is None:
                return {}
            response = model.invoke(prompts + history)
            if not isinstance(response, BaseMessage):
                response = AIMessage(content=str(response))
            updates = apply_update_state(update_directives, response, state)
            return {"messages": [response], **updates}

        return llm_node


def llm(**kwargs: Any) -> LLMFactory:
    return LLMFactory(**kwargs)
