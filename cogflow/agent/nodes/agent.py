"""Agent node — ReAct-style agent loop wrapped as a single node.

When both a model and tools are present we delegate the loop to
``langgraph.prebuilt.create_react_agent`` so the runtime is a proper ReAct
agent: model → tool calls (if any) → tool execution → model → … until a
terminal AIMessage with no tool_calls is produced. That matches what users
get when they import ``create_react_agent`` directly from langgraph.

The single-shot path is kept as a fallback for three cases: no tools were
provided (no loop is needed), the model lacks ``bind_tools`` (e.g. a bare
``FakeListChatModel`` instance), or ``create_react_agent`` couldn't be
imported (older langgraph). In every case the slice we return to the outer
state contains only the new messages — ``add_messages`` would dedupe by id
either way, but slicing keeps the diff cleaner in traces.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from ..ir.model import IRNode
from ._update_state import apply_update_state
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
        update_directives = self.config.get("agentUpdateState")

        # Construct the ReAct subgraph once per ``to_callable`` invocation
        # so each agent_node call doesn't re-build the inner graph. The
        # import is lazy because langgraph is an optional extra; the
        # single-shot fallback below covers the import-failed case.
        #
        # Two import paths tried in order:
        #   1. ``langchain.agents.create_agent`` — only present if the user
        #      installed the ``langchain`` package separately. Newer
        #      surface introduced after the ``langgraph.prebuilt`` API was
        #      deprecated.
        #   2. ``langgraph.prebuilt.create_react_agent`` — the supported
        #      home for the SDK's declared dep range
        #      (``langgraph >=0.2,<0.5`` in ``pyproject.toml``). This is the
        #      path 99% of users land on.
        # Either entry point returns a CompiledStateGraph with the same
        # ``.invoke`` contract, so the rest of this function is path-agnostic.
        react_app = None
        if model is not None and tools and hasattr(model, "bind_tools"):
            create_react: Any | None = None
            try:
                from langchain.agents import create_agent as create_react  # type: ignore[import-not-found,no-redef]
            except ImportError:
                try:
                    from langgraph.prebuilt import (
                        create_react_agent as create_react,  # type: ignore[attr-defined,no-redef]
                    )
                except ImportError:
                    create_react = None
            if create_react is not None:
                try:
                    react_app = create_react(model, tools)
                except Exception:
                    # ``create_react_agent`` may reject duck-typed models;
                    # the single-shot fallback below still runs.
                    react_app = None

        def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            if model is None:
                return {}
            history = list(state.get("messages") or [])
            input_messages = prompts + history

            if react_app is not None:
                result = react_app.invoke({"messages": input_messages})
                # The inner agent echoes the inputs back at the head of its
                # message list; slice them off so only the new turns enter
                # the outer state. ``add_messages`` would dedupe by id but
                # the slice keeps trace output cleaner.
                all_messages = list(result.get("messages") or [])
                new_msgs = (
                    all_messages[len(input_messages) :] if len(all_messages) >= len(input_messages) else all_messages
                )
                if not new_msgs:
                    return {}
                last = next((m for m in reversed(new_msgs) if isinstance(m, BaseMessage)), new_msgs[-1])
                updates = apply_update_state(update_directives, last, state)
                return {"messages": new_msgs, **updates}

            # Single-shot fallback: no tools, no bind_tools, or older
            # langgraph without create_react_agent.
            #
            # ``hasattr`` isn't a strong enough guard: ``BaseChatModel``
            # defines ``bind_tools`` on the base class but its default impl
            # raises ``NotImplementedError`` (so do most fakes). Catch that
            # and proceed with the unbound model — single-shot can still
            # produce a response, just without tool-calling capability.
            bound = model
            if tools and hasattr(model, "bind_tools"):
                try:
                    bound = model.bind_tools(tools)
                except NotImplementedError:
                    bound = model
            response = bound.invoke(input_messages)
            if not isinstance(response, BaseMessage):
                response = AIMessage(content=str(response))
            updates = apply_update_state(update_directives, response, state)
            return {"messages": [response], **updates}

        return agent_node


def agent(**kwargs: Any) -> AgentFactory:
    return AgentFactory(**kwargs)
