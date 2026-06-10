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
        # imports are lazy because langgraph is an optional extra; the
        # single-shot fallback below covers the all-imports-failed case.
        #
        # Two factory entrypoints tried in order:
        #   1. ``langchain.agents.create_agent`` — only present if the user
        #      installed the ``langchain`` package separately. Newer
        #      surface introduced after the ``langgraph.prebuilt`` API was
        #      deprecated.
        #   2. ``langgraph.prebuilt.create_react_agent`` — the supported
        #      home for the SDK's declared dep range
        #      (``langgraph >=0.2,<0.5`` in ``pyproject.toml``). The path
        #      99% of users land on.
        # Both entrypoints return a CompiledStateGraph with the same
        # ``.invoke`` contract, so the rest of this function is
        # entrypoint-agnostic.
        #
        # The loop matters: if the first entrypoint imports cleanly but
        # rejects our model/tools shape (a narrow set of construction
        # errors caught below), we still try the second one — otherwise
        # users with both packages installed could silently lose tool
        # dispatch when only one factory accepts their model.
        react_app = None
        if model is not None and tools and hasattr(model, "bind_tools"):
            entrypoints: list[Any] = []
            try:
                from langchain.agents import create_agent  # type: ignore[import-not-found]

                entrypoints.append(create_agent)
            except ImportError:
                pass
            try:
                from langgraph.prebuilt import create_react_agent  # type: ignore[attr-defined]

                entrypoints.append(create_react_agent)
            except ImportError:
                pass

            for create_react in entrypoints:
                try:
                    react_app = create_react(model, tools)
                    break
                except (TypeError, ValueError, NotImplementedError, AttributeError):
                    # Narrow tuple covers what these factories raise when
                    # they can't accept the model/tools shape:
                    #   - ``TypeError`` — model isn't a Runnable
                    #     ("Expected a Runnable, callable or dict").
                    #   - ``NotImplementedError`` — ``bind_tools`` isn't
                    #     implemented for this chat-model class.
                    #   - ``AttributeError`` — duck-typed model missing
                    #     a method the factory introspects.
                    #   - ``ValueError`` — malformed tool definition.
                    # Anything else (e.g. an OOMError, a network error
                    # from a model's eager init, an authentication
                    # failure) propagates so the user sees the real
                    # problem instead of silently degrading. Try the next
                    # entrypoint — maybe a different factory accepts the
                    # same model.
                    continue

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
                # ``messages`` written LAST so even a future hole in the
                # reserved-key filter can't clobber the loop output. The
                # filter in ``apply_update_state`` is the primary defence;
                # this ordering is defence in depth.
                return {**updates, "messages": new_msgs}

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
            # Same ordering as the ReAct path — see comment above.
            return {**updates, "messages": [response]}

        return agent_node


def agent(**kwargs: Any) -> AgentFactory:
    return AgentFactory(**kwargs)
