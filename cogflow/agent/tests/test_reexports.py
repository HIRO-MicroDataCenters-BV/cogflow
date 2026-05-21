"""Smoke tests for the LangChain / LangGraph re-export modules.

Each test does an explicit ``from cogflow.agent.<module> import <symbol>``
and exercises the re-exported object enough to prove it is the real
LangChain / LangGraph class (not a stub). Anything beyond that would be
testing LangChain itself.

The proposal that drove these re-exports lives at
``docs/agent/proposals/langchain-reexports.md``.
"""

from __future__ import annotations


def test_messages_reexports():
    from cogflow.agent.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    assert AIMessage(content="x").content == "x"
    assert HumanMessage(content="hi").content == "hi"
    assert SystemMessage(content="sys").content == "sys"
    assert ToolMessage(content="tool", tool_call_id="t0").content == "tool"
    assert isinstance(AIMessageChunk(content="chunk"), BaseMessage)


def test_prompts_reexports():
    from cogflow.agent.prompts import (
        ChatPromptTemplate,
        MessagesPlaceholder,
        PromptTemplate,
    )

    t = ChatPromptTemplate.from_messages(
        [
            ("system", "You are helpful."),
            ("human", "{q}"),
        ]
    )
    rendered = t.invoke({"q": "ok"}).to_messages()
    assert rendered[1].content == "ok"

    pt = PromptTemplate.from_template("hi {name}")
    assert pt.invoke({"name": "ada"}).to_string() == "hi ada"

    # MessagesPlaceholder just needs to construct.
    MessagesPlaceholder(variable_name="history")


def test_runnables_reexports():
    from cogflow.agent.runnables import (
        Runnable,
        RunnableConfig,
        RunnableLambda,
        RunnableParallel,
        RunnablePassthrough,
    )

    inc: Runnable = RunnableLambda(lambda x: x + 1)
    assert inc.invoke(1) == 2

    pair = RunnableParallel(a=RunnableLambda(lambda x: x), b=RunnableLambda(lambda x: x * 2))
    out = pair.invoke(3)
    assert out == {"a": 3, "b": 6}

    pt = RunnablePassthrough()
    assert pt.invoke({"k": 1}) == {"k": 1}

    # RunnableConfig is a TypedDict; just confirm it's importable + usable shape.
    _: RunnableConfig = {"tags": ["unit-test"]}


def test_fakes_reexports():
    from cogflow.agent.fakes import FakeListChatModel, GenericFakeChatModel

    m = FakeListChatModel(responses=["a", "b"])
    assert m.invoke("anything").content == "a"
    assert m.invoke("anything").content == "b"

    # GenericFakeChatModel just needs to construct (its API is slightly
    # different across langchain-core versions; constructability is enough).
    GenericFakeChatModel(messages=iter([]))


def test_runtime_basecheckpointsaver_reexport():
    from cogflow.agent.runtime import BaseCheckpointSaver, MemorySaver

    assert issubclass(MemorySaver, BaseCheckpointSaver)


def test_chat_models_namespace_does_not_eagerly_load_providers():
    """Importing ``cogflow.agent.chat_models`` must not require any provider."""
    import cogflow.agent.chat_models  # noqa: F401

    # No exported symbols at the namespace level — providers live in
    # submodules so each can gate its own optional install.
    assert cogflow.agent.chat_models.__all__ == []
