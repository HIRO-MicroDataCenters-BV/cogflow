"""Test doubles for cogflow.agent users.

Re-exports LangChain's scripted chat-model fakes. The module is named
``fakes`` rather than ``testing`` to avoid implying it's the cogflow.agent
test suite — that's elsewhere under ``cogflow/agent/tests/``.

Typical use::

    from cogflow.agent.fakes import FakeListChatModel
    model = FakeListChatModel(responses=["hello", "world"])
    assert model.invoke("hi").content == "hello"
"""

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)

__all__ = ["FakeListChatModel", "GenericFakeChatModel"]
