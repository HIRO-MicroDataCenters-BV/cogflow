"""OpenAI chat-model re-export, gated on the ``openai`` install extra.

We catch ``ImportError`` (broader than ``ModuleNotFoundError``) so two
``langchain_openai``-specific failures hit the same friendly hint:

  - ``ModuleNotFoundError`` with ``name == "langchain_openai"`` — package
    not installed at all.
  - ``ImportError`` with ``name == "langchain_openai"`` — package
    installed but the ``ChatOpenAI`` symbol is missing (broken / mismatched
    install).

Anything else originating from *inside* ``langchain_openai`` (e.g. its
``openai`` transitive dep missing, or any other unrelated import failure
during the package's own ``__init__``) is **re-raised unchanged** so the
user sees the real cause instead of a misleading "install
langchain-openai" hint.
"""

try:
    from langchain_openai import ChatOpenAI
except ImportError as _exc:  # pragma: no cover - install-time guard
    _name = getattr(_exc, "name", "") or ""
    if _name == "langchain_openai" or _name.startswith("langchain_openai."):
        raise ModuleNotFoundError(
            "cogflow.agent.chat_models.openai requires `langchain-openai`. Install with `pip install cogflow[openai]`."
        ) from _exc
    raise

__all__ = ["ChatOpenAI"]
