"""OpenAI chat-model re-export, gated on the ``openai`` install extra.

We catch the broader ``ImportError`` (not just ``ModuleNotFoundError``)
because ``from langchain_openai import ChatOpenAI`` can fail two ways:

  - ``ModuleNotFoundError`` — ``langchain_openai`` package not installed.
  - ``ImportError`` — package installed but the ``ChatOpenAI`` symbol
    is missing (broken install / version mismatch).

Both look the same to a downstream caller: "install the openai extra."
"""

try:
    from langchain_openai import ChatOpenAI
except ImportError as _exc:  # pragma: no cover - install-time guard
    raise ModuleNotFoundError(
        "cogflow.agent.chat_models.openai requires `langchain-openai`. "
        "Install with `pip install cogflow[openai]`."
    ) from _exc

__all__ = ["ChatOpenAI"]
