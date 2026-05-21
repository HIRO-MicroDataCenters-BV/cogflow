"""LangChain Runnable composition re-exports.

Surface needed for return-type annotations on chain-factory functions
and for composing nodes outside the ``StateGraph`` builder (where
``RunnableLambda`` / ``RunnableParallel`` are useful).
"""

from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

__all__ = [
    "Runnable",
    "RunnableConfig",
    "RunnableLambda",
    "RunnableParallel",
    "RunnablePassthrough",
]
