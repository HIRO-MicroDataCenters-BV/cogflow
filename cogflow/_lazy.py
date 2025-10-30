""""
Lazy loading for cogflow submodules.
"""

from importlib import import_module
from types import ModuleType
import sys


class _LazyLoader(ModuleType):
    def __getattr__(self, name):
        if name in {"models"}:
            module = import_module(f"cogflow.core.{name}")
            globals()[name] = module
            return module
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules["cogflow"] = _LazyLoader("cogflow")
