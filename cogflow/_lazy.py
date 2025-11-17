"""
Lazy loading for cogflow submodules.
"""

from importlib import import_module
from types import ModuleType


class _LazyLoader(ModuleType):
    def __init__(self, name, lazy_submodules):
        super().__init__(name)
        self._lazy_submodules = set(lazy_submodules)

    def __getattr__(self, name):
        if name in self._lazy_submodules:
            # 1st: Try core path (e.g., cogflow.core.models)
            try:
                module = import_module(f"cogflow.core.{name}")
            except ModuleNotFoundError:
                # 2nd: Try root path (e.g., cogflow.api)
                module = import_module(f"cogflow.{name}")

            # Cache result in module attribute
            setattr(self, name, module)
            return module

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
