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

        # Only lazy-load allowed names
        if name in self._lazy_submodules:

            # Try cogflow.core.<name>
            try:
                module = import_module(f"cogflow.core.{name}")
            except ModuleNotFoundError as e:
                # Only fallback if module truly doesn't exist,
                # not if import inside module failed
                if e.name == f"cogflow.core.{name}":
                    module = import_module(f"cogflow.{name}")
                else:
                    raise

            # Cache result to avoid repeated imports
            setattr(self, name, module)
            return module

        # correct module name in error message
        raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")
