"""
CogFlow SDK
===========

This package exposes high-level ML workflow operations under `cogflow.*`
and advanced functionality via submodules like `cogflow.models`, `cogflow.datasets`, etc.
"""

import sys
from ._lazy import _LazyLoader

__version__ = "2.0.0"

# Submodules that should be lazily imported when accessed
_LAZY_SUBMODULES = [
    "models",  # resolves to cogflow.core.models
    "api",  # resolves to cogflow.api
]

# Create lazy module and copy current attributes
_lazy = _LazyLoader(__name__, _LAZY_SUBMODULES)
_lazy.__dict__.update(globals())

# Replace existing module with lazy module
sys.modules[__name__] = _lazy
