"""
CogFlow SDK
===========

This package exposes high-level ML workflow operations under `cogflow.*`
and advanced functionality via submodules like `cogflow.models`, `cogflow.datasets`, etc.
"""

import sys

from ._lazy import _LazyLoader

__version__ = "3.0.0b11"

# Submodules that should be lazily imported when accessed
_LAZY_SUBMODULES = [
    "models",  # resolves to cogflow.core.models
    "api",  # resolves to cogflow.api
    "datasets",  # resolves to cogflow.core.datasets
    "serving",  # resolves to cogflow.core.serving
    "common",  # resolves to cogflow.utils.common
    "network",  # resolves to cogflow.utils.network
    "config",  # resolves to cogflow.config
    "pipelines",  # resolves to cogflow.pipelines
    "agent",  # resolves to cogflow.agent (Cognitive Framework Agent SDK)
]

# Create lazy module and copy current attributes
_lazy = _LazyLoader(__name__, _LAZY_SUBMODULES)
_lazy.__dict__.update(globals())

# Replace existing module with lazy module
sys.modules[__name__] = _lazy
