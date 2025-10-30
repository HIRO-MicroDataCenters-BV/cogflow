"""
CogFlow SDK
===========

This package exposes high-level ML workflow operations under `cogflow.*`
and advanced functionality via submodules like `cogflow.models`, `cogflow.datasets`, etc.
"""

from ._lazy import _LazyLoader  # noqa: F401
from .core import models
from .core.models import *  # noqa: F403
from . import config


__version__ = "1.11.15"

__all__ = [*models.__all__, "models", "__version__"]
