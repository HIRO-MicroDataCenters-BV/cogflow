# cogflow/utils/imports.py
"""
Utility module for safe, lazy dependency imports.
Used to defer importing heavy libraries (mlflow, ray, kfp, etc.)
until they're actually needed.
"""

from importlib import import_module
from .logging import get_logger

logger = get_logger(__name__)


def lazy_import(module_name: str):
    """
    Import a module only when it's first required.

    Args:
        module_name (str): Name of the Python module to import.

    Returns:
        The imported module object.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        module = import_module(module_name)
        logger.debug("Lazily imported dependency: %s", module_name)
        return module
    except ModuleNotFoundError as e:
        msg = (
            f"⚠️ Missing optional dependency '{module_name}'.\n"
            f"Install it with: pip install cogflow[{module_name}]"
        )
        logger.error(msg)
        raise ImportError(msg) from e
