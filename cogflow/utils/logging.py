""" "
Utility module for CogFlow logging.
Provides a standardized logger configuration for all CogFlow modules.
"""

import logging
import os
from functools import lru_cache


@lru_cache
def get_logger(name: str = "cogflow") -> logging.Logger:
    """Return a configured logger for CogFlow modules."""
    from ..config import config  # lazy import to avoid circulars

    logger = logging.getLogger(name)
    # IMPORTANT: prevent duplicate logs by stopping propagation
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Resolve level from config or env
        level_name = getattr(config, "LOG_LEVEL", os.getenv("COGFLOW_LOG_LEVEL", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

        logger.info("Logger initialized for '%s' at level: %s", name, level_name)  # Use % formatting

    return logger
