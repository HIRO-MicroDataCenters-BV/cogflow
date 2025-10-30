"""
CogFlow Unified Exception Framework
===================================

This module defines CogFlow’s standardized error classes and handling utilities.
It ensures consistent, structured, and debuggable exception propagation across
all CogFlow subsystems — including models, pipelines, datasets, and connectors.

-------------------------------------------------------------------------------
📘 MODULE OVERVIEW
-------------------------------------------------------------------------------

🔹 Base Classes
    - CogflowError (root class)
    - CogflowConnectionError
    - CogflowModelError
    - CogflowExperimentError
    - CogflowRunError
    - CogflowArtifactError
    - CogflowValidationError

🔹 Error Handler Utility
    - CogflowErrorHandler
        • to_dict()
        • handle_exception()
        • log_and_raise()
-------------------------------------------------------------------------------
"""

from typing import Any, Dict, Optional, Type
import traceback
import logging

logger = logging.getLogger("cogflow")


# -----------------------------------------------------------------------------
# ⚙️ Base Exception Hierarchy
# -----------------------------------------------------------------------------
class CogflowError(Exception):
    """
    Base class for all CogFlow-related errors.

    Every CogFlow-specific exception inherits from this class.
    It provides standardized fields for message and cause chaining.

    Attributes:
        message (str): Human-readable error description.
        cause (Optional[Exception]): The underlying exception (if any).

    Example:
        >>> raise CogflowError("Something went wrong in CogFlow")
        >>> raise CogflowError("Failed to fetch model", cause=ValueError("Bad ID"))
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        self.message = message
        self.cause = cause
        super().__init__(self.__str__())

    def __str__(self) -> str:
        """Return a formatted string representation of the error."""
        base = f"[CogFlowError] {self.message}"
        if self.cause:
            return f"{base} | Cause: {type(self.cause).__name__}: {self.cause}"
        return base


# -----------------------------------------------------------------------------
# 🔹 Domain-specific error classes
# -----------------------------------------------------------------------------
class CogflowConnectionError(CogflowError):
    """Raised when the MLflow tracking server or network endpoint is unreachable."""


class CogflowModelError(CogflowError):
    """Raised when a model operation (load, register, evaluate, detect, etc.) fails."""


class CogflowExperimentError(CogflowError):
    """Raised for errors during experiment creation, retrieval, or manipulation."""


class CogflowRunError(CogflowError):
    """Raised for run lifecycle issues (start, end, tag, logging, etc.)."""


class CogflowArtifactError(CogflowError):
    """Raised when artifact upload, download, or URI resolution fails."""


class CogflowValidationError(CogflowError):
    """Raised for invalid parameters, configurations, or schema mismatches."""


# -----------------------------------------------------------------------------
# 🧩 CogFlow Error Handler Utility
# -----------------------------------------------------------------------------
class CogflowErrorHandler:
    """
    Unified error handling utility for CogFlow.

    Provides methods for:
        - serializing CogFlow errors into dicts (for REST APIs)
        - logging structured exceptions
        - wrapping and re-raising unexpected exceptions

    This handler is useful for FastAPI/Flask-based microservices, notebooks,
    and CLI utilities to provide consistent error reporting.

    Example:
        >>> try:
        ...     manager.load_model("bad_uri")
        ... except Exception as e:
        ...     CogflowErrorHandler.handle_exception(e, context="Model Loading")
    """

    @staticmethod
    def to_dict(error: Exception) -> Dict[str, Any]:
        """
        Convert a CogFlow exception to a serializable dictionary.

        Args:
            error (Exception): The exception to serialize.

        Returns:
            dict: A structured representation of the error suitable for API responses.
        """
        if isinstance(error, CogflowError):
            return {
                "error_type": type(error).__name__,
                "message": error.message,
                "cause": str(error.cause) if error.cause else None,
                "traceback": traceback.format_exc(),
            }
        return {
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    @staticmethod
    def handle_exception(
        error: Exception,
        context: Optional[str] = None,
        re_raise: bool = False,
        raise_as: Optional[Type[CogflowError]] = None,
    ) -> Dict[str, Any]:
        """
        Handle and log an exception gracefully.

        Args:
            error (Exception): The caught exception instance.
            context (Optional[str]): Description of the operation being performed.
            re_raise (bool): Whether to re-raise the exception after handling.
            raise_as (Optional[Type[CogflowError]]): Convert to a CogFlow-specific subclass before re-raising.

        Returns:
            dict: Structured dictionary of error details for reporting.

        Example:
            >>> try:
            ...     raise ValueError("Invalid input")
            ... except Exception as e:
            ...     CogflowErrorHandler.handle_exception(e, context="training", raise_as=CogflowValidationError)
        """
        context_msg = f" in {context}" if context else ""
        logger.error("❌ Exception occurred%s: %s", context_msg, error, exc_info=True)

        # Convert to CogFlow subclass if specified
        if raise_as:
            wrapped_error = raise_as(str(error), cause=error)
            if re_raise:
                raise wrapped_error
            return CogflowErrorHandler.to_dict(wrapped_error)

        # Return generic serialization for unknown errors
        if re_raise:
            raise error
        return CogflowErrorHandler.to_dict(error)

    @staticmethod
    def log_and_raise(
        message: str, raise_as: Type[CogflowError], cause: Optional[Exception] = None
    ):
        """
        Log and immediately raise a CogFlow error with structured information.

        Args:
            message (str): Error message to log.
            raise_as (Type[CogflowError]): Specific CogFlow error subclass to raise.
            cause (Optional[Exception]): Underlying cause exception.

        Raises:
            CogflowError: The raised CogFlow-specific exception.

        Example:
            >>> CogflowErrorHandler.log_and_raise(
            ...     "Failed to connect to MLflow",
            ...     CogflowConnectionError,
            ...     cause=ConnectionError("Timeout")
            ... )
        """
        err = raise_as(message, cause=cause)
        logger.error("🚨 %s", err)
        raise err
