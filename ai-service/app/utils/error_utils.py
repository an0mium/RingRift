"""Error handling utilities for FastAPI responses."""

from __future__ import annotations

import os

__all__ = [
    "sanitize_error_detail",
    "IS_PRODUCTION",
]

# Check if we're in production mode
IS_PRODUCTION = os.getenv("RINGRIFT_ENV", "development").lower() == "production"


def sanitize_error_detail(error: Exception, fallback: str = "Internal server error") -> str:
    """Return sanitized error message for HTTP responses.

    In production, returns a generic message to prevent information leakage.
    In development, returns the actual error message for debugging.

    Args:
        error: The exception to sanitize
        fallback: Fallback message for production mode

    Returns:
        Sanitized error message string
    """
    if IS_PRODUCTION:
        return fallback
    return str(error)
