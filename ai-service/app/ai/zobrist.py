"""DEPRECATED: Use app.core.zobrist instead.

This module is a wrapper for backward compatibility.
Import directly from app.core.zobrist:

    from app.core.zobrist import ZobristHash

Scheduled for removal: Q2 2026
"""
import warnings

# Issue deprecation warning on import
warnings.warn(
    "app.ai.zobrist is deprecated. Use 'from app.core.zobrist import ZobristHash' instead. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export ZobristHash from core module for backward compatibility
from ..core.zobrist import ZobristHash

__all__ = ["ZobristHash"]
