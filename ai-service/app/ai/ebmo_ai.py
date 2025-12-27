"""Compatibility shim for deprecated EBMO AI.

Historically, EBMO_AI lived at `app.ai.ebmo_ai`. During the December 2025
consolidation, the implementation moved under `archive.deprecated_ai`.

This module provides a stable import surface for code that depends on EBMO.
Prefer using `app.ai.neural_net` for new code.
"""

from __future__ import annotations

# Re-export the deprecated implementation.
# The underlying module already emits a DeprecationWarning.
from archive.deprecated_ai.ebmo_ai import *  # noqa: F401,F403
