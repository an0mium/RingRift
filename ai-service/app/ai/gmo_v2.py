"""Compatibility shim for deprecated GMO v2 AI.

Historically, GMOv2AI lived at `app.ai.gmo_v2`. During the December 2025
consolidation, the implementation moved under `archive.deprecated_ai`.

This module provides a stable import surface for code that depends on GMOv2.
Prefer using `app.ai.neural_net` for new code.
"""

from __future__ import annotations

# Re-export the deprecated implementation.
# The underlying module already emits a DeprecationWarning.
from archive.deprecated_ai.gmo_v2 import *  # noqa: F401,F403
