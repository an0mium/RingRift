"""Compatibility shim for deprecated EBMO modules.

Historically, EBMO lived at `app.ai.ebmo_network`. During the December 2025
consolidation, the implementation moved under `archive.deprecated_ai`.

Some training utilities and tests (including `tests/test_ebmo_ai.py`) still
import from the legacy location. Keep this module as a thin re-export layer so
those imports remain stable.

Canonical engine/rules code does NOT depend on EBMO.
"""

from __future__ import annotations

# Re-export the deprecated implementation.
# The underlying module already emits a DeprecationWarning.
from archive.deprecated_ai.ebmo_network import *  # noqa: F401,F403
