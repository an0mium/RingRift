"""Compatibility shim for deprecated GMO modules.

Historically, GMO lived at `app.ai.gmo_ai`. During the December 2025
consolidation, the implementation moved under `archive.deprecated_ai`.

Some tests and older tooling still import from the legacy location.
Keep this module as a thin re-export layer so imports remain stable.

Canonical engine/rules code does NOT depend on GMO.
"""

from __future__ import annotations

# Re-export the deprecated implementation.
# The underlying module already emits a DeprecationWarning.
from archive.deprecated_ai.gmo_ai import *  # noqa: F401,F403
