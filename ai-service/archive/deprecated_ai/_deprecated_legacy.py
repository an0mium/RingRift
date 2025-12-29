"""Centralized legacy AI imports.

This module provides a single import point for all deprecated AI implementations.
It centralizes access to legacy code, making it easier to track usage and
plan migration.

Usage:
    # Instead of:
    from archive.deprecated_ai.gmo_ai import GMOAI
    from archive.deprecated_ai.ebmo_ai import EBMO_AI

    # Use:
    from app.ai.legacy import GMOAI, EBMO_AI

    # Or import from individual shims:
    from app.ai.gmo_ai import GMOAI
    from app.ai.ebmo_ai import EBMO_AI

Available Shims:
    - app.ai.gmo_ai - GMOAI, GMOConfig
    - app.ai.ebmo_ai - EBMO_AI
    - app.ai.ebmo_network - EBMONetwork, ActionFeatureExtractor, EBMOConfig
    - app.ai.gmo_v2 - GMOv2AI, create_gmo_v2
    - app.ai.ig_gmo - IGGMO

Migration Guide:
    All deprecated AI implementations will be removed in Q2 2026.
    New code should use:
    - app.ai.neural_net - For neural network models
    - app.ai.gumbel_search_engine - For MCTS search
    - app.ai.factory - For AI instance creation

December 2025: Created as part of archive import consolidation.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "app.ai.legacy is a compatibility shim for deprecated AI implementations. "
    "These will be removed in Q2 2026. Use app.ai.neural_net for new code.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all deprecated AI implementations
# These are lazy-imported to avoid loading heavy modules until needed

__all__ = [
    # GMO
    "GMOAI",
    "GMOConfig",
    "estimate_uncertainty",
    # EBMO
    "EBMO_AI",
    "EBMONetwork",
    "EBMOConfig",
    "ActionFeatureExtractor",
    # GMO v2
    "GMOv2AI",
    "create_gmo_v2",
    # IG-GMO
    "IGGMO",
    "IGGMOConfig",
]


def __getattr__(name: str):
    """Lazy import deprecated modules."""
    if name in ("GMOAI", "GMOConfig", "estimate_uncertainty"):
        from app.ai.gmo_ai import GMOAI, GMOConfig, estimate_uncertainty

        return {"GMOAI": GMOAI, "GMOConfig": GMOConfig, "estimate_uncertainty": estimate_uncertainty}[name]

    if name == "EBMO_AI":
        from app.ai.ebmo_ai import EBMO_AI

        return EBMO_AI

    if name in ("EBMONetwork", "EBMOConfig", "ActionFeatureExtractor"):
        from app.ai.ebmo_network import ActionFeatureExtractor, EBMOConfig, EBMONetwork

        return {"EBMONetwork": EBMONetwork, "EBMOConfig": EBMOConfig, "ActionFeatureExtractor": ActionFeatureExtractor}[name]

    if name in ("GMOv2AI", "create_gmo_v2"):
        from app.ai.gmo_v2 import GMOv2AI, create_gmo_v2

        return {"GMOv2AI": GMOv2AI, "create_gmo_v2": create_gmo_v2}[name]

    if name in ("IGGMO", "IGGMOConfig"):
        from app.ai.ig_gmo import IGGMO

        # IGGMOConfig may not exist, try importing
        try:
            from archive.deprecated_ai.ig_gmo import IGGMOConfig

            return {"IGGMO": IGGMO, "IGGMOConfig": IGGMOConfig}[name]
        except ImportError:
            if name == "IGGMO":
                return IGGMO
            raise AttributeError(f"module 'app.ai.legacy' has no attribute '{name}'")

    raise AttributeError(f"module 'app.ai.legacy' has no attribute '{name}'")
