"""Deprecated: Use v5_heavy_large instead.

This module has been renamed to v5_heavy_large.py to accurately reflect
that "v6" is not a separate architecture - it's V5 Heavy with larger
hyperparameters.

Migration:
    # Old (deprecated)
    from app.ai.neural_net.v6_large import create_v6_model, V6_LARGE_CONFIG

    # New (use this instead)
    from app.ai.neural_net.v5_heavy_large import (
        create_v5_heavy_large,
        V5_HEAVY_LARGE_CONFIG,
    )

This redirect module will be removed in Q2 2026.
"""

import warnings

warnings.warn(
    "v6_large module is deprecated. Use v5_heavy_large instead. "
    "v6 is not a separate architecture - it's V5 Heavy with larger parameters.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new module
from .v5_heavy_large import (
    # New canonical names
    V5_HEAVY_LARGE_CONFIG,
    V5_HEAVY_XL_CONFIG,
    V5_HEAVY_EFFICIENT_CONFIG,
    V5_1_CONFIG,
    create_v5_heavy_large,
    estimate_parameters,
    V5_HEAVY_LARGE_VERSION,
    # Deprecated aliases
    V6_LARGE_CONFIG,
    V6_XL_CONFIG,
    V6_EFFICIENT_CONFIG,
    V6_ARCHITECTURE_VERSION,
    create_v6_model,
)

__all__ = [
    "V5_HEAVY_LARGE_CONFIG",
    "V5_HEAVY_XL_CONFIG",
    "V5_HEAVY_EFFICIENT_CONFIG",
    "V5_1_CONFIG",
    "create_v5_heavy_large",
    "estimate_parameters",
    "V5_HEAVY_LARGE_VERSION",
    "V6_LARGE_CONFIG",
    "V6_XL_CONFIG",
    "V6_EFFICIENT_CONFIG",
    "V6_ARCHITECTURE_VERSION",
    "create_v6_model",
]
