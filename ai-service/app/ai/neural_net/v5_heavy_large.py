"""V5 Heavy Large Configuration: Scaled-up V5 Heavy for 2000+ Elo.

This module provides scaled-up configurations for V5 Heavy models to achieve
higher Elo ratings. These are NOT new architectures - they use the same V5 Heavy
building blocks but with increased capacity.

Key Scaling Changes (V5 Heavy → V5 Heavy Large):
    - num_filters: 160 → 256 (+60% capacity)
    - num_se_blocks: 6 → 10 (+67% attention depth)
    - num_attention_blocks: 5 → 8 (+60% long-range reasoning)
    - num_heuristics: 21 → 49 (full features enabled)
    - use_geometry_encoding: False → True (spatial awareness)

Estimated Parameters:
    - V5 Heavy (standard): ~6.5M parameters (~27MB)
    - V5 Heavy Large: ~25-30M parameters (~100-120MB)
    - V5 Heavy XL (with GNN): ~30-35M parameters (~120-140MB)

Architecture Philosophy:
    AlphaZero demonstrated that model capacity directly correlates with
    strength until data saturation. At 1600 Elo, we observe capacity
    saturation - the model learns patterns but lacks depth for complex
    tactics. V5 Heavy Large provides headroom for 1800+ → 2000+ Elo progression.

Usage:
    from app.ai.neural_net.v5_heavy_large import (
        create_v5_heavy_large,
        V5_HEAVY_LARGE_CONFIG,
    )

    # Standard V5 Heavy Large model
    model = create_v5_heavy_large("hex8", num_players=2)

    # V5 Heavy XL with GNN
    model = create_v5_heavy_large("square8", num_players=2, variant="xl")

Training Recommendations:
    - Use batch_size=256 (was 512) for better generalization
    - Use learning_rate=0.0005 (was 0.001) for stable training
    - Enable quality_weighting, outcome_weighted_policy, hard_example_mining
    - Use sampling_weights="combined_source" for balanced sampling
    - Train on 10,000+ games per configuration for capacity utilization

December 2025: Renamed from v6_large.py to accurately reflect architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .v5_heavy import HexNeuralNet_v5_Heavy, RingRiftCNN_v5_Heavy

logger = logging.getLogger(__name__)

# =============================================================================
# V5 Heavy Large Configuration Presets
# =============================================================================

# V5 Heavy Large: Conservative scaling for 1800+ Elo
# ~25M parameters - good balance of capacity and training speed
V5_HEAVY_LARGE_CONFIG = {
    "num_filters": 256,           # +60% vs V5 Heavy (was 160)
    "num_se_blocks": 10,          # +67% vs V5 Heavy (was 6)
    "num_attention_blocks": 8,    # +60% vs V5 Heavy (was 5)
    "num_heuristics": 49,         # Full features (was 21)
    "use_geometry_encoding": True,  # Enable spatial encoding (was False)
    "use_gnn": False,             # Disable GNN for training speed
    "se_reduction": 16,           # Keep same
    "dropout": 0.1,               # Keep same
    "num_attention_heads": 8,     # Increase from 4 for richer attention
}

# V5 Heavy XL: Aggressive scaling for 2000+ Elo
# ~35M parameters - maximum capacity with GNN refinement
V5_HEAVY_XL_CONFIG = {
    "num_filters": 320,           # +100% vs V5 Heavy (was 160)
    "num_se_blocks": 12,          # +100% vs V5 Heavy (was 6)
    "num_attention_blocks": 10,   # +100% vs V5 Heavy (was 5)
    "num_heuristics": 49,         # Full features
    "use_geometry_encoding": True,
    "use_gnn": True,              # Enable GNN for connectivity awareness
    "se_reduction": 16,
    "dropout": 0.1,
    "num_attention_heads": 8,
}

# V5.1: Exactly 256 channels, 20 blocks as specified for Elo improvement
# ~22M parameters - balanced for 1800+ Elo
V5_1_CONFIG = {
    "num_filters": 256,           # 256 channels as specified
    "num_se_blocks": 10,          # 10 SE blocks
    "num_attention_blocks": 10,   # 10 attention blocks = 20 total blocks
    "num_heuristics": 49,         # Full features
    "use_geometry_encoding": True,
    "use_gnn": False,
    "se_reduction": 16,
    "dropout": 0.1,
    "num_attention_heads": 8,
}

# V5 Heavy Efficient: Speed-optimized for high-throughput selfplay
# ~15M parameters - faster than V5 Heavy with better architecture
V5_HEAVY_EFFICIENT_CONFIG = {
    "num_filters": 192,           # +20% vs V5 Heavy
    "num_se_blocks": 8,           # +33% vs V5 Heavy
    "num_attention_blocks": 6,    # +20% vs V5 Heavy
    "num_heuristics": 49,         # Full features
    "use_geometry_encoding": True,
    "use_gnn": False,
    "se_reduction": 16,
    "dropout": 0.08,              # Slightly lower dropout for stability
    "num_attention_heads": 4,
}


def create_v5_heavy_large(
    board_type: str = "square8",
    num_players: int = 2,
    variant: str = "large",
    **kwargs,
) -> "RingRiftCNN_v5_Heavy | HexNeuralNet_v5_Heavy":
    """Factory function for V5 Heavy Large models.

    Creates a V5 Heavy model with larger scaling parameters applied.

    Args:
        board_type: 'square8', 'square19', 'hex8', or 'hexagonal'
        num_players: Number of players (2-4)
        variant: Configuration variant ('large', 'xl', 'efficient')
        **kwargs: Override individual parameters

    Returns:
        Configured V5 Heavy model with larger parameters

    Example:
        >>> model = create_v5_heavy_large("hex8", num_players=2)
        >>> model.count_parameters()
        25000000  # ~25M params
    """
    # Import here to avoid circular imports
    from .v5_heavy import create_v5_heavy_model

    # Select base configuration
    if variant == "xl":
        config = V5_HEAVY_XL_CONFIG.copy()
    elif variant == "efficient":
        config = V5_HEAVY_EFFICIENT_CONFIG.copy()
    elif variant == "v5.1" or variant == "v5_1":
        config = V5_1_CONFIG.copy()
    else:
        config = V5_HEAVY_LARGE_CONFIG.copy()

    # Apply any overrides
    config.update(kwargs)

    logger.info(
        f"Creating V5 Heavy {variant} model for {board_type} {num_players}p: "
        f"filters={config['num_filters']}, "
        f"se_blocks={config['num_se_blocks']}, "
        f"attn_blocks={config['num_attention_blocks']}, "
        f"heuristics={config['num_heuristics']}"
    )

    return create_v5_heavy_model(
        board_type=board_type,
        num_players=num_players,
        **config,
    )


def estimate_parameters(
    board_type: str = "square8",
    variant: str = "large",
) -> dict[str, int | str]:
    """Estimate parameter count for V5 Heavy Large configurations.

    Args:
        board_type: Board type for sizing
        variant: Configuration variant

    Returns:
        Dictionary with parameter estimates and memory usage
    """
    # Approximate formulas based on architecture
    if variant == "xl":
        config = V5_HEAVY_XL_CONFIG
    elif variant == "efficient":
        config = V5_HEAVY_EFFICIENT_CONFIG
    else:
        config = V5_HEAVY_LARGE_CONFIG

    filters = config["num_filters"]
    se_blocks = config["num_se_blocks"]
    attn_blocks = config["num_attention_blocks"]
    has_gnn = config.get("use_gnn", False)

    # Rough estimate:
    # Initial conv: in_channels * filters * 25 (5x5 kernel)
    # SE block: 2 * filters^2 * 9 + SE overhead
    # Attention block: 4 * filters^2 + FF overhead
    # Policy/Value heads: ~2M combined

    base_params = filters * 40 * 25  # Initial conv (assume 40 input channels)
    se_params = se_blocks * (2 * filters * filters * 9 + filters * filters // 16 * 2)
    attn_params = attn_blocks * (4 * filters * filters + 2 * filters * filters * 4)
    head_params = 2_000_000  # Approximate for policy + value heads
    gnn_params = 500_000 if has_gnn else 0

    total = base_params + se_params + attn_params + head_params + gnn_params

    return {
        "variant": variant,
        "total_parameters": total,
        "memory_mb": total * 4 / 1_000_000,  # 4 bytes per float32
        "filters": filters,
        "se_blocks": se_blocks,
        "attn_blocks": attn_blocks,
    }


# Architecture version for checkpoint compatibility
V5_HEAVY_LARGE_VERSION = "v5.2.0"


# =============================================================================
# Deprecated Aliases (Remove in Q2 2026)
# =============================================================================
# These aliases maintain backward compatibility with code using old "v6" names

V6_LARGE_CONFIG = V5_HEAVY_LARGE_CONFIG
V6_XL_CONFIG = V5_HEAVY_XL_CONFIG
V6_EFFICIENT_CONFIG = V5_HEAVY_EFFICIENT_CONFIG
V6_ARCHITECTURE_VERSION = V5_HEAVY_LARGE_VERSION


def create_v6_model(
    board_type: str = "square8",
    num_players: int = 2,
    variant: str = "large",
    **kwargs,
) -> "RingRiftCNN_v5_Heavy | HexNeuralNet_v5_Heavy":
    """Deprecated: Use create_v5_heavy_large() instead."""
    import warnings
    warnings.warn(
        "create_v6_model is deprecated, use create_v5_heavy_large instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_v5_heavy_large(board_type, num_players, variant, **kwargs)


__all__ = [
    # New canonical names
    "V5_HEAVY_LARGE_CONFIG",
    "V5_HEAVY_XL_CONFIG",
    "V5_HEAVY_EFFICIENT_CONFIG",
    "V5_1_CONFIG",
    "create_v5_heavy_large",
    "estimate_parameters",
    "V5_HEAVY_LARGE_VERSION",
    # Deprecated aliases (for backward compatibility)
    "V6_LARGE_CONFIG",
    "V6_XL_CONFIG",
    "V6_EFFICIENT_CONFIG",
    "V6_ARCHITECTURE_VERSION",
    "create_v6_model",
]
