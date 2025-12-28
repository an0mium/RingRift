"""V6 Large Architecture Configuration: Scaled-up Neural Network for 2000+ Elo.

This module provides scaled-up configurations for V5 Heavy models to achieve
higher Elo ratings. The V6 architecture uses the same building blocks as V5
but with increased capacity:

Key Scaling Changes (V5 → V6):
    - num_filters: 160 → 256 (+60% capacity)
    - num_se_blocks: 6 → 10 (+67% attention depth)
    - num_attention_blocks: 5 → 8 (+60% long-range reasoning)
    - num_heuristics: 21 → 49 (full features enabled)
    - use_geometry_encoding: False → True (spatial awareness)

Estimated Parameters:
    - V5 Heavy (standard): ~6.5M parameters (~27MB)
    - V6 Large: ~25-30M parameters (~100-120MB)
    - V6 XL (with GNN): ~30-35M parameters (~120-140MB)

Architecture Philosophy:
    AlphaZero demonstrated that model capacity directly correlates with
    strength until data saturation. At 1600 Elo, we observe capacity
    saturation - the model learns patterns but lacks depth for complex
    tactics. V6 provides headroom for 1800+ → 2000+ Elo progression.

Usage:
    from app.ai.neural_net.v6_large import create_v6_model, V6_LARGE_CONFIG

    # Standard V6 model
    model = create_v6_model("hex8", num_players=2)

    # V6 XL with GNN
    model = create_v6_model("square8", num_players=2, variant="xl")

    # Custom configuration
    model = create_v6_model("hex8", num_players=4, **V6_XL_CONFIG)

Training Recommendations:
    - Use batch_size=256 (was 512) for better generalization
    - Use learning_rate=0.0005 (was 0.001) for stable training
    - Enable quality_weighting, outcome_weighted_policy, hard_example_mining
    - Use sampling_weights="combined_source" for balanced sampling
    - Train on 10,000+ games per configuration for capacity utilization

December 2025: Initial release for ML acceleration to 2000+ Elo.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .v5_heavy import HexNeuralNet_v5_Heavy, RingRiftCNN_v5_Heavy

logger = logging.getLogger(__name__)

# =============================================================================
# V6 Configuration Presets
# =============================================================================

# V6 Large: Conservative scaling for 1800+ Elo
# ~25M parameters - good balance of capacity and training speed
V6_LARGE_CONFIG = {
    "num_filters": 256,           # +60% vs V5 (was 160)
    "num_se_blocks": 10,          # +67% vs V5 (was 6)
    "num_attention_blocks": 8,    # +60% vs V5 (was 5)
    "num_heuristics": 49,         # Full features (was 21)
    "use_geometry_encoding": True,  # Enable spatial encoding (was False)
    "use_gnn": False,             # Disable GNN for training speed
    "se_reduction": 16,           # Keep same
    "dropout": 0.1,               # Keep same
    "num_attention_heads": 8,     # Increase from 4 for richer attention
}

# V6 XL: Aggressive scaling for 2000+ Elo
# ~35M parameters - maximum capacity with GNN refinement
V6_XL_CONFIG = {
    "num_filters": 320,           # +100% vs V5 (was 160)
    "num_se_blocks": 12,          # +100% vs V5 (was 6)
    "num_attention_blocks": 10,   # +100% vs V5 (was 5)
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

# V6 Efficient: Speed-optimized for high-throughput selfplay
# ~15M parameters - faster than V5 Heavy with better architecture
V6_EFFICIENT_CONFIG = {
    "num_filters": 192,           # +20% vs V5
    "num_se_blocks": 8,           # +33% vs V5
    "num_attention_blocks": 6,    # +20% vs V5
    "num_heuristics": 49,         # Full features
    "use_geometry_encoding": True,
    "use_gnn": False,
    "se_reduction": 16,
    "dropout": 0.08,              # Slightly lower dropout for stability
    "num_attention_heads": 4,
}


def create_v6_model(
    board_type: str = "square8",
    num_players: int = 2,
    variant: str = "large",
    **kwargs,
) -> "RingRiftCNN_v5_Heavy | HexNeuralNet_v5_Heavy":
    """Factory function for V6 Large models.

    Creates a V5 Heavy model with V6 scaling parameters applied.

    Args:
        board_type: 'square8', 'square19', 'hex8', or 'hexagonal'
        num_players: Number of players (2-4)
        variant: Configuration variant ('large', 'xl', 'efficient')
        **kwargs: Override individual parameters

    Returns:
        Configured V6 model (same architecture as V5 with larger params)

    Example:
        >>> model = create_v6_model("hex8", num_players=2)
        >>> model.count_parameters()
        25000000  # ~25M params
    """
    # Import here to avoid circular imports
    from .v5_heavy import create_v5_heavy_model

    # Select base configuration
    if variant == "xl":
        config = V6_XL_CONFIG.copy()
    elif variant == "efficient":
        config = V6_EFFICIENT_CONFIG.copy()
    elif variant == "v5.1" or variant == "v5_1":
        config = V5_1_CONFIG.copy()
    else:
        config = V6_LARGE_CONFIG.copy()

    # Apply any overrides
    config.update(kwargs)

    logger.info(
        f"Creating V6 {variant} model for {board_type} {num_players}p: "
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
    """Estimate parameter count for V6 configurations.

    Args:
        board_type: Board type for sizing
        variant: Configuration variant

    Returns:
        Dictionary with parameter estimates and memory usage
    """
    # Approximate formulas based on architecture
    if variant == "xl":
        config = V6_XL_CONFIG
    elif variant == "efficient":
        config = V6_EFFICIENT_CONFIG
    else:
        config = V6_LARGE_CONFIG

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
V6_ARCHITECTURE_VERSION = "v6.0.0"


__all__ = [
    # Configuration presets
    "V5_1_CONFIG",
    "V6_LARGE_CONFIG",
    "V6_XL_CONFIG",
    "V6_EFFICIENT_CONFIG",
    # Factory function
    "create_v6_model",
    # Utilities
    "estimate_parameters",
    # Version
    "V6_ARCHITECTURE_VERSION",
]
