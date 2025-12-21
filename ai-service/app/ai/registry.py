"""AI Registry with Versioning and Capability Tracking.

Lane 3 Consolidation (2025-12):
    Provides a centralized registry of all AI implementations with:
    - Explicit versioning for reproducibility
    - Capability flags for feature gating
    - Release notes and status tracking
    - Support for deterministic replay tests

Usage:
    from app.ai.registry import AIRegistry, get_ai_info, list_production_ais

    # Get info about a specific AI type
    info = get_ai_info(AIType.GUMBEL_MCTS)
    print(f"Version: {info.version}, Supports determinism: {info.supports_determinism}")

    # List all production-ready AI types
    for ai in list_production_ais():
        print(f"{ai.name} v{ai.version}: {ai.description}")

    # Check if an AI supports a specific capability
    if AIRegistry.supports_capability(AIType.MCTS, "neural_guidance"):
        print("MCTS supports neural guidance")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from app.models.core import AIType


class AICapability(Enum):
    """Capabilities that AI implementations may support."""

    # Core capabilities
    DETERMINISTIC = auto()  # Supports deterministic replay with same seed
    NEURAL_GUIDANCE = auto()  # Uses neural network for evaluation
    TREE_SEARCH = auto()  # Uses tree search (minimax, MCTS, etc.)
    GRADIENT_BASED = auto()  # Uses gradient optimization

    # Board/player support
    MULTI_PLAYER = auto()  # Supports 3-4 player games
    HEX_BOARDS = auto()  # Supports hexagonal boards
    LARGE_BOARDS = auto()  # Efficient on square19/hexagonal

    # Performance characteristics
    GPU_ACCELERATED = auto()  # Benefits from GPU
    BATCHED_EVAL = auto()  # Supports batched evaluation
    LOW_MEMORY = auto()  # Suitable for memory-constrained devices

    # Training capabilities
    GENERATES_TRAINING_DATA = auto()  # Can generate labeled training data
    SELF_PLAY_CAPABLE = auto()  # Suitable for self-play training loops


class AIStatus(Enum):
    """Status of an AI implementation."""

    PRODUCTION = "production"  # Stable, used in production
    EXPERIMENTAL = "experimental"  # Under development/testing
    DEPRECATED = "deprecated"  # Being phased out
    ARCHIVED = "archived"  # No longer maintained


@dataclass(frozen=True)
class AIInfo:
    """Metadata about an AI implementation."""

    ai_type: AIType
    name: str
    version: str
    description: str
    status: AIStatus
    capabilities: frozenset[AICapability]
    min_difficulty: int  # Lowest difficulty level using this AI
    max_difficulty: int  # Highest difficulty level using this AI
    release_date: str  # YYYY-MM-DD format
    release_notes: str = ""
    deprecated_by: Optional[AIType] = None  # If deprecated, what replaces it

    @property
    def supports_determinism(self) -> bool:
        """Check if this AI supports deterministic replay."""
        return AICapability.DETERMINISTIC in self.capabilities

    @property
    def uses_neural_net(self) -> bool:
        """Check if this AI uses neural network guidance."""
        return AICapability.NEURAL_GUIDANCE in self.capabilities

    @property
    def is_production(self) -> bool:
        """Check if this AI is production-ready."""
        return self.status == AIStatus.PRODUCTION


# AI Registry - Single source of truth for AI implementations
_AI_REGISTRY: dict[AIType, AIInfo] = {
    AIType.RANDOM: AIInfo(
        ai_type=AIType.RANDOM,
        name="Random AI",
        version="1.0.0",
        description="Uniform random move selection for baseline comparisons",
        status=AIStatus.PRODUCTION,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.MULTI_PLAYER,
            AICapability.HEX_BOARDS,
            AICapability.LARGE_BOARDS,
            AICapability.LOW_MEMORY,
        }),
        min_difficulty=1,
        max_difficulty=1,
        release_date="2024-01-01",
        release_notes="Original baseline AI implementation",
    ),
    AIType.HEURISTIC: AIInfo(
        ai_type=AIType.HEURISTIC,
        name="Heuristic AI",
        version="2.1.0",
        description="Hand-crafted heuristic evaluation without search",
        status=AIStatus.PRODUCTION,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.MULTI_PLAYER,
            AICapability.HEX_BOARDS,
            AICapability.LARGE_BOARDS,
            AICapability.LOW_MEMORY,
        }),
        min_difficulty=2,
        max_difficulty=2,
        release_date="2024-03-15",
        release_notes="v2.1: Improved territory evaluation weights",
    ),
    AIType.POLICY_ONLY: AIInfo(
        ai_type=AIType.POLICY_ONLY,
        name="Policy-Only AI",
        version="1.2.0",
        description="Neural network policy without tree search",
        status=AIStatus.PRODUCTION,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.NEURAL_GUIDANCE,
            AICapability.MULTI_PLAYER,
            AICapability.HEX_BOARDS,
            AICapability.GPU_ACCELERATED,
            AICapability.BATCHED_EVAL,
        }),
        min_difficulty=3,
        max_difficulty=3,
        release_date="2024-06-01",
        release_notes="v1.2: Added NNUE policy support",
    ),
    AIType.MINIMAX: AIInfo(
        ai_type=AIType.MINIMAX,
        name="Minimax AI",
        version="3.0.0",
        description="Alpha-beta pruned minimax with neural or heuristic eval",
        status=AIStatus.PRODUCTION,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.NEURAL_GUIDANCE,
            AICapability.TREE_SEARCH,
            AICapability.GENERATES_TRAINING_DATA,
        }),
        min_difficulty=4,
        max_difficulty=5,
        release_date="2024-04-01",
        release_notes="v3.0: NNUE evaluation integration",
    ),
    AIType.DESCENT: AIInfo(
        ai_type=AIType.DESCENT,
        name="Descent AI",
        version="2.5.0",
        description="AlphaZero-style MCTS with neural policy and value",
        status=AIStatus.PRODUCTION,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.NEURAL_GUIDANCE,
            AICapability.TREE_SEARCH,
            AICapability.MULTI_PLAYER,
            AICapability.HEX_BOARDS,
            AICapability.GPU_ACCELERATED,
            AICapability.GENERATES_TRAINING_DATA,
            AICapability.SELF_PLAY_CAPABLE,
        }),
        min_difficulty=6,
        max_difficulty=6,
        release_date="2024-08-01",
        release_notes="v2.5: Improved value head calibration",
    ),
    AIType.MCTS: AIInfo(
        ai_type=AIType.MCTS,
        name="MCTS AI",
        version="4.0.0",
        description="Monte Carlo Tree Search with UCB exploration",
        status=AIStatus.PRODUCTION,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.NEURAL_GUIDANCE,
            AICapability.TREE_SEARCH,
            AICapability.MULTI_PLAYER,
            AICapability.HEX_BOARDS,
            AICapability.GPU_ACCELERATED,
            AICapability.GENERATES_TRAINING_DATA,
            AICapability.SELF_PLAY_CAPABLE,
        }),
        min_difficulty=7,
        max_difficulty=8,
        release_date="2024-09-01",
        release_notes="v4.0: Progressive widening for move selection",
    ),
    AIType.GUMBEL_MCTS: AIInfo(
        ai_type=AIType.GUMBEL_MCTS,
        name="Gumbel MCTS AI",
        version="2.0.0",
        description="Gumbel Top-K MCTS with improved exploration",
        status=AIStatus.PRODUCTION,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.NEURAL_GUIDANCE,
            AICapability.TREE_SEARCH,
            AICapability.MULTI_PLAYER,
            AICapability.HEX_BOARDS,
            AICapability.LARGE_BOARDS,
            AICapability.GPU_ACCELERATED,
            AICapability.GENERATES_TRAINING_DATA,
            AICapability.SELF_PLAY_CAPABLE,
        }),
        min_difficulty=9,
        max_difficulty=11,
        release_date="2025-01-15",
        release_notes="v2.0: Strongest 2P AI per Dec 2025 benchmarks",
    ),
    AIType.MAXN: AIInfo(
        ai_type=AIType.MAXN,
        name="MaxN AI",
        version="1.5.0",
        description="Multi-player minimax variant for 3-4 player games",
        status=AIStatus.PRODUCTION,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.NEURAL_GUIDANCE,
            AICapability.TREE_SEARCH,
            AICapability.MULTI_PLAYER,
        }),
        min_difficulty=4,
        max_difficulty=6,
        release_date="2024-11-01",
        release_notes="v1.5: Used for 3-4P games in difficulty ladder",
    ),
    AIType.BRS: AIInfo(
        ai_type=AIType.BRS,
        name="Best Reply Search AI",
        version="1.2.0",
        description="Best Reply Search for multi-player games",
        status=AIStatus.PRODUCTION,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.TREE_SEARCH,
            AICapability.MULTI_PLAYER,
        }),
        min_difficulty=5,
        max_difficulty=7,
        release_date="2024-11-15",
        release_notes="v1.2: Alternative to MaxN for 3-4P",
    ),
    # Experimental AI types
    AIType.EBMO: AIInfo(
        ai_type=AIType.EBMO,
        name="EBMO AI",
        version="0.5.0",
        description="Energy-Based Move Optimization using gradient descent",
        status=AIStatus.EXPERIMENTAL,
        capabilities=frozenset({
            AICapability.NEURAL_GUIDANCE,
            AICapability.GRADIENT_BASED,
            AICapability.GPU_ACCELERATED,
        }),
        min_difficulty=12,
        max_difficulty=12,
        release_date="2025-10-01",
        release_notes="Experimental gradient-based approach",
    ),
    AIType.GMO: AIInfo(
        ai_type=AIType.GMO,
        name="GMO AI",
        version="0.8.0",
        description="Gradient Move Optimization with entropy guidance",
        status=AIStatus.EXPERIMENTAL,
        capabilities=frozenset({
            AICapability.NEURAL_GUIDANCE,
            AICapability.GRADIENT_BASED,
            AICapability.GPU_ACCELERATED,
        }),
        min_difficulty=13,
        max_difficulty=13,
        release_date="2025-11-01",
        release_notes="Information-theoretic move optimization",
    ),
    AIType.IG_GMO: AIInfo(
        ai_type=AIType.IG_GMO,
        name="IG-GMO AI",
        version="0.3.0",
        description="Information-Gain GMO with GNN encoding",
        status=AIStatus.EXPERIMENTAL,
        capabilities=frozenset({
            AICapability.NEURAL_GUIDANCE,
            AICapability.GRADIENT_BASED,
            AICapability.GPU_ACCELERATED,
        }),
        min_difficulty=14,
        max_difficulty=14,
        release_date="2025-11-15",
        release_notes="Most experimental gradient approach",
    ),
    AIType.GPU_MINIMAX: AIInfo(
        ai_type=AIType.GPU_MINIMAX,
        name="GPU Minimax AI",
        version="1.0.0",
        description="GPU-accelerated minimax with batched evaluation",
        status=AIStatus.EXPERIMENTAL,
        capabilities=frozenset({
            AICapability.DETERMINISTIC,
            AICapability.NEURAL_GUIDANCE,
            AICapability.TREE_SEARCH,
            AICapability.GPU_ACCELERATED,
            AICapability.BATCHED_EVAL,
        }),
        min_difficulty=15,
        max_difficulty=15,
        release_date="2025-12-01",
        release_notes="Hardware-optimized minimax",
    ),
}


class AIRegistry:
    """Central registry for AI implementations."""

    @staticmethod
    def get(ai_type: AIType) -> Optional[AIInfo]:
        """Get information about an AI type."""
        return _AI_REGISTRY.get(ai_type)

    @staticmethod
    def get_all() -> list[AIInfo]:
        """Get all registered AI types."""
        return list(_AI_REGISTRY.values())

    @staticmethod
    def get_production() -> list[AIInfo]:
        """Get all production-ready AI types."""
        return [ai for ai in _AI_REGISTRY.values() if ai.is_production]

    @staticmethod
    def get_experimental() -> list[AIInfo]:
        """Get all experimental AI types."""
        return [ai for ai in _AI_REGISTRY.values() if ai.status == AIStatus.EXPERIMENTAL]

    @staticmethod
    def supports_capability(ai_type: AIType, capability: AICapability) -> bool:
        """Check if an AI type supports a specific capability."""
        info = _AI_REGISTRY.get(ai_type)
        if info is None:
            return False
        return capability in info.capabilities

    @staticmethod
    def get_by_capability(capability: AICapability) -> list[AIInfo]:
        """Get all AI types that support a specific capability."""
        return [ai for ai in _AI_REGISTRY.values() if capability in ai.capabilities]

    @staticmethod
    def get_deterministic() -> list[AIInfo]:
        """Get all AI types that support deterministic replay."""
        return AIRegistry.get_by_capability(AICapability.DETERMINISTIC)

    @staticmethod
    def get_self_play_capable() -> list[AIInfo]:
        """Get all AI types suitable for self-play training."""
        return AIRegistry.get_by_capability(AICapability.SELF_PLAY_CAPABLE)


# Convenience functions
def get_ai_info(ai_type: AIType) -> Optional[AIInfo]:
    """Get information about an AI type."""
    return AIRegistry.get(ai_type)


def list_production_ais() -> list[AIInfo]:
    """List all production-ready AI implementations."""
    return AIRegistry.get_production()


def list_deterministic_ais() -> list[AIInfo]:
    """List all AI implementations that support deterministic replay."""
    return AIRegistry.get_deterministic()


def list_selfplay_ais() -> list[AIInfo]:
    """List all AI implementations suitable for self-play training."""
    return AIRegistry.get_self_play_capable()
