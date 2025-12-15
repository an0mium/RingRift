"""AI implementations for RingRift.

This package intentionally avoids importing concrete AI classes at import
time to prevent circular dependencies between the rules engine and AI
modules. For concrete AI classes, import directly from their modules:

    from app.ai.base import BaseAI
    from app.ai.heuristic_ai import HeuristicAI

For creating AI instances, use the factory module:

    from app.ai.factory import AIFactory, create_ai_from_difficulty

    # Create from difficulty level (recommended for gameplay)
    ai = create_ai_from_difficulty(difficulty=5, player_number=1)

    # Create with explicit type
    ai = AIFactory.create(AIType.MCTS, player_number=1, config=config)

    # Create for tournament use
    ai = AIFactory.create_for_tournament("mcts_500", player_number=1)
"""

# Factory is safe to import (uses lazy loading internally)
from app.ai.factory import (
    AIFactory,
    DifficultyProfile,
    CANONICAL_DIFFICULTY_PROFILES,
    DIFFICULTY_DESCRIPTIONS,
    get_difficulty_profile,
    select_ai_type,
    get_randomness_for_difficulty,
    get_think_time_for_difficulty,
    uses_neural_net,
    get_all_difficulties,
    get_difficulty_description,
    create_ai,
    create_ai_from_difficulty,
    create_tournament_ai,
)

__all__ = [
    # Factory class
    "AIFactory",
    # Type definitions
    "DifficultyProfile",
    # Profile data
    "CANONICAL_DIFFICULTY_PROFILES",
    "DIFFICULTY_DESCRIPTIONS",
    # Helper functions
    "get_difficulty_profile",
    "select_ai_type",
    "get_randomness_for_difficulty",
    "get_think_time_for_difficulty",
    "uses_neural_net",
    "get_all_difficulties",
    "get_difficulty_description",
    # Convenience aliases
    "create_ai",
    "create_ai_from_difficulty",
    "create_tournament_ai",
]
