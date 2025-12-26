"""Fast heuristic feature extraction for V5 neural network training.

This module provides efficient extraction of heuristic features from game states
for use as neural network inputs. Unlike the slow `extract_linear_features` approach
which requires 50 evaluations per state, this uses `_compute_component_scores`
directly for O(1) extraction per state.

The extracted features capture the key strategic signals used by HeuristicAI:
- Material: stack control, rings, eliminations, markers
- Positional: territory, center control, closure, safety
- Tactical: opponent threats, vulnerability, overtake potential
- Mobility: movement options, stack mobility
- Strategic: victory proximity, opponent threats, LPS advantage

Usage:
    from app.training.fast_heuristic_features import (
        extract_heuristic_features,
        HEURISTIC_FEATURE_NAMES,
        NUM_HEURISTIC_FEATURES,
    )

    # Single state
    features = extract_heuristic_features(game_state, player_number=1)
    print(f"Features shape: {features.shape}")  # (20,)

    # Batch extraction
    features_batch = extract_heuristic_features_batch(states, player_numbers)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import GameState

# Canonical ordered list of heuristic feature names
# These correspond to the keys returned by HeuristicAI._compute_component_scores()
HEURISTIC_FEATURE_NAMES: tuple[str, ...] = (
    # Tier 0 (core)
    "stack_control",
    "territory",
    "rings_in_hand",
    "center_control",
    "eliminated_rings",
    "marker_count",
    # Tier 1 (local/mobility)
    "opponent_threats",
    "mobility",
    "stack_mobility",
    # Tier 1 (strategic)
    "victory_proximity",
    "opponent_victory_threat",
    "multi_leader_threat",
    # Tier 2 (structural) - only computed in "full" mode
    "line_potential",
    "line_connectivity",
    "vulnerability",
    "overtake_potential",
    "territory_closure",
    "territory_safety",
    "forced_elimination_risk",
    "lps_action_advantage",
    "recovery_potential",
)

NUM_HEURISTIC_FEATURES = len(HEURISTIC_FEATURE_NAMES)

# Default feature order for consistent indexing
_FEATURE_INDEX = {name: i for i, name in enumerate(HEURISTIC_FEATURE_NAMES)}


def extract_heuristic_features(
    game_state: "GameState",
    player_number: int,
    eval_mode: str = "full",
    normalize: bool = True,
) -> np.ndarray:
    """Extract heuristic features from a game state.

    This is a fast O(1) extraction that directly computes heuristic component
    scores without the 49x overhead of the linear feature decomposition.

    Args:
        game_state: The game state to evaluate
        player_number: Perspective player (1-indexed)
        eval_mode: Evaluation mode ("full" or "light")
        normalize: Whether to normalize features to reasonable ranges

    Returns:
        Feature vector of shape (NUM_HEURISTIC_FEATURES,) = (21,)
    """
    from app.ai.heuristic_ai import HeuristicAI
    from app.models import AIConfig

    # Create a lightweight HeuristicAI instance
    ai = HeuristicAI(
        player_number=player_number,
        config=AIConfig(difficulty=4, randomness=0.0),
    )
    ai.eval_mode = eval_mode

    # Get component scores (this is the efficient path)
    scores = ai._compute_component_scores(game_state)

    # Convert to ordered feature vector
    features = np.zeros(NUM_HEURISTIC_FEATURES, dtype=np.float32)
    for name, value in scores.items():
        if name in _FEATURE_INDEX:
            features[_FEATURE_INDEX[name]] = value

    if normalize:
        # Normalize to roughly [-1, 1] range based on typical value ranges
        # These are heuristic scaling factors based on observed ranges
        normalization_scales = {
            "stack_control": 100.0,
            "territory": 100.0,
            "rings_in_hand": 30.0,
            "center_control": 50.0,
            "eliminated_rings": 100.0,
            "marker_count": 50.0,
            "opponent_threats": 50.0,
            "mobility": 100.0,
            "stack_mobility": 50.0,
            "victory_proximity": 200.0,
            "opponent_victory_threat": 100.0,
            "multi_leader_threat": 50.0,
            "line_potential": 100.0,
            "line_connectivity": 50.0,
            "vulnerability": 100.0,
            "overtake_potential": 100.0,
            "territory_closure": 100.0,
            "territory_safety": 50.0,
            "forced_elimination_risk": 50.0,
            "lps_action_advantage": 30.0,
            "recovery_potential": 50.0,
        }
        for name, scale in normalization_scales.items():
            if name in _FEATURE_INDEX:
                idx = _FEATURE_INDEX[name]
                features[idx] = np.clip(features[idx] / scale, -1.0, 1.0)

    return features


def extract_heuristic_features_batch(
    game_states: list["GameState"],
    player_numbers: list[int],
    eval_mode: str = "full",
    normalize: bool = True,
) -> np.ndarray:
    """Batch extract heuristic features from multiple game states.

    Args:
        game_states: List of game states
        player_numbers: List of perspective players (1-indexed)
        eval_mode: Evaluation mode ("full" or "light")
        normalize: Whether to normalize features

    Returns:
        Feature array of shape (N, NUM_HEURISTIC_FEATURES)
    """
    if len(game_states) != len(player_numbers):
        raise ValueError(
            f"game_states and player_numbers must have same length, "
            f"got {len(game_states)} and {len(player_numbers)}"
        )

    if not game_states:
        return np.zeros((0, NUM_HEURISTIC_FEATURES), dtype=np.float32)

    features_list = []
    for state, player in zip(game_states, player_numbers):
        features = extract_heuristic_features(
            state, player, eval_mode=eval_mode, normalize=normalize
        )
        features_list.append(features)

    return np.stack(features_list, axis=0)


# Mapping from 49 HEURISTIC_WEIGHT_KEYS to fast feature indices
# For weights that don't have direct component mappings, we use -1
def get_weight_to_feature_mapping() -> dict[str, int]:
    """Get mapping from weight keys to fast feature indices.

    Some weights (like line pattern weights) don't have direct mappings
    because they're internal to the line_potential computation.

    Returns:
        Dict mapping weight key -> feature index (or -1 if no direct mapping)
    """
    from app.ai.heuristic_weights import HEURISTIC_WEIGHT_KEYS

    # Map weight keys to feature names (strip WEIGHT_ prefix, lowercase)
    mapping = {}
    for weight_key in HEURISTIC_WEIGHT_KEYS:
        # Convert WEIGHT_STACK_CONTROL -> stack_control
        feature_name = weight_key.replace("WEIGHT_", "").lower()

        # Handle special cases
        special_mappings = {
            "opponent_threat": "opponent_threats",
            "stack_height": "stack_control",  # Combined in stack_control
            "cap_height": "stack_control",    # Combined in stack_control
            "adjacency": "opponent_threats",  # Part of threat computation
        }
        feature_name = special_mappings.get(feature_name, feature_name)

        if feature_name in _FEATURE_INDEX:
            mapping[weight_key] = _FEATURE_INDEX[feature_name]
        else:
            mapping[weight_key] = -1  # No direct mapping

    return mapping


__all__ = [
    "extract_heuristic_features",
    "extract_heuristic_features_batch",
    "HEURISTIC_FEATURE_NAMES",
    "NUM_HEURISTIC_FEATURES",
    "get_weight_to_feature_mapping",
]
