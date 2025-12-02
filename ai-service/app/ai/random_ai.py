"""Random AI implementation for RingRift.

This agent selects uniformly random legal moves using the per‑instance RNG.
It is primarily intended for testing, baselines, and very low difficulties.
"""

from typing import Optional, Dict

from .base import BaseAI
from ..models import GameState, Move


class RandomAI(BaseAI):
    """AI that selects random valid moves."""

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """Select a random valid move for ``game_state``.

        Args:
            game_state: Current game state.

        Returns:
            A random valid :class:`Move` or ``None`` if no legal moves exist.
        """
        # Get all valid moves using the canonical rules engine
        valid_moves = self.rules_engine.get_valid_moves(
            game_state, self.player_number
        )
        
        if not valid_moves:
            return None
        
        # Select random move using the per-instance RNG.
        selected = self.get_random_element(valid_moves)
        
        self.move_count += 1
        return selected
    
    def evaluate_position(self, game_state: GameState) -> float:
        """Return a small random evaluation for ``game_state``.

        RandomAI does not attempt to evaluate positions meaningfully. It
        returns a small random value to introduce variance in diagnostic
        tooling that inspects scalar evaluations.

        Args:
            game_state: Current game state (unused).

        Returns:
            A small random float in ``[-0.1, 0.1]``.
        """
        return self.rng.uniform(-0.1, 0.1)
    
    def get_evaluation_breakdown(
        self, game_state: GameState
    ) -> Dict[str, float]:
        """
        Get evaluation breakdown for random AI
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary with random evaluation
        """
        return {
            "total": 0.0,
            "random_variance": self.rng.uniform(-0.1, 0.1)
        }
    
