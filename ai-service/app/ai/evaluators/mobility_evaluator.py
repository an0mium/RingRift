"""
MobilityEvaluator: Mobility evaluation for HeuristicAI.

This module provides the MobilityEvaluator class which handles all
mobility evaluation features:
- Pseudo-mobility (available legal-looking moves per stack)
- Stack mobility (individual stack movement options)
- Blocked stack penalties

Extracted from HeuristicAI as part of the decomposition plan
(docs/architecture/HEURISTIC_AI_DECOMPOSITION_PLAN.md).

Example usage:
    evaluator = MobilityEvaluator()
    score = evaluator.evaluate_mobility_all(game_state, player_idx=1)
    
    # Get detailed breakdown
    breakdown = evaluator.get_breakdown(game_state, player_idx=1)
    print(breakdown)  # {'mobility': 12.0, 'stack_mobility': 8.0, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...models import GameState
    from ..fast_geometry import FastGeometry


@dataclass
class MobilityWeights:
    """Weight configuration for mobility evaluation.
    
    These weights control the relative importance of each mobility feature
    in the overall evaluation. Default values match the HeuristicAI class
    constants for backward compatibility.
    
    Attributes:
        mobility: Weight for pseudo-mobility (legal-looking moves).
        stack_mobility: Weight for individual stack movement options.
        blocked_stack_penalty: Penalty for completely blocked stacks.
        safe_move_bonus: Bonus per safe move available.
        no_safe_moves_penalty: Penalty when no safe moves exist.
    """
    mobility: float = 4.0
    stack_mobility: float = 4.0
    blocked_stack_penalty: float = 5.0
    safe_move_bonus: float = 1.0
    no_safe_moves_penalty: float = 2.0
    
    @classmethod
    def from_heuristic_ai(cls, ai: "HeuristicAI") -> "MobilityWeights":
        """Create MobilityWeights from HeuristicAI instance weights.
        
        This factory method extracts the relevant WEIGHT_* attributes from
        a HeuristicAI instance to create a MobilityWeights configuration.
        
        Args:
            ai: HeuristicAI instance to extract weights from.
            
        Returns:
            MobilityWeights with values matching the AI's configuration.
        """
        return cls(
            mobility=getattr(ai, "WEIGHT_MOBILITY", 4.0),
            stack_mobility=getattr(ai, "WEIGHT_STACK_MOBILITY", 4.0),
            blocked_stack_penalty=getattr(
                ai, "WEIGHT_BLOCKED_STACK_PENALTY", 5.0
            ),
            safe_move_bonus=getattr(ai, "WEIGHT_SAFE_MOVE_BONUS", 1.0),
            no_safe_moves_penalty=getattr(
                ai, "WEIGHT_NO_SAFE_MOVES_PENALTY", 2.0
            ),
        )
    
    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary format.
        
        Returns:
            Dictionary with weight names as keys (WEIGHT_* format).
        """
        return {
            "WEIGHT_MOBILITY": self.mobility,
            "WEIGHT_STACK_MOBILITY": self.stack_mobility,
            "WEIGHT_BLOCKED_STACK_PENALTY": self.blocked_stack_penalty,
            "WEIGHT_SAFE_MOVE_BONUS": self.safe_move_bonus,
            "WEIGHT_NO_SAFE_MOVES_PENALTY": self.no_safe_moves_penalty,
        }


@dataclass
class MobilityScore:
    """Result from mobility evaluation with feature breakdown.
    
    Attributes:
        total: Sum of all mobility feature scores.
        mobility: Score from pseudo-mobility (legal-looking moves).
        stack_mobility: Score from individual stack movement options.
    """
    total: float = 0.0
    mobility: float = 0.0
    stack_mobility: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for breakdown reporting."""
        return {
            "total": self.total,
            "mobility": self.mobility,
            "stack_mobility": self.stack_mobility,
        }


# Type alias for HeuristicAI to avoid circular import
if TYPE_CHECKING:
    from ..heuristic_ai import HeuristicAI


class MobilityEvaluator:
    """Evaluates mobility factors: legal moves, movement options.
    
    This evaluator computes mobility-related features for position evaluation:
    
    Features computed:
    - mobility: Pseudo-mobility based on adjacent squares available for each
      stack, considering collapsed spaces and capturable enemy stacks.
    - stack_mobility: Per-stack movement options with penalties for completely
      blocked stacks.
    
    Both features are computed symmetrically (my_value - max_opponent_value)
    to ensure zero-sum across players.
    
    Example:
        evaluator = MobilityEvaluator()
        score = evaluator.evaluate_mobility_all(game_state, player_idx=1)
        
        # With custom weights
        weights = MobilityWeights(mobility=6.0, stack_mobility=5.0)
        evaluator = MobilityEvaluator(weights=weights)
    """
    
    def __init__(
        self,
        weights: Optional[MobilityWeights] = None,
        fast_geo: Optional["FastGeometry"] = None,
    ) -> None:
        """Initialize MobilityEvaluator with optional weight overrides.
        
        Args:
            weights: Optional MobilityWeights configuration. If None, uses
                default weights matching HeuristicAI class constants.
            fast_geo: Optional FastGeometry instance for adjacency lookup.
                If None, lazily fetches singleton instance when needed.
        """
        self.weights = weights or MobilityWeights()
        self._fast_geo = fast_geo
    
    @property
    def fast_geo(self) -> "FastGeometry":
        """Lazily get FastGeometry singleton for board geometry operations."""
        if self._fast_geo is None:
            from ..fast_geometry import FastGeometry
            self._fast_geo = FastGeometry.get_instance()
        return self._fast_geo
    
    def evaluate_mobility_all(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Compute total mobility score for a player.
        
        This is the main entry point for mobility evaluation. It computes
        all mobility features and returns the weighted sum.
        
        Args:
            state: Current game state to evaluate.
            player_idx: Player number (1-indexed) to evaluate for.
            
        Returns:
            Total weighted mobility score (positive = advantage for player).
        """
        result = self._compute_all_features(state, player_idx)
        return result.total
    
    def count_legal_moves(
        self,
        state: "GameState",
        player_idx: int,
    ) -> int:
        """Count pseudo-legal moves available for a player.
        
        This counts adjacent squares that look legal for movement (empty or
        capturable) without doing full move validation. This is faster than
        full move generation but may overcount in some edge cases.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            Count of pseudo-legal moves available.
        """
        board = state.board
        board_type = board.type
        stacks = board.stacks
        collapsed = board.collapsed_spaces
        
        move_count = 0
        for stack in stacks.values():
            if stack.controlling_player != player_idx:
                continue
            
            pos_key = stack.position.to_key()
            adj_keys = self.fast_geo.get_adjacent_keys(pos_key, board_type)
            for adj_key in adj_keys:
                if adj_key in collapsed:
                    continue
                if adj_key in stacks:
                    target = stacks[adj_key]
                    # Capture based on cap height per rules §10.1
                    if (target.controlling_player != player_idx
                            and stack.cap_height >= target.cap_height):
                        move_count += 1
                else:
                    move_count += 1
        
        return move_count
    
    def get_breakdown(
        self,
        state: "GameState",
        player_idx: int,
    ) -> dict[str, float]:
        """Get detailed breakdown of mobility evaluation.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            Dictionary with all feature scores and total.
        """
        result = self._compute_all_features(state, player_idx)
        return result.to_dict()
    
    def set_weights(self, weights: dict[str, float]) -> None:
        """Update weight values from a profile dictionary.
        
        This method allows dynamic weight updates from
        HEURISTIC_WEIGHT_PROFILES or other configuration sources.
        
        Args:
            weights: Dictionary with WEIGHT_* keys to update.
        """
        if "WEIGHT_MOBILITY" in weights:
            self.weights.mobility = weights["WEIGHT_MOBILITY"]
        if "WEIGHT_STACK_MOBILITY" in weights:
            self.weights.stack_mobility = weights["WEIGHT_STACK_MOBILITY"]
        if "WEIGHT_BLOCKED_STACK_PENALTY" in weights:
            val = weights["WEIGHT_BLOCKED_STACK_PENALTY"]
            self.weights.blocked_stack_penalty = val
        if "WEIGHT_SAFE_MOVE_BONUS" in weights:
            self.weights.safe_move_bonus = weights["WEIGHT_SAFE_MOVE_BONUS"]
        if "WEIGHT_NO_SAFE_MOVES_PENALTY" in weights:
            val = weights["WEIGHT_NO_SAFE_MOVES_PENALTY"]
            self.weights.no_safe_moves_penalty = val
    
    def _compute_all_features(
        self,
        state: "GameState",
        player_idx: int,
    ) -> MobilityScore:
        """Compute all mobility features and return detailed result.
        
        This is the internal workhorse that computes each feature
        independently. Made symmetric where appropriate.
        
        Args:
            state: Current game state.
            player_idx: Player number (1-indexed).
            
        Returns:
            MobilityScore with all feature values and total.
        """
        result = MobilityScore()
        
        # Pseudo-mobility (symmetric)
        result.mobility = self._evaluate_mobility(state, player_idx)
        
        # Stack mobility (symmetric)
        result.stack_mobility = self._evaluate_stack_mobility(
            state, player_idx
        )
        
        # Compute total
        result.total = result.mobility + result.stack_mobility
        
        return result
    
    def _evaluate_mobility(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate mobility (number of valid-looking moves).
        
        Uses pseudo-mobility instead of full move generation for performance.
        Full move generation is too expensive for evaluation function.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Mobility score (symmetric: my_mobility - opp_mobility).
        """
        board = state.board
        board_type = board.type
        stacks = board.stacks
        collapsed = board.collapsed_spaces
        
        # My pseudo-mobility
        my_stacks = [
            s for s in stacks.values()
            if s.controlling_player == player_idx
        ]
        my_mobility = 0
        for stack in my_stacks:
            # Use fast key-based adjacency lookup
            pos_key = stack.position.to_key()
            adj_keys = self.fast_geo.get_adjacent_keys(pos_key, board_type)
            for adj_key in adj_keys:
                if adj_key in collapsed:
                    continue
                if adj_key in stacks:
                    target = stacks[adj_key]
                    # Capture based on cap height per rules §10.1
                    if (target.controlling_player != player_idx
                            and stack.cap_height >= target.cap_height):
                        my_mobility += 1
                else:
                    my_mobility += 1
        
        # Opponent pseudo-mobility
        opp_stacks = [
            s for s in stacks.values()
            if s.controlling_player != player_idx
        ]
        opp_mobility = 0
        for stack in opp_stacks:
            # Use fast key-based adjacency lookup
            pos_key = stack.position.to_key()
            adj_keys = self.fast_geo.get_adjacent_keys(pos_key, board_type)
            for adj_key in adj_keys:
                if adj_key in collapsed:
                    continue
                if adj_key in stacks:
                    target = stacks[adj_key]
                    # Capture based on cap height per rules §10.1
                    if (target.controlling_player == player_idx
                            and stack.cap_height >= target.cap_height):
                        opp_mobility += 1
                else:
                    opp_mobility += 1
        
        score = (my_mobility - opp_mobility) * self.weights.mobility
        return score
    
    def _evaluate_stack_mobility(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate mobility of individual stacks (relative to opponents).
        
        Made symmetric: computes (my_mobility - max_opponent_mobility) so
        that the evaluation sums to approximately zero across all players.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Stack mobility score (symmetric: my_value - max_opponent_value).
        """
        board = state.board
        board_type = board.type
        stacks = board.stacks
        collapsed = board.collapsed_spaces
        
        def compute_mobility_for_player(player_num: int) -> float:
            """Compute raw mobility score for a player."""
            player_stacks = [
                s for s in stacks.values()
                if s.controlling_player == player_num
            ]
            mobility = 0.0
            for stack in player_stacks:
                pos_key = stack.position.to_key()
                adjacent_keys = self.fast_geo.get_adjacent_keys(
                    pos_key, board_type
                )
                valid_moves = 0
                for adj_key in adjacent_keys:
                    if adj_key in collapsed:
                        continue
                    if adj_key in stacks:
                        target = stacks[adj_key]
                        if (target.controlling_player != player_num
                                and stack.cap_height >= target.cap_height):
                            valid_moves += 1
                        continue
                    valid_moves += 1
                mobility += valid_moves
                if valid_moves == 0:
                    mobility -= self.weights.blocked_stack_penalty
            return mobility
        
        my_mobility = compute_mobility_for_player(player_idx)

        # Find max opponent mobility for symmetric evaluation
        # NOTE: We use float('-inf') as initial value to correctly handle
        # negative opponent mobilities (e.g., when opponent has only blocked
        # stacks). This ensures P1+P2 evals sum to 0 for true symmetry.
        opp_mobilities = [
            compute_mobility_for_player(p.player_number)
            for p in state.players
            if p.player_number != player_idx
        ]
        # If opponents have no stacks, their mobility is 0
        max_opp_mobility = max(opp_mobilities) if opp_mobilities else 0.0

        # Symmetric: advantage over best opponent
        advantage = my_mobility - max_opp_mobility
        return advantage * self.weights.stack_mobility
    
    # === Compatibility methods for HeuristicAI delegation ===
    # These methods match the original HeuristicAI method signatures
    # to enable gradual migration.
    
    def evaluate_mobility(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate pseudo-mobility feature only.
        
        Compatibility method matching HeuristicAI._evaluate_mobility
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Mobility score.
        """
        return self._evaluate_mobility(state, player_idx)
    
    def evaluate_stack_mobility(
        self,
        state: "GameState",
        player_idx: int,
    ) -> float:
        """Evaluate stack mobility feature only.
        
        Compatibility method matching HeuristicAI._evaluate_stack_mobility
        signature.
        
        Args:
            state: Current game state.
            player_idx: Player number.
            
        Returns:
            Stack mobility score.
        """
        return self._evaluate_stack_mobility(state, player_idx)
