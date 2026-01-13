"""Concrete harness implementations.

This module contains the actual harness classes that wrap the underlying
AI implementations and adapt them to the unified AIHarness interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base_harness import AIHarness, HarnessConfig, ModelType

if TYPE_CHECKING:
    from ...models import GameState, Move

logger = logging.getLogger(__name__)


class GumbelMCTSHarness(AIHarness):
    """Harness for Gumbel MCTS with Sequential Halving."""

    supports_nn = True
    supports_nnue = False
    requires_policy_head = True

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..gumbel_mcts_ai import GumbelMCTSAI

        # Pass model_path via nn_model_id in config (not as direct parameter)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            simulations=self.config.simulations,
            nn_model_id=self.config.model_path,  # Jan 2026: Fix - use config field
            **self.config.extra,
        )
        ai = GumbelMCTSAI(player_number, ai_config, board_type=self.config.board_type)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        # Ensure AI is for correct player
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        # Extract visit distribution from tree
        self._last_visit_distribution = self._extract_visit_distribution()
        self._last_policy_distribution = self._extract_policy_distribution()

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_value', 0.0),
            "nodes_visited": getattr(self._underlying_ai, 'nodes_visited', 0),
            "search_depth": getattr(self._underlying_ai, 'max_depth_reached', None),
            "simulations": self.config.simulations,
        }
        return move, metadata

    def _extract_visit_distribution(self) -> dict[str, float] | None:
        """Extract visit distribution from Gumbel MCTS tree."""
        if not hasattr(self._underlying_ai, '_root_visits'):
            return None
        root_visits = getattr(self._underlying_ai, '_root_visits', None)
        if root_visits is None:
            return None
        return {str(k): float(v) for k, v in root_visits.items()}

    def _extract_policy_distribution(self) -> dict[str, float] | None:
        """Extract policy distribution from neural network."""
        if not hasattr(self._underlying_ai, '_root_policy'):
            return None
        root_policy = getattr(self._underlying_ai, '_root_policy', None)
        if root_policy is None:
            return None
        return {str(k): float(v) for k, v in root_policy.items()}


class GPUGumbelHarness(AIHarness):
    """Harness for GPU-accelerated Gumbel MCTS.

    Uses tensor_gumbel_tree.GPUGumbelMCTS for GPU-accelerated tree search.
    The search runs on GPU with batched rollouts and GPU-accelerated game simulation.

    Jan 2026 fix: Updated to match GPUGumbelMCTS interface which takes a config
    object and requires neural_net passed to search() method.
    """

    supports_nn = True
    supports_nnue = False
    requires_policy_head = True

    def __init__(self, config: HarnessConfig) -> None:
        """Initialize GPU Gumbel harness.

        Args:
            config: Harness configuration.
        """
        super().__init__(config)
        self._neural_net: Any = None  # Cached neural net for search()

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..tensor_gumbel_tree import GPUGumbelMCTS, GPUGumbelMCTSConfig

        # Create neural net for evaluation (cached for search calls)
        # Pass model_path via nn_model_id in config (not as direct parameter)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            simulations=self.config.simulations,
            nn_model_id=self.config.model_path,  # Jan 2026: Fix - use config field
            **self.config.extra,
        )
        from ..neural_net import NeuralNetAI
        self._neural_net = NeuralNetAI(player_number, ai_config, board_type=self.config.board_type)

        # Create GPU MCTS config
        gpu_config = GPUGumbelMCTSConfig(
            simulation_budget=self.config.simulations,
            num_sampled_actions=self.config.extra.get("num_sampled_actions", 16),
            max_nodes=self.config.extra.get("max_nodes", 1024),
            device=self.config.extra.get("device", "cuda"),
            eval_mode=self.config.extra.get("eval_mode", "heuristic"),
        )

        return GPUGumbelMCTS(gpu_config)

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        # Ensure neural net is created for this player
        if self._neural_net is None or self._neural_net.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        # GPUGumbelMCTS.search() returns (move, policy_dict) or (move, policy_dict, stats)
        # depending on code path (early exit vs full search)
        result = self._underlying_ai.search(game_state, self._neural_net)

        if len(result) == 3:
            move, policy_dict, stats = result
        else:
            move, policy_dict = result
            stats = None

        # Store visit distribution from returned policy
        if policy_dict:
            self._last_visit_distribution = {
                str(k): float(v) for k, v in policy_dict.items()
            }

        # Extract stats from SearchStats dataclass (if available)
        metadata = {
            "value_estimate": getattr(stats, 'root_value', 0.0) if stats else 0.0,
            "nodes_visited": getattr(stats, 'nodes_explored', 0) if stats else 0,
            "search_depth": getattr(stats, 'search_depth', None) if stats else None,
            "simulations": getattr(stats, 'total_simulations', self.config.simulations) if stats else 1,
            "extra": {
                "uncertainty": getattr(stats, 'uncertainty', 0.0) if stats else 0.0,
                "q_values": getattr(stats, 'q_values', {}) if stats else {},
            },
        }
        return move, metadata


class MinimaxHarness(AIHarness):
    """Harness for Minimax with alpha-beta pruning and Paranoid extension.

    For 2-player games: Standard alpha-beta minimax.
    For 3-4 player games: Paranoid algorithm where all opponents are treated
    as a minimizing coalition against the maximizing player.

    The underlying MinimaxAI handles both cases automatically.
    """

    supports_nn = True
    supports_nnue = True
    requires_policy_head = False

    # Supports 2-4 players (Paranoid algorithm for 3-4p)
    min_players = 2
    max_players = 4

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..minimax_ai import MinimaxAI

        use_nn = self.config.model_type in (ModelType.NEURAL_NET, ModelType.NNUE)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            use_neural_net=use_nn,
            nn_model_id=self.config.model_id if use_nn else None,
            **self.config.extra,
        )
        ai = MinimaxAI(player_number, ai_config)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_eval', 0.0),
            "nodes_visited": self._underlying_ai.nodes_visited,
            "search_depth": self._underlying_ai._get_max_depth(),
        }
        return move, metadata


class MaxNHarness(AIHarness):
    """Harness for Max-N multiplayer search.

    Max-N search models each player as self-interested (maximizing their own
    score). Each node returns a score vector (one per player).

    Works for 2-4 players:
    - 2 players: Equivalent to minimax but without alpha-beta pruning
    - 3-4 players: True multiplayer Max-N with score vectors

    For 2-player games, MinimaxHarness may be faster due to alpha-beta pruning.
    """

    supports_nn = True
    supports_nnue = True
    requires_policy_head = False

    # Supports 2-4 players (score vectors work for any count)
    min_players = 2
    max_players = 4

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..maxn_ai import MaxNAI

        use_nn = self.config.model_type in (ModelType.NEURAL_NET, ModelType.NNUE)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            use_neural_net=use_nn,
            **self.config.extra,
        )
        ai = MaxNAI(player_number, ai_config)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": 0.0,  # Max-N returns score vectors
            "nodes_visited": self._underlying_ai.nodes_visited,
            "search_depth": self._underlying_ai._get_max_depth(),
        }
        return move, metadata


class BRSHarness(AIHarness):
    """Harness for Best-Reply Search (BRS).

    BRS is a fast approximation to Max-N. It assumes opponents play greedily
    against the maximizing player, reducing the branching factor significantly.

    Works for 2-4 players:
    - 2 players: Similar to minimax with greedy opponent modeling
    - 3-4 players: Fast multiplayer search via best-reply approximation

    BRS is generally faster than Max-N due to reduced search space.
    """

    supports_nn = True
    supports_nnue = True
    requires_policy_head = False

    # Supports 2-4 players (best-reply works for any count)
    min_players = 2
    max_players = 4

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..maxn_ai import BRSAI

        use_nn = self.config.model_type in (ModelType.NEURAL_NET, ModelType.NNUE)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            use_neural_net=use_nn,
            **self.config.extra,
        )
        ai = BRSAI(player_number, ai_config)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": 0.0,
            "nodes_visited": self._underlying_ai.nodes_visited,
            "search_depth": self._underlying_ai._get_lookahead_rounds(),
        }
        return move, metadata


class PolicyOnlyHarness(AIHarness):
    """Harness for direct policy sampling (no search)."""

    supports_nn = True
    supports_nnue = True  # If NNUE has policy head
    requires_policy_head = True

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..policy_only_ai import PolicyOnlyAI

        # Pass model_path via nn_model_id in config (not as direct parameter)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            nn_model_id=self.config.model_path,  # Jan 2026: Fix - use config field
            **self.config.extra,
        )
        ai = PolicyOnlyAI(player_number, ai_config, board_type=self.config.board_type)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        # Get policy distribution
        policy = getattr(self._underlying_ai, '_last_policy', None)
        if policy is not None:
            self._last_policy_distribution = {
                str(k): float(v) for k, v in policy.items()
            }

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_value', 0.0),
            "nodes_visited": 1,  # Single evaluation
        }
        return move, metadata


class DescentHarness(AIHarness):
    """Harness for gradient descent move selection."""

    supports_nn = True
    supports_nnue = False
    requires_policy_head = True

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..descent_ai import DescentAI

        # Pass model_path via nn_model_id in config (not as direct parameter)
        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            nn_model_id=self.config.model_path,  # Jan 2026: Fix - use config field
            **self.config.extra,
        )
        ai = DescentAI(player_number, ai_config, board_type=self.config.board_type)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_value', 0.0),
            "nodes_visited": getattr(self._underlying_ai, 'iterations', 1),
        }
        return move, metadata


class HeuristicHarness(AIHarness):
    """Harness for pure heuristic evaluation (no neural network)."""

    supports_nn = False
    supports_nnue = False
    requires_policy_head = False

    def _create_underlying_ai(self, player_number: int) -> Any:
        from ...models import AIConfig
        from ..heuristic_ai import HeuristicAI

        ai_config = AIConfig(
            difficulty=self.config.difficulty,
            think_time=self.config.think_time_ms,
            **self.config.extra,
        )
        ai = HeuristicAI(player_number, ai_config)
        return ai

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        if self._underlying_ai.player_number != player_number:
            self._underlying_ai = self._create_underlying_ai(player_number)

        move = self._underlying_ai.select_move(game_state)

        metadata = {
            "value_estimate": getattr(self._underlying_ai, '_last_eval', 0.0),
            "nodes_visited": 1,
        }
        return move, metadata


class RandomHarness(AIHarness):
    """Harness that selects moves randomly (uniform distribution).

    Useful for:
    - Baseline comparison in gauntlets/evaluations
    - Training data diversity (playing against random opponent)
    - Sanity checking that models beat random
    """

    supports_nn = False
    supports_nnue = False
    requires_policy_head = False

    def _create_underlying_ai(self, player_number: int) -> Any:
        """No underlying AI needed - we just select randomly."""
        return None

    def _select_move_impl(
        self,
        game_state: GameState,
        player_number: int,
    ) -> tuple[Move | None, dict[str, Any]]:
        import random
        from ...rules import GameEngine

        valid_moves = GameEngine.get_valid_moves(game_state, player_number)
        if not valid_moves:
            return None, {"value_estimate": 0.0, "nodes_visited": 0}

        move = random.choice(valid_moves)
        metadata = {
            "value_estimate": 0.0,  # Random has no value estimate
            "nodes_visited": 0,  # No search performed
        }
        return move, metadata
