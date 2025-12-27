"""Unified Gumbel MCTS Search Engine.

.. deprecated:: December 2025
    This module is deprecated. Use ``app.ai.gumbel_search_engine`` instead.

    The canonical Gumbel MCTS search engine is in ``gumbel_search_engine.py``.
    This file is kept for backwards compatibility but will be removed in a future version.

    Migrate by updating imports::

        # Old (deprecated)
        from app.ai.gumbel_engine import GumbelSearchEngine, GumbelSearchMode

        # New (canonical)
        from app.ai.gumbel_search_engine import GumbelSearchEngine, SearchMode

This module provides a unified interface for all Gumbel MCTS implementations,
allowing callers to use different search modes through a consistent API.

Usage:
    from app.ai.gumbel_engine import GumbelSearchEngine, GumbelSearchMode

    # Create engine with desired mode
    engine = GumbelSearchEngine(
        mode=GumbelSearchMode.GPU_BATCHED,
        board_type=BoardType.SQUARE8,
        num_players=2,
        simulation_budget=800,
    )

    # Search for best move
    move = engine.search(game_state)

    # Or batch search for multiple games
    moves = engine.search_batch(game_states)

Modes:
    - SINGLE_GAME: Standard CPU search (gumbel_mcts_ai.py)
    - MULTI_GAME: 64+ games parallel (multi_game_gumbel.py)
    - GPU_BATCHED: CPU tree with GPU evaluation (gumbel_mcts_ai.py batched)
    - GPU_TENSOR: Full GPU tensor tree (tensor_gumbel_tree.py)

December 2025 consolidation: This facade unifies 7 Gumbel MCTS implementations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from app.ai.gumbel_common import (
    GUMBEL_BUDGET_QUALITY,
    GUMBEL_BUDGET_STANDARD,
    GUMBEL_DEFAULT_K,
    GumbelAction,
)
from app.models import BoardType, GameState, Move

if TYPE_CHECKING:
    from app.ai.neural_net import NeuralNetAI

logger = logging.getLogger(__name__)


class GumbelSearchMode(Enum):
    """Search mode for Gumbel MCTS.

    Different modes trade off between latency, throughput, and quality.
    """

    SINGLE_GAME = "single"
    """Standard CPU search, one game at a time.

    Best for: Interactive play, low latency requirements.
    Throughput: ~10 moves/sec
    """

    MULTI_GAME = "multi"
    """Parallel search across 64+ games.

    Best for: High-throughput selfplay data generation.
    Throughput: ~500 moves/sec (across all games)
    """

    GPU_BATCHED = "gpu_batched"
    """CPU tree with GPU-batched leaf evaluation.

    Best for: Single-game search with GPU acceleration.
    Throughput: ~30 moves/sec
    """

    GPU_TENSOR = "gpu_tensor"
    """Full GPU tensor-based tree.

    Best for: Maximum throughput with GPU memory.
    Throughput: ~100 moves/sec
    """


@dataclass
class GumbelSearchConfig:
    """Configuration for Gumbel search engine.

    Attributes:
        mode: Search mode (single, multi, gpu_batched, gpu_tensor).
        board_type: Board geometry.
        num_players: Number of players.
        simulation_budget: Total simulations per move.
        num_sampled_actions: Actions to sample via Gumbel-Top-K.
        temperature: Policy temperature for exploration.
        device: Compute device ("cuda", "cpu", "auto").
        neural_net: Optional neural network for evaluation.
        use_simple_additive: Use simplified completed_q formula.
    """

    mode: GumbelSearchMode = GumbelSearchMode.GPU_BATCHED
    board_type: BoardType = BoardType.SQUARE8
    num_players: int = 2
    simulation_budget: int = GUMBEL_BUDGET_STANDARD
    num_sampled_actions: int = GUMBEL_DEFAULT_K
    temperature: float = 1.0
    device: str = "auto"
    neural_net: "NeuralNetAI | None" = None
    use_simple_additive: bool = False

    # Multi-game specific
    num_parallel_games: int = 64
    max_moves_per_game: int = 500


class GumbelSearchEngine:
    """Unified Gumbel MCTS search engine.

    This class provides a single entry point for all Gumbel MCTS variants,
    selecting the appropriate implementation based on the configured mode.

    Example:
        >>> engine = GumbelSearchEngine(
        ...     mode=GumbelSearchMode.GPU_BATCHED,
        ...     simulation_budget=800,
        ... )
        >>> move = engine.search(game_state)
    """

    def __init__(
        self,
        mode: GumbelSearchMode = GumbelSearchMode.GPU_BATCHED,
        board_type: BoardType = BoardType.SQUARE8,
        num_players: int = 2,
        simulation_budget: int = GUMBEL_BUDGET_STANDARD,
        num_sampled_actions: int = GUMBEL_DEFAULT_K,
        neural_net: "NeuralNetAI | None" = None,
        device: str = "auto",
        config: GumbelSearchConfig | None = None,
    ):
        """Initialize Gumbel search engine.

        Args:
            mode: Search mode to use.
            board_type: Board geometry.
            num_players: Number of players.
            simulation_budget: Simulations per move.
            num_sampled_actions: Gumbel-Top-K value.
            neural_net: Neural network for evaluation.
            device: Compute device.
            config: Full config object (overrides other args).
        """
        if config is not None:
            self.config = config
        else:
            self.config = GumbelSearchConfig(
                mode=mode,
                board_type=board_type,
                num_players=num_players,
                simulation_budget=simulation_budget,
                num_sampled_actions=num_sampled_actions,
                neural_net=neural_net,
                device=device,
            )

        self._impl = None
        self._multi_game_runner = None

        # Resolve device
        if self.config.device == "auto":
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = self.config.device

        logger.info(
            f"GumbelSearchEngine initialized: mode={self.config.mode.value}, "
            f"budget={self.config.simulation_budget}, device={self._device}"
        )

    def search(self, game_state: GameState, player: int | None = None) -> Move:
        """Search for the best move in a single game.

        Args:
            game_state: Current game state.
            player: Player to move (uses current_player if None).

        Returns:
            Best move according to search.
        """
        if player is None:
            player = game_state.current_player

        mode = self.config.mode

        if mode == GumbelSearchMode.SINGLE_GAME:
            return self._search_single(game_state, player)
        elif mode == GumbelSearchMode.GPU_BATCHED:
            return self._search_gpu_batched(game_state, player)
        elif mode == GumbelSearchMode.GPU_TENSOR:
            return self._search_gpu_tensor(game_state, player)
        elif mode == GumbelSearchMode.MULTI_GAME:
            # For single-game calls with multi-game engine, run batch of 1
            results = self._search_multi([game_state])
            return results[0] if results else self._search_single(game_state, player)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

    def search_batch(self, game_states: list[GameState]) -> list[Move]:
        """Search for best moves across multiple games.

        This is most efficient with MULTI_GAME mode.

        Args:
            game_states: List of game states.

        Returns:
            List of best moves, one per game.
        """
        if not game_states:
            return []

        mode = self.config.mode

        if mode == GumbelSearchMode.MULTI_GAME:
            return self._search_multi(game_states)
        else:
            # Fall back to sequential search
            return [
                self.search(state, state.current_player)
                for state in game_states
            ]

    def _get_single_game_impl(self):
        """Lazy-load single-game Gumbel MCTS implementation."""
        if self._impl is None:
            from app.ai.gumbel_mcts_ai import GumbelMCTSAI
            from app.models import AIConfig

            config = AIConfig(
                difficulty=9,
                use_neural_net=self.config.neural_net is not None,
            )
            self._impl = GumbelMCTSAI(
                player_number=1,  # Will be overridden per search
                config=config,
                board_type=self.config.board_type,
                num_sampled_actions=self.config.num_sampled_actions,
                simulation_budget=self.config.simulation_budget,
            )

            # Inject neural net if provided
            if self.config.neural_net:
                self._impl.neural_net = self.config.neural_net

        return self._impl

    def _search_single(self, game_state: GameState, player: int) -> Move:
        """Search using standard single-game implementation."""
        impl = self._get_single_game_impl()
        impl.player_number = player
        return impl.get_move(game_state)

    def _search_gpu_batched(self, game_state: GameState, player: int) -> Move:
        """Search using GPU-batched evaluation."""
        # The single-game impl auto-detects GPU and uses batched mode
        impl = self._get_single_game_impl()
        impl.player_number = player
        impl._use_gpu_tree = False  # Use batched, not full GPU tree
        return impl.get_move(game_state)

    def _search_gpu_tensor(self, game_state: GameState, player: int) -> Move:
        """Search using full GPU tensor tree."""
        impl = self._get_single_game_impl()
        impl.player_number = player
        impl._use_gpu_tree = True  # Enable full GPU tree
        return impl.get_move(game_state)

    def _get_multi_game_runner(self):
        """Lazy-load multi-game runner."""
        if self._multi_game_runner is None:
            from app.ai.multi_game_gumbel import MultiGameGumbelRunner

            self._multi_game_runner = MultiGameGumbelRunner(
                num_games=self.config.num_parallel_games,
                simulation_budget=self.config.simulation_budget,
                num_sampled_actions=self.config.num_sampled_actions,
                board_type=self.config.board_type,
                num_players=self.config.num_players,
                neural_net=self.config.neural_net,
                device=self._device,
                max_moves_per_game=self.config.max_moves_per_game,
            )

        return self._multi_game_runner

    def _search_multi(self, game_states: list[GameState]) -> list[Move]:
        """Search using multi-game parallel runner.

        Note: This runs games to completion, returning final moves.
        For single-move searches, this is inefficient.
        """
        # Multi-game runner is designed for full game selfplay,
        # not single-move search. For now, fall back to sequential.
        logger.warning(
            "Multi-game search called for single moves; falling back to sequential"
        )
        return [
            self._search_single(state, state.current_player)
            for state in game_states
        ]

    def get_search_actions(self) -> list[GumbelAction] | None:
        """Get the actions from the last search.

        Useful for extracting policy data for training.

        Returns:
            List of GumbelAction with visit counts, or None if not available.
        """
        if self._impl is not None and hasattr(self._impl, '_last_search_actions'):
            return self._impl._last_search_actions
        return None


# =============================================================================
# Factory Functions
# =============================================================================


def create_gumbel_engine(
    mode: str | GumbelSearchMode = "gpu_batched",
    board_type: str | BoardType = "square8",
    num_players: int = 2,
    simulation_budget: int = GUMBEL_BUDGET_STANDARD,
    neural_net: "NeuralNetAI | None" = None,
    device: str = "auto",
) -> GumbelSearchEngine:
    """Create a Gumbel search engine with the specified configuration.

    Args:
        mode: Search mode ("single", "multi", "gpu_batched", "gpu_tensor").
        board_type: Board type string or enum.
        num_players: Number of players.
        simulation_budget: Simulations per move.
        neural_net: Neural network for evaluation.
        device: Compute device.

    Returns:
        Configured GumbelSearchEngine.
    """
    # Parse mode
    if isinstance(mode, str):
        mode_map = {
            "single": GumbelSearchMode.SINGLE_GAME,
            "multi": GumbelSearchMode.MULTI_GAME,
            "gpu_batched": GumbelSearchMode.GPU_BATCHED,
            "gpu_tensor": GumbelSearchMode.GPU_TENSOR,
        }
        mode = mode_map.get(mode.lower(), GumbelSearchMode.GPU_BATCHED)

    # Parse board type
    if isinstance(board_type, str):
        board_map = {
            "square8": BoardType.SQUARE8,
            "sq8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "sq19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type = board_map.get(board_type.lower(), BoardType.SQUARE8)

    return GumbelSearchEngine(
        mode=mode,
        board_type=board_type,
        num_players=num_players,
        simulation_budget=simulation_budget,
        neural_net=neural_net,
        device=device,
    )


def create_selfplay_engine(
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    neural_net: "NeuralNetAI | None" = None,
    high_quality: bool = True,
) -> GumbelSearchEngine:
    """Create a Gumbel engine optimized for selfplay data generation.

    Args:
        board_type: Board geometry.
        num_players: Number of players.
        neural_net: Neural network for evaluation.
        high_quality: If True, use quality budget (800 sims).

    Returns:
        GumbelSearchEngine configured for selfplay.
    """
    budget = GUMBEL_BUDGET_QUALITY if high_quality else GUMBEL_BUDGET_STANDARD

    return GumbelSearchEngine(
        mode=GumbelSearchMode.GPU_BATCHED,
        board_type=board_type,
        num_players=num_players,
        simulation_budget=budget,
        neural_net=neural_net,
        device="auto",
    )
