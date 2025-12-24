"""Unified SelfplayRunner base class for all selfplay variants.

This module consolidates common patterns across the 20+ selfplay scripts into
a single base class that handles:
- Configuration loading (from selfplay_config.py)
- Model selection and hot reload (from selfplay_model_selector.py)
- Event coordination (from selfplay_orchestrator.py)
- Temperature scheduling
- Output handling (DB, JSONL, NPZ)
- Metrics and logging

Usage:
    from app.training.selfplay_runner import SelfplayRunner

    class MyCustomSelfplay(SelfplayRunner):
        def run_game(self, game_idx: int) -> GameResult:
            # Custom game logic
            ...

    runner = MyCustomSelfplay.from_cli()
    runner.run()
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .selfplay_config import SelfplayConfig, EngineMode, parse_selfplay_args

if TYPE_CHECKING:
    from ..models import BoardType, GameState, Move

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single selfplay game."""
    game_id: str
    winner: int | None
    num_moves: int
    duration_ms: float
    moves: list[dict] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)  # Training samples
    metadata: dict = field(default_factory=dict)

    @property
    def games_per_second(self) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return 1000.0 / self.duration_ms


@dataclass
class RunStats:
    """Aggregate statistics for a selfplay run."""
    games_completed: int = 0
    games_failed: int = 0
    total_moves: int = 0
    total_samples: int = 0
    total_duration_ms: float = 0.0
    wins_by_player: dict[int, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def games_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.games_completed / self.elapsed_seconds

    def record_game(self, result: GameResult) -> None:
        self.games_completed += 1
        self.total_moves += result.num_moves
        self.total_samples += len(result.samples)
        self.total_duration_ms += result.duration_ms
        if result.winner:
            self.wins_by_player[result.winner] = self.wins_by_player.get(result.winner, 0) + 1


class SelfplayRunner(ABC):
    """Base class for all selfplay implementations.

    Subclasses must implement:
    - run_game(game_idx) -> GameResult

    The base class handles:
    - Configuration parsing
    - Model loading and hot reload
    - Event emission
    - Output writing
    - Signal handling
    - Progress logging
    """

    def __init__(self, config: SelfplayConfig):
        self.config = config
        self.stats = RunStats()
        self.running = True
        self._model = None
        self._callbacks: list[Callable[[GameResult], None]] = []

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "SelfplayRunner":
        """Create runner from command-line arguments."""
        config = parse_selfplay_args(argv)
        return cls(config)

    @classmethod
    def from_config(cls, **kwargs) -> "SelfplayRunner":
        """Create runner from keyword arguments."""
        config = SelfplayConfig(**kwargs)
        return cls(config)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, stopping...")
        self.running = False

    @abstractmethod
    def run_game(self, game_idx: int) -> GameResult:
        """Run a single selfplay game. Must be implemented by subclasses."""
        ...

    def setup(self) -> None:
        """Called before run loop. Override for custom initialization."""
        logger.info(f"SelfplayRunner starting: {self.config.board_type}_{self.config.num_players}p")
        logger.info(f"  Engine: {self.config.engine_mode.value}")
        logger.info(f"  Target games: {self.config.num_games}")
        self._load_model()

    def teardown(self) -> None:
        """Called after run loop. Override for custom cleanup."""
        logger.info(f"SelfplayRunner finished: {self.stats.games_completed} games")
        logger.info(f"  Duration: {self.stats.elapsed_seconds:.1f}s")
        logger.info(f"  Throughput: {self.stats.games_per_second:.2f} games/sec")

    def _load_model(self) -> None:
        """Load neural network model if configured."""
        if not self.config.use_neural_net:
            return

        try:
            from .selfplay_model_selector import get_model_for_config
            model_path = get_model_for_config(
                self.config.board_type,
                self.config.num_players,
                prefer_nnue=self.config.prefer_nnue,
            )
            if model_path:
                logger.info(f"  Model: {model_path}")
                self._model = model_path
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")

    def on_game_complete(self, callback: Callable[[GameResult], None]) -> None:
        """Register callback for game completion events."""
        self._callbacks.append(callback)

    def _emit_game_complete(self, result: GameResult) -> None:
        """Emit game completion to registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def _emit_orchestrator_event(self) -> None:
        """Emit completion event to selfplay orchestrator if available."""
        try:
            from ..coordination.event_emitters import emit_event
            emit_event("SELFPLAY_BATCH_COMPLETE", {
                "games": self.stats.games_completed,
                "samples": self.stats.total_samples,
                "board_type": self.config.board_type,
                "num_players": self.config.num_players,
                "throughput": self.stats.games_per_second,
            })
        except ImportError:
            pass  # Event system not available

    def get_temperature(self, move_number: int) -> float:
        """Get temperature for move selection based on scheduling."""
        if move_number < self.config.temperature_threshold:
            return self.config.opening_temperature
        return self.config.base_temperature

    def run(self) -> RunStats:
        """Main run loop. Executes setup, games, teardown."""
        self.setup()

        try:
            game_idx = 0
            while self.running and game_idx < self.config.num_games:
                try:
                    result = self.run_game(game_idx)
                    self.stats.record_game(result)
                    self._emit_game_complete(result)

                    # Progress logging
                    if (game_idx + 1) % self.config.log_interval == 0:
                        logger.info(
                            f"  Progress: {game_idx + 1}/{self.config.num_games} games, "
                            f"{self.stats.games_per_second:.2f} g/s"
                        )

                except Exception as e:
                    logger.warning(f"Game {game_idx} failed: {e}")
                    self.stats.games_failed += 1

                game_idx += 1

        finally:
            self._emit_orchestrator_event()
            self.teardown()

        return self.stats


class HeuristicSelfplayRunner(SelfplayRunner):
    """Selfplay using heuristic AI (fast, no neural network)."""

    def __init__(self, config: SelfplayConfig):
        config.engine_mode = EngineMode.HEURISTIC
        config.use_neural_net = False
        super().__init__(config)
        self._engine = None
        self._ai = None

    def setup(self) -> None:
        super().setup()
        from ..game_engine import GameEngine
        from ..ai.factory import AIFactory
        from ..models import AIType, BoardType

        self._engine = GameEngine
        board_type = BoardType(self.config.board_type)

        # Create AI for each player
        self._ais = {}
        for p in range(1, self.config.num_players + 1):
            self._ais[p] = AIFactory.create(
                AIType.HEURISTIC,
                player_number=p,
                board_type=board_type,
                num_players=self.config.num_players,
            )

    def run_game(self, game_idx: int) -> GameResult:
        import uuid
        from ..training.initial_state import create_initial_state
        from ..models import BoardType

        start_time = time.time()
        game_id = str(uuid.uuid4())

        board_type = BoardType(self.config.board_type)
        state = create_initial_state(board_type, self.config.num_players)
        moves = []

        while not state.game_over and len(moves) < self.config.max_moves:
            current_player = state.current_player
            ai = self._ais[current_player]

            valid_moves = self._engine.get_valid_moves(state)
            if not valid_moves:
                break

            temperature = self.get_temperature(len(moves))
            move = ai.select_move(state, valid_moves, temperature=temperature)

            state = self._engine.apply_move(state, move)
            moves.append({"player": current_player, "move": str(move)})

        duration_ms = (time.time() - start_time) * 1000

        return GameResult(
            game_id=game_id,
            winner=getattr(state, "winner", None),
            num_moves=len(moves),
            duration_ms=duration_ms,
            moves=moves,
            metadata={"engine": "heuristic"},
        )


class GumbelMCTSSelfplayRunner(SelfplayRunner):
    """Selfplay using Gumbel MCTS (high quality, slower)."""

    def __init__(self, config: SelfplayConfig):
        config.engine_mode = EngineMode.GUMBEL_MCTS
        super().__init__(config)
        self._mcts = None

    def setup(self) -> None:
        super().setup()
        from ..ai.factory import create_mcts
        from ..models import BoardType
        from ..ai.gumbel_common import get_budget_for_difficulty

        board_type = BoardType(self.config.board_type)

        # Use budget based on config or difficulty
        budget = self.config.simulation_budget or get_budget_for_difficulty(
            self.config.difficulty or 8
        )

        self._mcts = create_mcts(
            board_type=board_type.value,
            num_players=self.config.num_players,
            mode="tensor" if self.config.use_gpu else "standard",
            simulation_budget=budget,
            device=self.config.device or "cuda",
        )

    def run_game(self, game_idx: int) -> GameResult:
        import uuid
        from ..training.initial_state import create_initial_state
        from ..models import BoardType
        from ..game_engine import GameEngine

        start_time = time.time()
        game_id = str(uuid.uuid4())

        board_type = BoardType(self.config.board_type)
        state = create_initial_state(board_type, self.config.num_players)
        moves = []
        samples = []

        while not state.game_over and len(moves) < self.config.max_moves:
            valid_moves = GameEngine.get_valid_moves(state)
            if not valid_moves:
                break

            # Get move from MCTS
            move = self._mcts.select_move(state, valid_moves)

            # Record sample for training
            if self.config.record_samples:
                samples.append({
                    "state": state,
                    "move": move,
                    "player": state.current_player,
                })

            state = GameEngine.apply_move(state, move)
            moves.append({"player": state.current_player, "move": str(move)})

        duration_ms = (time.time() - start_time) * 1000

        return GameResult(
            game_id=game_id,
            winner=getattr(state, "winner", None),
            num_moves=len(moves),
            duration_ms=duration_ms,
            moves=moves,
            samples=samples,
            metadata={"engine": "gumbel_mcts"},
        )


# Convenience function for quick selfplay
def run_selfplay(
    board_type: str = "square8",
    num_players: int = 2,
    num_games: int = 10,
    engine: str = "heuristic",
    **kwargs,
) -> RunStats:
    """Quick selfplay with minimal configuration.

    Args:
        board_type: Board type (square8, hex8, etc.)
        num_players: Number of players (2, 3, 4)
        num_games: Number of games to generate
        engine: Engine mode (heuristic, gumbel_mcts, etc.)
        **kwargs: Additional SelfplayConfig options

    Returns:
        RunStats with game results
    """
    config = SelfplayConfig(
        board_type=board_type,
        num_players=num_players,
        num_games=num_games,
        engine_mode=EngineMode(engine),
        **kwargs,
    )

    if engine == "heuristic":
        runner = HeuristicSelfplayRunner(config)
    elif engine == "gumbel_mcts":
        runner = GumbelMCTSSelfplayRunner(config)
    else:
        raise ValueError(f"Unknown engine: {engine}")

    return runner.run()
