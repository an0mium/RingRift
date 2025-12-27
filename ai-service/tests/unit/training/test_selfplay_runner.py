"""Tests for unified SelfplayRunner.

Tests cover:
- SelfplayRunner imports and class structure
- GameResult and RunStats data classes
- Engine mode alias resolution
"""

import pytest

from app.training.selfplay_runner import (
    SelfplayRunner,
    HeuristicSelfplayRunner,
    GumbelMCTSSelfplayRunner,
    GameResult,
    RunStats,
    run_selfplay,
)
from app.training.selfplay_config import SelfplayConfig, EngineMode


class TestGameResult:
    """Test GameResult data class."""

    def test_create_basic_result(self):
        """Test creating a basic game result."""
        result = GameResult(
            game_id="test-123",
            winner=1,
            num_moves=42,
            duration_ms=1500.0,
        )
        assert result.game_id == "test-123"
        assert result.winner == 1
        assert result.num_moves == 42
        assert result.duration_ms == 1500.0
        assert result.moves == []
        assert result.samples == []

    def test_games_per_second_calculation(self):
        """Test games per second property."""
        result = GameResult(
            game_id="test",
            winner=None,
            num_moves=10,
            duration_ms=100.0,  # 0.1 seconds
        )
        # 1 game in 0.1 seconds = 10 games/sec
        assert result.games_per_second == 10.0

    def test_games_per_second_zero_duration(self):
        """Test games per second with zero duration."""
        result = GameResult(
            game_id="test",
            winner=None,
            num_moves=10,
            duration_ms=0.0,
        )
        assert result.games_per_second == 0.0


class TestRunStats:
    """Test RunStats aggregation class."""

    def test_initial_stats(self):
        """Test initial stats values."""
        stats = RunStats()
        assert stats.games_completed == 0
        assert stats.games_failed == 0
        assert stats.total_moves == 0
        assert stats.total_samples == 0

    def test_record_game(self):
        """Test recording a game result."""
        stats = RunStats()
        result = GameResult(
            game_id="test",
            winner=1,
            num_moves=50,
            duration_ms=1000.0,
            samples=[{"state": None}] * 5,
        )
        stats.record_game(result)

        assert stats.games_completed == 1
        assert stats.total_moves == 50
        assert stats.total_samples == 5
        assert stats.wins_by_player[1] == 1

    def test_record_multiple_games(self):
        """Test recording multiple games."""
        stats = RunStats()

        # Player 1 wins
        stats.record_game(GameResult(
            game_id="1", winner=1, num_moves=30, duration_ms=500.0,
        ))
        # Player 2 wins
        stats.record_game(GameResult(
            game_id="2", winner=2, num_moves=40, duration_ms=600.0,
        ))
        # Player 1 wins again
        stats.record_game(GameResult(
            game_id="3", winner=1, num_moves=35, duration_ms=550.0,
        ))

        assert stats.games_completed == 3
        assert stats.total_moves == 105
        assert stats.wins_by_player[1] == 2
        assert stats.wins_by_player[2] == 1


class TestHeuristicSelfplayRunner:
    """Test HeuristicSelfplayRunner initialization."""

    def test_init_sets_engine_mode(self):
        """Test that init sets correct engine mode."""
        config = SelfplayConfig(
            board_type="square8",
            num_players=2,
            num_games=1,
        )
        runner = HeuristicSelfplayRunner(config)
        assert runner.config.engine_mode == EngineMode.HEURISTIC
        assert runner.config.use_neural_net is False

    def test_init_from_config(self):
        """Test creating runner from config kwargs."""
        runner = HeuristicSelfplayRunner.from_config(
            board_type="hex8",
            num_players=2,
            num_games=5,
        )
        assert runner.config.board_type == "hex8"
        assert runner.config.num_players == 2
        assert runner.config.num_games == 5


class TestEngineModeAliases:
    """Test engine mode alias resolution in run_selfplay."""

    def test_heuristic_alias(self):
        """Test 'heuristic' resolves correctly."""
        # Just test the import path works - actual run is slow
        from app.training.selfplay_config import ENGINE_MODE_ALIASES
        assert "heuristic" in ENGINE_MODE_ALIASES
        assert ENGINE_MODE_ALIASES["heuristic"] == "heuristic-only"

    def test_gumbel_aliases(self):
        """Test gumbel aliases resolve correctly."""
        from app.training.selfplay_config import ENGINE_MODE_ALIASES
        assert "gumbel_mcts" in ENGINE_MODE_ALIASES
        assert "gumbel" in ENGINE_MODE_ALIASES


class TestSelfplayRunnerBase:
    """Test SelfplayRunner abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that SelfplayRunner cannot be instantiated directly."""
        config = SelfplayConfig(
            board_type="square8",
            num_players=2,
        )
        # Abstract class should raise TypeError on instantiation
        with pytest.raises(TypeError, match="abstract"):
            SelfplayRunner(config)

    def test_subclass_has_required_methods(self):
        """Test that concrete subclass has all required methods."""
        runner = HeuristicSelfplayRunner.from_config(
            board_type="square8",
            num_players=2,
            num_games=1,
        )
        assert hasattr(runner, 'run_game')
        assert hasattr(runner, 'run')
        assert hasattr(runner, 'setup')
        assert hasattr(runner, 'teardown')
        assert callable(runner.run_game)
        assert callable(runner.run)
