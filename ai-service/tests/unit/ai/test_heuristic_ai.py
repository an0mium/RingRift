"""
Unit tests for app.ai.heuristic_ai module.

Tests cover:
- HeuristicAI initialization
- Weight profile application
- Weight noise for training diversity
- Lazy evaluator properties
- Victory proximity calculation
- Module-level helper functions

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch

from app.ai.heuristic_ai import (
    HeuristicAI,
    _get_parallel_executor,
    _shutdown_parallel_executor,
)
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Create a basic AI config."""
    return AIConfig(difficulty=5, think_time=1000)


@pytest.fixture
def config_with_seed():
    """Create an AI config with RNG seed."""
    return AIConfig(difficulty=5, think_time=1000, rng_seed=42)


@pytest.fixture
def config_with_profile():
    """Create an AI config with heuristic profile."""
    config = AIConfig(difficulty=5, think_time=1000)
    config.heuristic_profile_id = "v1-heuristic-5"
    return config


@pytest.fixture
def config_with_noise():
    """Create an AI config with weight noise."""
    config = AIConfig(difficulty=5, think_time=1000, rng_seed=42)
    config.weight_noise = 0.1
    return config


@pytest.fixture
def mock_game_state():
    """Create a mock game state for testing."""
    state = MagicMock(spec=GameState)
    state.board = MagicMock(spec=BoardState)
    state.board.board_type = BoardType.SQUARE8
    state.board.size = 8
    state.current_player = 1
    state.phase = GamePhase.MOVEMENT
    state.status = GameStatus.ACTIVE
    state.victory_threshold = 3
    state.territory_victory_threshold = 10
    state.players = [
        MagicMock(player_number=1, eliminated_rings=0, territory_spaces=0),
        MagicMock(player_number=2, eliminated_rings=0, territory_spaces=0),
    ]
    return state


# =============================================================================
# HeuristicAI Initialization Tests
# =============================================================================


class TestHeuristicAIInitialization:
    """Tests for HeuristicAI initialization."""

    def test_basic_initialization(self, basic_config):
        """AI can be initialized with basic config."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        assert ai.player_number == 1
        assert ai.config == basic_config
        assert ai.eval_mode == "full"  # Default mode

    def test_initialization_with_light_eval_mode(self):
        """AI can be initialized with light evaluation mode."""
        config = AIConfig(difficulty=5, think_time=1000)
        config.heuristic_eval_mode = "light"

        ai = HeuristicAI(player_number=1, config=config)

        assert ai.eval_mode == "light"

    def test_initialization_with_full_eval_mode(self):
        """AI defaults to full mode for non-light values."""
        config = AIConfig(difficulty=5, think_time=1000)
        config.heuristic_eval_mode = "full"

        ai = HeuristicAI(player_number=1, config=config)

        assert ai.eval_mode == "full"

    def test_initialization_with_unknown_eval_mode(self):
        """AI defaults to full mode for unknown values."""
        config = AIConfig(difficulty=5, think_time=1000)
        config.heuristic_eval_mode = "unknown"

        ai = HeuristicAI(player_number=1, config=config)

        assert ai.eval_mode == "full"

    def test_initialization_with_incremental_search(self):
        """AI reads use_incremental_search from config."""
        config = AIConfig(difficulty=5, think_time=1000)
        config.use_incremental_search = True

        ai = HeuristicAI(player_number=1, config=config)

        assert ai.use_incremental_search is True

    def test_default_weight_constants(self, basic_config):
        """AI has default weight constants."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        # Check some key weights exist and are reasonable
        assert hasattr(ai, "WEIGHT_STACK_CONTROL")
        assert hasattr(ai, "WEIGHT_TERRITORY")
        assert hasattr(ai, "WEIGHT_MOBILITY")
        assert ai.WEIGHT_STACK_CONTROL > 0
        assert ai.WEIGHT_TERRITORY > 0


# =============================================================================
# Weight Profile Tests
# =============================================================================


class TestWeightProfileApplication:
    """Tests for weight profile application."""

    def test_no_profile_keeps_defaults(self, basic_config):
        """Without profile, class defaults are used."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        # Should have class-level default
        assert ai.WEIGHT_STACK_CONTROL == HeuristicAI.WEIGHT_STACK_CONTROL

    def test_profile_overrides_weights(self):
        """Profile overrides specific weights."""
        config = AIConfig(difficulty=5, think_time=1000)
        config.heuristic_profile_id = "v1-heuristic-5"

        with patch.dict(
            "app.ai.heuristic_ai.HEURISTIC_WEIGHT_PROFILES",
            {"v1-heuristic-5": {"WEIGHT_STACK_CONTROL": 15.0}},
        ):
            ai = HeuristicAI(player_number=1, config=config)

            # Should be overridden by profile
            assert ai.WEIGHT_STACK_CONTROL == 15.0

    def test_inferred_profile_from_difficulty(self):
        """Profile is inferred from difficulty when not explicit."""
        config = AIConfig(difficulty=3, think_time=1000)

        with patch.dict(
            "app.ai.heuristic_ai.HEURISTIC_WEIGHT_PROFILES",
            {"v1-heuristic-3": {"WEIGHT_TERRITORY": 12.0}},
        ):
            ai = HeuristicAI(player_number=1, config=config)

            assert ai.WEIGHT_TERRITORY == 12.0

    def test_unknown_profile_keeps_defaults(self):
        """Unknown profile ID keeps defaults."""
        config = AIConfig(difficulty=5, think_time=1000)
        config.heuristic_profile_id = "nonexistent-profile"

        ai = HeuristicAI(player_number=1, config=config)

        # Should still have class-level defaults
        assert ai.WEIGHT_STACK_CONTROL == HeuristicAI.WEIGHT_STACK_CONTROL


# =============================================================================
# Weight Noise Tests
# =============================================================================


class TestWeightNoise:
    """Tests for weight noise application."""

    def test_no_noise_by_default(self, basic_config):
        """Weights are not modified without noise config."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        # Should have exact class defaults
        assert ai.WEIGHT_STACK_CONTROL == HeuristicAI.WEIGHT_STACK_CONTROL

    def test_noise_modifies_weights(self, config_with_noise):
        """Weight noise modifies weights within range."""
        ai = HeuristicAI(player_number=1, config=config_with_noise)

        # With noise=0.1, weights should be within ±10% of default
        default = HeuristicAI.WEIGHT_STACK_CONTROL
        min_val = default * 0.9
        max_val = default * 1.1

        assert min_val <= ai.WEIGHT_STACK_CONTROL <= max_val

    def test_noise_is_reproducible_with_seed(self):
        """Same seed produces same noisy weights."""
        config1 = AIConfig(difficulty=5, think_time=1000, rng_seed=42)
        config1.weight_noise = 0.1
        config2 = AIConfig(difficulty=5, think_time=1000, rng_seed=42)
        config2.weight_noise = 0.1

        ai1 = HeuristicAI(player_number=1, config=config1)
        ai2 = HeuristicAI(player_number=1, config=config2)

        assert ai1.WEIGHT_STACK_CONTROL == ai2.WEIGHT_STACK_CONTROL

    def test_different_seeds_produce_different_noise(self):
        """Different seeds produce different noisy weights."""
        config1 = AIConfig(difficulty=5, think_time=1000, rng_seed=42)
        config1.weight_noise = 0.2
        config2 = AIConfig(difficulty=5, think_time=1000, rng_seed=99)
        config2.weight_noise = 0.2

        ai1 = HeuristicAI(player_number=1, config=config1)
        ai2 = HeuristicAI(player_number=1, config=config2)

        # Very unlikely to be exactly equal with different seeds
        assert ai1.WEIGHT_STACK_CONTROL != ai2.WEIGHT_STACK_CONTROL

    def test_noise_clamped_to_valid_range(self):
        """Noise above 1.0 is clamped."""
        config = AIConfig(difficulty=5, think_time=1000, rng_seed=42)
        config.weight_noise = 5.0  # Way too high

        ai = HeuristicAI(player_number=1, config=config)

        # Should not crash and weight should be within ±100% of default
        default = HeuristicAI.WEIGHT_STACK_CONTROL
        assert 0 <= ai.WEIGHT_STACK_CONTROL <= 2 * default


# =============================================================================
# Lazy Evaluator Tests
# =============================================================================


class TestLazyEvaluators:
    """Tests for lazy evaluator properties."""

    def test_swap_evaluator_created_on_access(self, basic_config):
        """Swap evaluator is created on first access."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        assert ai._swap_evaluator is None
        evaluator = ai.swap_evaluator
        assert evaluator is not None
        assert ai._swap_evaluator is evaluator

    def test_swap_evaluator_cached(self, basic_config):
        """Swap evaluator is cached after creation."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        evaluator1 = ai.swap_evaluator
        evaluator2 = ai.swap_evaluator

        assert evaluator1 is evaluator2

    def test_material_evaluator_created_on_access(self, basic_config):
        """Material evaluator is created on first access."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        assert ai._material_evaluator is None
        evaluator = ai.material_evaluator
        assert evaluator is not None

    def test_positional_evaluator_created_on_access(self, basic_config):
        """Positional evaluator is created on first access."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        assert ai._positional_evaluator is None
        evaluator = ai.positional_evaluator
        assert evaluator is not None

    def test_tactical_evaluator_created_on_access(self, basic_config):
        """Tactical evaluator is created on first access."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        assert ai._tactical_evaluator is None
        evaluator = ai.tactical_evaluator
        assert evaluator is not None

    def test_mobility_evaluator_created_on_access(self, basic_config):
        """Mobility evaluator is created on first access."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        assert ai._mobility_evaluator is None
        evaluator = ai.mobility_evaluator
        assert evaluator is not None

    def test_strategic_evaluator_created_on_access(self, basic_config):
        """Strategic evaluator is created on first access."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        assert ai._strategic_evaluator is None
        evaluator = ai.strategic_evaluator
        assert evaluator is not None

    def test_endgame_evaluator_created_on_access(self, basic_config):
        """Endgame evaluator is created on first access."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        assert ai._endgame_evaluator is None
        evaluator = ai.endgame_evaluator
        assert evaluator is not None


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestParallelExecutorFunctions:
    """Tests for parallel executor helper functions."""

    def test_get_parallel_executor_creates_executor(self):
        """Get parallel executor creates one if needed."""
        # Clean up any existing executor first
        _shutdown_parallel_executor()

        executor = _get_parallel_executor()
        assert executor is not None

        # Clean up
        _shutdown_parallel_executor()

    def test_get_parallel_executor_reuses_existing(self):
        """Get parallel executor reuses existing one."""
        _shutdown_parallel_executor()

        executor1 = _get_parallel_executor()
        executor2 = _get_parallel_executor()

        assert executor1 is executor2

        _shutdown_parallel_executor()

    def test_shutdown_parallel_executor(self):
        """Shutdown cleans up executor."""
        _get_parallel_executor()  # Ensure one exists
        _shutdown_parallel_executor()

        # After shutdown, getting executor should create new one
        import app.ai.heuristic_ai as module

        assert module._parallel_executor is None


# =============================================================================
# Victory Proximity Tests
# =============================================================================


class TestVictoryProximity:
    """Tests for victory proximity calculation."""

    def test_victory_proximity_at_victory(self, basic_config):
        """At victory threshold, returns high score."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        state = MagicMock()
        state.victory_threshold = 3
        state.territory_victory_threshold = 10
        state.lps_consecutive_exclusive_player = None

        player = MagicMock()
        player.player_number = 1
        player.eliminated_rings = 3  # At victory threshold
        player.territory_spaces = 5

        score = ai._victory_proximity_base_for_player(state, player)

        assert score == ai.WEIGHT_VICTORY_THRESHOLD_BONUS

    def test_victory_proximity_territory_win(self, basic_config):
        """Territory victory threshold triggers high score."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        state = MagicMock()
        state.victory_threshold = 10
        state.territory_victory_threshold = 5
        state.lps_consecutive_exclusive_player = None

        player = MagicMock()
        player.player_number = 1
        player.eliminated_rings = 0
        player.territory_spaces = 5  # At territory threshold

        score = ai._victory_proximity_base_for_player(state, player)

        assert score == ai.WEIGHT_VICTORY_THRESHOLD_BONUS

    def test_lps_victory_proximity(self, basic_config):
        """LPS rounds near threshold returns high score."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        state = MagicMock()
        state.victory_threshold = 10
        state.territory_victory_threshold = 20
        state.lps_consecutive_exclusive_player = 1
        state.lps_consecutive_exclusive_rounds = 3
        state.lps_rounds_required = 3

        player = MagicMock()
        player.player_number = 1
        player.eliminated_rings = 0
        player.territory_spaces = 0

        score = ai._victory_proximity_base_for_player(state, player)

        assert score == ai.WEIGHT_VICTORY_THRESHOLD_BONUS

    def test_lps_partial_progress(self, basic_config):
        """LPS partial progress returns scaled score."""
        ai = HeuristicAI(player_number=1, config=basic_config)

        state = MagicMock()
        state.victory_threshold = 10
        state.territory_victory_threshold = 20
        state.lps_consecutive_exclusive_player = 1
        state.lps_consecutive_exclusive_rounds = 1
        state.lps_rounds_required = 3

        player = MagicMock()
        player.player_number = 1
        player.eliminated_rings = 0
        player.territory_spaces = 0

        score = ai._victory_proximity_base_for_player(state, player)

        # Should be close to but less than max
        assert score > ai.WEIGHT_VICTORY_THRESHOLD_BONUS * 0.9
        assert score < ai.WEIGHT_VICTORY_THRESHOLD_BONUS
