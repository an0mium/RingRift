"""Unit tests for app/ai/hybrid_gpu.py

Tests cover:
- State conversion utilities (game_state_to_gpu_arrays, batch_game_states_to_gpu)
- HybridGPUEvaluator configuration and initialization
- AsyncEvalRequest and AsyncPipelineEvaluator
- HybridSelfPlayRunner configuration
- HybridNNAI initialization and move selection
- Factory functions (create_hybrid_evaluator)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Test imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_game_state():
    """Create a mock GameState that matches the expected structure."""
    state = MagicMock()

    # Mock board with stacks - need stack_height, controlling_player, cap_height
    mock_stack_1 = MagicMock()
    mock_stack_1.controlling_player = 1
    mock_stack_1.stack_height = 2
    mock_stack_1.cap_height = 0

    mock_stack_2 = MagicMock()
    mock_stack_2.controlling_player = 2
    mock_stack_2.stack_height = 1
    mock_stack_2.cap_height = 0

    mock_stacks = {
        "0,0": mock_stack_1,
        "1,1": mock_stack_2,
    }

    # Mock marker
    mock_marker = MagicMock()
    mock_marker.player = 1
    mock_markers = {"2,2": mock_marker}

    state.board.stacks = mock_stacks
    state.board.markers = mock_markers
    state.board.collapsed_spaces = {}  # Dict or set of "x,y" keys

    # Mock players as a list with player_number, rings_in_hand, eliminated_rings, territory_spaces
    mock_player_1 = MagicMock()
    mock_player_1.player_number = 1
    mock_player_1.rings_in_hand = 10
    mock_player_1.eliminated_rings = 2
    mock_player_1.territory_spaces = 5

    mock_player_2 = MagicMock()
    mock_player_2.player_number = 2
    mock_player_2.rings_in_hand = 12
    mock_player_2.eliminated_rings = 0
    mock_player_2.territory_spaces = 3

    state.players = [mock_player_1, mock_player_2]
    state.current_player = 1
    state.game_status = "active"
    state.winner = None

    return state


@pytest.fixture
def mock_config():
    """Create a mock AIConfig."""
    config = MagicMock()
    config.difficulty = 5
    config.think_time = 1000
    config.randomness = 0.1
    config.use_neural_net = True
    config.nn_model_id = None
    config.board_type = "square8"
    config.num_players = 2
    config.hybrid_top_k = 8
    config.hybrid_temperature = 0.1
    return config


# =============================================================================
# State Conversion Tests
# =============================================================================


class TestGameStateToGPUArrays:
    """Tests for game_state_to_gpu_arrays function."""

    def test_converts_empty_board(self, mock_game_state):
        """Empty board should produce zero arrays for stacks."""
        mock_game_state.board.stacks = {}
        mock_game_state.board.markers = {}
        mock_game_state.board.collapsed_spaces = {}

        from app.ai.hybrid_gpu import game_state_to_gpu_arrays

        result = game_state_to_gpu_arrays(mock_game_state, board_size=8)

        assert "stack_owner" in result
        assert "stack_height" in result
        assert result["stack_owner"].shape == (64,)
        assert result["stack_height"].shape == (64,)
        assert np.all(result["stack_owner"] == 0)
        assert np.all(result["stack_height"] == 0)

    def test_converts_populated_board(self, mock_game_state):
        """Board with stacks should have non-zero arrays."""
        from app.ai.hybrid_gpu import game_state_to_gpu_arrays

        result = game_state_to_gpu_arrays(mock_game_state, board_size=8)

        # Check stack_owner has the correct values at correct positions
        assert result["stack_owner"].shape == (64,)
        # Position (0,0) = index 0 should have player 1
        assert result["stack_owner"][0] == 1
        # Position (1,1) = index 1*8+1 = 9 should have player 2
        assert result["stack_owner"][9] == 2

    def test_different_board_sizes(self, mock_game_state):
        """Should work with different board sizes."""
        mock_game_state.board.stacks = {}
        mock_game_state.board.markers = {}
        mock_game_state.board.collapsed_spaces = {}

        from app.ai.hybrid_gpu import game_state_to_gpu_arrays

        result_8 = game_state_to_gpu_arrays(mock_game_state, board_size=8)
        result_19 = game_state_to_gpu_arrays(mock_game_state, board_size=19)

        assert result_8["stack_owner"].shape == (64,)
        assert result_19["stack_owner"].shape == (361,)


class TestBatchGameStatesToGPU:
    """Tests for batch_game_states_to_gpu function."""

    @patch("app.ai.hybrid_gpu.GPUBoardState")
    def test_empty_batch(self, mock_gpu_state, mock_game_state):
        """Empty batch should work."""
        from app.ai.hybrid_gpu import batch_game_states_to_gpu

        mock_gpu_state.from_numpy_batch.return_value = MagicMock()
        device = torch.device("cpu")

        result = batch_game_states_to_gpu([], device, board_size=8)

        mock_gpu_state.from_numpy_batch.assert_called_once()

    @patch("app.ai.hybrid_gpu.GPUBoardState")
    def test_single_state_batch(self, mock_gpu_state, mock_game_state):
        """Single state batch should work."""
        from app.ai.hybrid_gpu import batch_game_states_to_gpu

        mock_gpu_state.from_numpy_batch.return_value = MagicMock()
        device = torch.device("cpu")

        result = batch_game_states_to_gpu([mock_game_state], device, board_size=8)

        mock_gpu_state.from_numpy_batch.assert_called_once()
        call_args = mock_gpu_state.from_numpy_batch.call_args
        assert len(call_args[0][0]) == 1  # One state dict

    @patch("app.ai.hybrid_gpu.GPUBoardState")
    def test_multiple_states_batch(self, mock_gpu_state, mock_game_state):
        """Multiple states should be batched."""
        from app.ai.hybrid_gpu import batch_game_states_to_gpu

        mock_gpu_state.from_numpy_batch.return_value = MagicMock()
        device = torch.device("cpu")
        states = [mock_game_state, mock_game_state, mock_game_state]

        result = batch_game_states_to_gpu(states, device, board_size=8)

        call_args = mock_gpu_state.from_numpy_batch.call_args
        assert len(call_args[0][0]) == 3  # Three state dicts


# =============================================================================
# HybridGPUEvaluator Tests
# =============================================================================


class TestHybridGPUEvaluator:
    """Tests for HybridGPUEvaluator class."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_initialization_cpu(self, mock_heuristic, mock_get_device):
        """Should initialize with CPU device."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device="cpu", board_size=8, num_players=2)

        assert evaluator.board_size == 8
        assert evaluator.num_players == 2
        assert evaluator.device == torch.device("cpu")

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_default_board_size(self, mock_heuristic, mock_get_device):
        """Default board size should be 8."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device=None)

        assert evaluator.board_size == 8

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_use_heuristic_flag(self, mock_heuristic, mock_get_device):
        """Should respect use_heuristic flag."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device="cpu", use_heuristic=False)

        assert evaluator.use_heuristic is False


# =============================================================================
# AsyncEvalRequest Tests
# =============================================================================


class TestAsyncEvalRequest:
    """Tests for AsyncEvalRequest dataclass."""

    def test_creation(self, mock_game_state):
        """Should create request with required fields."""
        from app.ai.hybrid_gpu import AsyncEvalRequest

        callback = MagicMock()
        request = AsyncEvalRequest(
            game_state=mock_game_state,
            moves=["move1", "move2"],
            player_number=1,
            callback=callback,
        )

        assert request.game_state is mock_game_state
        assert request.moves == ["move1", "move2"]
        assert request.player_number == 1
        assert request.callback is callback

    def test_default_timestamp(self, mock_game_state):
        """Should have default timestamp."""
        from app.ai.hybrid_gpu import AsyncEvalRequest

        callback = MagicMock()
        request = AsyncEvalRequest(
            game_state=mock_game_state,
            moves=[],
            player_number=1,
            callback=callback,
        )

        assert request.timestamp > 0


# =============================================================================
# AsyncPipelineEvaluator Tests
# =============================================================================


class TestAsyncPipelineEvaluator:
    """Tests for AsyncPipelineEvaluator class."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_initialization(self, mock_heuristic, mock_get_device):
        """Should initialize with hybrid evaluator."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator, AsyncPipelineEvaluator

        evaluator = HybridGPUEvaluator(device="cpu")
        rules_engine = MagicMock()

        pipeline = AsyncPipelineEvaluator(evaluator, rules_engine, batch_size=32)

        assert pipeline.evaluator is evaluator
        assert pipeline.batch_size == 32

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_start_stop(self, mock_heuristic, mock_get_device):
        """Should start and stop cleanly."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator, AsyncPipelineEvaluator

        evaluator = HybridGPUEvaluator(device="cpu")
        rules_engine = MagicMock()

        pipeline = AsyncPipelineEvaluator(evaluator, rules_engine)
        pipeline.start()

        assert pipeline._running is True

        pipeline.stop()

        assert pipeline._running is False


# =============================================================================
# HybridSelfPlayRunner Tests
# =============================================================================


class TestHybridSelfPlayRunner:
    """Tests for HybridSelfPlayRunner class."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_initialization(self, mock_heuristic, mock_get_device):
        """Should initialize with evaluator."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator, HybridSelfPlayRunner

        evaluator = HybridGPUEvaluator(device="cpu")

        runner = HybridSelfPlayRunner(evaluator, board_type="square8", num_players=2)

        assert runner.evaluator is evaluator
        assert runner.board_type == "square8"
        assert runner.num_players == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateHybridEvaluator:
    """Tests for create_hybrid_evaluator function."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_creates_evaluator(self, mock_heuristic, mock_get_device):
        """Should create evaluator with specified config."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import create_hybrid_evaluator

        evaluator = create_hybrid_evaluator(
            board_type="square8",
            num_players=2,
            prefer_gpu=False,
        )

        assert evaluator.num_players == 2

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_default_device(self, mock_heuristic, mock_get_device):
        """Should use auto-detected device."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import create_hybrid_evaluator

        evaluator = create_hybrid_evaluator()

        assert evaluator.device is not None


# =============================================================================
# HybridNNAI Tests
# =============================================================================


class TestHybridNNAI:
    """Tests for HybridNNAI class."""

    @patch("app.ai.hybrid_gpu.HybridNNValuePlayer")
    def test_initialization(self, mock_value_player, mock_config):
        """Should initialize HybridNNAI."""
        mock_value_player.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridNNAI

        ai = HybridNNAI(player_number=1, config=mock_config)

        assert ai.player_number == 1

    @patch("app.ai.hybrid_gpu.HybridNNValuePlayer")
    def test_inherits_from_base_ai(self, mock_value_player, mock_config):
        """Should inherit from BaseAI."""
        mock_value_player.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridNNAI
        from app.ai.base import BaseAI

        ai = HybridNNAI(player_number=1, config=mock_config)

        assert isinstance(ai, BaseAI)


# =============================================================================
# HybridNNValuePlayer Tests
# =============================================================================


class TestHybridNNValuePlayer:
    """Tests for HybridNNValuePlayer class."""

    @patch("app.ai.hybrid_gpu.create_hybrid_evaluator")
    def test_initialization(self, mock_create_eval):
        """Should initialize with fallback when no NN available."""
        mock_create_eval.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridNNValuePlayer

        # Should not raise even if NeuralNetAI fails (no model available)
        player = HybridNNValuePlayer(
            board_type="square8",
            num_players=2,
            player_number=1,
        )

        assert player.board_type == "square8"
        assert player.num_players == 2
        # neural_net may be None or a NeuralNetAI instance depending on model availability

    @patch("app.ai.hybrid_gpu.create_hybrid_evaluator")
    def test_with_top_k(self, mock_create_eval):
        """Should accept top_k parameter."""
        mock_create_eval.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridNNValuePlayer

        player = HybridNNValuePlayer(
            board_type="square8",
            num_players=2,
            player_number=1,
            top_k=10,
        )

        assert player.top_k == 10


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_invalid_board_size(self, mock_heuristic, mock_get_device):
        """Should handle unusual board sizes."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        # Very small board
        evaluator = HybridGPUEvaluator(device="cpu", board_size=4)
        assert evaluator.board_size == 4

        # Very large board
        evaluator = HybridGPUEvaluator(device="cpu", board_size=25)
        assert evaluator.board_size == 25

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_empty_moves_evaluation(self, mock_heuristic, mock_get_device, mock_game_state):
        """Should handle empty moves list."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device="cpu")
        rules_engine = MagicMock()

        results = evaluator.evaluate_moves(mock_game_state, [], 1, rules_engine)

        assert results == []

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_empty_positions_evaluation(self, mock_heuristic, mock_get_device):
        """Should handle empty positions list."""
        mock_get_device.return_value = torch.device("cpu")
        mock_heuristic.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device="cpu")

        scores = evaluator.evaluate_positions([], 1)

        assert len(scores) == 0
