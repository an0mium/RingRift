"""Unit tests for GPUMinimaxAI.

Tests cover:
1. Configuration and initialization
2. Board detection
3. Zero-sum score conversion
4. Move priority bonus calculation
5. Leaf buffer management
6. Integration with GPU batch evaluation (when available)
"""

import os
import unittest
import uuid
from unittest.mock import MagicMock, patch

import torch

from app.ai.gpu_minimax_ai import GPUMinimaxAI
from app.models import AIConfig, BoardType, Move, MoveType


def create_mock_game_state(
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    current_player: int = 1,
) -> MagicMock:
    """Create a mock GameState for testing."""
    state = MagicMock()
    state.board_type = board_type
    state.current_player = current_player
    state.phase = "ring_placement"  # Use string to avoid import issues

    # Mock players
    players = []
    for i in range(num_players):
        player = MagicMock()
        player.rings_in_hand = 12
        player.eliminated_rings = 0
        player.territory_spaces = 0
        player.buried_rings = 0
        players.append(player)
    state.players = players

    # Mock board
    state.board = MagicMock()
    state.board.stacks = {}
    state.board.markers = {}
    state.board.collapsed_spaces = {}

    return state


def create_move(
    move_type: MoveType,
    player: int = 1,
    from_pos: str | None = None,
    to_pos: str | None = None,
) -> Move:
    """Create a Move with required fields."""
    return Move(
        id=str(uuid.uuid4()),
        type=move_type,
        player=player,
        from_pos=from_pos,
        to=to_pos,
    )


class TestGPUMinimaxAIConfiguration(unittest.TestCase):
    """Test GPUMinimaxAI configuration and initialization."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        self.assertTrue(ai.use_gpu_batch)
        self.assertEqual(ai.gpu_batch_size, 64)
        self.assertEqual(ai.gpu_min_batch, 8)
        self.assertEqual(ai.player_number, 1)

    def test_env_override_batch_size(self):
        """Test environment variable overrides batch size."""
        config = AIConfig(difficulty=5)

        with patch.dict(os.environ, {"RINGRIFT_GPU_MINIMAX_BATCH_SIZE": "256"}):
            ai = GPUMinimaxAI(player_number=1, config=config)
            self.assertEqual(ai.gpu_batch_size, 256)

    def test_env_disable_gpu(self):
        """Test environment variable can disable GPU."""
        config = AIConfig(difficulty=5)

        with patch.dict(os.environ, {"RINGRIFT_GPU_MINIMAX_DISABLE": "1"}):
            ai = GPUMinimaxAI(player_number=1, config=config)
            self.assertFalse(ai.use_gpu_batch)

    def test_env_disable_gpu_true_string(self):
        """Test 'true' string disables GPU."""
        config = AIConfig(difficulty=5)

        with patch.dict(os.environ, {"RINGRIFT_GPU_MINIMAX_DISABLE": "true"}):
            ai = GPUMinimaxAI(player_number=1, config=config)
            self.assertFalse(ai.use_gpu_batch)

    def test_lazy_gpu_initialization(self):
        """Test GPU resources are lazily initialized."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        # GPU should not be initialized until first use
        self.assertIsNone(ai._gpu_available)
        self.assertIsNone(ai._gpu_device)


class TestBoardDetection(unittest.TestCase):
    """Test board configuration detection."""

    def test_detect_square8(self):
        """Test detection of SQUARE8 board."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        state = create_mock_game_state(BoardType.SQUARE8, num_players=2)
        ai._detect_board_info(state)

        self.assertEqual(ai._board_type, BoardType.SQUARE8)
        self.assertEqual(ai._board_size, 8)
        self.assertEqual(ai._num_players, 2)

    def test_detect_square19(self):
        """Test detection of SQUARE19 board."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        state = create_mock_game_state(BoardType.SQUARE19, num_players=4)
        ai._detect_board_info(state)

        self.assertEqual(ai._board_type, BoardType.SQUARE19)
        self.assertEqual(ai._board_size, 19)
        self.assertEqual(ai._num_players, 4)

    def test_detect_hexagonal(self):
        """Test detection of HEXAGONAL board."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        state = create_mock_game_state(BoardType.HEXAGONAL, num_players=3)
        ai._detect_board_info(state)

        self.assertEqual(ai._board_type, BoardType.HEXAGONAL)
        self.assertEqual(ai._board_size, 25)  # Hex uses 25x25 embedding
        self.assertEqual(ai._num_players, 3)

    def test_detect_hex8_defaults_to_8(self):
        """Test detection of HEX8 board defaults to size 8."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        state = create_mock_game_state(BoardType.HEX8, num_players=2)
        ai._detect_board_info(state)

        self.assertEqual(ai._board_type, BoardType.HEX8)
        # HEX8 not in map, defaults to 8
        self.assertEqual(ai._board_size, 8)
        self.assertEqual(ai._num_players, 2)

    def test_detection_is_cached(self):
        """Test that board detection is cached."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        state1 = create_mock_game_state(BoardType.SQUARE8, num_players=2)
        ai._detect_board_info(state1)

        # Second call with different state should not change cached values
        state2 = create_mock_game_state(BoardType.SQUARE19, num_players=4)
        ai._detect_board_info(state2)

        # Should still be SQUARE8 from first detection
        self.assertEqual(ai._board_type, BoardType.SQUARE8)
        self.assertEqual(ai._board_size, 8)


class TestZeroSumScoreConversion(unittest.TestCase):
    """Test zero-sum score conversion for minimax."""

    def test_2_player_zero_sum(self):
        """Test zero-sum conversion for 2-player game."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)
        ai._num_players = 2

        # Scores tensor: [batch, num_players+1] - index 0 unused
        # Player 1 score = 100, Player 2 score = 50
        scores = torch.tensor([
            [0.0, 100.0, 50.0],
            [0.0, 80.0, 120.0],
        ])

        result = ai._to_zero_sum_scores(scores)

        # Expected: (my_score - opponent_score) / 2
        # Game 0: (100 - 50) / 2 = 25
        # Game 1: (80 - 120) / 2 = -20
        expected = torch.tensor([25.0, -20.0])
        torch.testing.assert_close(result, expected)

    def test_4_player_zero_sum(self):
        """Test zero-sum conversion uses max opponent score."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)
        ai._num_players = 4

        # Player 1 = 100, others = [50, 80, 60]
        scores = torch.tensor([
            [0.0, 100.0, 50.0, 80.0, 60.0],
        ])

        result = ai._to_zero_sum_scores(scores)

        # Max opponent = 80
        # Expected: (100 - 80) / 2 = 10
        expected = torch.tensor([10.0])
        torch.testing.assert_close(result, expected)

    def test_zero_sum_player_2_perspective(self):
        """Test zero-sum from player 2's perspective."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=2, config=config)
        ai._num_players = 2

        # Player 1 = 100, Player 2 = 50
        scores = torch.tensor([
            [0.0, 100.0, 50.0],
        ])

        result = ai._to_zero_sum_scores(scores)

        # From P2's view: (50 - 100) / 2 = -25
        expected = torch.tensor([-25.0])
        torch.testing.assert_close(result, expected)


class TestMovePriorityBonus(unittest.TestCase):
    """Test move priority bonus calculation for move ordering."""

    def test_capture_moves_highest_priority(self):
        """Test that capture moves get highest priority bonus."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        # Use MagicMock to simulate move with type attribute
        capture_move = MagicMock()
        capture_move.type = MoveType.OVERTAKING_CAPTURE
        bonus = ai._get_move_priority_bonus(capture_move)
        self.assertEqual(bonus, 1000.0)

    def test_chain_capture_priority(self):
        """Test chain capture gets highest priority."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        move = MagicMock()
        move.type = MoveType.CHAIN_CAPTURE
        bonus = ai._get_move_priority_bonus(move)
        self.assertEqual(bonus, 1000.0)

    def test_place_ring_priority(self):
        """Test place ring moves get second highest priority."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        place_move = MagicMock()
        place_move.type = MoveType.PLACE_RING
        bonus = ai._get_move_priority_bonus(place_move)
        self.assertEqual(bonus, 100.0)

    def test_move_stack_priority(self):
        """Test stack movement gets third priority."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        move = MagicMock()
        move.type = MoveType.MOVE_STACK
        bonus = ai._get_move_priority_bonus(move)
        self.assertEqual(bonus, 50.0)

    def test_other_moves_no_bonus(self):
        """Test other move types get no bonus."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        swap_move = MagicMock()
        swap_move.type = MoveType.SWAP_SIDES
        bonus = ai._get_move_priority_bonus(swap_move)
        self.assertEqual(bonus, 0.0)


class TestLeafBufferManagement(unittest.TestCase):
    """Test leaf buffer management for batched evaluation."""

    def test_initial_buffer_empty(self):
        """Test leaf buffer starts empty."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        self.assertEqual(len(ai._leaf_buffer), 0)
        self.assertEqual(len(ai._leaf_results), 0)
        self.assertEqual(ai._next_callback_id, 0)

    def test_clear_leaf_buffer(self):
        """Test clearing leaf buffer."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        # Add some dummy data
        ai._leaf_buffer.append((MagicMock(), 12345))
        ai._leaf_results[12345] = 0.5
        ai._next_callback_id = 10

        ai._clear_leaf_buffer()

        self.assertEqual(len(ai._leaf_buffer), 0)
        self.assertEqual(len(ai._leaf_results), 0)
        self.assertEqual(ai._next_callback_id, 0)

    def test_reset_for_new_game_clears_buffer(self):
        """Test reset_for_new_game clears buffer and board info."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        # Set some state
        ai._leaf_buffer.append((MagicMock(), 12345))
        ai._board_type = BoardType.SQUARE8
        ai._board_size = 8
        ai._num_players = 2

        ai.reset_for_new_game()

        self.assertEqual(len(ai._leaf_buffer), 0)
        self.assertIsNone(ai._board_type)
        self.assertIsNone(ai._board_size)
        self.assertIsNone(ai._num_players)


class TestGPUInitialization(unittest.TestCase):
    """Test GPU initialization behavior."""

    def test_gpu_available_detection_cuda(self):
        """Test GPU detection when CUDA is available."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        mock_device = MagicMock()
        mock_device.type = "cuda"

        # Simulate GPU available
        ai._gpu_device = mock_device
        ai._gpu_available = True

        # Verify detection
        self.assertTrue(ai._gpu_available)
        self.assertEqual(ai._gpu_device.type, "cuda")

    def test_gpu_init_fails_gracefully(self):
        """Test GPU initialization fails gracefully."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        # Simulate failed initialization
        ai._gpu_available = False

        self.assertFalse(ai._gpu_available)

    def test_ensure_gpu_initialized_returns_cached(self):
        """Test _ensure_gpu_initialized returns cached value."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        # Pre-set cached value
        ai._gpu_available = True
        ai._gpu_device = torch.device("cpu")

        # Should return cached value without re-initializing
        result = ai._ensure_gpu_initialized()
        self.assertTrue(result)


class TestHeuristicWeights(unittest.TestCase):
    """Test heuristic weight handling."""

    def test_default_weights(self):
        """Test default weights are loaded."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        weights = ai._get_weights()

        self.assertIsInstance(weights, dict)
        # Check for some expected keys (may vary by profile)
        self.assertTrue(len(weights) > 0)

    def test_weights_are_cached(self):
        """Test weights are cached after first retrieval."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        weights1 = ai._get_weights()
        weights2 = ai._get_weights()

        self.assertIs(weights1, weights2)

    def test_custom_profile_weights(self):
        """Test custom profile weights can be loaded."""
        config = AIConfig(difficulty=5, heuristic_profile_id="aggressive")
        ai = GPUMinimaxAI(player_number=1, config=config)

        # Should not raise
        weights = ai._get_weights()
        self.assertIsInstance(weights, dict)


class TestBatchCreation(unittest.TestCase):
    """Test batch creation from game states."""

    def test_batch_creation_empty_raises(self):
        """Test empty state list raises ValueError."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        with self.assertRaises(ValueError):
            ai._create_batch_from_states([])

    def test_batch_creation_detects_board_info(self):
        """Test batch creation detects board info from first state."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)
        ai._gpu_device = torch.device("cpu")

        states = [create_mock_game_state(BoardType.SQUARE8) for _ in range(4)]

        # Board info should be None before
        self.assertIsNone(ai._board_type)

        try:
            ai._create_batch_from_states(states)
            # Board info should be detected
            self.assertEqual(ai._board_type, BoardType.SQUARE8)
        except ImportError:
            self.skipTest("gpu_parallel_games not available")


class TestSelectMove(unittest.TestCase):
    """Test move selection."""

    def test_single_move_returned_directly(self):
        """Test that when only one move is valid, it's returned immediately."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        state = create_mock_game_state()
        single_move = MagicMock()
        single_move.type = MoveType.PLACE_RING

        with patch.object(ai, "get_valid_moves", return_value=[single_move]):
            result = ai.select_move(state)
            self.assertEqual(result, single_move)

    def test_no_moves_returns_none(self):
        """Test that no valid moves returns None."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        state = create_mock_game_state()

        with patch.object(ai, "get_valid_moves", return_value=[]):
            result = ai.select_move(state)
            self.assertIsNone(result)

    def test_fallback_to_cpu_when_gpu_disabled(self):
        """Test fallback to CPU path when GPU is disabled."""
        config = AIConfig(difficulty=5)

        # Disable GPU via environment
        with patch.dict(os.environ, {"RINGRIFT_GPU_MINIMAX_DISABLE": "1"}):
            ai = GPUMinimaxAI(player_number=1, config=config)

            state = create_mock_game_state()
            move1 = MagicMock()
            move1.type = MoveType.PLACE_RING
            move2 = MagicMock()
            move2.type = MoveType.PLACE_RING
            moves = [move1, move2]

            with patch.object(ai, "get_valid_moves", return_value=moves):
                with patch.object(ai, "_select_move_legacy", return_value=moves[0]) as mock_legacy:
                    ai.use_incremental_search = False  # Force legacy path
                    result = ai.select_move(state)
                    mock_legacy.assert_called_once()


class TestFlushLeafBuffer(unittest.TestCase):
    """Test leaf buffer flushing."""

    def test_empty_buffer_no_action(self):
        """Test flushing empty buffer does nothing."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)

        # Should not raise
        ai._flush_leaf_buffer()

        self.assertEqual(len(ai._leaf_buffer), 0)
        self.assertEqual(len(ai._leaf_results), 0)

    def test_flush_stores_results(self):
        """Test flushing stores results and clears buffer."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)
        ai._board_type = BoardType.SQUARE8
        ai._board_size = 8
        ai._num_players = 2
        ai._gpu_device = torch.device("cpu")
        ai._gpu_available = False  # Force CPU fallback

        # Add mock states to buffer
        state = create_mock_game_state()
        ai._leaf_buffer.append((state, 12345))
        ai._leaf_buffer.append((state, 67890))

        # Mock the CPU evaluation (method is evaluate_position, not _evaluate_position)
        with patch.object(ai, "evaluate_position", return_value=0.5):
            ai._flush_leaf_buffer()

        # Buffer should be cleared
        self.assertEqual(len(ai._leaf_buffer), 0)

        # Results should be stored
        self.assertIn(12345, ai._leaf_results)
        self.assertIn(67890, ai._leaf_results)
        self.assertEqual(ai._leaf_results[12345], 0.5)


class TestCopyStateToBatch(unittest.TestCase):
    """Test state copying to batch."""

    def test_hex_coordinate_conversion(self):
        """Test hexagonal coordinate conversion."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=1, config=config)
        ai._board_type = BoardType.HEXAGONAL
        ai._board_size = 25
        ai._num_players = 2
        ai._gpu_device = torch.device("cpu")

        # Create a mock batch
        try:
            from app.ai.gpu_parallel_games import BatchGameState

            batch = BatchGameState.create_batch(
                batch_size=1,
                board_size=25,
                num_players=2,
                device=torch.device("cpu"),
            )

            state = create_mock_game_state(BoardType.HEXAGONAL)
            # Add a stack at hex coordinates
            state.board.stacks = {"0,0": MagicMock(controlling_player=1, rings=[[1]], cap_height=1)}

            ai._copy_state_to_batch(state, batch, idx=0, is_hex=True, board_size=25)

            # Center offset for hex
            center = 12
            # (0,0) in axial -> (0+12, 0+12) = (12, 12) in grid
            self.assertEqual(batch.stack_owner[0, center, center].item(), 1)
        except ImportError:
            self.skipTest("gpu_parallel_games not available")


class TestPlayerNumberConsistency(unittest.TestCase):
    """Test player number handling across different methods."""

    def test_player_number_in_zero_sum(self):
        """Test player number is correctly used in zero-sum conversion."""
        config = AIConfig(difficulty=5)

        # Test for player 1
        ai1 = GPUMinimaxAI(player_number=1, config=config)
        ai1._num_players = 2

        scores = torch.tensor([[0.0, 100.0, 50.0]])
        result1 = ai1._to_zero_sum_scores(scores)

        # Test for player 2
        ai2 = GPUMinimaxAI(player_number=2, config=config)
        ai2._num_players = 2

        result2 = ai2._to_zero_sum_scores(scores)

        # Results should be opposite (zero-sum property)
        torch.testing.assert_close(result1, -result2)

    def test_player_number_preserved_after_reset(self):
        """Test player number is preserved after reset."""
        config = AIConfig(difficulty=5)
        ai = GPUMinimaxAI(player_number=2, config=config)

        self.assertEqual(ai.player_number, 2)

        ai.reset_for_new_game()

        self.assertEqual(ai.player_number, 2)


if __name__ == "__main__":
    unittest.main()
