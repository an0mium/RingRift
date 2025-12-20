"""Tests for gpu_line_detection module.

Tests cover:
- detect_lines_vectorized: Fast vectorized line detection
- has_lines_batch_vectorized: Boolean line check
- detect_lines_with_metadata: Full line metadata including overlength
- detect_lines_batch: Line position extraction
- process_lines_batch: Line processing with collapse
"""

import pytest
import torch

from app.ai.gpu_game_types import DetectedLine


class MockBatchGameState:
    """Mock BatchGameState for testing line detection."""

    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 8,
        num_players: int = 2,
        device: torch.device = None,
    ):
        self.batch_size = batch_size
        self.board_size = board_size
        self.num_players = num_players
        self.device = device or torch.device('cpu')

        # Initialize tensors
        self.marker_owner = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.int16, device=self.device
        )
        self.stack_owner = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.int16, device=self.device
        )
        self.stack_height = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.int16, device=self.device
        )
        self.is_collapsed = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.bool, device=self.device
        )
        self.territory_owner = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.int16, device=self.device
        )
        self.territory_count = torch.zeros(
            batch_size, num_players + 1,
            dtype=torch.int32, device=self.device
        )
        self.eliminated_rings = torch.zeros(
            batch_size, num_players + 1,
            dtype=torch.int32, device=self.device
        )
        self.rings_caused_eliminated = torch.zeros(
            batch_size, num_players + 1,
            dtype=torch.int32, device=self.device
        )

    def get_active_mask(self) -> torch.Tensor:
        return torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    def place_markers(self, game_idx: int, player: int, positions: list):
        """Helper to place markers for testing."""
        for y, x in positions:
            self.marker_owner[game_idx, y, x] = player

    def place_stack(self, game_idx: int, player: int, y: int, x: int, height: int = 1):
        """Helper to place a stack for testing."""
        self.stack_owner[game_idx, y, x] = player
        self.stack_height[game_idx, y, x] = height


class TestDetectLinesVectorized:
    """Tests for detect_lines_vectorized function."""

    def test_no_markers_no_lines(self):
        """Empty board has no lines."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        in_line_mask, counts = detect_lines_vectorized(state, player=1)

        assert in_line_mask.shape == (1, 8, 8)
        assert counts.shape == (1,)
        assert counts[0].item() == 0
        assert not in_line_mask.any()

    def test_horizontal_line_detected(self):
        """Horizontal line of 4 markers is detected."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Place horizontal line at row 3, cols 2-5
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])

        in_line_mask, counts = detect_lines_vectorized(state, player=1)

        assert counts[0].item() == 4
        assert in_line_mask[0, 3, 2].item()
        assert in_line_mask[0, 3, 3].item()
        assert in_line_mask[0, 3, 4].item()
        assert in_line_mask[0, 3, 5].item()

    def test_vertical_line_detected(self):
        """Vertical line of 4 markers is detected."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Place vertical line at col 4, rows 1-4
        state.place_markers(0, 1, [(1, 4), (2, 4), (3, 4), (4, 4)])

        in_line_mask, counts = detect_lines_vectorized(state, player=1)

        assert counts[0].item() == 4
        assert in_line_mask[0, 1, 4].item()
        assert in_line_mask[0, 2, 4].item()
        assert in_line_mask[0, 3, 4].item()
        assert in_line_mask[0, 4, 4].item()

    def test_diagonal_line_detected(self):
        """Diagonal line (down-right) of 4 markers is detected."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Place diagonal line
        state.place_markers(0, 1, [(1, 1), (2, 2), (3, 3), (4, 4)])

        in_line_mask, counts = detect_lines_vectorized(state, player=1)

        assert counts[0].item() == 4
        assert in_line_mask[0, 1, 1].item()
        assert in_line_mask[0, 2, 2].item()
        assert in_line_mask[0, 3, 3].item()
        assert in_line_mask[0, 4, 4].item()

    def test_anti_diagonal_line_detected(self):
        """Anti-diagonal line (down-left) of 4 markers is detected."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Place anti-diagonal line
        state.place_markers(0, 1, [(1, 6), (2, 5), (3, 4), (4, 3)])

        in_line_mask, counts = detect_lines_vectorized(state, player=1)

        assert counts[0].item() == 4

    def test_short_line_not_detected(self):
        """Line shorter than required length is not detected."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Place only 3 markers (required is 4 for 2-player 8x8)
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4)])

        in_line_mask, counts = detect_lines_vectorized(state, player=1)

        assert counts[0].item() == 0
        assert not in_line_mask.any()

    def test_stacks_block_line(self):
        """Markers with stacks on them don't count for lines."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Place 4 markers
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])
        # Place a stack on one of them
        state.place_stack(0, 1, 3, 3, height=1)

        in_line_mask, counts = detect_lines_vectorized(state, player=1)

        # Should not detect a line because marker at (3,3) has a stack
        assert counts[0].item() == 0

    def test_different_player_markers_separate(self):
        """Markers from different players don't form lines together."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Place 2 markers for player 1, 2 markers for player 2
        state.place_markers(0, 1, [(3, 2), (3, 3)])
        state.place_markers(0, 2, [(3, 4), (3, 5)])

        in_line_mask_p1, counts_p1 = detect_lines_vectorized(state, player=1)
        in_line_mask_p2, counts_p2 = detect_lines_vectorized(state, player=2)

        assert counts_p1[0].item() == 0
        assert counts_p2[0].item() == 0

    def test_batch_processing(self):
        """Multiple games processed correctly."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=3, board_size=8, num_players=2)
        # Game 0: no line
        state.place_markers(0, 1, [(0, 0), (0, 1)])
        # Game 1: has line
        state.place_markers(1, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])
        # Game 2: no line
        state.place_markers(2, 1, [(7, 7)])

        in_line_mask, counts = detect_lines_vectorized(state, player=1)

        assert counts[0].item() == 0
        assert counts[1].item() == 4
        assert counts[2].item() == 0

    def test_game_mask_filtering(self):
        """game_mask filters which games to check."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=2, board_size=8, num_players=2)
        # Both games have lines
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])
        state.place_markers(1, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])

        # Only check game 0
        game_mask = torch.tensor([True, False], dtype=torch.bool)
        in_line_mask, counts = detect_lines_vectorized(state, player=1, game_mask=game_mask)

        assert counts[0].item() == 4
        assert counts[1].item() == 0  # Masked out


class TestHasLinesBatchVectorized:
    """Tests for has_lines_batch_vectorized function."""

    def test_returns_bool_tensor(self):
        """Returns boolean tensor of correct shape."""
        from app.ai.gpu_line_detection import has_lines_batch_vectorized

        state = MockBatchGameState(batch_size=4, board_size=8, num_players=2)
        result = has_lines_batch_vectorized(state, player=1)

        assert result.dtype == torch.bool
        assert result.shape == (4,)

    def test_no_lines_returns_false(self):
        """Returns False when no lines exist."""
        from app.ai.gpu_line_detection import has_lines_batch_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        result = has_lines_batch_vectorized(state, player=1)

        assert not result[0].item()

    def test_has_lines_returns_true(self):
        """Returns True when lines exist."""
        from app.ai.gpu_line_detection import has_lines_batch_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])

        result = has_lines_batch_vectorized(state, player=1)

        assert result[0].item()


class TestDetectLinesWithMetadata:
    """Tests for detect_lines_with_metadata function."""

    def test_returns_list_per_game(self):
        """Returns list of lists, one per game."""
        from app.ai.gpu_line_detection import detect_lines_with_metadata

        state = MockBatchGameState(batch_size=3, board_size=8, num_players=2)
        result = detect_lines_with_metadata(state, player=1)

        assert len(result) == 3
        assert all(isinstance(game_lines, list) for game_lines in result)

    def test_no_lines_returns_empty_lists(self):
        """Returns empty lists when no lines exist."""
        from app.ai.gpu_line_detection import detect_lines_with_metadata

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        result = detect_lines_with_metadata(state, player=1)

        assert result[0] == []

    def test_detected_line_has_correct_metadata(self):
        """Detected lines have correct metadata."""
        from app.ai.gpu_line_detection import detect_lines_with_metadata

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])

        result = detect_lines_with_metadata(state, player=1)

        assert len(result[0]) == 1
        line = result[0][0]
        assert isinstance(line, DetectedLine)
        assert line.length == 4
        assert line.is_overlength is False
        assert len(line.positions) == 4
        assert line.direction in [(0, 1), (1, 0), (1, 1), (1, -1)]

    def test_overlength_line_marked_correctly(self):
        """Overlength lines are marked as overlength."""
        from app.ai.gpu_line_detection import detect_lines_with_metadata

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Place 5 markers (required is 4)
        state.place_markers(0, 1, [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)])

        result = detect_lines_with_metadata(state, player=1)

        assert len(result[0]) == 1
        line = result[0][0]
        assert line.length == 5
        assert line.is_overlength is True

    def test_multiple_lines_detected(self):
        """Multiple lines in same game are all detected."""
        from app.ai.gpu_line_detection import detect_lines_with_metadata

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Horizontal line
        state.place_markers(0, 1, [(1, 0), (1, 1), (1, 2), (1, 3)])
        # Vertical line (separate)
        state.place_markers(0, 1, [(4, 6), (5, 6), (6, 6), (7, 6)])

        result = detect_lines_with_metadata(state, player=1)

        assert len(result[0]) == 2


class TestDetectLinesBatch:
    """Tests for detect_lines_batch function."""

    def test_returns_list_of_positions(self):
        """Returns list of position tuples."""
        from app.ai.gpu_line_detection import detect_lines_batch

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])

        result = detect_lines_batch(state, player=1)

        assert len(result) == 1
        assert len(result[0]) == 4
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in result[0])

    def test_empty_when_no_lines(self):
        """Returns empty list when no lines."""
        from app.ai.gpu_line_detection import detect_lines_batch

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        result = detect_lines_batch(state, player=1)

        assert result[0] == []


class TestEliminateOneRingFromAnyStack:
    """Tests for _eliminate_one_ring_from_any_stack function."""

    def test_eliminates_from_controlled_stack(self):
        """Eliminates one ring from a controlled stack."""
        from app.ai.gpu_line_detection import _eliminate_one_ring_from_any_stack

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_stack(0, 1, 3, 3, height=3)

        result = _eliminate_one_ring_from_any_stack(state, game_idx=0, player=1)

        assert result is True
        assert state.stack_height[0, 3, 3].item() == 2
        assert state.eliminated_rings[0, 1].item() == 1
        assert state.rings_caused_eliminated[0, 1].item() == 1

    def test_clears_stack_owner_when_empty(self):
        """Clears stack owner when height reaches 0."""
        from app.ai.gpu_line_detection import _eliminate_one_ring_from_any_stack

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_stack(0, 1, 3, 3, height=1)

        result = _eliminate_one_ring_from_any_stack(state, game_idx=0, player=1)

        assert result is True
        assert state.stack_height[0, 3, 3].item() == 0
        assert state.stack_owner[0, 3, 3].item() == 0

    def test_returns_false_when_no_stacks(self):
        """Returns False when player has no stacks."""
        from app.ai.gpu_line_detection import _eliminate_one_ring_from_any_stack

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # No stacks placed

        result = _eliminate_one_ring_from_any_stack(state, game_idx=0, player=1)

        assert result is False

    def test_ignores_other_player_stacks(self):
        """Only eliminates from player's own stacks."""
        from app.ai.gpu_line_detection import _eliminate_one_ring_from_any_stack

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_stack(0, 2, 3, 3, height=3)  # Player 2's stack

        result = _eliminate_one_ring_from_any_stack(state, game_idx=0, player=1)

        assert result is False
        assert state.stack_height[0, 3, 3].item() == 3  # Unchanged


class TestProcessLinesBatch:
    """Tests for process_lines_batch function."""

    def test_collapses_line_markers(self):
        """Line markers are collapsed to territory."""
        from app.ai.gpu_line_detection import process_lines_batch

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])
        state.place_stack(0, 1, 0, 0, height=2)  # Stack for elimination cost

        process_lines_batch(state, option2_probability=0.0)

        # Markers should be cleared
        assert state.marker_owner[0, 3, 2].item() == 0
        assert state.marker_owner[0, 3, 3].item() == 0
        assert state.marker_owner[0, 3, 4].item() == 0
        assert state.marker_owner[0, 3, 5].item() == 0

        # Territory should be claimed
        assert state.territory_owner[0, 3, 2].item() == 1
        assert state.territory_owner[0, 3, 3].item() == 1
        assert state.territory_owner[0, 3, 4].item() == 1
        assert state.territory_owner[0, 3, 5].item() == 1

        # Cells should be collapsed
        assert state.is_collapsed[0, 3, 2].item()
        assert state.is_collapsed[0, 3, 3].item()
        assert state.is_collapsed[0, 3, 4].item()
        assert state.is_collapsed[0, 3, 5].item()

    def test_elimination_cost_paid(self):
        """Exact-length line pays elimination cost."""
        from app.ai.gpu_line_detection import process_lines_batch

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])
        state.place_stack(0, 1, 0, 0, height=2)

        initial_height = state.stack_height[0, 0, 0].item()
        process_lines_batch(state, option2_probability=0.0)

        # One ring eliminated
        assert state.stack_height[0, 0, 0].item() == initial_height - 1
        assert state.eliminated_rings[0, 1].item() == 1

    def test_no_processing_when_no_lines(self):
        """No changes when no lines exist."""
        from app.ai.gpu_line_detection import process_lines_batch

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_markers(0, 1, [(3, 2), (3, 3)])  # Only 2 markers

        process_lines_batch(state)

        # Markers unchanged
        assert state.marker_owner[0, 3, 2].item() == 1
        assert state.marker_owner[0, 3, 3].item() == 1
        assert not state.is_collapsed[0, 3, 2].item()
        assert not state.is_collapsed[0, 3, 3].item()

    def test_territory_count_updated(self):
        """Territory count is updated correctly."""
        from app.ai.gpu_line_detection import process_lines_batch

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])
        state.place_stack(0, 1, 0, 0, height=2)

        initial_territory = state.territory_count[0, 1].item()
        process_lines_batch(state, option2_probability=0.0)

        assert state.territory_count[0, 1].item() == initial_territory + 4

    def test_game_mask_respected(self):
        """game_mask filters which games are processed."""
        from app.ai.gpu_line_detection import process_lines_batch

        state = MockBatchGameState(batch_size=2, board_size=8, num_players=2)
        # Both games have lines
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])
        state.place_markers(1, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])
        state.place_stack(0, 1, 0, 0, height=2)
        state.place_stack(1, 1, 0, 0, height=2)

        # Only process game 0
        game_mask = torch.tensor([True, False], dtype=torch.bool)
        process_lines_batch(state, game_mask=game_mask)

        # Game 0 processed
        assert state.marker_owner[0, 3, 2].item() == 0
        # Game 1 not processed
        assert state.marker_owner[1, 3, 2].item() == 1


class TestThreePlayerLineLength:
    """Tests for 3-player games which require shorter lines."""

    def test_three_player_requires_three(self):
        """3-player 8x8 requires only 3 markers for a line."""
        from app.ai.gpu_line_detection import detect_lines_vectorized

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=3)
        # Place only 3 markers
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4)])

        in_line_mask, counts = detect_lines_vectorized(state, player=1)

        # Should detect a line with 3 markers for 3-player
        assert counts[0].item() == 3

    def test_three_player_overlength(self):
        """4+ markers is overlength for 3-player."""
        from app.ai.gpu_line_detection import detect_lines_with_metadata

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=3)
        # Place 4 markers (overlength for 3-player)
        state.place_markers(0, 1, [(3, 2), (3, 3), (3, 4), (3, 5)])

        result = detect_lines_with_metadata(state, player=1)

        assert len(result[0]) == 1
        assert result[0][0].is_overlength is True
