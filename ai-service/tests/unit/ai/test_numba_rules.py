"""Unit tests for Numba JIT-compiled game rules.

Tests cover:
- BoardArrays dataclass creation and conversion
- Line detection algorithms
- Position indexing utilities
- Direction tables
- Numba availability detection
"""

import pytest
import numpy as np


class TestNumbaAvailability:
    """Tests for Numba import and fallback."""

    def test_numba_available_flag_exists(self):
        """Module exposes NUMBA_AVAILABLE flag."""
        from app.ai.numba_rules import NUMBA_AVAILABLE

        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_njit_decorator_exists(self):
        """Module exports njit decorator (real or fallback)."""
        from app.ai import numba_rules

        # Either real numba or fallback
        assert hasattr(numba_rules, 'njit')

    def test_prange_exists(self):
        """Module exports prange (real or fallback)."""
        from app.ai import numba_rules

        assert hasattr(numba_rules, 'prange')


class TestDirectionTables:
    """Tests for pre-computed direction arrays."""

    def test_square_dirs_shape(self):
        """SQUARE_DIRS has correct shape (8 directions, 2 components)."""
        from app.ai.numba_rules import SQUARE_DIRS

        assert SQUARE_DIRS.shape == (8, 2)

    def test_square_dirs_dtype(self):
        """SQUARE_DIRS uses int8 for memory efficiency."""
        from app.ai.numba_rules import SQUARE_DIRS

        assert SQUARE_DIRS.dtype == np.int8

    def test_square_line_dirs_shape(self):
        """SQUARE_LINE_DIRS has 4 directions (avoids double counting)."""
        from app.ai.numba_rules import SQUARE_LINE_DIRS

        assert SQUARE_LINE_DIRS.shape == (4, 2)

    def test_territory_dirs_shape(self):
        """TERRITORY_DIRS has 4 directions (Von Neumann neighborhood)."""
        from app.ai.numba_rules import TERRITORY_DIRS

        assert TERRITORY_DIRS.shape == (4, 2)

    def test_directions_are_unique(self):
        """Each direction table has unique direction vectors."""
        from app.ai.numba_rules import SQUARE_DIRS, SQUARE_LINE_DIRS, TERRITORY_DIRS

        # Convert to set of tuples for uniqueness check
        square_set = set(tuple(d) for d in SQUARE_DIRS)
        assert len(square_set) == 8

        line_set = set(tuple(d) for d in SQUARE_LINE_DIRS)
        assert len(line_set) == 4

        territory_set = set(tuple(d) for d in TERRITORY_DIRS)
        assert len(territory_set) == 4


class TestBoardArrays:
    """Tests for BoardArrays numpy representation."""

    def test_default_creation(self):
        """BoardArrays creates with default parameters."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()

        assert arrays.board_size == 8
        assert arrays.num_positions == 64
        assert arrays.num_players == 2

    def test_custom_board_size(self):
        """BoardArrays supports custom board sizes."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays(board_size=19, num_players=4)

        assert arrays.board_size == 19
        assert arrays.num_positions == 361
        assert arrays.num_players == 4

    def test_array_shapes(self):
        """All board arrays have correct shapes."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays(board_size=8)

        assert arrays.stack_owner.shape == (64,)
        assert arrays.stack_height.shape == (64,)
        assert arrays.cap_height.shape == (64,)
        assert arrays.marker_owner.shape == (64,)
        assert arrays.collapsed.shape == (64,)
        assert arrays.rings.shape == (64, 20)

    def test_player_arrays_shape(self):
        """Player state arrays have correct shape (indexed 1-4)."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()

        assert arrays.rings_in_hand.shape == (5,)  # Index 0 unused
        assert arrays.eliminated_rings.shape == (5,)
        assert arrays.territory_count.shape == (5,)

    def test_arrays_initialized_to_zero(self):
        """All arrays are zero-initialized."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()

        assert np.all(arrays.stack_owner == 0)
        assert np.all(arrays.stack_height == 0)
        assert np.all(arrays.marker_owner == 0)
        assert np.all(arrays.collapsed == False)
        assert np.all(arrays.rings == 0)

    def test_initial_game_state(self):
        """Initial game state values are correct."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()

        assert arrays.current_player == 1
        assert arrays.game_active == True
        assert arrays.winner == 0


class TestPositionIndexing:
    """Tests for position conversion utilities."""

    def test_is_valid_pos_inside_board(self):
        """Valid positions within board bounds return True."""
        from app.ai.numba_rules import _is_valid_pos

        assert _is_valid_pos(0, 0, 8) == True
        assert _is_valid_pos(7, 7, 8) == True
        assert _is_valid_pos(4, 4, 8) == True

    def test_is_valid_pos_outside_board(self):
        """Invalid positions return False."""
        from app.ai.numba_rules import _is_valid_pos

        assert _is_valid_pos(-1, 0, 8) == False
        assert _is_valid_pos(8, 0, 8) == False
        assert _is_valid_pos(0, -1, 8) == False
        assert _is_valid_pos(0, 8, 8) == False

    def test_pos_to_idx_corners(self):
        """Position to index conversion for corner positions."""
        from app.ai.numba_rules import _pos_to_idx

        # Top-left
        assert _pos_to_idx(0, 0, 8) == 0
        # Top-right
        assert _pos_to_idx(7, 0, 8) == 7
        # Bottom-left
        assert _pos_to_idx(0, 7, 8) == 56
        # Bottom-right
        assert _pos_to_idx(7, 7, 8) == 63

    def test_idx_to_pos_corners(self):
        """Index to position conversion for corner indices."""
        from app.ai.numba_rules import _idx_to_pos

        assert _idx_to_pos(0, 8) == (0, 0)
        assert _idx_to_pos(7, 8) == (7, 0)
        assert _idx_to_pos(56, 8) == (0, 7)
        assert _idx_to_pos(63, 8) == (7, 7)

    def test_pos_idx_roundtrip(self):
        """Position <-> index conversion is reversible."""
        from app.ai.numba_rules import _pos_to_idx, _idx_to_pos

        for x in range(8):
            for y in range(8):
                idx = _pos_to_idx(x, y, 8)
                rx, ry = _idx_to_pos(idx, 8)
                assert (rx, ry) == (x, y), f"Roundtrip failed for ({x}, {y})"


class TestLineDetection:
    """Tests for line detection algorithm."""

    def test_detect_line_no_marker(self):
        """No line detected when position has no marker."""
        from app.ai.numba_rules import detect_line_at_position, SQUARE_LINE_DIRS

        board_size = 8
        marker_owner = np.zeros(64, dtype=np.int8)
        collapsed = np.zeros(64, dtype=np.bool_)
        stack_owner = np.zeros(64, dtype=np.int8)

        length, positions = detect_line_at_position(
            marker_owner, collapsed, stack_owner,
            x=3, y=3, player=1, board_size=board_size,
            min_length=3, directions=SQUARE_LINE_DIRS,
        )

        assert length == 0

    def test_detect_line_single_marker(self):
        """Single marker is not a line."""
        from app.ai.numba_rules import detect_line_at_position, SQUARE_LINE_DIRS

        board_size = 8
        marker_owner = np.zeros(64, dtype=np.int8)
        collapsed = np.zeros(64, dtype=np.bool_)
        stack_owner = np.zeros(64, dtype=np.int8)

        # Single marker at (3, 3)
        marker_owner[3 * 8 + 3] = 1

        length, positions = detect_line_at_position(
            marker_owner, collapsed, stack_owner,
            x=3, y=3, player=1, board_size=board_size,
            min_length=3, directions=SQUARE_LINE_DIRS,
        )

        # Length 1 is below min_length=3
        assert length < 3

    def test_detect_horizontal_line(self):
        """Detects horizontal line of markers."""
        from app.ai.numba_rules import detect_line_at_position, SQUARE_LINE_DIRS

        board_size = 8
        marker_owner = np.zeros(64, dtype=np.int8)
        collapsed = np.zeros(64, dtype=np.bool_)
        stack_owner = np.zeros(64, dtype=np.int8)

        # Horizontal line: (2,3), (3,3), (4,3), (5,3)
        for x in range(2, 6):
            marker_owner[3 * board_size + x] = 1

        length, positions = detect_line_at_position(
            marker_owner, collapsed, stack_owner,
            x=3, y=3, player=1, board_size=board_size,
            min_length=3, directions=SQUARE_LINE_DIRS,
        )

        assert length >= 4

    def test_line_blocked_by_stack(self):
        """Line detection ignores positions blocked by stacks."""
        from app.ai.numba_rules import detect_line_at_position, SQUARE_LINE_DIRS

        board_size = 8
        marker_owner = np.zeros(64, dtype=np.int8)
        collapsed = np.zeros(64, dtype=np.bool_)
        stack_owner = np.zeros(64, dtype=np.int8)

        # Markers in a line
        for x in range(2, 6):
            marker_owner[3 * board_size + x] = 1

        # Stack blocks middle of line
        stack_owner[3 * board_size + 4] = 1

        length, positions = detect_line_at_position(
            marker_owner, collapsed, stack_owner,
            x=3, y=3, player=1, board_size=board_size,
            min_length=3, directions=SQUARE_LINE_DIRS,
        )

        # Stack at (4,3) blocks the line
        assert length < 4

    def test_line_blocked_by_collapsed(self):
        """Line detection ignores collapsed positions."""
        from app.ai.numba_rules import detect_line_at_position, SQUARE_LINE_DIRS

        board_size = 8
        marker_owner = np.zeros(64, dtype=np.int8)
        collapsed = np.zeros(64, dtype=np.bool_)
        stack_owner = np.zeros(64, dtype=np.int8)

        # Markers in a line
        for x in range(2, 6):
            marker_owner[3 * board_size + x] = 1

        # Collapsed space blocks the start
        collapsed[3 * board_size + 3] = True

        length, positions = detect_line_at_position(
            marker_owner, collapsed, stack_owner,
            x=3, y=3, player=1, board_size=board_size,
            min_length=3, directions=SQUARE_LINE_DIRS,
        )

        # Starting position is collapsed
        assert length == 0


class TestDtypes:
    """Tests for array dtype consistency."""

    def test_int8_arrays(self):
        """Integer arrays use int8 for memory efficiency."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()

        assert arrays.stack_owner.dtype == np.int8
        assert arrays.stack_height.dtype == np.int8
        assert arrays.cap_height.dtype == np.int8
        assert arrays.marker_owner.dtype == np.int8
        assert arrays.rings.dtype == np.int8

    def test_bool_arrays(self):
        """Boolean arrays use proper dtype."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()

        assert arrays.collapsed.dtype == np.bool_

    def test_int16_player_arrays(self):
        """Player state arrays use int16 for larger values."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()

        assert arrays.rings_in_hand.dtype == np.int16
        assert arrays.eliminated_rings.dtype == np.int16
        assert arrays.territory_count.dtype == np.int16


class TestBoardArraysModification:
    """Tests for modifying BoardArrays in-place."""

    def test_modify_marker_owner(self):
        """Can modify marker_owner array."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()
        arrays.marker_owner[0] = 1
        arrays.marker_owner[63] = 2

        assert arrays.marker_owner[0] == 1
        assert arrays.marker_owner[63] == 2

    def test_modify_stack_state(self):
        """Can modify stack state arrays."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()
        arrays.stack_owner[10] = 1
        arrays.stack_height[10] = 3
        arrays.cap_height[10] = 2

        assert arrays.stack_owner[10] == 1
        assert arrays.stack_height[10] == 3
        assert arrays.cap_height[10] == 2

    def test_modify_rings(self):
        """Can modify ring composition array."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()
        # Stack at position 10 has rings owned by player 1, 2, 1
        arrays.rings[10, 0] = 1
        arrays.rings[10, 1] = 2
        arrays.rings[10, 2] = 1

        assert arrays.rings[10, 0] == 1
        assert arrays.rings[10, 1] == 2
        assert arrays.rings[10, 2] == 1

    def test_modify_player_state(self):
        """Can modify player state arrays."""
        from app.ai.numba_rules import BoardArrays

        arrays = BoardArrays()
        arrays.rings_in_hand[1] = 10
        arrays.rings_in_hand[2] = 8

        assert arrays.rings_in_hand[1] == 10
        assert arrays.rings_in_hand[2] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
