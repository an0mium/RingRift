"""Tests for app.rules.generators.line module.

Tests the LineGenerator which enumerates line processing moves.

Canonical Spec References:
- RR-CANON-R076: Interactive decision moves only
- RR-CANON-R120: Line formation conditions
- RR-CANON-R123: Line elimination reward (pending_line_reward_elimination)

Note: Lines are formed by MARKERS, not stacks. Stacks act as blockers for lines.
Line length requirements per RR-CANON-R120:
- square8 2-player: 4 markers in a row
- square8 3-4 player: 3 markers in a row
"""

import pytest

from app.models import MarkerInfo, MoveType, Position
from app.rules.generators.line import LineGenerator
from app.testing.fixtures import (
    create_board_state,
    create_game_state,
)


def create_marker(x: int, y: int, player: int) -> MarkerInfo:
    """Create a MarkerInfo for testing."""
    return MarkerInfo(
        position=Position(x=x, y=y),
        player=player,
        type="regular",
    )


class TestLineGeneratorBasics:
    """Basic tests for LineGenerator initialization and interface."""

    def test_generator_instantiation(self):
        """Test LineGenerator can be instantiated."""
        generator = LineGenerator()
        assert generator is not None

    def test_generate_returns_list(self):
        """Test generate() returns a list."""
        generator = LineGenerator()
        state = create_game_state()
        moves = generator.generate(state, player=1)
        assert isinstance(moves, list)

    def test_generate_empty_board_no_lines(self):
        """Test no line moves on empty board."""
        generator = LineGenerator()
        state = create_game_state()
        moves = generator.generate(state, player=1)
        assert len(moves) == 0


class TestLineDetection:
    """Tests for line detection during move generation.

    Note: square8 2-player requires 4 markers for a line (RR-CANON-R120).
    square8 3+ player requires 3 markers.
    """

    @pytest.fixture
    def generator(self):
        return LineGenerator()

    def test_no_lines_when_insufficient_markers(self, generator):
        """Test no lines detected when fewer than required markers in a row."""
        # 2-player requires 4 markers, so 3 is insufficient
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=2,
        )
        moves = generator.generate(state, player=1)
        assert len(moves) == 0

    def test_line_detected_horizontal_four(self, generator):
        """Test horizontal line of 4 markers generates move (2-player)."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
                "3,0": create_marker(3, 0, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=2,
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1

    def test_line_detected_three_with_three_players(self, generator):
        """Test 3-marker line detected in 3-player game."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,  # 3-player requires only 3 markers
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1

    def test_line_not_detected_different_players(self, generator):
        """Test line not detected when markers owned by different players."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),  # Player 1
                "1,0": create_marker(1, 0, 2),  # Player 2 - breaks the line
                "2,0": create_marker(2, 0, 1),  # Player 1
                "3,0": create_marker(3, 0, 1),  # Player 1
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=2,
        )
        moves = generator.generate(state, player=1)
        assert len(moves) == 0

    def test_line_detected_vertical(self, generator):
        """Test vertical line detection (3-player)."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "0,1": create_marker(0, 1, 1),
                "0,2": create_marker(0, 2, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,  # 3-player requires only 3 markers
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1

    def test_line_detected_diagonal(self, generator):
        """Test diagonal line detection (3-player)."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,1": create_marker(1, 1, 1),
                "2,2": create_marker(2, 2, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,  # 3-player requires only 3 markers
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1


class TestMoveProperties:
    """Tests for generated move properties and structure."""

    @pytest.fixture
    def generator(self):
        return LineGenerator()

    @pytest.fixture
    def state_with_line(self):
        """Create state with a valid line for player 1 (3-player game)."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
            }
        )
        return create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,
        )

    def test_move_has_correct_player(self, generator, state_with_line):
        """Test generated moves have correct player attribute."""
        moves = generator.generate(state_with_line, player=1)
        assert len(moves) > 0  # Verify we got moves
        for move in moves:
            assert move.player == 1

    def test_move_has_id(self, generator, state_with_line):
        """Test generated moves have non-empty id."""
        moves = generator.generate(state_with_line, player=1)
        assert len(moves) > 0  # Verify we got moves
        for move in moves:
            assert move.id is not None
            assert len(move.id) > 0

    def test_process_line_move_type(self, generator, state_with_line):
        """Test PROCESS_LINE moves have correct type."""
        moves = generator.generate(state_with_line, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1
        for move in process_moves:
            assert move.type == MoveType.PROCESS_LINE


# Placeholder tests for future implementation
class TestEdgeCases:
    """Edge case tests - expand as needed."""

    def test_multiple_lines_same_marker(self):
        """Test handling when one marker is part of multiple lines."""
        # TODO: Implement when corner cases are defined
        pytest.skip("Not yet implemented")

    def test_line_at_board_edge(self):
        """Test line detection at board edges."""
        # TODO: Implement boundary testing
        pytest.skip("Not yet implemented")

    def test_longer_than_minimum_line(self):
        """Test lines longer than minimum length (e.g., 5 in a row)."""
        # TODO: Implement for extended line handling
        pytest.skip("Not yet implemented")
