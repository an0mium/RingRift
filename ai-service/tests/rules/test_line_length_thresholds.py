from datetime import datetime
import os
import sys

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.board_manager import BoardManager  # noqa: E402
from app.models import (  # noqa: E402
    BoardState,
    BoardType,
    MarkerInfo,
    Position,
)


def _make_empty_square8_board() -> BoardState:
    return BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={},
        markers={},
        collapsed_spaces={},
        eliminated_rings={},
        formed_lines=[],
        territories={},
    )


def test_square8_line_length_varies_by_player_count() -> None:
    """
    RR-CANON-R120: square8 line length depends on player count.
    - 2-player: line length = 4
    - 3-4 player: line length = 3
    """
    board = _make_empty_square8_board()

    # Place three P3 markers on a NE diagonal: (5,6), (6,5), (7,4)
    coords_3 = [(5, 6), (6, 5), (7, 4)]
    for x, y in coords_3:
        pos = Position(x=x, y=y)
        key = pos.to_key()
        board.markers[key] = MarkerInfo(
            player=3,
            position=pos,
            type="regular",
        )

    # For 3-4 player games, 3 markers should form a valid line (threshold = 3)
    for num_players in (3, 4):
        lines = BoardManager.find_all_lines(board, num_players=num_players)
        assert len(lines) == 1, f"Expected 1 line for {num_players} players, got {len(lines)}"
        line = lines[0]
        assert line.player == 3
        assert line.length == 3
        assert {p.to_key() for p in line.positions} == {
            f"{x},{y}" for x, y in coords_3
        }

    # For 2-player games, 3 markers should NOT form a valid line (threshold = 4)
    lines_2p = BoardManager.find_all_lines(board, num_players=2)
    assert len(lines_2p) == 0, "2-player games require 4 markers for a line, not 3"


def test_square8_2player_requires_four_markers() -> None:
    """RR-CANON-R120: square8 2-player requires 4 markers for a line."""
    board = _make_empty_square8_board()

    # Place four P1 markers in a horizontal line: (0,0), (1,0), (2,0), (3,0)
    coords_4 = [(0, 0), (1, 0), (2, 0), (3, 0)]
    for x, y in coords_4:
        pos = Position(x=x, y=y)
        key = pos.to_key()
        board.markers[key] = MarkerInfo(
            player=1,
            position=pos,
            type="regular",
        )

    # For 2-player games, 4 markers should form a valid line (threshold = 4)
    lines = BoardManager.find_all_lines(board, num_players=2)
    assert len(lines) == 1, f"Expected 1 line for 2 players with 4 markers, got {len(lines)}"
    line = lines[0]
    assert line.player == 1
    assert line.length == 4
    assert {p.to_key() for p in line.positions} == {
        f"{x},{y}" for x, y in coords_4
    }

    # For 3-4 player games, the same 4 markers also form a valid line
    for num_players in (3, 4):
        lines = BoardManager.find_all_lines(board, num_players=num_players)
        assert len(lines) == 1, f"Expected 1 line for {num_players} players with 4 markers"
        assert lines[0].length == 4
