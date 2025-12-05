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


def test_square8_line_length_is_three_for_all_player_counts() -> None:
    """RR-CANON-R120: square8 uses 3-in-a-row for all player counts."""
    board = _make_empty_square8_board()

    # Place three P3 markers on a NE diagonal: (5,6), (6,5), (7,4)
    coords = [(5, 6), (6, 5), (7, 4)]
    for x, y in coords:
        pos = Position(x=x, y=y)
        key = pos.to_key()
        board.markers[key] = MarkerInfo(
            player=3,
            position=pos,
            type="regular",
        )

    # For all supported player counts, this should be a valid line of length 3
    # for Player 3.
    for num_players in (2, 3, 4):
        lines = BoardManager.find_all_lines(board, num_players=num_players)
        assert len(lines) == 1
        line = lines[0]
        assert line.player == 3
        assert line.length == 3
        assert {p.to_key() for p in line.positions} == {
            f"{x},{y}" for x, y in coords
        }
