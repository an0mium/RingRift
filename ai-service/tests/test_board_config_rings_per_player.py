from __future__ import annotations

from app.models import BoardType
from app.rules.core import BOARD_CONFIGS


def test_board_config_rings_per_player_stable() -> None:
    assert BOARD_CONFIGS[BoardType.SQUARE19].rings_per_player == 72
    assert BOARD_CONFIGS[BoardType.HEXAGONAL].rings_per_player == 96

