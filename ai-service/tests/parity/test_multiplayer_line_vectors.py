"""
Test multiplayer line contract vectors per RR-CANON-R120.

Validates that line detection correctly implements player-count-dependent thresholds:
- square8 2-player: line length = 4
- square8 3-4 player: line length = 3
- square19 / hexagonal: line length = 4 (all player counts)
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure app package is importable when running tests directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from app.board_manager import BoardManager
from app.models import (
    BoardState,
    BoardType,
    MarkerInfo,
    Position,
    RingStack,
)
from app.rules.core import get_effective_line_length

VECTORS_PATH = Path(__file__).parent / "vectors" / "multiplayer_line.vectors.json"


def load_vectors():
    """Load contract vectors from JSON file."""
    with open(VECTORS_PATH) as f:
        data = json.load(f)
    return data["vectors"]


def build_board_from_vector(vector: dict) -> BoardState:
    """Build a BoardState from a contract vector input."""
    board_data = vector["input"]["state"]["board"]
    board_type = BoardType(board_data["type"])

    # Build stacks
    stacks = {}
    for key, stack_data in board_data.get("stacks", {}).items():
        pos = Position(
            x=stack_data["position"]["x"],
            y=stack_data["position"]["y"],
            z=stack_data["position"].get("z"),
        )
        stacks[key] = RingStack(
            position=pos,
            rings=stack_data["rings"],
            stackHeight=stack_data["stackHeight"],
            capHeight=stack_data["capHeight"],
            controllingPlayer=stack_data["controllingPlayer"],
        )

    # Build markers
    markers = {}
    for key, marker_data in board_data.get("markers", {}).items():
        pos = Position(
            x=marker_data["position"]["x"],
            y=marker_data["position"]["y"],
            z=marker_data["position"].get("z"),
        )
        markers[key] = MarkerInfo(
            position=pos,
            player=marker_data["player"],
            type=marker_data.get("type", "regular"),
        )

    return BoardState(
        type=board_type,
        size=board_data["size"],
        stacks=stacks,
        markers=markers,
        collapsed_spaces=board_data.get("collapsedSpaces", {}),
        eliminated_rings={},
        formed_lines=[],
        territories={},
    )


def get_num_players(vector: dict) -> int:
    """Get number of players from vector."""
    return len(vector["input"]["state"]["players"])


@pytest.fixture
def vectors():
    """Load all contract vectors."""
    return load_vectors()


class TestMultiplayerLineThresholds:
    """Test line detection thresholds vary by player count per RR-CANON-R120."""

    def test_3player_3markers_forms_line(self, vectors):
        """3-player square8: 3 markers should form a valid line (threshold=3)."""
        vector = next(v for v in vectors if v["id"] == "line.3player.exact_length_3")
        board = build_board_from_vector(vector)
        num_players = get_num_players(vector)

        lines = BoardManager.find_all_lines(board, num_players=num_players)

        assertions = vector["assertions"]["lineDetection"]
        assert len(lines) == assertions["lineCount"], (
            f"Expected {assertions['lineCount']} line(s), got {len(lines)}"
        )
        assert lines[0].length == assertions["lineLength"]
        assert lines[0].player == assertions["linePlayer"]

    def test_4player_3markers_forms_line(self, vectors):
        """4-player square8: 3 markers should form a valid line (threshold=3)."""
        vector = next(v for v in vectors if v["id"] == "line.4player.exact_length_3")
        board = build_board_from_vector(vector)
        num_players = get_num_players(vector)

        lines = BoardManager.find_all_lines(board, num_players=num_players)

        assertions = vector["assertions"]["lineDetection"]
        assert len(lines) == assertions["lineCount"]
        assert lines[0].length == assertions["lineLength"]

    def test_2player_3markers_no_line(self, vectors):
        """2-player square8: 3 markers should NOT form a line (threshold=4)."""
        vector = next(v for v in vectors if v["id"] == "line.2player.3markers_no_line")
        board = build_board_from_vector(vector)
        num_players = get_num_players(vector)

        lines = BoardManager.find_all_lines(board, num_players=num_players)

        assertions = vector["assertions"]["lineDetection"]
        assert len(lines) == assertions["lineCount"], (
            f"Expected {assertions['lineCount']} lines for 2-player with 3 markers, "
            f"got {len(lines)}. 2-player threshold is 4."
        )

    def test_2player_4markers_forms_line(self, vectors):
        """2-player square8: 4 markers should form a valid line (threshold=4)."""
        vector = next(v for v in vectors if v["id"] == "line.2player.exact_length_4")
        board = build_board_from_vector(vector)
        num_players = get_num_players(vector)

        lines = BoardManager.find_all_lines(board, num_players=num_players)

        assertions = vector["assertions"]["lineDetection"]
        assert len(lines) == assertions["lineCount"]
        assert lines[0].length == assertions["lineLength"]

    def test_3player_4markers_is_overlength(self, vectors):
        """3-player square8: 4-marker line is overlength (threshold=3)."""
        vector = next(v for v in vectors if v["id"] == "line.3player.overlength_4")
        board = build_board_from_vector(vector)
        num_players = get_num_players(vector)

        lines = BoardManager.find_all_lines(board, num_players=num_players)

        assertions = vector["assertions"]["lineDetection"]
        assert len(lines) == assertions["lineCount"]
        assert lines[0].length == assertions["lineLength"]
        # Overlength means length > threshold
        threshold = get_effective_line_length(board.type, num_players)
        assert threshold == assertions["effectiveThreshold"]
        assert lines[0].length > threshold, "Line should be overlength"


class TestEffectiveLineLengthFunction:
    """Direct tests of get_effective_line_length per RR-CANON-R120."""

    def test_square8_2player_threshold_is_4(self):
        """square8 2-player: threshold should be 4."""
        threshold = get_effective_line_length(BoardType.SQUARE8, 2)
        assert threshold == 4

    def test_square8_3player_threshold_is_3(self):
        """square8 3-player: threshold should be 3."""
        threshold = get_effective_line_length(BoardType.SQUARE8, 3)
        assert threshold == 3

    def test_square8_4player_threshold_is_3(self):
        """square8 4-player: threshold should be 3."""
        threshold = get_effective_line_length(BoardType.SQUARE8, 4)
        assert threshold == 3

    def test_square19_all_players_threshold_is_4(self):
        """square19: threshold should be 4 for all player counts."""
        for num_players in (2, 3, 4):
            threshold = get_effective_line_length(BoardType.SQUARE19, num_players)
            assert threshold == 4, f"square19 {num_players}p should have threshold 4"

    def test_hexagonal_all_players_threshold_is_4(self):
        """hexagonal: threshold should be 4 for all player counts."""
        for num_players in (2, 3, 4):
            threshold = get_effective_line_length(BoardType.HEXAGONAL, num_players)
            assert threshold == 4, f"hexagonal {num_players}p should have threshold 4"
