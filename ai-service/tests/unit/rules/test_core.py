"""Tests for app.rules.core module.

Tests the core rules utilities including board configurations, line lengths,
victory thresholds, distance calculations, and path positions.
"""

import pytest

from app.models import BoardType, Position
from app.rules.core import (
    BOARD_CONFIGS,
    BoardConfig,
    get_line_length_for_board,
    get_effective_line_length,
    get_victory_threshold,
    get_territory_victory_minimum,
    get_territory_victory_threshold,
    get_rings_per_player,
    get_board_size,
    get_total_spaces,
    calculate_cap_height,
    calculate_distance,
    get_path_positions,
)


class TestBoardConfigs:
    """Tests for BOARD_CONFIGS dictionary."""

    def test_all_board_types_configured(self):
        """Test all board types have configurations."""
        assert BoardType.SQUARE8 in BOARD_CONFIGS
        assert BoardType.SQUARE19 in BOARD_CONFIGS
        assert BoardType.HEXAGONAL in BOARD_CONFIGS
        assert BoardType.HEX8 in BOARD_CONFIGS

    def test_square8_config(self):
        """Test square8 configuration values."""
        config = BOARD_CONFIGS[BoardType.SQUARE8]
        assert config.size == 8
        assert config.total_spaces == 64
        assert config.rings_per_player == 18
        assert config.line_length == 3

    def test_square19_config(self):
        """Test square19 configuration values."""
        config = BOARD_CONFIGS[BoardType.SQUARE19]
        assert config.size == 19
        assert config.total_spaces == 361
        assert config.rings_per_player == 72
        assert config.line_length == 4

    def test_hexagonal_config(self):
        """Test hexagonal configuration values."""
        config = BOARD_CONFIGS[BoardType.HEXAGONAL]
        assert config.size == 25  # Bounding box for radius=12
        assert config.total_spaces == 469
        assert config.rings_per_player == 96
        assert config.line_length == 4

    def test_hex8_config(self):
        """Test hex8 configuration values."""
        config = BOARD_CONFIGS[BoardType.HEX8]
        assert config.size == 9  # Bounding box for radius=4
        assert config.total_spaces == 61  # 3r² + 3r + 1 = 61 for r=4
        assert config.rings_per_player == 18
        assert config.line_length == 4


class TestLineLengths:
    """Tests for line length functions."""

    def test_base_line_length_square8(self):
        """Test base line length for square8."""
        assert get_line_length_for_board(BoardType.SQUARE8) == 3

    def test_base_line_length_square19(self):
        """Test base line length for square19."""
        assert get_line_length_for_board(BoardType.SQUARE19) == 4

    def test_base_line_length_hex8(self):
        """Test base line length for hex8."""
        assert get_line_length_for_board(BoardType.HEX8) == 4

    def test_base_line_length_hexagonal(self):
        """Test base line length for hexagonal."""
        assert get_line_length_for_board(BoardType.HEXAGONAL) == 4

    def test_effective_line_length_square8_2p(self):
        """Test effective line length for square8 2-player (RR-CANON-R120)."""
        assert get_effective_line_length(BoardType.SQUARE8, 2) == 4

    def test_effective_line_length_square8_3p(self):
        """Test effective line length for square8 3-player."""
        assert get_effective_line_length(BoardType.SQUARE8, 3) == 3

    def test_effective_line_length_square8_4p(self):
        """Test effective line length for square8 4-player."""
        assert get_effective_line_length(BoardType.SQUARE8, 4) == 3

    def test_effective_line_length_hex8_2p(self):
        """Test effective line length for hex8 2-player."""
        assert get_effective_line_length(BoardType.HEX8, 2) == 4

    def test_effective_line_length_hex8_3p(self):
        """Test effective line length for hex8 3-player."""
        assert get_effective_line_length(BoardType.HEX8, 3) == 3

    def test_effective_line_length_hex8_4p(self):
        """Test effective line length for hex8 4-player."""
        assert get_effective_line_length(BoardType.HEX8, 4) == 3

    def test_effective_line_length_square19_all_players(self):
        """Test effective line length for square19 (always 4)."""
        assert get_effective_line_length(BoardType.SQUARE19, 2) == 4
        assert get_effective_line_length(BoardType.SQUARE19, 3) == 4
        assert get_effective_line_length(BoardType.SQUARE19, 4) == 4

    def test_effective_line_length_hexagonal_all_players(self):
        """Test effective line length for hexagonal (always 4)."""
        assert get_effective_line_length(BoardType.HEXAGONAL, 2) == 4
        assert get_effective_line_length(BoardType.HEXAGONAL, 3) == 4
        assert get_effective_line_length(BoardType.HEXAGONAL, 4) == 4


class TestVictoryThresholds:
    """Tests for victory threshold calculations."""

    def test_ring_victory_2_player(self):
        """Test ring victory threshold for 2-player games."""
        # For 2-player, must eliminate all opponent rings
        assert get_victory_threshold(BoardType.SQUARE8, 2) == 18
        assert get_victory_threshold(BoardType.SQUARE19, 2) == 72

    def test_ring_victory_3_player(self):
        """Test ring victory threshold for 3-player games."""
        # Per RR-CANON-R061: round(rings * (2/3 + 1/3 * (n-1)))
        # For n=3: round(18 * (2/3 + 2/3)) = round(18 * 4/3) = 24
        assert get_victory_threshold(BoardType.SQUARE8, 3) == 24

    def test_ring_victory_4_player(self):
        """Test ring victory threshold for 4-player games."""
        # For n=4: round(18 * (2/3 + 3/3)) = round(18 * 5/3) = 30
        assert get_victory_threshold(BoardType.SQUARE8, 4) == 30

    def test_ring_victory_with_override(self):
        """Test ring victory with rings per player override."""
        # With 10 rings per player, 2 players
        threshold = get_victory_threshold(BoardType.SQUARE8, 2, rings_per_player_override=10)
        assert threshold == 10  # Must eliminate all 10 opponent rings

    def test_territory_victory_minimum_2_player(self):
        """Test territory victory minimum for 2-player."""
        # floor(64/2) + 1 = 33
        assert get_territory_victory_minimum(BoardType.SQUARE8, 2) == 33
        # floor(361/2) + 1 = 181
        assert get_territory_victory_minimum(BoardType.SQUARE19, 2) == 181

    def test_territory_victory_minimum_3_player(self):
        """Test territory victory minimum for 3-player."""
        # floor(64/3) + 1 = 22
        assert get_territory_victory_minimum(BoardType.SQUARE8, 3) == 22

    def test_territory_victory_minimum_4_player(self):
        """Test territory victory minimum for 4-player."""
        # floor(64/4) + 1 = 17
        assert get_territory_victory_minimum(BoardType.SQUARE8, 4) == 17

    def test_territory_victory_threshold_legacy(self):
        """Test legacy territory victory threshold (>50%)."""
        assert get_territory_victory_threshold(BoardType.SQUARE8) == 33
        assert get_territory_victory_threshold(BoardType.SQUARE19) == 181


class TestBoardAccessors:
    """Tests for board property accessors."""

    def test_get_rings_per_player(self):
        """Test rings per player for each board type."""
        assert get_rings_per_player(BoardType.SQUARE8) == 18
        assert get_rings_per_player(BoardType.SQUARE19) == 72
        assert get_rings_per_player(BoardType.HEX8) == 18
        assert get_rings_per_player(BoardType.HEXAGONAL) == 96

    def test_get_rings_per_player_with_override(self):
        """Test rings per player with override."""
        assert get_rings_per_player(BoardType.SQUARE8, override=25) == 25
        assert get_rings_per_player(BoardType.SQUARE19, override=100) == 100

    def test_get_board_size(self):
        """Test board size for each board type."""
        assert get_board_size(BoardType.SQUARE8) == 8
        assert get_board_size(BoardType.SQUARE19) == 19
        assert get_board_size(BoardType.HEX8) == 9
        assert get_board_size(BoardType.HEXAGONAL) == 25

    def test_get_total_spaces(self):
        """Test total spaces for each board type."""
        assert get_total_spaces(BoardType.SQUARE8) == 64
        assert get_total_spaces(BoardType.SQUARE19) == 361
        assert get_total_spaces(BoardType.HEX8) == 61
        assert get_total_spaces(BoardType.HEXAGONAL) == 469


class TestCapHeight:
    """Tests for cap height calculation."""

    def test_empty_stack(self):
        """Test cap height of empty stack."""
        assert calculate_cap_height([]) == 0

    def test_single_ring(self):
        """Test cap height of single ring."""
        assert calculate_cap_height([0]) == 1
        assert calculate_cap_height([1]) == 1

    def test_same_player_stack(self):
        """Test cap height when all rings belong to same player."""
        assert calculate_cap_height([0, 0, 0]) == 3
        assert calculate_cap_height([1, 1]) == 2

    def test_mixed_stack_one_cap(self):
        """Test cap height with one ring on top."""
        assert calculate_cap_height([0, 0, 1]) == 1
        assert calculate_cap_height([1, 0]) == 1

    def test_mixed_stack_multi_cap(self):
        """Test cap height with multiple rings on top."""
        assert calculate_cap_height([0, 1, 1]) == 2
        assert calculate_cap_height([0, 0, 1, 1, 1]) == 3

    def test_alternating_stack(self):
        """Test cap height with alternating rings."""
        assert calculate_cap_height([0, 1, 0, 1]) == 1


class TestDistanceCalculation:
    """Tests for distance calculation."""

    def test_same_position_square(self):
        """Test distance between same positions on square board."""
        pos = Position(x=3, y=3)
        assert calculate_distance(BoardType.SQUARE8, pos, pos) == 0

    def test_horizontal_distance_square(self):
        """Test horizontal distance on square board."""
        from_pos = Position(x=0, y=3)
        to_pos = Position(x=5, y=3)
        assert calculate_distance(BoardType.SQUARE8, from_pos, to_pos) == 5

    def test_vertical_distance_square(self):
        """Test vertical distance on square board."""
        from_pos = Position(x=3, y=0)
        to_pos = Position(x=3, y=7)
        assert calculate_distance(BoardType.SQUARE8, from_pos, to_pos) == 7

    def test_diagonal_distance_square(self):
        """Test diagonal distance on square board (Chebyshev)."""
        from_pos = Position(x=0, y=0)
        to_pos = Position(x=4, y=4)
        # Chebyshev distance: max(dx, dy)
        assert calculate_distance(BoardType.SQUARE8, from_pos, to_pos) == 4

    def test_knight_move_distance_square(self):
        """Test knight-like move distance on square board."""
        from_pos = Position(x=0, y=0)
        to_pos = Position(x=2, y=3)
        # max(2, 3) = 3
        assert calculate_distance(BoardType.SQUARE8, from_pos, to_pos) == 3

    def test_same_position_hex(self):
        """Test distance between same positions on hex board."""
        pos = Position(x=4, y=4, z=-8)
        assert calculate_distance(BoardType.HEX8, pos, pos) == 0

    def test_hex_distance_adjacent(self):
        """Test distance to adjacent hex."""
        # Adjacent hexes in cube coords differ by 1 in two axes
        from_pos = Position(x=0, y=0, z=0)
        to_pos = Position(x=1, y=-1, z=0)
        # (|1| + |-1| + |0|) / 2 = 1
        assert calculate_distance(BoardType.HEX8, from_pos, to_pos) == 1


class TestPathPositions:
    """Tests for path position generation."""

    def test_same_position_path(self):
        """Test path between same positions."""
        pos = Position(x=3, y=3)
        path = get_path_positions(pos, pos)
        assert len(path) == 1
        assert path[0] == pos

    def test_horizontal_path(self):
        """Test horizontal path."""
        from_pos = Position(x=0, y=3)
        to_pos = Position(x=3, y=3)
        path = get_path_positions(from_pos, to_pos)

        assert len(path) == 4
        assert path[0] == from_pos
        assert path[-1] == to_pos

    def test_vertical_path(self):
        """Test vertical path."""
        from_pos = Position(x=3, y=0)
        to_pos = Position(x=3, y=2)
        path = get_path_positions(from_pos, to_pos)

        assert len(path) == 3
        assert path[0] == from_pos
        assert path[-1] == to_pos

    def test_diagonal_path(self):
        """Test diagonal path."""
        from_pos = Position(x=0, y=0)
        to_pos = Position(x=2, y=2)
        path = get_path_positions(from_pos, to_pos)

        assert len(path) == 3
        assert path[0] == from_pos
        assert path[1] == Position(x=1, y=1)
        assert path[-1] == to_pos

    def test_path_includes_endpoints(self):
        """Test path includes both start and end positions."""
        from_pos = Position(x=1, y=1)
        to_pos = Position(x=4, y=4)
        path = get_path_positions(from_pos, to_pos)

        assert from_pos in path
        assert to_pos in path


class TestBoardConfigNamedTuple:
    """Tests for BoardConfig named tuple."""

    def test_config_creation(self):
        """Test creating a BoardConfig."""
        config = BoardConfig(
            size=10,
            total_spaces=100,
            rings_per_player=20,
            line_length=3,
        )
        assert config.size == 10
        assert config.total_spaces == 100
        assert config.rings_per_player == 20
        assert config.line_length == 3

    def test_config_is_immutable(self):
        """Test that BoardConfig is immutable."""
        config = BoardConfig(
            size=10,
            total_spaces=100,
            rings_per_player=20,
            line_length=3,
        )
        with pytest.raises(AttributeError):
            config.size = 20  # type: ignore

    def test_config_indexable(self):
        """Test that BoardConfig is indexable."""
        config = BoardConfig(
            size=10,
            total_spaces=100,
            rings_per_player=20,
            line_length=3,
        )
        assert config[0] == 10  # size
        assert config[1] == 100  # total_spaces
        assert config[2] == 20  # rings_per_player
        assert config[3] == 3  # line_length
