"""Tests for app.validation.domain module.

Tests domain-specific validators for RingRift.
"""

import pytest
import tempfile
from pathlib import Path

from app.validation.domain import (
    is_valid_board_type,
    is_valid_config_key,
    is_valid_elo,
    is_valid_model_path,
    is_valid_num_players,
    VALID_BOARD_TYPES,
)


class TestIsValidConfigKey:
    """Tests for is_valid_config_key validator."""

    def test_valid_square8_2p(self):
        """Test valid square8_2p config key."""
        result = is_valid_config_key("square8_2p")
        assert result.valid is True

    def test_valid_hex8_4p(self):
        """Test valid hex8_4p config key."""
        result = is_valid_config_key("hex8_4p")
        assert result.valid is True

    def test_valid_square19_3p(self):
        """Test valid square19_3p config key."""
        result = is_valid_config_key("square19_3p")
        assert result.valid is True

    def test_hexagonal_not_valid_in_config_key(self):
        """Test hexagonal_2p is NOT valid (base must be 'square' or 'hex')."""
        # Note: is_valid_config_key only accepts boards that reduce to "square" or "hex"
        # "hexagonal" doesn't match this pattern - use "hex8" instead
        result = is_valid_config_key("hexagonal_2p")
        assert result.valid is False

    def test_invalid_format_no_p(self):
        """Test invalid format without 'p' suffix."""
        result = is_valid_config_key("square8_2")
        assert result.valid is False

    def test_invalid_format_no_underscore(self):
        """Test invalid format without underscore."""
        result = is_valid_config_key("square82p")
        assert result.valid is False

    def test_invalid_board_type(self):
        """Test invalid board type."""
        result = is_valid_config_key("triangle8_2p")
        assert result.valid is False

    def test_invalid_player_count_too_low(self):
        """Test player count too low (1)."""
        result = is_valid_config_key("square8_1p")
        assert result.valid is False

    def test_invalid_player_count_too_high(self):
        """Test player count too high (9+)."""
        result = is_valid_config_key("square8_9p")
        assert result.valid is False

    def test_valid_player_count_range(self):
        """Test valid player counts (2-8)."""
        for n in range(2, 9):
            result = is_valid_config_key(f"square8_{n}p")
            assert result.valid is True, f"Failed for {n} players"

    def test_non_string(self):
        """Test non-string value."""
        result = is_valid_config_key(123)
        assert result.valid is False

    def test_empty_string(self):
        """Test empty string."""
        result = is_valid_config_key("")
        assert result.valid is False


class TestIsValidBoardType:
    """Tests for is_valid_board_type validator."""

    def test_square8(self):
        """Test square8 is valid."""
        assert is_valid_board_type("square8").valid is True

    def test_square19(self):
        """Test square19 is valid."""
        assert is_valid_board_type("square19").valid is True

    def test_hex8(self):
        """Test hex8 is valid."""
        assert is_valid_board_type("hex8").valid is True

    def test_hexagonal(self):
        """Test hexagonal is valid."""
        assert is_valid_board_type("hexagonal").valid is True

    def test_generic_square(self):
        """Test generic 'square' alias."""
        assert is_valid_board_type("square").valid is True

    def test_generic_hex(self):
        """Test generic 'hex' alias."""
        assert is_valid_board_type("hex").valid is True

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        assert is_valid_board_type("SQUARE8").valid is True
        assert is_valid_board_type("Square8").valid is True
        assert is_valid_board_type("HEX8").valid is True

    def test_invalid_type(self):
        """Test invalid board type."""
        result = is_valid_board_type("triangle")
        assert result.valid is False

    def test_invalid_number(self):
        """Test invalid board with wrong number."""
        result = is_valid_board_type("square99")
        assert result.valid is False

    def test_non_string(self):
        """Test non-string value."""
        result = is_valid_board_type(123)
        assert result.valid is False

    def test_fullhex_alias(self):
        """Test fullhex alias."""
        assert is_valid_board_type("fullhex").valid is True
        assert is_valid_board_type("full_hex").valid is True

    def test_all_valid_types(self):
        """Test all documented valid types."""
        for board_type in VALID_BOARD_TYPES:
            result = is_valid_board_type(board_type)
            assert result.valid is True, f"Failed for {board_type}"


class TestIsValidElo:
    """Tests for is_valid_elo validator."""

    def test_typical_elo(self):
        """Test typical Elo rating."""
        assert is_valid_elo(1200).valid is True
        assert is_valid_elo(1500).valid is True
        assert is_valid_elo(2000).valid is True

    def test_starting_elo(self):
        """Test starting Elo (1000-1200 typically)."""
        assert is_valid_elo(1000).valid is True

    def test_zero_elo(self):
        """Test zero Elo is valid."""
        assert is_valid_elo(0).valid is True

    def test_high_elo(self):
        """Test high but valid Elo."""
        assert is_valid_elo(3000).valid is True
        assert is_valid_elo(4000).valid is True

    def test_negative_elo(self):
        """Test negative Elo is invalid."""
        result = is_valid_elo(-100)
        assert result.valid is False

    def test_suspiciously_high_elo(self):
        """Test suspiciously high Elo (>5000)."""
        result = is_valid_elo(6000)
        assert result.valid is False

    def test_float_elo(self):
        """Test float Elo values."""
        assert is_valid_elo(1500.5).valid is True
        assert is_valid_elo(1234.67).valid is True

    def test_string_numeric(self):
        """Test string that can be converted to number."""
        assert is_valid_elo("1500").valid is True

    def test_non_numeric(self):
        """Test non-numeric value."""
        result = is_valid_elo("not a number")
        assert result.valid is False

    def test_none(self):
        """Test None value."""
        result = is_valid_elo(None)
        assert result.valid is False


class TestIsValidModelPath:
    """Tests for is_valid_model_path validator."""

    def test_valid_pt_extension(self):
        """Test valid .pt extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_valid_model_path(f"{tmpdir}/model.pt")
            assert result.valid is True

    def test_valid_pth_extension(self):
        """Test valid .pth extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_valid_model_path(f"{tmpdir}/model.pth")
            assert result.valid is True

    def test_valid_onnx_extension(self):
        """Test valid .onnx extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_valid_model_path(f"{tmpdir}/model.onnx")
            assert result.valid is True

    def test_valid_weights_extension(self):
        """Test valid .weights extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_valid_model_path(f"{tmpdir}/model.weights")
            assert result.valid is True

    def test_valid_bin_extension(self):
        """Test valid .bin extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_valid_model_path(f"{tmpdir}/model.bin")
            assert result.valid is True

    def test_invalid_extension(self):
        """Test invalid extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_valid_model_path(f"{tmpdir}/model.txt")
            assert result.valid is False

    def test_no_extension(self):
        """Test path without extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_valid_model_path(f"{tmpdir}/model")
            assert result.valid is False

    def test_parent_not_exists(self):
        """Test path with non-existent parent directory."""
        result = is_valid_model_path("/nonexistent/directory/model.pt")
        assert result.valid is False

    def test_path_object(self):
        """Test Path object input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            result = is_valid_model_path(path)
            assert result.valid is True

    def test_non_string_or_path(self):
        """Test non-string/Path value."""
        result = is_valid_model_path(123)
        assert result.valid is False

    def test_case_insensitive_extension(self):
        """Test case-insensitive extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = is_valid_model_path(f"{tmpdir}/model.PT")
            assert result.valid is True
            result = is_valid_model_path(f"{tmpdir}/model.PTH")
            assert result.valid is True


class TestIsValidNumPlayers:
    """Tests for is_valid_num_players validator."""

    def test_two_players(self):
        """Test 2 players is valid."""
        assert is_valid_num_players(2).valid is True

    def test_four_players(self):
        """Test 4 players is valid."""
        assert is_valid_num_players(4).valid is True

    def test_eight_players(self):
        """Test 8 players is valid."""
        assert is_valid_num_players(8).valid is True

    def test_all_valid_counts(self):
        """Test all valid player counts (2-8)."""
        for n in range(2, 9):
            result = is_valid_num_players(n)
            assert result.valid is True, f"Failed for {n} players"

    def test_one_player(self):
        """Test 1 player is invalid."""
        result = is_valid_num_players(1)
        assert result.valid is False

    def test_zero_players(self):
        """Test 0 players is invalid."""
        result = is_valid_num_players(0)
        assert result.valid is False

    def test_nine_players(self):
        """Test 9 players is invalid (>8)."""
        result = is_valid_num_players(9)
        assert result.valid is False

    def test_negative_players(self):
        """Test negative players is invalid."""
        result = is_valid_num_players(-1)
        assert result.valid is False

    def test_string_numeric(self):
        """Test string that can be converted to int."""
        assert is_valid_num_players("4").valid is True

    def test_float_value(self):
        """Test float value (should convert to int)."""
        assert is_valid_num_players(4.0).valid is True

    def test_non_numeric(self):
        """Test non-numeric value."""
        result = is_valid_num_players("four")
        assert result.valid is False

    def test_none(self):
        """Test None value."""
        result = is_valid_num_players(None)
        assert result.valid is False
