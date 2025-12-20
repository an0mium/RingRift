"""Tests for canonical naming utilities."""
import pytest

from app.models import BoardType
from app.utils.canonical_naming import (
    CANONICAL_CONFIG_KEYS,
    get_all_config_keys,
    get_board_type_enum,
    is_valid_board_type,
    make_config_key,
    normalize_board_type,
    normalize_database_filename,
    parse_config_key,
)


class TestNormalizeBoardType:
    """Tests for normalize_board_type function."""

    def test_canonical_values_unchanged(self):
        """Canonical values should return unchanged."""
        assert normalize_board_type("square8") == "square8"
        assert normalize_board_type("square19") == "square19"
        assert normalize_board_type("hex8") == "hex8"
        assert normalize_board_type("hexagonal") == "hexagonal"

    def test_square8_aliases(self):
        """Various square8 aliases should normalize correctly."""
        assert normalize_board_type("sq8") == "square8"
        assert normalize_board_type("square_8") == "square8"
        assert normalize_board_type("square-8") == "square8"
        assert normalize_board_type("8x8") == "square8"

    def test_square19_aliases(self):
        """Various square19 aliases should normalize correctly."""
        assert normalize_board_type("sq19") == "square19"
        assert normalize_board_type("square_19") == "square19"
        assert normalize_board_type("19x19") == "square19"

    def test_hex8_aliases(self):
        """Various hex8 aliases should normalize correctly."""
        assert normalize_board_type("hex_8") == "hex8"
        assert normalize_board_type("hex-8") == "hex8"
        assert normalize_board_type("smallhex") == "hex8"
        assert normalize_board_type("small_hex") == "hex8"

    def test_hexagonal_aliases(self):
        """Various hexagonal aliases should normalize correctly."""
        assert normalize_board_type("hex") == "hexagonal"
        assert normalize_board_type("hex24") == "hexagonal"  # Diameter convention (2 * radius 12)
        assert normalize_board_type("hex_24") == "hexagonal"
        assert normalize_board_type("largehex") == "hexagonal"
        assert normalize_board_type("bighex") == "hexagonal"

    def test_case_insensitive(self):
        """Normalization should be case-insensitive."""
        assert normalize_board_type("SQUARE8") == "square8"
        assert normalize_board_type("Square8") == "square8"
        assert normalize_board_type("HEX") == "hexagonal"

    def test_whitespace_stripped(self):
        """Whitespace should be stripped."""
        assert normalize_board_type("  square8  ") == "square8"

    def test_board_type_enum_input(self):
        """BoardType enum values should be handled."""
        assert normalize_board_type(BoardType.SQUARE8) == "square8"
        assert normalize_board_type(BoardType.HEX8) == "hex8"
        assert normalize_board_type(BoardType.HEXAGONAL) == "hexagonal"

    def test_invalid_board_type_raises(self):
        """Invalid board types should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown board type"):
            normalize_board_type("invalid")
        with pytest.raises(ValueError, match="Unknown board type"):
            normalize_board_type("square10")


class TestGetBoardTypeEnum:
    """Tests for get_board_type_enum function."""

    def test_string_to_enum(self):
        """String board types should convert to enum."""
        assert get_board_type_enum("square8") == BoardType.SQUARE8
        assert get_board_type_enum("sq8") == BoardType.SQUARE8
        assert get_board_type_enum("hex") == BoardType.HEXAGONAL
        assert get_board_type_enum("hex8") == BoardType.HEX8

    def test_enum_passthrough(self):
        """Enum values should pass through unchanged."""
        assert get_board_type_enum(BoardType.SQUARE8) == BoardType.SQUARE8
        assert get_board_type_enum(BoardType.HEXAGONAL) == BoardType.HEXAGONAL


class TestMakeConfigKey:
    """Tests for make_config_key function."""

    def test_basic_config_keys(self):
        """Basic config key generation."""
        assert make_config_key("square8", 2) == "square8_2p"
        assert make_config_key("hexagonal", 4) == "hexagonal_4p"
        assert make_config_key("hex8", 3) == "hex8_3p"

    def test_normalizes_board_type(self):
        """Config key should normalize board type."""
        assert make_config_key("sq8", 2) == "square8_2p"
        assert make_config_key("hex", 4) == "hexagonal_4p"

    def test_enum_input(self):
        """BoardType enum should work as input."""
        assert make_config_key(BoardType.SQUARE8, 2) == "square8_2p"

    def test_invalid_player_count(self):
        """Invalid player counts should raise ValueError."""
        with pytest.raises(ValueError, match="num_players must be 2-4"):
            make_config_key("square8", 1)
        with pytest.raises(ValueError, match="num_players must be 2-4"):
            make_config_key("square8", 5)


class TestParseConfigKey:
    """Tests for parse_config_key function."""

    def test_basic_parsing(self):
        """Basic config key parsing."""
        assert parse_config_key("square8_2p") == ("square8", 2)
        assert parse_config_key("hexagonal_4p") == ("hexagonal", 4)
        assert parse_config_key("hex8_3p") == ("hex8", 3)

    def test_case_insensitive(self):
        """Parsing should be case-insensitive."""
        assert parse_config_key("SQUARE8_2P") == ("square8", 2)

    def test_normalizes_board_type(self):
        """Parsed board type should be normalized."""
        assert parse_config_key("sq8_2p") == ("square8", 2)

    def test_invalid_format_raises(self):
        """Invalid formats should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid config key format"):
            parse_config_key("square8")
        with pytest.raises(ValueError, match="Invalid config key format"):
            parse_config_key("square8-2p")

    def test_invalid_player_count_raises(self):
        """Invalid player counts in config key should raise."""
        with pytest.raises(ValueError, match="Invalid player count"):
            parse_config_key("square8_1p")
        with pytest.raises(ValueError, match="Invalid player count"):
            parse_config_key("square8_5p")


class TestIsValidBoardType:
    """Tests for is_valid_board_type function."""

    def test_valid_types(self):
        """Valid board types should return True."""
        assert is_valid_board_type("square8") is True
        assert is_valid_board_type("sq8") is True
        assert is_valid_board_type("hex") is True
        assert is_valid_board_type("hexagonal") is True

    def test_invalid_types(self):
        """Invalid board types should return False."""
        assert is_valid_board_type("invalid") is False
        assert is_valid_board_type("square10") is False


class TestGetAllConfigKeys:
    """Tests for get_all_config_keys function."""

    def test_returns_all_combinations(self):
        """Should return all board type + player count combinations."""
        keys = get_all_config_keys()
        # 4 board types * 3 player counts = 12 keys
        assert len(keys) == 12
        assert "square8_2p" in keys
        assert "hexagonal_4p" in keys
        assert "hex8_3p" in keys

    def test_sorted_order(self):
        """Keys should be sorted."""
        keys = get_all_config_keys()
        assert keys == sorted(keys)


class TestNormalizeDatabaseFilename:
    """Tests for normalize_database_filename function."""

    def test_basic_filename(self):
        """Basic filename generation."""
        assert normalize_database_filename("square8", 2) == "selfplay_square8_2p.db"
        assert normalize_database_filename("hexagonal", 4) == "selfplay_hexagonal_4p.db"

    def test_normalizes_board_type(self):
        """Filename should use normalized board type."""
        assert normalize_database_filename("sq8", 2) == "selfplay_square8_2p.db"
        assert normalize_database_filename("hex", 4) == "selfplay_hexagonal_4p.db"

    def test_custom_prefix(self):
        """Custom prefix should be used."""
        assert normalize_database_filename("square8", 2, prefix="games") == "games_square8_2p.db"
        assert normalize_database_filename("hex8", 3, prefix="tournament") == "tournament_hex8_3p.db"

    def test_suffix(self):
        """Suffix should be appended before extension."""
        assert normalize_database_filename("square8", 2, suffix="_vast123") == "selfplay_square8_2p_vast123.db"


class TestCanonicalConfigKeys:
    """Tests for CANONICAL_CONFIG_KEYS constant."""

    def test_is_list(self):
        """Should be a list."""
        assert isinstance(CANONICAL_CONFIG_KEYS, list)

    def test_has_expected_count(self):
        """Should have 12 keys (4 boards * 3 player counts)."""
        assert len(CANONICAL_CONFIG_KEYS) == 12

    def test_contains_expected_keys(self):
        """Should contain expected keys."""
        assert "square8_2p" in CANONICAL_CONFIG_KEYS
        assert "square8_3p" in CANONICAL_CONFIG_KEYS
        assert "square8_4p" in CANONICAL_CONFIG_KEYS
        assert "hexagonal_2p" in CANONICAL_CONFIG_KEYS
