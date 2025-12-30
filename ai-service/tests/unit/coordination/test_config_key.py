"""Unit tests for config_key module.

Tests the canonical ConfigKey implementation for parsing and representing
board configuration keys like 'hex8_2p'.

December 2025: Created as part of parse_config_key() consolidation.
"""

from __future__ import annotations

import pytest

from app.coordination.config_key import (
    VALID_BOARD_TYPES,
    VALID_PLAYER_COUNTS,
    ConfigKey,
    ConfigKeyTuple,
    ParsedConfigKey,
    format_config_key,
    get_all_config_keys,
    is_valid_config_key,
    parse_config_key,
    parse_config_key_safe,
    parse_config_key_tuple,
)


# =============================================================================
# ConfigKey Class Tests
# =============================================================================


class TestConfigKeyParsing:
    """Tests for ConfigKey.parse() method."""

    def test_parse_standard_format(self) -> None:
        """Should parse standard 'board_Np' format."""
        key = ConfigKey.parse("hex8_2p")
        assert key is not None
        assert key.board_type == "hex8"
        assert key.num_players == 2
        assert key.raw == "hex8_2p"

    def test_parse_without_p_suffix(self) -> None:
        """Should parse 'board_N' format without 'p' suffix."""
        key = ConfigKey.parse("square19_4")
        assert key is not None
        assert key.board_type == "square19"
        assert key.num_players == 4
        assert key.raw == "square19_4"

    def test_parse_all_valid_board_types(self) -> None:
        """Should parse all valid board types."""
        for board in VALID_BOARD_TYPES:
            key = ConfigKey.parse(f"{board}_2p")
            assert key is not None
            assert key.board_type == board

    def test_parse_all_valid_player_counts(self) -> None:
        """Should parse all valid player counts."""
        for players in VALID_PLAYER_COUNTS:
            key = ConfigKey.parse(f"hex8_{players}p")
            assert key is not None
            assert key.num_players == players

    def test_parse_returns_none_for_empty_string(self) -> None:
        """Should return None for empty string."""
        assert ConfigKey.parse("") is None

    def test_parse_returns_none_for_none(self) -> None:
        """Should return None for None input."""
        assert ConfigKey.parse(None) is None  # type: ignore

    def test_parse_returns_none_for_invalid_format(self) -> None:
        """Should return None for invalid format."""
        assert ConfigKey.parse("invalid") is None
        assert ConfigKey.parse("hex8") is None
        assert ConfigKey.parse("_2p") is None

    def test_parse_returns_none_for_invalid_player_count(self) -> None:
        """Should return None for player counts outside [2,4]."""
        assert ConfigKey.parse("hex8_1p") is None
        assert ConfigKey.parse("hex8_5p") is None
        assert ConfigKey.parse("hex8_0p") is None
        assert ConfigKey.parse("hex8_-1p") is None

    def test_parse_returns_none_for_non_numeric_players(self) -> None:
        """Should return None for non-numeric player count."""
        assert ConfigKey.parse("hex8_xp") is None
        assert ConfigKey.parse("hex8_twop") is None

    def test_parse_unknown_board_type(self) -> None:
        """Should parse unknown board types (with debug log)."""
        key = ConfigKey.parse("newboard_2p")
        assert key is not None
        assert key.board_type == "newboard"
        assert key.num_players == 2


class TestConfigKeyCreation:
    """Tests for ConfigKey direct creation."""

    def test_create_directly(self) -> None:
        """Should create ConfigKey directly with components."""
        key = ConfigKey(board_type="hex8", num_players=2)
        assert key.board_type == "hex8"
        assert key.num_players == 2
        assert key.raw == ""

    def test_create_with_raw(self) -> None:
        """Should create ConfigKey with raw string."""
        key = ConfigKey(board_type="hex8", num_players=2, raw="hex8_2p")
        assert key.raw == "hex8_2p"

    def test_from_components(self) -> None:
        """Should create via from_components factory."""
        key = ConfigKey.from_components("square8", 3)
        assert key.board_type == "square8"
        assert key.num_players == 3

    def test_create_raises_for_invalid_players(self) -> None:
        """Should raise ValueError for invalid player count."""
        with pytest.raises(ValueError, match="Invalid num_players"):
            ConfigKey(board_type="hex8", num_players=5)

        with pytest.raises(ValueError, match="Invalid num_players"):
            ConfigKey(board_type="hex8", num_players=1)


class TestConfigKeyProperties:
    """Tests for ConfigKey properties."""

    def test_config_key_property(self) -> None:
        """Should return canonical config_key format."""
        key = ConfigKey(board_type="hex8", num_players=2)
        assert key.config_key == "hex8_2p"

    def test_config_key_always_has_p_suffix(self) -> None:
        """config_key should always have 'p' suffix even if parsed without."""
        key = ConfigKey.parse("square19_4")
        assert key.config_key == "square19_4p"

    def test_str_returns_config_key(self) -> None:
        """__str__ should return canonical config_key."""
        key = ConfigKey(board_type="hexagonal", num_players=4)
        assert str(key) == "hexagonal_4p"


class TestConfigKeyEquality:
    """Tests for ConfigKey equality and hashing."""

    def test_equal_keys(self) -> None:
        """Keys with same board_type and num_players should be equal."""
        key1 = ConfigKey(board_type="hex8", num_players=2)
        key2 = ConfigKey(board_type="hex8", num_players=2)
        assert key1 == key2

    def test_equal_keys_different_raw(self) -> None:
        """Keys with same values but different raw should be equal."""
        key1 = ConfigKey.parse("hex8_2p")
        key2 = ConfigKey.parse("hex8_2")
        assert key1 == key2

    def test_unequal_board_type(self) -> None:
        """Keys with different board_type should not be equal."""
        key1 = ConfigKey(board_type="hex8", num_players=2)
        key2 = ConfigKey(board_type="square8", num_players=2)
        assert key1 != key2

    def test_unequal_players(self) -> None:
        """Keys with different num_players should not be equal."""
        key1 = ConfigKey(board_type="hex8", num_players=2)
        key2 = ConfigKey(board_type="hex8", num_players=4)
        assert key1 != key2

    def test_equal_to_string(self) -> None:
        """Should allow comparison with string config keys."""
        key = ConfigKey(board_type="hex8", num_players=2)
        assert key == "hex8_2p"
        assert key == "hex8_2"

    def test_hash_equal_for_equal_keys(self) -> None:
        """Equal keys should have equal hashes."""
        key1 = ConfigKey.parse("hex8_2p")
        key2 = ConfigKey.parse("hex8_2")
        assert hash(key1) == hash(key2)

    def test_hashable_for_dict_key(self) -> None:
        """ConfigKey should be usable as dict key."""
        key = ConfigKey(board_type="hex8", num_players=2)
        d = {key: "value"}
        assert d[key] == "value"

    def test_hashable_for_set_member(self) -> None:
        """ConfigKey should be usable in sets."""
        key1 = ConfigKey(board_type="hex8", num_players=2)
        key2 = ConfigKey.parse("hex8_2p")
        s = {key1, key2}
        assert len(s) == 1  # Both represent same key


class TestConfigKeyConversion:
    """Tests for ConfigKey conversion methods."""

    def test_to_tuple(self) -> None:
        """Should convert to tuple."""
        key = ConfigKey(board_type="hex8", num_players=2)
        assert key.to_tuple() == ("hex8", 2)

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        key = ConfigKey(board_type="hex8", num_players=2)
        d = key.to_dict()
        assert d["board_type"] == "hex8"
        assert d["num_players"] == 2
        assert d["config_key"] == "hex8_2p"


class TestConfigKeyImmutability:
    """Tests for ConfigKey immutability (frozen dataclass)."""

    def test_cannot_modify_board_type(self) -> None:
        """Should not allow modification of board_type."""
        key = ConfigKey(board_type="hex8", num_players=2)
        with pytest.raises(AttributeError):
            key.board_type = "square8"  # type: ignore

    def test_cannot_modify_num_players(self) -> None:
        """Should not allow modification of num_players."""
        key = ConfigKey(board_type="hex8", num_players=2)
        with pytest.raises(AttributeError):
            key.num_players = 4  # type: ignore


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestParseConfigKeyFunction:
    """Tests for parse_config_key() convenience function."""

    def test_returns_config_key(self) -> None:
        """Should return ConfigKey instance."""
        result = parse_config_key("hex8_2p")
        assert isinstance(result, ConfigKey)
        assert result.board_type == "hex8"
        assert result.num_players == 2

    def test_returns_none_for_invalid(self) -> None:
        """Should return None for invalid input."""
        assert parse_config_key("invalid") is None


class TestParseConfigKeyTuple:
    """Tests for parse_config_key_tuple() backward compat function."""

    def test_returns_tuple(self) -> None:
        """Should return tuple."""
        result = parse_config_key_tuple("hex8_2p")
        assert result == ("hex8", 2)
        assert isinstance(result, tuple)

    def test_returns_none_for_invalid(self) -> None:
        """Should return None for invalid input."""
        assert parse_config_key_tuple("invalid") is None


class TestParseConfigKeySafe:
    """Tests for parse_config_key_safe() backward compat function."""

    def test_returns_tuple_for_valid(self) -> None:
        """Should return tuple for valid input."""
        result = parse_config_key_safe("hex8_2p")
        assert result == ("hex8", 2)

    def test_returns_none_tuple_for_invalid(self) -> None:
        """Should return (None, None) for invalid input."""
        result = parse_config_key_safe("invalid")
        assert result == (None, None)


class TestFormatConfigKey:
    """Tests for format_config_key() function."""

    def test_formats_correctly(self) -> None:
        """Should format components into config key."""
        assert format_config_key("hex8", 2) == "hex8_2p"
        assert format_config_key("square19", 4) == "square19_4p"

    def test_raises_for_invalid_players(self) -> None:
        """Should raise ValueError for invalid player count."""
        with pytest.raises(ValueError):
            format_config_key("hex8", 5)


class TestIsValidConfigKey:
    """Tests for is_valid_config_key() function."""

    def test_returns_true_for_valid(self) -> None:
        """Should return True for valid config keys."""
        assert is_valid_config_key("hex8_2p") is True
        assert is_valid_config_key("square19_4") is True

    def test_returns_false_for_invalid(self) -> None:
        """Should return False for invalid config keys."""
        assert is_valid_config_key("invalid") is False
        assert is_valid_config_key("") is False
        assert is_valid_config_key("hex8_5p") is False


class TestGetAllConfigKeys:
    """Tests for get_all_config_keys() function."""

    def test_returns_all_combinations(self) -> None:
        """Should return all valid config key combinations."""
        keys = get_all_config_keys()
        # 4 board types * 3 player counts = 12 combinations
        assert len(keys) == len(VALID_BOARD_TYPES) * len(VALID_PLAYER_COUNTS)

    def test_all_have_p_suffix(self) -> None:
        """All returned keys should have 'p' suffix."""
        keys = get_all_config_keys()
        for key in keys:
            assert key.endswith("p")

    def test_all_are_valid(self) -> None:
        """All returned keys should be valid."""
        keys = get_all_config_keys()
        for key in keys:
            assert is_valid_config_key(key)


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_parsed_config_key_alias(self) -> None:
        """ParsedConfigKey should be alias for ConfigKey."""
        assert ParsedConfigKey is ConfigKey

    def test_config_key_tuple_namedtuple(self) -> None:
        """ConfigKeyTuple should be a NamedTuple."""
        t = ConfigKeyTuple(board_type="hex8", num_players=2)
        assert t.board_type == "hex8"
        assert t.num_players == 2
        assert t[0] == "hex8"
        assert t[1] == 2


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_valid_board_types(self) -> None:
        """Should have all expected board types."""
        assert "hex8" in VALID_BOARD_TYPES
        assert "square8" in VALID_BOARD_TYPES
        assert "square19" in VALID_BOARD_TYPES
        assert "hexagonal" in VALID_BOARD_TYPES
        assert len(VALID_BOARD_TYPES) == 4

    def test_valid_player_counts(self) -> None:
        """Should have all expected player counts."""
        assert 2 in VALID_PLAYER_COUNTS
        assert 3 in VALID_PLAYER_COUNTS
        assert 4 in VALID_PLAYER_COUNTS
        assert len(VALID_PLAYER_COUNTS) == 3

    def test_constants_are_frozenset(self) -> None:
        """Constants should be immutable frozensets."""
        assert isinstance(VALID_BOARD_TYPES, frozenset)
        assert isinstance(VALID_PLAYER_COUNTS, frozenset)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_board_type_with_numbers(self) -> None:
        """Should handle board types containing numbers."""
        key = ConfigKey.parse("board123_2p")
        assert key is not None
        assert key.board_type == "board123"

    def test_whitespace_handling(self) -> None:
        """Should not trim whitespace (exact matching)."""
        assert ConfigKey.parse(" hex8_2p") is None
        assert ConfigKey.parse("hex8_2p ") is None

    def test_case_sensitivity(self) -> None:
        """Should be case-sensitive."""
        key_lower = ConfigKey.parse("hex8_2p")
        key_upper = ConfigKey.parse("HEX8_2P")
        # Upper case should still parse (unknown board type is allowed)
        assert key_lower is not None
        assert key_upper is not None
        assert key_lower != key_upper  # Different board types

    def test_multiple_underscores(self) -> None:
        """Should handle board types that could have underscores."""
        # Uses rsplit with maxsplit=1, so only last underscore matters
        key = ConfigKey.parse("my_board_type_2p")
        assert key is not None
        assert key.board_type == "my_board_type"
        assert key.num_players == 2
