"""Tests for event_utils module.

December 30, 2025: Tests for the consolidated event extraction utilities.
"""

import pytest

from app.coordination.event_utils import (
    EvaluationEventData,
    ParsedConfigKey,
    TrainingEventData,
    extract_board_type_and_players,
    extract_config_key,
    extract_evaluation_data,
    extract_model_path,
    extract_training_data,
    make_config_key,
    parse_config_key,
)


class TestParseConfigKey:
    """Tests for parse_config_key function."""

    def test_parse_valid_2p(self):
        """Test parsing a valid 2-player config key."""
        result = parse_config_key("hex8_2p")
        assert result is not None
        assert result.board_type == "hex8"
        assert result.num_players == 2
        assert result.raw == "hex8_2p"
        assert result.config_key == "hex8_2p"

    def test_parse_valid_4p(self):
        """Test parsing a valid 4-player config key."""
        result = parse_config_key("square19_4p")
        assert result is not None
        assert result.board_type == "square19"
        assert result.num_players == 4

    def test_parse_valid_3p(self):
        """Test parsing a valid 3-player config key."""
        result = parse_config_key("hexagonal_3p")
        assert result is not None
        assert result.board_type == "hexagonal"
        assert result.num_players == 3

    def test_parse_without_p_suffix(self):
        """Test parsing config key without 'p' suffix."""
        result = parse_config_key("hex8_2")
        assert result is not None
        assert result.board_type == "hex8"
        assert result.num_players == 2

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        assert parse_config_key("") is None

    def test_parse_no_underscore(self):
        """Test parsing without underscore returns None."""
        assert parse_config_key("hex8") is None

    def test_parse_invalid_player_count(self):
        """Test parsing with invalid player count."""
        assert parse_config_key("hex8_1p") is None  # Less than 2
        assert parse_config_key("hex8_5p") is None  # More than 4

    def test_parse_non_numeric_players(self):
        """Test parsing with non-numeric player count."""
        assert parse_config_key("hex8_xp") is None

    def test_parse_multiple_underscores(self):
        """Test parsing with multiple underscores uses last one."""
        result = parse_config_key("my_custom_board_2p")
        assert result is not None
        assert result.board_type == "my_custom_board"
        assert result.num_players == 2


class TestExtractConfigKey:
    """Tests for extract_config_key function."""

    def test_extract_from_config_key(self):
        """Test extracting from config_key field."""
        event = {"config_key": "hex8_2p"}
        assert extract_config_key(event) == "hex8_2p"

    def test_extract_from_config(self):
        """Test extracting from config field as fallback."""
        event = {"config": "square8_4p"}
        assert extract_config_key(event) == "square8_4p"

    def test_config_key_takes_precedence(self):
        """Test that config_key takes precedence over config."""
        event = {"config_key": "hex8_2p", "config": "square8_4p"}
        assert extract_config_key(event) == "hex8_2p"

    def test_extract_empty_event(self):
        """Test extracting from empty event."""
        assert extract_config_key({}) == ""


class TestExtractModelPath:
    """Tests for extract_model_path function."""

    def test_extract_from_model_path(self):
        """Test extracting from model_path field."""
        event = {"model_path": "/path/to/model.pth"}
        assert extract_model_path(event) == "/path/to/model.pth"

    def test_extract_from_model_id(self):
        """Test extracting from model_id field as fallback."""
        event = {"model_id": "canonical_hex8_2p"}
        assert extract_model_path(event) == "canonical_hex8_2p"

    def test_extract_from_model(self):
        """Test extracting from model field as fallback."""
        event = {"model": "my_model.pth"}
        assert extract_model_path(event) == "my_model.pth"

    def test_model_path_takes_precedence(self):
        """Test that model_path takes precedence."""
        event = {"model_path": "/path/model.pth", "model_id": "other"}
        assert extract_model_path(event) == "/path/model.pth"

    def test_fallback_to_config(self):
        """Test fallback to config field."""
        event = {"config": "hex8_2p"}
        assert extract_model_path(event) == "hex8_2p"


class TestExtractBoardTypeAndPlayers:
    """Tests for extract_board_type_and_players function."""

    def test_extract_direct_fields(self):
        """Test extracting when fields are directly available."""
        event = {"board_type": "hex8", "num_players": 2}
        board_type, num_players = extract_board_type_and_players(event)
        assert board_type == "hex8"
        assert num_players == 2

    def test_fallback_to_config_key(self):
        """Test fallback to parsing config_key."""
        event = {"config_key": "square19_4p"}
        board_type, num_players = extract_board_type_and_players(event)
        assert board_type == "square19"
        assert num_players == 4

    def test_partial_fields_with_fallback(self):
        """Test when only board_type is present."""
        event = {"board_type": "hex8", "config_key": "hex8_3p"}
        board_type, num_players = extract_board_type_and_players(event)
        # Should use direct board_type but get num_players from config_key
        assert board_type == "hex8"
        assert num_players == 3

    def test_empty_event(self):
        """Test extracting from empty event."""
        board_type, num_players = extract_board_type_and_players({})
        assert board_type == ""
        assert num_players == 0


class TestExtractEvaluationData:
    """Tests for extract_evaluation_data function."""

    def test_extract_full_evaluation_event(self):
        """Test extracting from a complete evaluation event."""
        event = {
            "config_key": "hex8_2p",
            "model_path": "/path/to/model.pth",
            "elo": 1450.5,
            "games_played": 100,
            "win_rate": 0.65,
        }
        data = extract_evaluation_data(event)

        assert data.config_key == "hex8_2p"
        assert data.board_type == "hex8"
        assert data.num_players == 2
        assert data.model_path == "/path/to/model.pth"
        assert data.elo == 1450.5
        assert data.games_played == 100
        assert data.win_rate == 0.65
        assert data.is_valid

    def test_extract_with_games_field(self):
        """Test extracting when games field is used instead of games_played."""
        event = {"config_key": "hex8_2p", "games": 50}
        data = extract_evaluation_data(event)
        assert data.games_played == 50

    def test_extract_multi_harness_fields(self):
        """Test extracting multi-harness evaluation fields."""
        event = {
            "config_key": "hex8_4p",
            "is_multi_harness": True,
            "harness_results": {"gumbel": 1500, "minimax": 1450},
            "best_harness": "gumbel",
            "composite_participant_ids": ["model:gumbel:abc", "model:minimax:def"],
        }
        data = extract_evaluation_data(event)

        assert data.is_multi_harness is True
        assert data.harness_results == {"gumbel": 1500, "minimax": 1450}
        assert data.best_harness == "gumbel"
        assert len(data.composite_participant_ids) == 2

    def test_extract_defaults(self):
        """Test that defaults are applied correctly."""
        event = {"config_key": "hex8_2p"}
        data = extract_evaluation_data(event)

        assert data.elo == 1000.0
        assert data.games_played == 0
        assert data.win_rate == 0.0
        assert data.is_multi_harness is False
        assert data.harness_results is None

    def test_is_valid_property(self):
        """Test is_valid property."""
        valid = extract_evaluation_data({"config_key": "hex8_2p"})
        assert valid.is_valid

        invalid = extract_evaluation_data({})
        assert not invalid.is_valid


class TestExtractTrainingData:
    """Tests for extract_training_data function."""

    def test_extract_full_training_event(self):
        """Test extracting from a complete training event."""
        event = {
            "config_key": "square8_4p",
            "model_path": "/path/to/model.pth",
            "epochs": 50,
            "final_loss": 0.123,
            "samples_trained": 10000,
        }
        data = extract_training_data(event)

        assert data.config_key == "square8_4p"
        assert data.board_type == "square8"
        assert data.num_players == 4
        assert data.model_path == "/path/to/model.pth"
        assert data.epochs == 50
        assert data.final_loss == 0.123
        assert data.samples_trained == 10000
        assert data.is_valid

    def test_extract_with_alternate_field_names(self):
        """Test extracting with alternate field names."""
        event = {
            "config_key": "hex8_2p",
            "loss": 0.456,  # alternate name
            "samples": 5000,  # alternate name
        }
        data = extract_training_data(event)

        assert data.final_loss == 0.456
        assert data.samples_trained == 5000

    def test_extract_defaults(self):
        """Test that defaults are applied correctly."""
        event = {"config_key": "hex8_2p"}
        data = extract_training_data(event)

        assert data.epochs == 0
        assert data.final_loss == 0.0
        assert data.samples_trained == 0


class TestMakeConfigKey:
    """Tests for make_config_key function."""

    def test_make_2p_config_key(self):
        """Test making a 2-player config key."""
        assert make_config_key("hex8", 2) == "hex8_2p"

    def test_make_4p_config_key(self):
        """Test making a 4-player config key."""
        assert make_config_key("square19", 4) == "square19_4p"

    def test_roundtrip(self):
        """Test that make and parse are inverse operations."""
        for config in ["hex8_2p", "square8_3p", "hexagonal_4p"]:
            parsed = parse_config_key(config)
            assert parsed is not None
            reconstructed = make_config_key(parsed.board_type, parsed.num_players)
            assert reconstructed == config


class TestDataclassProperties:
    """Tests for dataclass property methods."""

    def test_parsed_config_key_config_key_property(self):
        """Test config_key property on ParsedConfigKey."""
        parsed = ParsedConfigKey(board_type="hex8", num_players=2, raw="hex8_2p")
        assert parsed.config_key == "hex8_2p"

        # Should normalize the format
        parsed2 = ParsedConfigKey(board_type="hex8", num_players=2, raw="hex8_2")
        assert parsed2.config_key == "hex8_2p"

    def test_evaluation_data_is_valid(self):
        """Test is_valid on EvaluationEventData."""
        valid = EvaluationEventData(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            model_path="/path",
            elo=1000,
            games_played=0,
            win_rate=0.0,
        )
        assert valid.is_valid

        invalid = EvaluationEventData(
            config_key="",
            board_type="",
            num_players=0,
            model_path="/path",
            elo=1000,
            games_played=0,
            win_rate=0.0,
        )
        assert not invalid.is_valid

    def test_training_data_is_valid(self):
        """Test is_valid on TrainingEventData."""
        valid = TrainingEventData(
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            model_path="/path",
            epochs=10,
            final_loss=0.1,
            samples_trained=1000,
        )
        assert valid.is_valid

        invalid = TrainingEventData(
            config_key="",
            board_type="",
            num_players=0,
            model_path="/path",
            epochs=10,
            final_loss=0.1,
            samples_trained=1000,
        )
        assert not invalid.is_valid
