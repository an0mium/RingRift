"""Unit tests for selfplay_config module.

December 2025: Tests for the unified selfplay configuration system
that eliminates duplicated argument parsing across 34+ scripts.
"""

import argparse
from dataclasses import fields
from unittest.mock import patch

import pytest

from app.training.selfplay_config import (
    CPU_COMPATIBLE_ENGINE_MODES,
    CURRICULUM_STAGES,
    ENGINE_MODE_ALIASES,
    GPU_REQUIRED_ENGINE_MODES,
    MIXED_ENGINE_MODES,
    CurriculumStage,
    EngineMode,
    OutputFormat,
    SelfplayConfig,
    create_argument_parser,
    engine_mode_is_cpu_compatible,
    engine_mode_requires_gpu,
    get_all_configs_curriculum,
    get_curriculum_config,
    get_default_config,
    get_full_curriculum,
    get_production_config,
    list_curriculum_stages,
    normalize_engine_mode,
    parse_selfplay_args,
)


class TestEngineMode:
    """Tests for EngineMode enum."""

    def test_all_engine_modes_exist(self) -> None:
        """Test that all expected engine modes are defined."""
        expected_modes = [
            "heuristic-only",
            "nnue-guided",
            "policy-only",
            "nn-minimax",
            "nn-descent",
            "gumbel-mcts",
            "mcts",
            "mixed",
            "diverse",
            "random",
            "descent-only",
            "maxn",
            "brs",
            "gmo",
            "ebmo",
            "ig-gmo",
            "cage",
            "gnn",
            "hybrid",
        ]
        for mode_value in expected_modes:
            assert any(
                e.value == mode_value for e in EngineMode
            ), f"Missing mode: {mode_value}"

    def test_engine_mode_count(self) -> None:
        """Test total number of engine modes."""
        assert len(EngineMode) == 19

    def test_engine_mode_is_str_enum(self) -> None:
        """Test that EngineMode is a string enum."""
        assert EngineMode.GUMBEL_MCTS == "gumbel-mcts"
        assert EngineMode.HEURISTIC.value == "heuristic-only"

    def test_deprecated_modes_marked(self) -> None:
        """Test that deprecated modes are documented in enum."""
        deprecated_modes = [
            EngineMode.GMO,
            EngineMode.EBMO,
            EngineMode.IG_GMO,
            EngineMode.CAGE,
        ]
        for mode in deprecated_modes:
            assert mode in EngineMode.__members__.values()


class TestEngineModeAliases:
    """Tests for ENGINE_MODE_ALIASES dict."""

    def test_common_aliases_exist(self) -> None:
        """Test that common aliases are defined."""
        assert "gumbel" in ENGINE_MODE_ALIASES
        assert "gumbel_mcts" in ENGINE_MODE_ALIASES
        assert "heuristic" in ENGINE_MODE_ALIASES
        assert "descent" in ENGINE_MODE_ALIASES

    def test_aliases_map_to_valid_modes(self) -> None:
        """Test that all aliases map to valid EngineMode values."""
        valid_values = {e.value for e in EngineMode}
        for alias, value in ENGINE_MODE_ALIASES.items():
            assert value in valid_values, f"Alias '{alias}' maps to invalid value '{value}'"

    def test_gnn_aliases(self) -> None:
        """Test GNN-related aliases."""
        assert ENGINE_MODE_ALIASES["gnn-policy"] == EngineMode.GNN.value
        assert ENGINE_MODE_ALIASES["hybrid-gnn"] == EngineMode.HYBRID.value
        assert ENGINE_MODE_ALIASES["cnn-gnn"] == EngineMode.HYBRID.value


class TestGpuRequirements:
    """Tests for GPU requirement frozensets and functions."""

    def test_gpu_required_modes_are_valid(self) -> None:
        """Test that GPU_REQUIRED_ENGINE_MODES contains valid EngineMode members."""
        for mode in GPU_REQUIRED_ENGINE_MODES:
            assert isinstance(mode, EngineMode)

    def test_cpu_compatible_modes_are_valid(self) -> None:
        """Test that CPU_COMPATIBLE_ENGINE_MODES contains valid EngineMode members."""
        for mode in CPU_COMPATIBLE_ENGINE_MODES:
            assert isinstance(mode, EngineMode)

    def test_mixed_modes_are_valid(self) -> None:
        """Test that MIXED_ENGINE_MODES contains valid EngineMode members."""
        for mode in MIXED_ENGINE_MODES:
            assert isinstance(mode, EngineMode)

    def test_no_overlap_between_gpu_and_cpu(self) -> None:
        """Test that GPU and CPU sets don't overlap."""
        overlap = GPU_REQUIRED_ENGINE_MODES & CPU_COMPATIBLE_ENGINE_MODES
        assert len(overlap) == 0

    def test_gumbel_mcts_requires_gpu(self) -> None:
        """Test that GUMBEL_MCTS is in GPU required set."""
        assert EngineMode.GUMBEL_MCTS in GPU_REQUIRED_ENGINE_MODES

    def test_heuristic_is_cpu_compatible(self) -> None:
        """Test that HEURISTIC is CPU compatible."""
        assert EngineMode.HEURISTIC in CPU_COMPATIBLE_ENGINE_MODES

    def test_mixed_is_in_mixed_modes(self) -> None:
        """Test that MIXED is in mixed modes set."""
        assert EngineMode.MIXED in MIXED_ENGINE_MODES


class TestEngineModeRequiresGpu:
    """Tests for engine_mode_requires_gpu function."""

    def test_with_enum(self) -> None:
        """Test with EngineMode enum."""
        assert engine_mode_requires_gpu(EngineMode.GUMBEL_MCTS) is True
        assert engine_mode_requires_gpu(EngineMode.HEURISTIC) is False

    def test_with_string(self) -> None:
        """Test with string values."""
        assert engine_mode_requires_gpu("gumbel-mcts") is True
        assert engine_mode_requires_gpu("heuristic-only") is False

    def test_with_alias(self) -> None:
        """Test with alias strings."""
        assert engine_mode_requires_gpu("gumbel") is True
        assert engine_mode_requires_gpu("heuristic") is False

    def test_unknown_mode_returns_true(self) -> None:
        """Test that unknown modes return True for safety."""
        assert engine_mode_requires_gpu("unknown-mode") is True

    def test_mixed_mode_not_gpu_required(self) -> None:
        """Test that mixed modes don't require GPU."""
        assert engine_mode_requires_gpu(EngineMode.MIXED) is False


class TestEngineModeIsCpuCompatible:
    """Tests for engine_mode_is_cpu_compatible function."""

    def test_with_enum(self) -> None:
        """Test with EngineMode enum."""
        assert engine_mode_is_cpu_compatible(EngineMode.HEURISTIC) is True
        assert engine_mode_is_cpu_compatible(EngineMode.GUMBEL_MCTS) is False

    def test_with_string(self) -> None:
        """Test with string values."""
        assert engine_mode_is_cpu_compatible("random") is True
        assert engine_mode_is_cpu_compatible("gumbel-mcts") is False

    def test_mixed_modes_are_compatible(self) -> None:
        """Test that mixed modes are CPU compatible."""
        assert engine_mode_is_cpu_compatible(EngineMode.MIXED) is True
        assert engine_mode_is_cpu_compatible(EngineMode.DIVERSE) is True

    def test_unknown_mode_returns_false(self) -> None:
        """Test that unknown modes return False."""
        assert engine_mode_is_cpu_compatible("unknown-mode") is False


class TestNormalizeEngineMode:
    """Tests for normalize_engine_mode function."""

    def test_pass_through_valid_mode(self) -> None:
        """Test that valid modes pass through unchanged."""
        assert normalize_engine_mode("gumbel-mcts") == "gumbel-mcts"

    def test_normalize_alias(self) -> None:
        """Test that aliases are normalized."""
        assert normalize_engine_mode("gumbel") == "gumbel-mcts"
        assert normalize_engine_mode("heuristic") == "heuristic-only"

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert normalize_engine_mode("GUMBEL") == "gumbel-mcts"
        assert normalize_engine_mode("Heuristic") == "heuristic-only"

    def test_strips_whitespace(self) -> None:
        """Test whitespace stripping."""
        assert normalize_engine_mode("  gumbel  ") == "gumbel-mcts"


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_all_formats_exist(self) -> None:
        """Test that all expected formats are defined."""
        assert OutputFormat.JSONL == "jsonl"
        assert OutputFormat.DB == "db"
        assert OutputFormat.NPZ == "npz"

    def test_format_count(self) -> None:
        """Test total number of formats."""
        assert len(OutputFormat) == 3


class TestSelfplayConfig:
    """Tests for SelfplayConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are sensible."""
        config = SelfplayConfig()

        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.num_games == 1000
        assert config.engine_mode == EngineMode.NNUE_GUIDED
        assert config.search_depth == 3
        assert config.mcts_simulations == 800
        assert config.temperature == 1.0
        assert config.output_format == OutputFormat.DB
        assert config.use_gpu is True

    def test_custom_values(self) -> None:
        """Test that custom values override defaults."""
        config = SelfplayConfig(
            board_type="hexagonal",
            num_players=4,
            num_games=500,
            engine_mode=EngineMode.GUMBEL_MCTS,
        )

        assert config.board_type == "hexagonal"
        assert config.num_players == 4
        assert config.num_games == 500
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

    def test_board_type_normalization(self) -> None:
        """Test that board types are normalized."""
        config = SelfplayConfig(board_type="hex")
        assert config.board_type == "hexagonal"

        config = SelfplayConfig(board_type="sq8")
        assert config.board_type == "square8"

        config = SelfplayConfig(board_type="full_hex")
        assert config.board_type == "hexagonal"

    def test_engine_mode_string_conversion(self) -> None:
        """Test that string engine modes are converted to enum."""
        config = SelfplayConfig(engine_mode="gumbel-mcts")  # type: ignore
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

    def test_output_format_string_conversion(self) -> None:
        """Test that string output formats are converted to enum."""
        config = SelfplayConfig(output_format="npz")  # type: ignore
        assert config.output_format == OutputFormat.NPZ

    def test_config_key_property(self) -> None:
        """Test config_key property."""
        config = SelfplayConfig(board_type="hexagonal", num_players=4)
        assert config.config_key == "hexagonal_4p"

    def test_board_type_enum_property(self) -> None:
        """Test board_type_enum property."""
        from app.models import BoardType

        config = SelfplayConfig(board_type="hex8")
        assert config.board_type_enum == BoardType.HEX8

    def test_default_output_dir(self) -> None:
        """Test default output_dir is set based on config_key."""
        config = SelfplayConfig(board_type="hex8", num_players=2)
        assert config.output_dir == "data/selfplay/hex8_2p"

    def test_default_record_db(self) -> None:
        """Test default record_db is set for DB format."""
        config = SelfplayConfig(board_type="hex8", num_players=2)
        assert config.record_db == "data/games/hex8_2p.db"

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        config = SelfplayConfig(board_type="hex8", num_players=2)
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["board_type"] == "hex8"
        assert data["num_players"] == 2
        assert data["config_key"] == "hex8_2p"
        assert data["engine_mode"] == "nnue-guided"

    def test_from_dict(self) -> None:
        """Test from_dict deserialization."""
        data = {
            "board_type": "hexagonal",
            "num_players": 4,
            "num_games": 500,
            "engine_mode": "gumbel-mcts",
        }
        config = SelfplayConfig.from_dict(data)

        assert config.board_type == "hexagonal"
        assert config.num_players == 4
        assert config.num_games == 500

    def test_is_dataclass(self) -> None:
        """Test that config is a proper dataclass."""
        config_fields = fields(SelfplayConfig)
        assert len(config_fields) > 40  # Has many configuration options


class TestSelfplayConfigEffectiveBudget:
    """Tests for SelfplayConfig.get_effective_budget method."""

    def test_explicit_budget_has_priority(self) -> None:
        """Test that explicit simulation_budget takes priority."""
        config = SelfplayConfig(simulation_budget=999)
        assert config.get_effective_budget() == 999

    def test_elo_adaptive_budget(self) -> None:
        """Test Elo-adaptive budget calculation."""
        with patch("app.ai.gumbel_common.get_elo_adaptive_budget") as mock:
            mock.return_value = 225
            config = SelfplayConfig(model_elo=1450, training_epoch=50)
            budget = config.get_effective_budget()
            assert budget == 225
            mock.assert_called_once_with(1450, 50)

    def test_difficulty_based_budget(self) -> None:
        """Test difficulty-based budget calculation."""
        with patch("app.ai.gumbel_common.get_budget_for_difficulty") as mock:
            mock.return_value = 800
            config = SelfplayConfig(difficulty=8)
            budget = config.get_effective_budget()
            assert budget == 800
            mock.assert_called_once_with(8)

    def test_default_budget(self) -> None:
        """Test default budget when nothing is set."""
        # Clear any set values
        config = SelfplayConfig()
        config.simulation_budget = None
        config.model_elo = None
        config.difficulty = None

        with patch("app.ai.gumbel_common.GUMBEL_BUDGET_STANDARD", 150):
            budget = config.get_effective_budget()
            assert budget == 150


class TestCreateArgumentParser:
    """Tests for create_argument_parser function."""

    def test_returns_parser(self) -> None:
        """Test that function returns an ArgumentParser."""
        parser = create_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_has_required_arguments(self) -> None:
        """Test that parser has required arguments."""
        parser = create_argument_parser()
        # Parse with defaults
        args = parser.parse_args([])

        assert hasattr(args, "board")
        assert hasattr(args, "num_players")
        assert hasattr(args, "num_games")
        assert hasattr(args, "engine_mode")

    def test_custom_description(self) -> None:
        """Test custom description."""
        parser = create_argument_parser(description="Custom selfplay")
        assert "Custom selfplay" in parser.description

    def test_include_ramdrive_false(self) -> None:
        """Test ramdrive arguments can be excluded."""
        parser = create_argument_parser(include_ramdrive=False)
        args = parser.parse_args([])
        # Should not have ramdrive args
        assert not hasattr(args, "use_ramdrive") or args.use_ramdrive is None

    def test_include_gpu_false(self) -> None:
        """Test GPU arguments can be excluded."""
        parser = create_argument_parser(include_gpu=False)
        args = parser.parse_args([])
        # Should not have gpu args
        assert not hasattr(args, "no_gpu") or args.no_gpu is None


class TestParseSelfplayArgs:
    """Tests for parse_selfplay_args function."""

    def test_returns_config(self) -> None:
        """Test that function returns a SelfplayConfig."""
        config = parse_selfplay_args([])
        assert isinstance(config, SelfplayConfig)

    def test_parses_board_type(self) -> None:
        """Test parsing board type."""
        config = parse_selfplay_args(["--board", "hexagonal"])
        assert config.board_type == "hexagonal"

    def test_parses_num_players(self) -> None:
        """Test parsing num_players."""
        config = parse_selfplay_args(["--num-players", "4"])
        assert config.num_players == 4

    def test_parses_engine_mode(self) -> None:
        """Test parsing engine mode."""
        config = parse_selfplay_args(["--engine-mode", "gumbel-mcts"])
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

    def test_parses_no_gpu(self) -> None:
        """Test parsing --no-gpu flag."""
        config = parse_selfplay_args(["--no-gpu"])
        assert config.use_gpu is False

    def test_parses_mixed_opponents(self) -> None:
        """Test parsing --mixed-opponents flag."""
        config = parse_selfplay_args(["--mixed-opponents"])
        assert config.mixed_opponents is True
        assert config.engine_mode == EngineMode.MIXED

    def test_parses_opponent_mix(self) -> None:
        """Test parsing --opponent-mix."""
        config = parse_selfplay_args([
            "--mixed-opponents",
            "--opponent-mix", "random:0.2,heuristic:0.5,mcts:0.3"
        ])
        assert config.opponent_mix == {"random": 0.2, "heuristic": 0.5, "mcts": 0.3}

    def test_parses_disable_pfsp(self) -> None:
        """Test parsing --disable-pfsp flag."""
        config = parse_selfplay_args(["--disable-pfsp"])
        assert config.use_pfsp is False

    def test_default_pfsp_enabled(self) -> None:
        """Test that PFSP is enabled by default."""
        config = parse_selfplay_args([])
        assert config.use_pfsp is True


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_config(self) -> None:
        """Test that function returns a SelfplayConfig."""
        config = get_default_config()
        assert isinstance(config, SelfplayConfig)

    def test_uses_provided_board_type(self) -> None:
        """Test that provided board_type is used."""
        config = get_default_config(board_type="hexagonal")
        assert config.board_type == "hexagonal"

    def test_uses_provided_num_players(self) -> None:
        """Test that provided num_players is used."""
        config = get_default_config(num_players=4)
        assert config.num_players == 4


class TestGetProductionConfig:
    """Tests for get_production_config function."""

    def test_returns_config(self) -> None:
        """Test that function returns a SelfplayConfig."""
        config = get_production_config("hex8", 2)
        assert isinstance(config, SelfplayConfig)

    def test_has_high_game_count(self) -> None:
        """Test that production config has high game count."""
        config = get_production_config("hex8", 2)
        assert config.num_games == 100000

    def test_uses_diverse_engine(self) -> None:
        """Test that production config uses DIVERSE engine."""
        config = get_production_config("hex8", 2)
        assert config.engine_mode == EngineMode.DIVERSE

    def test_has_high_mcts_simulations(self) -> None:
        """Test that production config has high MCTS simulations."""
        config = get_production_config("hex8", 2)
        assert config.mcts_simulations == 1600


class TestCurriculumStage:
    """Tests for CurriculumStage dataclass."""

    def test_create_stage(self) -> None:
        """Test creating a curriculum stage."""
        stage = CurriculumStage(
            name="test_stage",
            engine_mode=EngineMode.MCTS,
            temperature=1.0,
            mcts_simulations=400,
            search_depth=2,
            games_per_config=100,
        )

        assert stage.name == "test_stage"
        assert stage.engine_mode == EngineMode.MCTS
        assert stage.temperature == 1.0

    def test_default_random_opening_moves(self) -> None:
        """Test default random_opening_moves is 0."""
        stage = CurriculumStage(
            name="test",
            engine_mode=EngineMode.MCTS,
            temperature=1.0,
            mcts_simulations=400,
            search_depth=2,
            games_per_config=100,
        )
        assert stage.random_opening_moves == 0


class TestCurriculumStages:
    """Tests for CURRICULUM_STAGES dict."""

    def test_has_expected_stages(self) -> None:
        """Test that expected stages are defined."""
        expected = [
            "explore_random",
            "explore_weak",
            "moderate_mcts",
            "moderate_nnue",
            "strong_gumbel",
            "strong_full",
            "experimental_gmo",
            "robust_diverse",
        ]
        for stage_name in expected:
            assert stage_name in CURRICULUM_STAGES

    def test_all_stages_are_curriculum_stage(self) -> None:
        """Test that all stages are CurriculumStage instances."""
        for name, stage in CURRICULUM_STAGES.items():
            assert isinstance(stage, CurriculumStage), f"Stage '{name}' is not CurriculumStage"

    def test_stages_have_valid_engine_modes(self) -> None:
        """Test that all stages have valid engine modes."""
        for name, stage in CURRICULUM_STAGES.items():
            assert isinstance(
                stage.engine_mode, EngineMode
            ), f"Stage '{name}' has invalid engine_mode"

    def test_stages_have_descriptions(self) -> None:
        """Test that all stages have descriptions."""
        for name, stage in CURRICULUM_STAGES.items():
            assert stage.description, f"Stage '{name}' has no description"


class TestGetCurriculumConfig:
    """Tests for get_curriculum_config function."""

    def test_with_stage_name(self) -> None:
        """Test with stage name string."""
        config = get_curriculum_config("moderate_mcts", "hex8", 2)

        assert isinstance(config, SelfplayConfig)
        assert config.board_type == "hex8"
        assert config.num_players == 2
        assert config.engine_mode == EngineMode.MCTS

    def test_with_stage_object(self) -> None:
        """Test with CurriculumStage object."""
        stage = CURRICULUM_STAGES["strong_gumbel"]
        config = get_curriculum_config(stage, "hexagonal", 4)

        assert config.board_type == "hexagonal"
        assert config.num_players == 4
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

    def test_unknown_stage_raises(self) -> None:
        """Test that unknown stage name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown curriculum stage"):
            get_curriculum_config("nonexistent_stage")

    def test_copies_stage_settings(self) -> None:
        """Test that stage settings are copied to config."""
        stage = CURRICULUM_STAGES["explore_random"]
        config = get_curriculum_config(stage)

        assert config.temperature == stage.temperature
        assert config.mcts_simulations == stage.mcts_simulations
        assert config.search_depth == stage.search_depth
        assert config.random_opening_moves == stage.random_opening_moves


class TestGetFullCurriculum:
    """Tests for get_full_curriculum function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        configs = get_full_curriculum()
        assert isinstance(configs, list)

    def test_default_stages(self) -> None:
        """Test default stage progression."""
        configs = get_full_curriculum()
        # Default is 6 stages
        assert len(configs) == 6

    def test_custom_stages(self) -> None:
        """Test custom stage selection."""
        configs = get_full_curriculum(stages=["explore_random", "strong_full"])
        assert len(configs) == 2

    def test_all_configs_have_same_board(self) -> None:
        """Test that all configs have same board type."""
        configs = get_full_curriculum(board_type="hexagonal")
        for config in configs:
            assert config.board_type == "hexagonal"

    def test_all_configs_have_same_players(self) -> None:
        """Test that all configs have same player count."""
        configs = get_full_curriculum(num_players=4)
        for config in configs:
            assert config.num_players == 4


class TestGetAllConfigsCurriculum:
    """Tests for get_all_configs_curriculum function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        configs = get_all_configs_curriculum(stages=["explore_random"])
        assert isinstance(configs, list)

    def test_covers_all_board_types(self) -> None:
        """Test that all board types are covered."""
        configs = get_all_configs_curriculum(stages=["moderate_mcts"])
        board_types = {c.board_type for c in configs}
        assert "square8" in board_types
        assert "square19" in board_types
        assert "hexagonal" in board_types

    def test_covers_all_player_counts(self) -> None:
        """Test that all player counts are covered."""
        configs = get_all_configs_curriculum(stages=["moderate_mcts"])
        player_counts = {c.num_players for c in configs}
        assert 2 in player_counts
        assert 3 in player_counts
        assert 4 in player_counts

    def test_total_configs_for_single_stage(self) -> None:
        """Test total configs for single stage (9 = 3 boards x 3 player counts)."""
        configs = get_all_configs_curriculum(stages=["moderate_mcts"])
        assert len(configs) == 9


class TestListCurriculumStages:
    """Tests for list_curriculum_stages function."""

    def test_returns_dict(self) -> None:
        """Test that function returns a dict."""
        stages = list_curriculum_stages()
        assert isinstance(stages, dict)

    def test_has_all_stages(self) -> None:
        """Test that all stages are listed."""
        stages = list_curriculum_stages()
        assert len(stages) == len(CURRICULUM_STAGES)

    def test_values_are_descriptions(self) -> None:
        """Test that values are stage descriptions."""
        stages = list_curriculum_stages()
        for name, description in stages.items():
            assert description == CURRICULUM_STAGES[name].description


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_opponent_mix_string(self) -> None:
        """Test parsing empty opponent mix."""
        # Should not crash on empty string
        config = parse_selfplay_args(["--mixed-opponents"])
        assert config.opponent_mix is None

    def test_invalid_opponent_mix_format(self) -> None:
        """Test parsing invalid opponent mix format."""
        # Should warn but not crash
        config = parse_selfplay_args([
            "--mixed-opponents",
            "--opponent-mix", "invalid-format"
        ])
        assert config.opponent_mix is None

    def test_board_type_case_insensitive(self) -> None:
        """Test that board type normalization is case insensitive."""
        config = SelfplayConfig(board_type="HEX")
        assert config.board_type == "hexagonal"

    def test_unknown_board_type_passes_through(self) -> None:
        """Test that unknown board type passes through."""
        config = SelfplayConfig(board_type="custom_board")
        assert config.board_type == "custom_board"
