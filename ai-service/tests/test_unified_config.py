"""Tests for unified_config module."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import os

from app.config.unified_config import (
    UnifiedConfig,
    TrainingConfig,
    EvaluationConfig,
    PromotionConfig,
    CurriculumConfig,
    SafeguardsConfig,
    SafetyConfig,
    TournamentConfig,
    SelfplayConfig,
    get_config,
    get_training_threshold,
    get_min_elo_improvement,
)


class TestUnifiedConfigValidation:
    """Tests for config validation."""

    def test_valid_config_returns_empty_errors(self):
        """Valid default config should return no errors."""
        config = UnifiedConfig()
        errors = config.validate()
        assert errors == []

    def test_training_threshold_too_low(self):
        """Training threshold below 10 should fail."""
        config = UnifiedConfig()
        config.training.trigger_threshold_games = 5
        errors = config.validate()
        assert any("trigger_threshold_games=5 too low" in e for e in errors)

    def test_training_threshold_too_high(self):
        """Training threshold above 100000 should fail."""
        config = UnifiedConfig()
        config.training.trigger_threshold_games = 200000
        errors = config.validate()
        assert any("trigger_threshold_games=200000 too high" in e for e in errors)

    def test_training_min_interval_too_low(self):
        """Min interval below 60 seconds should fail."""
        config = UnifiedConfig()
        config.training.min_interval_seconds = 30
        errors = config.validate()
        assert any("min_interval_seconds=30 too low" in e for e in errors)

    def test_validation_split_negative(self):
        """Negative validation split should fail."""
        config = UnifiedConfig()
        config.training.validation_split = -0.1
        errors = config.validate()
        assert any("validation_split=-0.1 out of range" in e for e in errors)

    def test_validation_split_too_high(self):
        """Validation split above 0.5 should fail."""
        config = UnifiedConfig()
        config.training.validation_split = 0.6
        errors = config.validate()
        assert any("validation_split=0.6 out of range" in e for e in errors)

    def test_shadow_games_too_low(self):
        """Shadow games below 1 should fail."""
        config = UnifiedConfig()
        config.evaluation.shadow_games_per_config = 0
        errors = config.validate()
        assert any("shadow_games_per_config=0 too low" in e for e in errors)

    def test_min_games_for_elo_too_low(self):
        """Min games for elo below 1 should fail."""
        config = UnifiedConfig()
        config.evaluation.min_games_for_elo = 0
        errors = config.validate()
        assert any("min_games_for_elo=0 too low" in e for e in errors)

    def test_elo_k_factor_out_of_range(self):
        """K factor outside 1-100 should fail."""
        config = UnifiedConfig()
        config.evaluation.elo_k_factor = 0
        errors = config.validate()
        assert any("elo_k_factor=0 out of range" in e for e in errors)

        config.evaluation.elo_k_factor = 150
        errors = config.validate()
        assert any("elo_k_factor=150 out of range" in e for e in errors)

    def test_elo_threshold_negative(self):
        """Negative elo threshold should fail."""
        config = UnifiedConfig()
        config.promotion.elo_threshold = -10
        errors = config.validate()
        assert any("elo_threshold=-10 cannot be negative" in e for e in errors)

    def test_significance_level_out_of_range(self):
        """Significance level outside 0.001-0.5 should fail."""
        config = UnifiedConfig()
        config.promotion.significance_level = 0.0001
        errors = config.validate()
        assert any("significance_level=0.0001 out of range" in e for e in errors)

    def test_curriculum_weight_multipliers(self):
        """Weight multiplier constraints should be enforced."""
        config = UnifiedConfig()
        config.curriculum.max_weight_multiplier = 0.5
        errors = config.validate()
        assert any("max_weight_multiplier=0.5 must be >= 1.0" in e for e in errors)

        config = UnifiedConfig()
        config.curriculum.min_weight_multiplier = 1.5
        errors = config.validate()
        assert any("min_weight_multiplier=1.5 must be <= 1.0" in e for e in errors)

    def test_curriculum_min_greater_than_max(self):
        """Min weight multiplier > max should fail."""
        config = UnifiedConfig()
        config.curriculum.min_weight_multiplier = 0.9
        config.curriculum.max_weight_multiplier = 0.8
        errors = config.validate()
        assert any("min_weight_multiplier > max_weight_multiplier" in e for e in errors)

    def test_max_python_processes_too_low(self):
        """Max python processes below 1 should fail."""
        config = UnifiedConfig()
        config.safeguards.max_python_processes_per_host = 0
        errors = config.validate()
        assert any("max_python_processes_per_host=0 too low" in e for e in errors)

    def test_max_process_age_too_low(self):
        """Max process age below 0.5 hours should fail."""
        config = UnifiedConfig()
        config.safeguards.max_process_age_hours = 0.2
        errors = config.validate()
        assert any("max_process_age_hours=0.2 too low" in e for e in errors)

    def test_overfit_threshold_out_of_range(self):
        """Overfit threshold outside 0-1 should fail."""
        config = UnifiedConfig()
        config.safety.overfit_threshold = 1.5
        errors = config.validate()
        assert any("overfit_threshold=1.5 out of range" in e for e in errors)

    def test_data_quality_score_out_of_range(self):
        """Data quality score outside 0-1 should fail."""
        config = UnifiedConfig()
        config.safety.data_quality_score_min = -0.1
        errors = config.validate()
        assert any("data_quality_score_min=-0.1 out of range" in e for e in errors)

    def test_tournament_k_factor_out_of_range(self):
        """Tournament k factor outside 1-100 should fail."""
        config = UnifiedConfig()
        config.tournament.k_factor = 200
        errors = config.validate()
        assert any("tournament.k_factor=200 out of range" in e for e in errors)

    def test_tournament_initial_elo_negative(self):
        """Negative initial elo should fail."""
        config = UnifiedConfig()
        config.tournament.initial_elo = -100
        errors = config.validate()
        assert any("initial_elo=-100 cannot be negative" in e for e in errors)

    def test_selfplay_mcts_simulations_too_low(self):
        """MCTS simulations below 1 should fail."""
        config = UnifiedConfig()
        config.selfplay.mcts_simulations = 0
        errors = config.validate()
        assert any("mcts_simulations=0 too low" in e for e in errors)

    def test_selfplay_temperature_negative(self):
        """Negative temperature should fail."""
        config = UnifiedConfig()
        config.selfplay.temperature = -0.5
        errors = config.validate()
        assert any("temperature=-0.5 cannot be negative" in e for e in errors)

    def test_validate_or_raise_with_valid_config(self):
        """validate_or_raise should not raise for valid config."""
        config = UnifiedConfig()
        config.validate_or_raise()  # Should not raise

    def test_validate_or_raise_with_invalid_config(self):
        """validate_or_raise should raise ValueError for invalid config."""
        config = UnifiedConfig()
        config.training.trigger_threshold_games = 5
        with pytest.raises(ValueError) as exc_info:
            config.validate_or_raise()
        assert "Config validation failed" in str(exc_info.value)
        assert "trigger_threshold_games=5" in str(exc_info.value)

    def test_multiple_errors_collected(self):
        """Multiple validation errors should all be collected."""
        config = UnifiedConfig()
        config.training.trigger_threshold_games = 5
        config.training.validation_split = 0.8
        config.promotion.elo_threshold = -10
        errors = config.validate()
        assert len(errors) >= 3


class TestUnifiedConfigFromYaml:
    """Tests for loading config from YAML."""

    def test_from_yaml_with_valid_file(self):
        """Loading from valid YAML should work."""
        yaml_content = """
training:
  trigger_threshold_games: 500
  min_interval_seconds: 300
evaluation:
  shadow_games_per_config: 20
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = UnifiedConfig.from_yaml(f.name)
            assert config.training.trigger_threshold_games == 500
            assert config.training.min_interval_seconds == 300
            assert config.evaluation.shadow_games_per_config == 20
            os.unlink(f.name)

    def test_from_yaml_with_missing_file(self):
        """Loading from missing file should return defaults."""
        config = UnifiedConfig.from_yaml("/nonexistent/path/config.yaml")
        # Should return a valid config (defaults are applied)
        assert config.training.trigger_threshold_games >= 10  # Must be valid per validation


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_training_threshold(self):
        """get_training_threshold should return configured value."""
        threshold = get_training_threshold()
        assert isinstance(threshold, int)
        assert threshold >= 10

    def test_get_min_elo_improvement(self):
        """get_min_elo_improvement should return configured value."""
        improvement = get_min_elo_improvement()
        assert isinstance(improvement, float)
        assert improvement >= 0


class TestEnvOverrides:
    """Tests for environment variable overrides."""

    def test_training_threshold_env_override(self):
        """RINGRIFT_TRAINING_THRESHOLD should override config."""
        config = UnifiedConfig()
        config.training.trigger_threshold_games = 1000

        with patch.dict(os.environ, {"RINGRIFT_TRAINING_THRESHOLD": "2000"}):
            config.apply_env_overrides()
            assert config.training.trigger_threshold_games == 2000

    def test_legacy_env_var_override(self):
        """RINGRIFT_MIN_GAMES_FOR_TRAINING should override config."""
        config = UnifiedConfig()
        config.training.trigger_threshold_games = 1000

        with patch.dict(os.environ, {"RINGRIFT_MIN_GAMES_FOR_TRAINING": "3000"}):
            config.apply_env_overrides()
            assert config.training.trigger_threshold_games == 3000

    def test_elo_db_env_override(self):
        """RINGRIFT_ELO_DB should override config."""
        config = UnifiedConfig()
        config.elo_db = "data/elo.db"

        with patch.dict(os.environ, {"RINGRIFT_ELO_DB": "/custom/path/elo.db"}):
            config.apply_env_overrides()
            assert config.elo_db == "/custom/path/elo.db"
