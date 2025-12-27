"""Tests for app.config.hyperparameters module.

Tests hyperparameter loading and management utilities.
"""

import pytest
from unittest.mock import patch, mock_open

from app.config.hyperparameters import (
    DEFAULT_HYPERPARAMETERS,
    get_config_key,
    get_hyperparameters,
    reload_config,
)


class TestDefaultHyperparameters:
    """Tests for default hyperparameter values."""

    def test_learning_rate_valid(self):
        """Test learning rate is valid."""
        assert 0 < DEFAULT_HYPERPARAMETERS["learning_rate"] < 1
        assert DEFAULT_HYPERPARAMETERS["learning_rate"] == 0.0003

    def test_batch_size_power_of_two(self):
        """Test batch size is power of two."""
        batch_size = DEFAULT_HYPERPARAMETERS["batch_size"]
        assert batch_size > 0
        assert (batch_size & (batch_size - 1)) == 0  # Power of 2

    def test_hidden_dim_positive(self):
        """Test hidden dimension is positive."""
        assert DEFAULT_HYPERPARAMETERS["hidden_dim"] > 0
        assert DEFAULT_HYPERPARAMETERS["hidden_dim"] == 256

    def test_num_hidden_layers_positive(self):
        """Test number of hidden layers is positive."""
        assert DEFAULT_HYPERPARAMETERS["num_hidden_layers"] >= 1

    def test_weight_decay_non_negative(self):
        """Test weight decay is non-negative."""
        assert DEFAULT_HYPERPARAMETERS["weight_decay"] >= 0
        assert DEFAULT_HYPERPARAMETERS["weight_decay"] < 1

    def test_dropout_valid_range(self):
        """Test dropout is in valid range [0, 1)."""
        assert 0 <= DEFAULT_HYPERPARAMETERS["dropout"] < 1
        assert DEFAULT_HYPERPARAMETERS["dropout"] == 0.1

    def test_epochs_positive(self):
        """Test epochs is positive."""
        assert DEFAULT_HYPERPARAMETERS["epochs"] > 0
        assert DEFAULT_HYPERPARAMETERS["epochs"] == 50

    def test_early_stopping_patience_positive(self):
        """Test early stopping patience is positive."""
        assert DEFAULT_HYPERPARAMETERS["early_stopping_patience"] > 0

    def test_warmup_epochs_less_than_total(self):
        """Test warmup epochs is less than total epochs."""
        assert DEFAULT_HYPERPARAMETERS["warmup_epochs"] < DEFAULT_HYPERPARAMETERS["epochs"]

    def test_value_weight_positive(self):
        """Test value weight is positive."""
        assert DEFAULT_HYPERPARAMETERS["value_weight"] > 0

    def test_policy_weight_positive(self):
        """Test policy weight is positive."""
        assert DEFAULT_HYPERPARAMETERS["policy_weight"] > 0


class TestConfigKey:
    """Tests for config key generation."""

    def test_square8_2p(self):
        """Test config key for square8 2p."""
        assert get_config_key("square8", 2) == "square8_2p"

    def test_square19_4p(self):
        """Test config key for square19 4p."""
        assert get_config_key("square19", 4) == "square19_4p"

    def test_hex8_3p(self):
        """Test config key for hex8 3p."""
        assert get_config_key("hex8", 3) == "hex8_3p"

    def test_hexagonal_2p(self):
        """Test config key for hexagonal 2p."""
        assert get_config_key("hexagonal", 2) == "hexagonal_2p"


class TestGetHyperparameters:
    """Tests for get_hyperparameters function."""

    def test_returns_dict(self):
        """Test that get_hyperparameters returns a dictionary."""
        params = get_hyperparameters("square8", 2)
        assert isinstance(params, dict)

    def test_has_required_keys(self):
        """Test that returned params have required keys."""
        params = get_hyperparameters("square8", 2)
        required_keys = ["learning_rate", "batch_size", "hidden_dim", "epochs"]
        for key in required_keys:
            assert key in params, f"Missing required key: {key}"

    def test_learning_rate_valid(self):
        """Test learning rate is valid in returned params."""
        params = get_hyperparameters("square8", 2)
        assert 0 < params["learning_rate"] < 1

    def test_different_configs_may_differ(self):
        """Test that different configs may have different params."""
        params_sq8 = get_hyperparameters("square8", 2)
        params_sq19 = get_hyperparameters("square19", 2)
        # Both should have valid params (may or may not be same)
        assert params_sq8["learning_rate"] > 0
        assert params_sq19["learning_rate"] > 0


class TestReloadConfig:
    """Tests for reload_config function."""

    def test_reload_does_not_raise(self):
        """Test that reload_config doesn't raise exceptions."""
        # Should not raise even if config file doesn't exist
        reload_config()

    def test_reload_clears_cache(self):
        """Test that reload clears the cache."""
        # Get params to populate cache
        params1 = get_hyperparameters("square8", 2)

        # Reload to clear cache
        reload_config()

        # Get params again - should work
        params2 = get_hyperparameters("square8", 2)

        # Both should be valid
        assert params1["learning_rate"] > 0
        assert params2["learning_rate"] > 0


class TestHyperparameterConsistency:
    """Tests for hyperparameter consistency across configs."""

    def test_all_configs_have_learning_rate(self):
        """Test all configs have learning rate."""
        configs = [
            ("square8", 2), ("square8", 3), ("square8", 4),
            ("square19", 2), ("hex8", 2),
        ]
        for board_type, num_players in configs:
            params = get_hyperparameters(board_type, num_players)
            assert "learning_rate" in params

    def test_all_configs_have_batch_size(self):
        """Test all configs have batch size."""
        configs = [
            ("square8", 2), ("square19", 2), ("hex8", 2),
        ]
        for board_type, num_players in configs:
            params = get_hyperparameters(board_type, num_players)
            assert "batch_size" in params
            assert params["batch_size"] > 0

    def test_learning_rates_reasonable_range(self):
        """Test learning rates are in reasonable range."""
        configs = [
            ("square8", 2), ("square19", 2), ("hex8", 2),
        ]
        for board_type, num_players in configs:
            params = get_hyperparameters(board_type, num_players)
            lr = params["learning_rate"]
            # Reasonable range for neural network training
            assert 1e-6 <= lr <= 0.1, f"LR {lr} out of range for {board_type}_{num_players}p"
