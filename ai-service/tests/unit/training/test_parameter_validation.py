"""Tests for parameter_validation.py - Training parameter validation.

Tests validation logic for model/dataset compatibility, value head validation,
and architecture-data compatibility.
"""

from __future__ import annotations

from unittest import mock

import pytest

from app.training.parameter_validation import (
    ValidationResult,
    validate_architecture_data_compatibility,
    validate_board_type_compatibility,
    validate_checkpoint_compatibility,
    validate_model_value_head,
    validate_policy_size_compatibility,
    validate_sample_data,
    validate_training_compatibility,
)


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_init_default_valid(self):
        """Test result is valid by default."""
        result = ValidationResult()

        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_makes_invalid(self):
        """Test add_error marks result as invalid."""
        result = ValidationResult()

        result.add_error("Test error")

        assert result.valid is False
        assert "Test error" in result.errors

    def test_add_warning_keeps_valid(self):
        """Test add_warning doesn't affect validity."""
        result = ValidationResult()

        result.add_warning("Test warning")

        assert result.valid is True
        assert "Test warning" in result.warnings

    def test_raise_if_invalid_raises(self):
        """Test raise_if_invalid raises for invalid result."""
        result = ValidationResult()
        result.add_error("Error 1")
        result.add_error("Error 2")

        with pytest.raises(ValueError, match="Error 1"):
            result.raise_if_invalid()

    def test_raise_if_invalid_passes(self):
        """Test raise_if_invalid passes for valid result."""
        result = ValidationResult()
        result.add_warning("Just a warning")

        # Should not raise
        result.raise_if_invalid()


# =============================================================================
# Policy Size Validation Tests
# =============================================================================


class TestValidatePolicySizeCompatibility:
    """Tests for validate_policy_size_compatibility."""

    def test_none_values_return_valid(self):
        """Test None values return valid result."""
        result = validate_policy_size_compatibility(None, None)
        assert result.valid is True

        result = validate_policy_size_compatibility(1000, None)
        assert result.valid is True

        result = validate_policy_size_compatibility(None, 1000)
        assert result.valid is True

    def test_equal_sizes_valid(self):
        """Test equal sizes are valid."""
        result = validate_policy_size_compatibility(1000, 1000)

        assert result.valid is True
        assert len(result.warnings) == 0

    def test_dataset_larger_than_model_invalid(self):
        """Test dataset larger than model is invalid."""
        result = validate_policy_size_compatibility(
            model_policy_size=1000,
            dataset_policy_size=2000,
        )

        assert result.valid is False
        assert "Dataset policy_size (2000) > model policy_size (1000)" in result.errors[0]

    def test_dataset_smaller_than_model_warning(self):
        """Test dataset smaller than model gives warning."""
        result = validate_policy_size_compatibility(
            model_policy_size=2000,
            dataset_policy_size=1000,
        )

        assert result.valid is True
        assert len(result.warnings) == 1
        assert "zero-padded" in result.warnings[0]


# =============================================================================
# Board Type Validation Tests
# =============================================================================


class TestValidateBoardTypeCompatibility:
    """Tests for validate_board_type_compatibility."""

    def test_none_values_return_valid(self):
        """Test None values return valid result."""
        result = validate_board_type_compatibility(None, None)
        assert result.valid is True

    def test_matching_types_valid(self):
        """Test matching board types are valid."""
        result = validate_board_type_compatibility("hex8", "hex8")

        assert result.valid is True

    def test_mismatched_types_invalid(self):
        """Test mismatched board types are invalid."""
        result = validate_board_type_compatibility("hex8", "square8")

        assert result.valid is False
        assert "CROSS-CONFIG CONTAMINATION" in result.errors[0]


# =============================================================================
# Sample Data Validation Tests
# =============================================================================


class TestValidateSampleData:
    """Tests for validate_sample_data."""

    def test_empty_dataset(self):
        """Test empty dataset validation."""
        mock_dataset = mock.MagicMock()
        mock_dataset.__len__ = mock.MagicMock(return_value=0)

        result = validate_sample_data(mock_dataset)

        assert result.valid is True

    def test_valid_samples(self):
        """Test validation of valid samples."""
        mock_policy = mock.MagicMock()
        mock_policy.sum.return_value.item.return_value = 1.0

        mock_dataset = mock.MagicMock()
        mock_dataset.__len__ = mock.MagicMock(return_value=5)
        mock_dataset.__getitem__ = mock.MagicMock(
            return_value=(None, None, None, mock_policy)
        )

        result = validate_sample_data(mock_dataset, num_samples=3)

        assert result.valid is True

    def test_invalid_policy_sum(self):
        """Test validation catches invalid policy sum."""
        mock_policy = mock.MagicMock()
        mock_policy.sum.return_value.item.return_value = 5.0  # Invalid

        mock_dataset = mock.MagicMock()
        mock_dataset.__len__ = mock.MagicMock(return_value=5)
        mock_dataset.__getitem__ = mock.MagicMock(
            return_value=(None, None, None, mock_policy)
        )

        result = validate_sample_data(mock_dataset, num_samples=3)

        # Result is still valid, just has warnings
        assert result.valid is True
        assert len(result.warnings) > 0


# =============================================================================
# Training Compatibility Validation Tests
# =============================================================================


class TestValidateTrainingCompatibility:
    """Tests for validate_training_compatibility."""

    def test_compatible_model_and_dataset(self):
        """Test compatible model and dataset pass validation."""
        mock_model = mock.MagicMock()
        mock_model.policy_size = 1000
        mock_model.board_type = "hex8"

        mock_dataset = mock.MagicMock()
        mock_dataset.policy_size = 1000
        mock_dataset.board_type = "hex8"
        mock_dataset.__len__ = mock.MagicMock(return_value=100)

        mock_config = mock.MagicMock()

        # Should not raise
        validate_training_compatibility(mock_model, mock_dataset, mock_config)

    def test_board_type_mismatch_raises(self):
        """Test board type mismatch raises ValueError."""
        mock_model = mock.MagicMock()
        mock_model.policy_size = 1000
        mock_model.board_type = "hex8"

        mock_dataset = mock.MagicMock()
        mock_dataset.policy_size = 1000
        mock_dataset.board_type = "square8"
        mock_dataset.__len__ = mock.MagicMock(return_value=100)

        mock_config = mock.MagicMock()

        with pytest.raises(ValueError, match="CROSS-CONFIG CONTAMINATION"):
            validate_training_compatibility(mock_model, mock_dataset, mock_config)


# =============================================================================
# Model Value Head Validation Tests
# =============================================================================


class TestValidateModelValueHead:
    """Tests for validate_model_value_head."""

    def test_matching_num_players_passes(self):
        """Test matching num_players attribute passes."""
        mock_model = mock.MagicMock()
        mock_model.num_players = 4

        # Should not raise
        validate_model_value_head(mock_model, expected_players=4)

    def test_mismatched_num_players_raises(self):
        """Test mismatched num_players raises ValueError."""
        mock_model = mock.MagicMock()
        mock_model.num_players = 2

        with pytest.raises(ValueError, match="model.num_players=2"):
            validate_model_value_head(mock_model, expected_players=4)

    def test_value_fc2_mismatch_raises(self):
        """Test value_fc2 output mismatch raises."""
        mock_model = mock.MagicMock()
        del mock_model.num_players  # Remove num_players attribute
        mock_model.value_fc2.out_features = 2
        del mock_model.value_fc3  # Remove value_fc3

        with pytest.raises(ValueError, match="value_fc2 output mismatch"):
            validate_model_value_head(mock_model, expected_players=4)

    def test_context_included_in_error(self):
        """Test context is included in error message."""
        mock_model = mock.MagicMock()
        mock_model.num_players = 2

        with pytest.raises(ValueError, match="\\(after checkpoint load\\)"):
            validate_model_value_head(
                mock_model,
                expected_players=4,
                context="after checkpoint load",
            )


# =============================================================================
# Architecture Data Compatibility Tests
# =============================================================================


class TestValidateArchitectureDataCompatibility:
    """Tests for validate_architecture_data_compatibility."""

    def test_non_heuristic_version_skips(self):
        """Test non-heuristic versions skip validation."""
        # Should not raise for v2 even with no heuristics
        validate_architecture_data_compatibility(
            model_version="v2",
            detected_num_heuristics=0,
            board_type="hex8",
        )

    def test_missing_encoder_config_skips(self):
        """Test missing encoder config skips validation gracefully."""
        # Mock the import to fail
        with mock.patch(
            "app.training.parameter_validation.get_encoder_config",
            side_effect=ValueError("Not found"),
        ):
            # Should not raise
            validate_architecture_data_compatibility(
                model_version="v5-heavy",
                detected_num_heuristics=5,
                board_type="unknown_board",
            )


# =============================================================================
# Checkpoint Compatibility Tests
# =============================================================================


class TestValidateCheckpointCompatibility:
    """Tests for validate_checkpoint_compatibility."""

    def test_missing_model_state_dict(self):
        """Test missing model_state_dict returns error."""
        checkpoint = {"other_key": "value"}
        mock_model = mock.MagicMock()
        mock_model.state_dict.return_value = {}

        result = validate_checkpoint_compatibility(checkpoint, mock_model)

        assert result.valid is False
        assert "missing 'model_state_dict'" in result.errors[0]

    def test_exact_match_valid(self):
        """Test exact key match is valid."""
        state = {"layer1.weight": None, "layer2.bias": None}
        checkpoint = {"model_state_dict": state}

        mock_model = mock.MagicMock()
        mock_model.state_dict.return_value = state

        result = validate_checkpoint_compatibility(checkpoint, mock_model, strict=True)

        assert result.valid is True

    def test_missing_keys_strict_invalid(self):
        """Test missing keys in strict mode are invalid."""
        checkpoint = {"model_state_dict": {"layer1.weight": None}}

        mock_model = mock.MagicMock()
        mock_model.state_dict.return_value = {
            "layer1.weight": None,
            "layer2.bias": None,
        }

        result = validate_checkpoint_compatibility(checkpoint, mock_model, strict=True)

        assert result.valid is False
        assert "Missing keys" in result.errors[0]

    def test_missing_keys_non_strict_warning(self):
        """Test missing keys in non-strict mode are warnings."""
        checkpoint = {"model_state_dict": {"layer1.weight": None}}

        mock_model = mock.MagicMock()
        mock_model.state_dict.return_value = {
            "layer1.weight": None,
            "layer2.bias": None,
        }

        result = validate_checkpoint_compatibility(checkpoint, mock_model, strict=False)

        assert result.valid is True
        assert len(result.warnings) > 0
