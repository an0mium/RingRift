"""Tests for app/coordination/npz_validation.py.

This module provides deep validation of NPZ training data files beyond checksum
verification. Tests cover validation of file structure, array shapes, data types,
and detection of corruption.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.coordination.npz_validation import (
    EXPECTED_DTYPES,
    MAX_REASONABLE_DIMENSION,
    MAX_REASONABLE_SAMPLES,
    NPZValidationResult,
    POLICY_PREFIXES,
    REQUIRED_ARRAYS,
    _get_expected_cells,
    quick_npz_check,
    validate_npz_for_training,
    validate_npz_structure,
)


# =============================================================================
# NPZValidationResult Tests
# =============================================================================
class TestNPZValidationResult:
    """Tests for NPZValidationResult dataclass."""

    def test_default_values(self):
        """Test default initialization values."""
        result = NPZValidationResult(valid=False)

        assert result.valid is False
        assert result.sample_count == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.array_shapes == {}
        assert result.array_dtypes == {}
        assert result.file_size == 0

    def test_custom_values(self):
        """Test initialization with custom values."""
        result = NPZValidationResult(
            valid=True,
            sample_count=1000,
            errors=["error1"],
            warnings=["warning1"],
            array_shapes={"features": (1000, 64, 32)},
            array_dtypes={"features": "float32"},
            file_size=1024 * 1024,
        )

        assert result.valid is True
        assert result.sample_count == 1000
        assert result.errors == ["error1"]
        assert result.warnings == ["warning1"]
        assert result.array_shapes == {"features": (1000, 64, 32)}
        assert result.array_dtypes == {"features": "float32"}
        assert result.file_size == 1024 * 1024

    def test_summary_valid(self):
        """Test summary method for valid result."""
        result = NPZValidationResult(
            valid=True,
            sample_count=5000,
            array_shapes={"features": (5000, 64), "values": (5000, 2)},
            file_size=10 * 1024 * 1024,  # 10MB
        )

        summary = result.summary()

        assert "Valid NPZ" in summary
        assert "5000 samples" in summary
        assert "2 arrays" in summary
        assert "10.0MB" in summary

    def test_summary_invalid(self):
        """Test summary method for invalid result."""
        result = NPZValidationResult(
            valid=False,
            errors=["Missing features array", "Corrupted data"],
        )

        summary = result.summary()

        assert "Invalid NPZ" in summary
        assert "Missing features array" in summary
        assert "Corrupted data" in summary


# =============================================================================
# Module Constants Tests
# =============================================================================
class TestModuleConstants:
    """Tests for module-level constants."""

    def test_max_reasonable_samples(self):
        """Test MAX_REASONABLE_SAMPLES is reasonable."""
        assert MAX_REASONABLE_SAMPLES == 100_000_000
        assert MAX_REASONABLE_SAMPLES > 0

    def test_max_reasonable_dimension(self):
        """Test MAX_REASONABLE_DIMENSION is reasonable."""
        assert MAX_REASONABLE_DIMENSION == 1_000_000_000
        assert MAX_REASONABLE_DIMENSION > MAX_REASONABLE_SAMPLES

    def test_required_arrays(self):
        """Test REQUIRED_ARRAYS contains expected values."""
        assert "features" in REQUIRED_ARRAYS
        assert "values" in REQUIRED_ARRAYS

    def test_policy_prefixes(self):
        """Test POLICY_PREFIXES contains expected values."""
        assert "policy" in POLICY_PREFIXES

    def test_expected_dtypes(self):
        """Test EXPECTED_DTYPES has reasonable type constraints."""
        assert "features" in EXPECTED_DTYPES
        assert "float32" in EXPECTED_DTYPES["features"]

        assert "values" in EXPECTED_DTYPES
        assert "float32" in EXPECTED_DTYPES["values"]


# =============================================================================
# validate_npz_structure Tests
# =============================================================================
class TestValidateNpzStructure:
    """Tests for validate_npz_structure function."""

    @pytest.fixture
    def valid_npz_file(self, tmp_path: Path) -> Path:
        """Create a valid NPZ file for testing."""
        file_path = tmp_path / "valid.npz"
        features = np.random.randn(100, 64, 32).astype(np.float32)
        values = np.random.randn(100, 2).astype(np.float32)
        policy_logits = np.random.randn(100, 64).astype(np.float32)

        np.savez(
            file_path,
            features=features,
            values=values,
            policy_logits=policy_logits,
        )
        return file_path

    @pytest.fixture
    def minimal_npz_file(self, tmp_path: Path) -> Path:
        """Create a minimal valid NPZ file."""
        file_path = tmp_path / "minimal.npz"
        features = np.random.randn(10, 8).astype(np.float32)
        values = np.random.randn(10, 2).astype(np.float32)

        np.savez(file_path, features=features, values=values)
        return file_path

    def test_valid_file(self, valid_npz_file: Path):
        """Test validation of a valid NPZ file."""
        result = validate_npz_structure(valid_npz_file)

        assert result.valid is True
        assert result.sample_count == 100
        assert len(result.errors) == 0
        assert result.file_size > 0
        assert "features" in result.array_shapes
        assert "values" in result.array_shapes
        assert "policy_logits" in result.array_shapes

    def test_file_not_found(self, tmp_path: Path):
        """Test validation of non-existent file."""
        result = validate_npz_structure(tmp_path / "nonexistent.npz")

        assert result.valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_empty_file(self, tmp_path: Path):
        """Test validation of empty file."""
        file_path = tmp_path / "empty.npz"
        file_path.touch()

        result = validate_npz_structure(file_path)

        assert result.valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_invalid_npz_format(self, tmp_path: Path):
        """Test validation of non-NPZ file."""
        file_path = tmp_path / "not_npz.npz"
        file_path.write_text("This is not a valid NPZ file")

        result = validate_npz_structure(file_path)

        assert result.valid is False
        assert any("cannot open" in e.lower() for e in result.errors)

    def test_missing_required_array(self, tmp_path: Path):
        """Test validation fails when required array is missing."""
        file_path = tmp_path / "missing_values.npz"
        features = np.random.randn(10, 8).astype(np.float32)
        np.savez(file_path, features=features)  # Missing 'values'

        result = validate_npz_structure(file_path)

        assert result.valid is False
        assert any("missing required array" in e.lower() for e in result.errors)
        assert any("values" in e.lower() for e in result.errors)

    def test_missing_policy_warning(self, minimal_npz_file: Path):
        """Test warning when policy arrays are missing but required."""
        result = validate_npz_structure(minimal_npz_file, require_policy=True)

        # Should still be valid but with warning
        assert result.valid is True
        assert any("policy" in w.lower() for w in result.warnings)

    def test_no_policy_requirement(self, minimal_npz_file: Path):
        """Test no warning when policy is not required."""
        result = validate_npz_structure(minimal_npz_file, require_policy=False)

        assert result.valid is True
        # No policy warning when not required
        policy_warnings = [w for w in result.warnings if "policy" in w.lower()]
        assert len(policy_warnings) == 0

    def test_inconsistent_sample_counts(self, tmp_path: Path):
        """Test validation fails when arrays have different sample counts."""
        file_path = tmp_path / "inconsistent.npz"
        features = np.random.randn(100, 64).astype(np.float32)
        values = np.random.randn(50, 2).astype(np.float32)  # Different count!

        np.savez(file_path, features=features, values=values)

        result = validate_npz_structure(file_path)

        assert result.valid is False
        assert any("inconsistent" in e.lower() for e in result.errors)

    def test_excessive_sample_count(self, tmp_path: Path):
        """Test validation fails when sample count exceeds maximum."""
        # We can't actually create a file with billions of samples,
        # so we use a lower max_samples parameter
        file_path = tmp_path / "normal.npz"
        features = np.random.randn(1000, 8).astype(np.float32)
        values = np.random.randn(1000, 2).astype(np.float32)
        np.savez(file_path, features=features, values=values)

        result = validate_npz_structure(file_path, max_samples=500)

        assert result.valid is False
        assert any("exceeding maximum" in e.lower() for e in result.errors)

    def test_records_array_shapes(self, valid_npz_file: Path):
        """Test that array shapes are properly recorded."""
        result = validate_npz_structure(valid_npz_file)

        assert "features" in result.array_shapes
        assert result.array_shapes["features"] == (100, 64, 32)

        assert "values" in result.array_shapes
        assert result.array_shapes["values"] == (100, 2)

    def test_records_array_dtypes(self, valid_npz_file: Path):
        """Test that array dtypes are properly recorded."""
        result = validate_npz_structure(valid_npz_file)

        assert "features" in result.array_dtypes
        assert result.array_dtypes["features"] == "float32"

        assert "values" in result.array_dtypes
        assert result.array_dtypes["values"] == "float32"

    def test_unexpected_dtype_warning(self, tmp_path: Path):
        """Test warning for unexpected data types."""
        file_path = tmp_path / "wrong_dtype.npz"
        features = np.random.randn(10, 8).astype(np.float64)  # Unexpected dtype
        values = np.random.randn(10, 2).astype(np.float32)

        np.savez(file_path, features=features, values=values)

        result = validate_npz_structure(file_path, require_policy=False)

        assert result.valid is True  # Still valid, just a warning
        # Should have dtype warning (float64 not in expected list)
        dtype_warnings = [w for w in result.warnings if "dtype" in w.lower()]
        assert len(dtype_warnings) > 0

    def test_file_size_recorded(self, valid_npz_file: Path):
        """Test that file size is properly recorded."""
        result = validate_npz_structure(valid_npz_file)

        assert result.file_size > 0
        assert result.file_size == valid_npz_file.stat().st_size


# =============================================================================
# validate_npz_for_training Tests
# =============================================================================
class TestValidateNpzForTraining:
    """Tests for validate_npz_for_training function."""

    @pytest.fixture
    def hex8_npz(self, tmp_path: Path) -> Path:
        """Create NPZ for hex8 board type."""
        file_path = tmp_path / "hex8.npz"
        # hex8 has 61 cells
        features = np.random.randn(100, 61, 32).astype(np.float32)
        values = np.random.randn(100, 2).astype(np.float32)
        policy_logits = np.random.randn(100, 61).astype(np.float32)

        np.savez(file_path, features=features, values=values, policy_logits=policy_logits)
        return file_path

    @pytest.fixture
    def square8_npz(self, tmp_path: Path) -> Path:
        """Create NPZ for square8 board type."""
        file_path = tmp_path / "square8.npz"
        # square8 has 64 cells
        features = np.random.randn(100, 64, 32).astype(np.float32)
        values = np.random.randn(100, 4).astype(np.float32)  # 4 players
        policy_logits = np.random.randn(100, 64).astype(np.float32)

        np.savez(file_path, features=features, values=values, policy_logits=policy_logits)
        return file_path

    def test_valid_hex8_2p(self, hex8_npz: Path):
        """Test validation for hex8 2-player training data."""
        result = validate_npz_for_training(hex8_npz, board_type="hex8", num_players=2)

        assert result.valid is True
        assert result.sample_count == 100

    def test_valid_square8_4p(self, square8_npz: Path):
        """Test validation for square8 4-player training data."""
        result = validate_npz_for_training(square8_npz, board_type="square8", num_players=4)

        assert result.valid is True
        assert result.sample_count == 100

    def test_wrong_board_type_warning(self, hex8_npz: Path):
        """Test warning when features don't match board type."""
        # hex8 NPZ has 61 cells, but we claim it's square8 (64 cells)
        result = validate_npz_for_training(hex8_npz, board_type="square8", num_players=2)

        # Should still be valid, just with warning
        assert result.valid is True
        assert any("may not match board type" in w for w in result.warnings)

    def test_wrong_player_count_warning(self, hex8_npz: Path):
        """Test warning when values don't match player count."""
        # hex8_npz has 2-player values, but we claim 4 players
        result = validate_npz_for_training(hex8_npz, board_type="hex8", num_players=4)

        assert result.valid is True
        assert any("does not match num_players" in w for w in result.warnings)

    def test_no_board_type_specified(self, hex8_npz: Path):
        """Test validation without specifying board type."""
        result = validate_npz_for_training(hex8_npz, board_type=None, num_players=2)

        assert result.valid is True
        # No board-type related warnings when not specified

    def test_no_player_count_specified(self, hex8_npz: Path):
        """Test validation without specifying player count."""
        result = validate_npz_for_training(hex8_npz, board_type="hex8", num_players=None)

        assert result.valid is True
        # No player-count related warnings when not specified


# =============================================================================
# _get_expected_cells Tests
# =============================================================================
class TestGetExpectedCells:
    """Tests for _get_expected_cells helper function."""

    def test_hex8(self):
        """Test cell count for hex8 board."""
        assert _get_expected_cells("hex8") == 61

    def test_square8(self):
        """Test cell count for square8 board."""
        assert _get_expected_cells("square8") == 64

    def test_square19(self):
        """Test cell count for square19 board."""
        assert _get_expected_cells("square19") == 361

    def test_hexagonal(self):
        """Test cell count for hexagonal board."""
        assert _get_expected_cells("hexagonal") == 469

    def test_unknown_board_type(self):
        """Test unknown board type returns None."""
        assert _get_expected_cells("unknown") is None
        assert _get_expected_cells("") is None


# =============================================================================
# quick_npz_check Tests
# =============================================================================
class TestQuickNpzCheck:
    """Tests for quick_npz_check function."""

    @pytest.fixture
    def valid_npz(self, tmp_path: Path) -> Path:
        """Create a valid NPZ file."""
        file_path = tmp_path / "valid.npz"
        features = np.random.randn(10, 8).astype(np.float32)
        values = np.random.randn(10, 2).astype(np.float32)
        np.savez(file_path, features=features, values=values)
        return file_path

    def test_valid_file(self, valid_npz: Path):
        """Test quick check on valid file."""
        is_valid, error = quick_npz_check(valid_npz)

        assert is_valid is True
        assert error == ""

    def test_file_not_found(self, tmp_path: Path):
        """Test quick check on non-existent file."""
        is_valid, error = quick_npz_check(tmp_path / "nonexistent.npz")

        assert is_valid is False
        assert "not found" in error.lower()

    def test_empty_file(self, tmp_path: Path):
        """Test quick check on empty file."""
        file_path = tmp_path / "empty.npz"
        file_path.touch()

        is_valid, error = quick_npz_check(file_path)

        assert is_valid is False
        assert "empty" in error.lower()

    def test_missing_features(self, tmp_path: Path):
        """Test quick check on file missing features array."""
        file_path = tmp_path / "no_features.npz"
        values = np.random.randn(10, 2).astype(np.float32)
        np.savez(file_path, values=values)

        is_valid, error = quick_npz_check(file_path)

        assert is_valid is False
        assert "features" in error.lower()

    def test_missing_values(self, tmp_path: Path):
        """Test quick check on file missing values array."""
        file_path = tmp_path / "no_values.npz"
        features = np.random.randn(10, 8).astype(np.float32)
        np.savez(file_path, features=features)

        is_valid, error = quick_npz_check(file_path)

        assert is_valid is False
        assert "values" in error.lower()

    def test_invalid_format(self, tmp_path: Path):
        """Test quick check on non-NPZ file."""
        file_path = tmp_path / "not_npz.npz"
        file_path.write_text("This is not NPZ")

        is_valid, error = quick_npz_check(file_path)

        assert is_valid is False
        assert len(error) > 0  # Some error message


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================
class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_npz_with_extra_arrays(self, tmp_path: Path):
        """Test validation with extra arrays beyond required (same sample count)."""
        file_path = tmp_path / "extra.npz"
        features = np.random.randn(100, 64).astype(np.float32)
        values = np.random.randn(100, 2).astype(np.float32)
        policy_logits = np.random.randn(100, 64).astype(np.float32)
        extra_array = np.random.randn(100, 10).astype(np.float32)
        # Note: all arrays must have same first dimension for validation to pass
        metadata = np.array(["m" + str(i) for i in range(100)])  # Same sample count

        np.savez(
            file_path,
            features=features,
            values=values,
            policy_logits=policy_logits,
            extra_array=extra_array,
            metadata=metadata,
        )

        result = validate_npz_structure(file_path)

        assert result.valid is True
        assert "extra_array" in result.array_shapes

    def test_npz_with_scalar_arrays(self, tmp_path: Path):
        """Test validation with scalar (0-dim) arrays."""
        file_path = tmp_path / "scalar.npz"
        features = np.random.randn(100, 64).astype(np.float32)
        values = np.random.randn(100, 2).astype(np.float32)
        policy_logits = np.random.randn(100, 64).astype(np.float32)
        version = np.array(1.0)  # Scalar

        np.savez(
            file_path,
            features=features,
            values=values,
            policy_logits=policy_logits,
            version=version,
        )

        result = validate_npz_structure(file_path)

        # Should still be valid - scalar doesn't affect sample count
        assert result.valid is True

    def test_compressed_npz(self, tmp_path: Path):
        """Test validation with compressed NPZ file."""
        file_path = tmp_path / "compressed.npz"
        features = np.random.randn(100, 64).astype(np.float32)
        values = np.random.randn(100, 2).astype(np.float32)
        policy_logits = np.random.randn(100, 64).astype(np.float32)

        np.savez_compressed(
            file_path,
            features=features,
            values=values,
            policy_logits=policy_logits,
        )

        result = validate_npz_structure(file_path)

        assert result.valid is True
        assert result.sample_count == 100

    def test_large_array_dimensions(self, tmp_path: Path):
        """Test that large but valid dimensions pass."""
        file_path = tmp_path / "large_dims.npz"
        # 1000 samples, 469 cells (hexagonal), 128 channels
        features = np.random.randn(1000, 469, 128).astype(np.float32)
        values = np.random.randn(1000, 4).astype(np.float32)
        policy_logits = np.random.randn(1000, 469).astype(np.float32)

        np.savez(file_path, features=features, values=values, policy_logits=policy_logits)

        result = validate_npz_structure(file_path)

        assert result.valid is True
        assert result.sample_count == 1000

    def test_different_policy_array_names(self, tmp_path: Path):
        """Test recognition of various policy array naming conventions."""
        file_path = tmp_path / "policy_variants.npz"
        features = np.random.randn(50, 64).astype(np.float32)
        values = np.random.randn(50, 2).astype(np.float32)
        policy_mask = np.ones((50, 64), dtype=np.bool_)
        policy_targets = np.random.randn(50, 64).astype(np.float32)

        np.savez(
            file_path,
            features=features,
            values=values,
            policy_mask=policy_mask,
            policy_targets=policy_targets,
        )

        result = validate_npz_structure(file_path, require_policy=True)

        assert result.valid is True
        # No policy warning since we have policy_* arrays
        policy_warnings = [w for w in result.warnings if "no policy" in w.lower()]
        assert len(policy_warnings) == 0

    def test_memory_mapped_reading(self, tmp_path: Path):
        """Test that memory-mapped reading works for larger files."""
        file_path = tmp_path / "mmap_test.npz"
        # Create a moderately sized file
        features = np.random.randn(10000, 64, 32).astype(np.float32)
        values = np.random.randn(10000, 2).astype(np.float32)
        policy = np.random.randn(10000, 64).astype(np.float32)

        np.savez(file_path, features=features, values=values, policy_logits=policy)

        result = validate_npz_structure(file_path)

        assert result.valid is True
        assert result.sample_count == 10000
        # File should be in the MB range
        assert result.file_size > 1024 * 1024
