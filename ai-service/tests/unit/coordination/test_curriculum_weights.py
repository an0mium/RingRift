"""Tests for curriculum weight management.

Tests curriculum_weights.py which provides persistence for selfplay
prioritization weights used by SelfplayScheduler, QueuePopulator, and P2P.

December 28, 2025: Created comprehensive test coverage.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from app.coordination.curriculum_weights import (
    CURRICULUM_WEIGHTS_PATH,
    CURRICULUM_WEIGHTS_STALE_SECONDS,
    export_curriculum_weights,
    get_curriculum_weight,
    load_curriculum_weights,
)


class TestCurriculumWeightsConstants:
    """Tests for module constants."""

    def test_path_constant_is_path(self) -> None:
        """Verify CURRICULUM_WEIGHTS_PATH is a Path object."""
        assert isinstance(CURRICULUM_WEIGHTS_PATH, Path)

    def test_path_ends_with_json(self) -> None:
        """Verify path has .json extension."""
        assert CURRICULUM_WEIGHTS_PATH.suffix == ".json"

    def test_path_in_data_directory(self) -> None:
        """Verify path is in data directory."""
        assert "data" in str(CURRICULUM_WEIGHTS_PATH)

    def test_staleness_constant_is_positive(self) -> None:
        """Verify staleness threshold is a positive number."""
        assert CURRICULUM_WEIGHTS_STALE_SECONDS > 0

    def test_staleness_constant_is_reasonable(self) -> None:
        """Verify staleness threshold is between 1 minute and 24 hours."""
        assert 60 <= CURRICULUM_WEIGHTS_STALE_SECONDS <= 86400  # 1 min to 24 hours


class TestExportCurriculumWeights:
    """Tests for export_curriculum_weights function."""

    def test_successful_export_creates_file(self, tmp_path: Path) -> None:
        """Verify export creates the weights file."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            weights = {"hex8_2p": 1.5, "square8_4p": 0.8}
            result = export_curriculum_weights(weights)

        assert result is True
        assert weights_path.exists()

    def test_export_creates_valid_json(self, tmp_path: Path) -> None:
        """Verify export creates valid JSON with correct structure."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            weights = {"hex8_2p": 1.5, "square8_4p": 0.8}
            export_curriculum_weights(weights)

        with open(weights_path) as f:
            data = json.load(f)

        assert "weights" in data
        assert "updated_at" in data
        assert "updated_at_iso" in data
        assert data["weights"] == weights

    def test_export_includes_timestamp(self, tmp_path: Path) -> None:
        """Verify export includes timestamp for staleness checking."""
        weights_path = tmp_path / "curriculum_weights.json"

        before = time.time()
        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights({"hex8_2p": 1.0})
        after = time.time()

        with open(weights_path) as f:
            data = json.load(f)

        assert before <= data["updated_at"] <= after

    def test_export_uses_atomic_write(self, tmp_path: Path) -> None:
        """Verify export uses temp file + rename pattern (atomic write)."""
        weights_path = tmp_path / "curriculum_weights.json"
        temp_path = weights_path.with_suffix(".tmp")

        # Pre-create to verify overwrite
        weights_path.write_text('{"old": "data"}')

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights({"new": 1.0})

        # Temp file should not exist after successful export
        assert not temp_path.exists()
        # Main file should have new data
        with open(weights_path) as f:
            data = json.load(f)
        assert "new" in data["weights"]

    def test_export_creates_parent_directory(self, tmp_path: Path) -> None:
        """Verify export creates parent directories if needed."""
        weights_path = tmp_path / "nested" / "dir" / "weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = export_curriculum_weights({"hex8_2p": 1.0})

        assert result is True
        assert weights_path.exists()

    def test_export_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Verify export overwrites existing file."""
        weights_path = tmp_path / "curriculum_weights.json"
        weights_path.write_text('{"weights": {"old": 1.0}, "updated_at": 0}')

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights({"new": 2.0})

        with open(weights_path) as f:
            data = json.load(f)

        assert "new" in data["weights"]
        assert "old" not in data["weights"]

    def test_export_returns_false_on_permission_error(self, tmp_path: Path) -> None:
        """Verify export returns False on permission error."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            with mock.patch("builtins.open", side_effect=PermissionError("denied")):
                result = export_curriculum_weights({"hex8_2p": 1.0})

        assert result is False

    def test_export_handles_empty_weights(self, tmp_path: Path) -> None:
        """Verify export works with empty weights dict."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = export_curriculum_weights({})

        assert result is True
        with open(weights_path) as f:
            data = json.load(f)
        assert data["weights"] == {}


class TestLoadCurriculumWeights:
    """Tests for load_curriculum_weights function."""

    def test_load_returns_empty_dict_if_file_missing(self, tmp_path: Path) -> None:
        """Verify load returns empty dict if file doesn't exist."""
        weights_path = tmp_path / "nonexistent.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_returns_valid_weights(self, tmp_path: Path) -> None:
        """Verify load returns weights from valid file."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5, "square8_4p": 0.8},
            "updated_at": time.time(),  # Fresh
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {"hex8_2p": 1.5, "square8_4p": 0.8}

    def test_load_respects_max_age_seconds(self, tmp_path: Path) -> None:
        """Verify load respects custom max_age_seconds parameter."""
        weights_path = tmp_path / "curriculum_weights.json"
        # Write data that's 100 seconds old
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time() - 100,
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            # With 50s max age, should be stale
            result_stale = load_curriculum_weights(max_age_seconds=50)
            # With 200s max age, should be fresh
            result_fresh = load_curriculum_weights(max_age_seconds=200)

        assert result_stale == {}
        assert result_fresh == {"hex8_2p": 1.5}

    def test_load_returns_empty_dict_if_stale(self, tmp_path: Path) -> None:
        """Verify load returns empty dict for stale weights."""
        weights_path = tmp_path / "curriculum_weights.json"
        # Write data that's definitely stale (1 day old)
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time() - 86400,  # 1 day ago
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_handles_malformed_json(self, tmp_path: Path) -> None:
        """Verify load returns empty dict for malformed JSON."""
        weights_path = tmp_path / "curriculum_weights.json"
        weights_path.write_text("not valid json{{{")

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_handles_missing_weights_key(self, tmp_path: Path) -> None:
        """Verify load returns empty dict if weights key missing."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "updated_at": time.time(),
            # Missing "weights" key
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}

    def test_load_handles_missing_updated_at(self, tmp_path: Path) -> None:
        """Verify load returns empty dict if updated_at missing (treated as stale)."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            # Missing "updated_at" - defaults to 0, so always stale
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = load_curriculum_weights()

        assert result == {}


class TestGetCurriculumWeight:
    """Tests for get_curriculum_weight convenience function."""

    def test_get_weight_returns_value_if_exists(self, tmp_path: Path) -> None:
        """Verify get_weight returns correct value for existing config."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5, "square8_4p": 0.8},
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p")

        assert result == 1.5

    def test_get_weight_returns_default_if_missing(self, tmp_path: Path) -> None:
        """Verify get_weight returns default for missing config."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("nonexistent_config")

        assert result == 1.0  # Default

    def test_get_weight_uses_custom_default(self, tmp_path: Path) -> None:
        """Verify get_weight respects custom default parameter."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {},
            "updated_at": time.time(),
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p", default=0.5)

        assert result == 0.5

    def test_get_weight_returns_default_if_file_missing(self, tmp_path: Path) -> None:
        """Verify get_weight returns default if weights file doesn't exist."""
        weights_path = tmp_path / "nonexistent.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p", default=2.0)

        assert result == 2.0

    def test_get_weight_returns_default_if_stale(self, tmp_path: Path) -> None:
        """Verify get_weight returns default for stale weights."""
        weights_path = tmp_path / "curriculum_weights.json"
        data = {
            "weights": {"hex8_2p": 1.5},
            "updated_at": time.time() - 86400,  # 1 day old (stale)
        }
        weights_path.write_text(json.dumps(data))

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            result = get_curriculum_weight("hex8_2p", default=0.7)

        assert result == 0.7


class TestIntegration:
    """Integration tests for export/load cycle."""

    def test_export_then_load_roundtrip(self, tmp_path: Path) -> None:
        """Verify weights can be exported and loaded correctly."""
        weights_path = tmp_path / "curriculum_weights.json"
        original_weights = {
            "hex8_2p": 1.5,
            "hex8_3p": 1.2,
            "hex8_4p": 0.8,
            "square8_2p": 1.0,
        }

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_result = export_curriculum_weights(original_weights)
            loaded_weights = load_curriculum_weights()

        assert export_result is True
        assert loaded_weights == original_weights

    def test_multiple_exports_overwrite(self, tmp_path: Path) -> None:
        """Verify multiple exports correctly overwrite previous data."""
        weights_path = tmp_path / "curriculum_weights.json"

        with mock.patch(
            "app.coordination.curriculum_weights.CURRICULUM_WEIGHTS_PATH",
            weights_path,
        ):
            export_curriculum_weights({"v1": 1.0})
            export_curriculum_weights({"v2": 2.0})
            export_curriculum_weights({"v3": 3.0})
            loaded = load_curriculum_weights()

        assert loaded == {"v3": 3.0}
        assert "v1" not in loaded
        assert "v2" not in loaded
