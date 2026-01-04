"""Tests for app.models.discovery module.

Tests model discovery functionality including board type detection,
sidecar file handling, and model enumeration.
"""

import json
import pytest
import tempfile
from pathlib import Path

from app.models.discovery import (
    ModelInfo,
    detect_board_type_from_name,
    read_model_sidecar,
    write_model_sidecar,
)


class TestDetectBoardTypeFromName:
    """Tests for detect_board_type_from_name function."""

    def test_detect_square8_variants(self):
        """Test detection of square8 board type."""
        test_cases = [
            "model_sq8_2p.pth",
            "ringrift_square8_v1.pth",
            "canonical_8x8_model.pth",
            "sq8_production.pth",
        ]
        for name in test_cases:
            board_type, _ = detect_board_type_from_name(name)
            assert board_type == "square8", f"Failed for {name}"

    def test_detect_square19_variants(self):
        """Test detection of square19 board type."""
        test_cases = [
            "model_sq19_2p.pth",
            "ringrift_square19_v2.pth",
            "canonical_19x19_model.pth",
        ]
        for name in test_cases:
            board_type, _ = detect_board_type_from_name(name)
            assert board_type == "square19", f"Failed for {name}"

    def test_detect_hexagonal_variants(self):
        """Test detection of hexagonal board type."""
        test_cases = [
            "model_hex_2p.pth",
            "hexagonal_v1.pth",
            "hex8_production.pth",
        ]
        for name in test_cases:
            board_type, _ = detect_board_type_from_name(name)
            assert board_type == "hexagonal", f"Failed for {name}"

    def test_detect_num_players(self):
        """Test detection of number of players."""
        test_cases = [
            ("model_sq8_2p.pth", 2),
            ("model_sq8_3p.pth", 3),
            ("model_sq8_4p.pth", 4),
            ("2p_model.pth", 2),
            ("3p_model.pth", 3),
            ("4p_model.pth", 4),
        ]
        for name, expected in test_cases:
            _, num_players = detect_board_type_from_name(name)
            assert num_players == expected, f"Failed for {name}"

    def test_default_num_players(self):
        """Test that unknown patterns default to 2 players."""
        _, num_players = detect_board_type_from_name("unknown_model.pth")
        assert num_players == 2

    def test_legacy_ringrift_model(self):
        """Test legacy ringrift_v* models default to square8."""
        board_type, _ = detect_board_type_from_name("ringrift_v3.pth")
        assert board_type == "square8"


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_to_dict(self):
        """Test ModelInfo serialization to dict."""
        info = ModelInfo(
            path="/path/to/model.pth",
            name="model",
            model_type="nn",
            board_type="square8",
            num_players=2,
            elo=1500.0,
        )
        d = info.to_dict()
        assert d["path"] == "/path/to/model.pth"
        assert d["board_type"] == "square8"
        assert d["elo"] == 1500.0

    def test_from_dict(self):
        """Test ModelInfo deserialization from dict."""
        data = {
            "path": "/path/to/model.pth",
            "name": "model",
            "model_type": "nn",
            "board_type": "square8",
            "num_players": 2,
        }
        info = ModelInfo.from_dict(data)
        assert info.path == "/path/to/model.pth"
        assert info.board_type == "square8"

    def test_from_dict_ignores_extra_fields(self):
        """Test that from_dict ignores unknown fields."""
        data = {
            "path": "/path/to/model.pth",
            "name": "model",
            "model_type": "nn",
            "board_type": "square8",
            "unknown_field": "ignored",
        }
        info = ModelInfo.from_dict(data)
        assert info.board_type == "square8"
        assert not hasattr(info, "unknown_field")


class TestSidecarFiles:
    """Tests for sidecar file reading and writing."""

    def test_write_sidecar(self):
        """Test writing a model sidecar file."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            model_path = Path(f.name)

        try:
            sidecar_path = write_model_sidecar(
                model_path,
                board_type="square8",
                num_players=2,
                elo=1500.0,
                architecture_version="v2",
            )

            assert sidecar_path.exists()
            with open(sidecar_path) as f:
                data = json.load(f)

            assert data["board_type"] == "square8"
            assert data["num_players"] == 2
            assert data["elo"] == 1500.0
            assert data["architecture_version"] == "v2"
            assert "created_at" in data
        finally:
            model_path.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)

    def test_read_sidecar(self):
        """Test reading a model sidecar file."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            model_path = Path(f.name)
        sidecar_path = Path(str(model_path) + ".json")

        try:
            # Write sidecar manually
            sidecar_data = {
                "board_type": "hexagonal",
                "num_players": 4,
                "elo": 1600.0,
            }
            with open(sidecar_path, "w") as f:
                json.dump(sidecar_data, f)

            # Read it back
            data = read_model_sidecar(model_path)
            assert data is not None
            assert data["board_type"] == "hexagonal"
            assert data["num_players"] == 4
        finally:
            model_path.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)

    def test_read_missing_sidecar(self):
        """Test reading sidecar that doesn't exist returns None."""
        data = read_model_sidecar(Path("/nonexistent/model.pth"))
        assert data is None

    def test_write_sidecar_with_extra_metadata(self):
        """Test writing sidecar with extra metadata."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            model_path = Path(f.name)

        try:
            sidecar_path = write_model_sidecar(
                model_path,
                board_type="square8",
                num_players=2,
                extra_metadata={
                    "training_epochs": 100,
                    "dataset_size": 50000,
                },
            )

            with open(sidecar_path) as f:
                data = json.load(f)

            assert data["training_epochs"] == 100
            assert data["dataset_size"] == 50000
        finally:
            model_path.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)
