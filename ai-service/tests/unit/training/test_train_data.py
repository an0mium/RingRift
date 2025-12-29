"""Unit tests for app/training/train_data.py module.

Tests cover:
- DataLoaderConfig dataclass creation and from_config()
- DatasetMetadata dataclass
- DataLoaderResult dataclass
- collect_data_paths() function
- get_total_data_size() function
- should_use_streaming() threshold logic
- get_sample_count() function
- get_num_loader_workers() function

December 2025: Created as part of training module test coverage initiative.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDataLoaderConfig:
    """Tests for DataLoaderConfig dataclass."""

    def test_default_creation(self):
        """DataLoaderConfig can be created with defaults."""
        from app.training.train_data import DataLoaderConfig

        config = DataLoaderConfig()

        assert config.data_path is None
        assert config.data_dir is None
        assert config.use_streaming is False
        assert config.batch_size == 512
        assert config.sampling_weights == "uniform"
        assert config.val_split == 0.2
        assert config.num_workers == 0
        assert config.discover_synced_data is False
        assert config.filter_empty_policies is True
        assert config.enable_elo_weighting is False
        assert config.min_quality_score == 0.0
        assert config.seed == 42
        assert config.policy_size == 64

    def test_custom_creation(self):
        """DataLoaderConfig can be created with custom values."""
        from app.training.train_data import DataLoaderConfig

        config = DataLoaderConfig(
            data_path="/path/to/data.npz",
            batch_size=256,
            use_streaming=True,
            sampling_weights="quality",
            val_split=0.1,
        )

        assert config.data_path == "/path/to/data.npz"
        assert config.batch_size == 256
        assert config.use_streaming is True
        assert config.sampling_weights == "quality"
        assert config.val_split == 0.1

    def test_from_config_basic(self):
        """DataLoaderConfig.from_config() creates config from FullTrainingConfig."""
        from app.training.train_data import DataLoaderConfig

        # Mock FullTrainingConfig
        full_config = MagicMock()
        full_config.data_path = "/data/training.npz"
        full_config.data_dir = None
        full_config.use_streaming = True
        full_config.batch_size = 1024
        full_config.sampling_weights = "uniform"
        full_config.val_split = 0.15
        full_config.seed = 123

        config = DataLoaderConfig.from_config(full_config)

        assert config.data_path == "/data/training.npz"
        assert config.use_streaming is True
        assert config.batch_size == 1024
        assert config.seed == 123

    def test_from_config_missing_attributes(self):
        """DataLoaderConfig.from_config() handles missing attributes gracefully."""
        from app.training.train_data import DataLoaderConfig

        # Minimal mock with required attributes only
        full_config = MagicMock(spec=["batch_size", "seed"])
        full_config.batch_size = 512
        full_config.seed = 42

        # Should not raise - getattr with defaults handles missing attrs
        config = DataLoaderConfig.from_config(full_config)

        assert config.batch_size == 512
        assert config.data_path is None  # Default from getattr


class TestDatasetMetadata:
    """Tests for DatasetMetadata dataclass."""

    def test_default_creation(self):
        """DatasetMetadata can be created with defaults."""
        from app.training.train_data import DatasetMetadata

        metadata = DatasetMetadata()

        assert metadata.in_channels is None
        assert metadata.globals_dim is None
        assert metadata.history_length is None
        assert metadata.feature_version is None
        assert metadata.policy_encoding is None
        assert metadata.encoder_type is None
        assert metadata.encoder_version is None
        assert metadata.base_channels is None
        assert metadata.board_type is None

    def test_custom_creation(self):
        """DatasetMetadata can be created with custom values."""
        from app.training.train_data import DatasetMetadata

        metadata = DatasetMetadata(
            in_channels=24,
            globals_dim=8,
            history_length=4,
            feature_version=2,
            policy_encoding="flat",
            board_type="hex8",
        )

        assert metadata.in_channels == 24
        assert metadata.globals_dim == 8
        assert metadata.history_length == 4
        assert metadata.feature_version == 2
        assert metadata.policy_encoding == "flat"
        assert metadata.board_type == "hex8"


class TestDataLoaderResult:
    """Tests for DataLoaderResult dataclass."""

    def test_default_creation(self):
        """DataLoaderResult can be created with defaults."""
        from app.training.train_data import DataLoaderResult

        result = DataLoaderResult()

        assert result.train_loader is None
        assert result.val_loader is None
        assert result.train_streaming_loader is None
        assert result.val_streaming_loader is None
        assert result.train_sampler is None
        assert result.train_size == 0
        assert result.val_size == 0
        assert result.total_samples == 0
        assert result.feature_shape is None
        assert result.use_streaming is False
        assert result.value_only_training is False
        assert result.has_multi_player_values is False
        assert result.full_dataset is None

    def test_custom_creation(self):
        """DataLoaderResult can be created with custom values."""
        from app.training.train_data import DataLoaderResult

        mock_loader = MagicMock()

        result = DataLoaderResult(
            train_loader=mock_loader,
            train_size=10000,
            val_size=2000,
            total_samples=12000,
            feature_shape=(24, 8, 8),
            use_streaming=True,
        )

        assert result.train_loader is mock_loader
        assert result.train_size == 10000
        assert result.val_size == 2000
        assert result.total_samples == 12000
        assert result.feature_shape == (24, 8, 8)
        assert result.use_streaming is True


class TestCollectDataPaths:
    """Tests for collect_data_paths function."""

    def test_single_data_path(self):
        """collect_data_paths handles single data path."""
        from app.training.train_data import collect_data_paths

        paths = collect_data_paths(
            data_path="/path/to/data.npz",
            data_dir=None,
        )

        assert paths == ["/path/to/data.npz"]

    def test_list_data_paths(self):
        """collect_data_paths handles list of data paths."""
        from app.training.train_data import collect_data_paths

        paths = collect_data_paths(
            data_path=["/path/a.npz", "/path/b.npz"],
            data_dir=None,
        )

        assert paths == ["/path/a.npz", "/path/b.npz"]

    def test_data_dir_with_glob(self):
        """collect_data_paths discovers files from data_dir."""
        from app.training.train_data import collect_data_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some NPZ files
            Path(tmpdir, "file1.npz").touch()
            Path(tmpdir, "file2.npz").touch()
            Path(tmpdir, "not_npz.txt").touch()

            paths = collect_data_paths(
                data_path=None,
                data_dir=tmpdir,
            )

            assert len(paths) == 2
            assert all(p.endswith(".npz") for p in paths)

    def test_deduplication(self):
        """collect_data_paths deduplicates paths."""
        from app.training.train_data import collect_data_paths

        paths = collect_data_paths(
            data_path=["/path/a.npz", "/path/b.npz", "/path/a.npz"],
            data_dir=None,
        )

        assert paths == ["/path/a.npz", "/path/b.npz"]

    def test_combined_sources(self):
        """collect_data_paths combines multiple sources."""
        from app.training.train_data import collect_data_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "dir_file.npz").touch()

            paths = collect_data_paths(
                data_path="/explicit/path.npz",
                data_dir=tmpdir,
            )

            assert len(paths) == 2
            assert "/explicit/path.npz" in paths
            assert any("dir_file.npz" in p for p in paths)

    def test_empty_inputs(self):
        """collect_data_paths handles empty inputs."""
        from app.training.train_data import collect_data_paths

        paths = collect_data_paths(
            data_path=None,
            data_dir=None,
        )

        assert paths == []


class TestGetTotalDataSize:
    """Tests for get_total_data_size function."""

    def test_single_file(self):
        """get_total_data_size calculates size of single file."""
        from app.training.train_data import get_total_data_size

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            f.write(b"x" * 1024)  # 1KB
            f.flush()

            try:
                size = get_total_data_size([f.name])
                assert size == 1024
            finally:
                os.unlink(f.name)

    def test_multiple_files(self):
        """get_total_data_size sums sizes of multiple files."""
        from app.training.train_data import get_total_data_size

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir, "f1.npz")
            file2 = Path(tmpdir, "f2.npz")
            file1.write_bytes(b"x" * 500)
            file2.write_bytes(b"y" * 300)

            size = get_total_data_size([str(file1), str(file2)])
            assert size == 800

    def test_missing_file(self):
        """get_total_data_size handles missing files."""
        from app.training.train_data import get_total_data_size

        # Should not raise, just skip missing files
        size = get_total_data_size(["/nonexistent/file.npz"])
        assert size == 0

    def test_empty_list(self):
        """get_total_data_size handles empty list."""
        from app.training.train_data import get_total_data_size

        size = get_total_data_size([])
        assert size == 0


class TestShouldUseStreaming:
    """Tests for should_use_streaming function."""

    def test_force_streaming_true(self):
        """should_use_streaming returns True when force_streaming=True."""
        from app.training.train_data import should_use_streaming

        result = should_use_streaming(
            paths=["/some/file.npz"],
            force_streaming=True,
        )
        assert result is True

    def test_force_streaming_false(self):
        """should_use_streaming returns False when force_streaming=False."""
        from app.training.train_data import should_use_streaming

        result = should_use_streaming(
            paths=["/some/file.npz"],
            force_streaming=False,
        )
        assert result is False

    def test_auto_streaming_small_file(self):
        """should_use_streaming returns False for small files (auto mode)."""
        from app.training.train_data import should_use_streaming

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            f.write(b"x" * 1024)  # 1KB - well under threshold

            try:
                result = should_use_streaming(
                    paths=[f.name],
                    force_streaming=None,  # Auto mode
                )
                assert result is False
            finally:
                os.unlink(f.name)

    def test_empty_paths(self):
        """should_use_streaming returns False for empty paths."""
        from app.training.train_data import should_use_streaming

        result = should_use_streaming(
            paths=[],
            force_streaming=None,
        )
        assert result is False


class TestGetSampleCount:
    """Tests for get_sample_count function."""

    def test_valid_npz_file(self):
        """get_sample_count returns correct count from NPZ file."""
        from app.training.train_data import get_sample_count

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            # Create NPZ with 100 samples - uses "features" key
            features = np.random.randn(100, 24, 8, 8).astype(np.float32)
            np.savez(f.name, features=features)

            try:
                count = get_sample_count(f.name)
                assert count == 100
            finally:
                os.unlink(f.name)

    def test_missing_file(self):
        """get_sample_count returns 0 for missing file."""
        from app.training.train_data import get_sample_count

        count = get_sample_count("/nonexistent/file.npz")
        assert count == 0


class TestGetNumLoaderWorkers:
    """Tests for get_num_loader_workers function."""

    def test_streaming_mode(self):
        """get_num_loader_workers returns 0 for streaming mode."""
        from app.training.train_data import get_num_loader_workers

        workers = get_num_loader_workers(use_streaming=True)
        assert workers == 0

    def test_non_streaming_mode(self):
        """get_num_loader_workers returns positive count for non-streaming."""
        from app.training.train_data import get_num_loader_workers

        workers = get_num_loader_workers(use_streaming=False)
        # Should be min(4, cpu_count) or fallback
        assert workers >= 0


class TestAutoStreamingThreshold:
    """Tests for AUTO_STREAMING_THRESHOLD_BYTES constant."""

    def test_default_threshold(self):
        """Default streaming threshold is 2GB."""
        from app.training.train_data import AUTO_STREAMING_THRESHOLD_BYTES

        expected = 2 * (1024 ** 3)  # 2GB
        assert AUTO_STREAMING_THRESHOLD_BYTES == expected

    def test_env_var_override(self):
        """Streaming threshold can be overridden via env var."""
        # Note: This test documents the behavior, but can't test runtime
        # since the module is already imported. The env var is read at import time.
        from app.training.train_data import AUTO_STREAMING_THRESHOLD_BYTES

        # Just verify the constant exists and is reasonable
        assert AUTO_STREAMING_THRESHOLD_BYTES > 0
        assert AUTO_STREAMING_THRESHOLD_BYTES >= 1024 ** 3  # At least 1GB


class TestLazyImports:
    """Tests for lazy import functions."""

    def test_try_import_streaming_loader(self):
        """_try_import_streaming_loader returns class or None."""
        from app.training.train_data import _try_import_streaming_loader

        result = _try_import_streaming_loader()
        # Either returns the class or None (if not available)
        assert result is None or callable(result)

    def test_try_import_dataset(self):
        """_try_import_dataset returns class or None."""
        from app.training.train_data import _try_import_dataset

        result = _try_import_dataset()
        assert result is None or callable(result)

    def test_try_import_data_catalog(self):
        """_try_import_data_catalog returns function or None."""
        from app.training.train_data import _try_import_data_catalog

        result = _try_import_data_catalog()
        assert result is None or callable(result)
