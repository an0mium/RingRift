"""Tests for app/training/data_loader_factory.py.

Comprehensive tests for the data loader factory module which handles
data loading for neural network training, including streaming loaders,
curriculum weights, and platform-aware configuration.

December 2025: Created for test coverage expansion.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.models import BoardType


# =============================================================================
# Test DataLoaderConfig
# =============================================================================

class TestDataLoaderConfig:
    """Tests for DataLoaderConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.training.data_loader_factory import DataLoaderConfig

        config = DataLoaderConfig()

        assert config.batch_size == 256
        assert config.use_streaming is False
        assert config.sampling_weights == 'uniform'
        assert config.augment_hex_symmetry is False
        assert config.multi_player is False
        assert config.filter_empty_policies is True
        assert config.seed == 42
        assert config.board_type == BoardType.SQUARE8
        assert config.policy_size == 512
        assert config.distributed is False
        assert config.rank == 0
        assert config.world_size == 1
        assert config.data_path is None
        assert config.data_dir is None
        assert config.use_curriculum_weights is False
        assert config.curriculum_weights is None
        assert config.return_heuristics is False
        assert config.num_workers is None

    def test_custom_values(self):
        """Test custom configuration values."""
        from app.training.data_loader_factory import DataLoaderConfig

        config = DataLoaderConfig(
            batch_size=512,
            use_streaming=True,
            sampling_weights='quality',
            board_type=BoardType.HEX8,
            policy_size=256,
            distributed=True,
            rank=1,
            world_size=4,
            data_path='/path/to/data.npz',
            use_curriculum_weights=True,
            curriculum_weights={'hex8_2p': 1.5},
        )

        assert config.batch_size == 512
        assert config.use_streaming is True
        assert config.sampling_weights == 'quality'
        assert config.board_type == BoardType.HEX8
        assert config.policy_size == 256
        assert config.distributed is True
        assert config.rank == 1
        assert config.world_size == 4
        assert config.data_path == '/path/to/data.npz'
        assert config.use_curriculum_weights is True
        assert config.curriculum_weights == {'hex8_2p': 1.5}

    def test_list_data_path(self):
        """Test configuration with list of data paths."""
        from app.training.data_loader_factory import DataLoaderConfig

        config = DataLoaderConfig(
            data_path=['/path/a.npz', '/path/b.npz'],
        )

        assert isinstance(config.data_path, list)
        assert len(config.data_path) == 2


# =============================================================================
# Test DataLoaderResult
# =============================================================================

class TestDataLoaderResult:
    """Tests for DataLoaderResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        from app.training.data_loader_factory import DataLoaderResult

        result = DataLoaderResult()

        assert result.train_loader is None
        assert result.val_loader is None
        assert result.train_sampler is None
        assert result.val_sampler is None
        assert result.train_size == 0
        assert result.val_size == 0
        assert result.use_streaming is False
        assert result.has_multi_player_values is False
        assert result.data_paths is None
        assert result.num_workers == 0

    def test_custom_values(self):
        """Test result with custom values."""
        from app.training.data_loader_factory import DataLoaderResult

        mock_loader = MagicMock()
        mock_sampler = MagicMock()

        result = DataLoaderResult(
            train_loader=mock_loader,
            val_loader=mock_loader,
            train_sampler=mock_sampler,
            train_size=8000,
            val_size=2000,
            use_streaming=True,
            has_multi_player_values=True,
            data_paths=['/a.npz', '/b.npz'],
            num_workers=4,
        )

        assert result.train_loader is mock_loader
        assert result.train_size == 8000
        assert result.val_size == 2000
        assert result.use_streaming is True
        assert result.has_multi_player_values is True
        assert len(result.data_paths) == 2
        assert result.num_workers == 4


# =============================================================================
# Test compute_num_workers
# =============================================================================

class TestComputeNumWorkers:
    """Tests for compute_num_workers function."""

    def test_env_override(self):
        """Test environment variable override."""
        from app.training.data_loader_factory import compute_num_workers

        with patch.dict(os.environ, {'RINGRIFT_DATALOADER_WORKERS': '8'}):
            assert compute_num_workers(None) == 8
            # Env should override explicit config too
            assert compute_num_workers(4) == 8

    def test_explicit_config(self):
        """Test explicit config value."""
        from app.training.data_loader_factory import compute_num_workers

        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if present
            os.environ.pop('RINGRIFT_DATALOADER_WORKERS', None)
            assert compute_num_workers(6) == 6

    def test_macos_default(self):
        """Test macOS returns 0 workers (mmap incompatible with fork)."""
        from app.training.data_loader_factory import compute_num_workers

        os.environ.pop('RINGRIFT_DATALOADER_WORKERS', None)

        with patch.object(sys, 'platform', 'darwin'):
            result = compute_num_workers(None)
            assert result == 0

    def test_linux_default(self):
        """Test Linux computes workers from CPU count."""
        from app.training.data_loader_factory import compute_num_workers

        os.environ.pop('RINGRIFT_DATALOADER_WORKERS', None)

        with patch.object(sys, 'platform', 'linux'):
            with patch('multiprocessing.cpu_count', return_value=16):
                result = compute_num_workers(None)
                # min(4, 16 // 2) = min(4, 8) = 4
                assert result == 4

    def test_linux_few_cpus(self):
        """Test Linux with few CPUs."""
        from app.training.data_loader_factory import compute_num_workers

        os.environ.pop('RINGRIFT_DATALOADER_WORKERS', None)

        with patch.object(sys, 'platform', 'linux'):
            with patch('multiprocessing.cpu_count', return_value=4):
                result = compute_num_workers(None)
                # min(4, 4 // 2) = min(4, 2) = 2
                assert result == 2


# =============================================================================
# Test should_use_streaming
# =============================================================================

class TestShouldUseStreaming:
    """Tests for should_use_streaming function."""

    def test_empty_paths(self):
        """Test returns False with no paths."""
        from app.training.data_loader_factory import should_use_streaming

        assert should_use_streaming(None, None) is False
        assert should_use_streaming('', None) is False
        assert should_use_streaming([], None) is False

    def test_small_file(self):
        """Test returns False for small file."""
        from app.training.data_loader_factory import should_use_streaming

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            f.write(b'x' * 1000)  # 1KB
            f.flush()
            temp_path = f.name

        try:
            result = should_use_streaming(temp_path, None)
            assert result is False
        finally:
            os.unlink(temp_path)

    def test_large_file_exceeds_threshold(self):
        """Test returns True for large file exceeding threshold."""
        from app.training.data_loader_factory import should_use_streaming

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            f.write(b'x' * 1000)
            f.flush()
            temp_path = f.name

        try:
            # Use a very low threshold so the file exceeds it
            result = should_use_streaming(temp_path, None, threshold_bytes=500)
            assert result is True
        finally:
            os.unlink(temp_path)

    def test_list_of_paths(self):
        """Test handles list of paths."""
        from app.training.data_loader_factory import should_use_streaming

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f1:
            f1.write(b'x' * 500)
            f1.flush()
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f2:
            f2.write(b'x' * 600)
            f2.flush()
            path2 = f2.name

        try:
            # Combined size 1100, threshold 1000 -> True
            result = should_use_streaming([path1, path2], None, threshold_bytes=1000)
            assert result is True
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_directory_with_npz_files(self):
        """Test handles data_dir with .npz files."""
        from app.training.data_loader_factory import should_use_streaming

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some .npz files
            for i in range(3):
                path = os.path.join(tmpdir, f'data_{i}.npz')
                with open(path, 'wb') as f:
                    f.write(b'x' * 400)

            # Combined 1200 bytes, threshold 1000 -> True
            result = should_use_streaming(None, tmpdir, threshold_bytes=1000)
            assert result is True


# =============================================================================
# Test infer_config_key_from_path
# =============================================================================

class TestInferConfigKeyFromPath:
    """Tests for infer_config_key_from_path function."""

    def test_hex8_2p(self):
        """Test inferring hex8_2p."""
        from app.training.data_loader_factory import infer_config_key_from_path

        assert infer_config_key_from_path('/path/hex8_2p_games.npz') == 'hex8_2p'
        assert infer_config_key_from_path('/path/canonical_hex8_2p.npz') == 'hex8_2p'
        assert infer_config_key_from_path('/path/training_hex8_2p_v2.npz') == 'hex8_2p'

    def test_square8_4p(self):
        """Test inferring square8_4p."""
        from app.training.data_loader_factory import infer_config_key_from_path

        assert infer_config_key_from_path('/path/square8_4p.npz') == 'square8_4p'
        assert infer_config_key_from_path('square8_4p_selfplay.npz') == 'square8_4p'

    def test_hexagonal_3p(self):
        """Test inferring hexagonal_3p."""
        from app.training.data_loader_factory import infer_config_key_from_path

        assert infer_config_key_from_path('/data/hexagonal_3p.npz') == 'hexagonal_3p'

    def test_square19_2p(self):
        """Test inferring square19_2p."""
        from app.training.data_loader_factory import infer_config_key_from_path

        assert infer_config_key_from_path('/models/square19_2p_large.npz') == 'square19_2p'

    def test_no_match(self):
        """Test returns None for non-matching paths."""
        from app.training.data_loader_factory import infer_config_key_from_path

        assert infer_config_key_from_path('/path/random_data.npz') is None
        assert infer_config_key_from_path('/path/training.npz') is None
        assert infer_config_key_from_path('/path/') is None

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        from app.training.data_loader_factory import infer_config_key_from_path

        assert infer_config_key_from_path('/path/HEX8_2P.npz') == 'hex8_2p'
        assert infer_config_key_from_path('/path/Square8_3P.NPZ') == 'square8_3p'


# =============================================================================
# Test compute_curriculum_file_weights
# =============================================================================

class TestComputeCurriculumFileWeights:
    """Tests for compute_curriculum_file_weights function."""

    def test_empty_paths(self):
        """Test returns empty dict for empty paths."""
        from app.training.data_loader_factory import compute_curriculum_file_weights

        result = compute_curriculum_file_weights([], {})
        assert result == {}

    def test_matching_weights(self):
        """Test applies matching curriculum weights."""
        from app.training.data_loader_factory import compute_curriculum_file_weights

        paths = ['/data/hex8_2p.npz', '/data/square8_4p.npz']
        weights = {'hex8_2p': 1.5, 'square8_4p': 0.8}

        result = compute_curriculum_file_weights(paths, weights)

        assert result['/data/hex8_2p.npz'] == 1.5
        assert result['/data/square8_4p.npz'] == 0.8

    def test_default_weight_for_unknown(self):
        """Test uses default weight for unknown config keys."""
        from app.training.data_loader_factory import compute_curriculum_file_weights

        paths = ['/data/hex8_2p.npz', '/data/unknown.npz']
        weights = {'hex8_2p': 2.0}

        result = compute_curriculum_file_weights(paths, weights, default_weight=1.0)

        assert result['/data/hex8_2p.npz'] == 2.0
        assert result['/data/unknown.npz'] == 1.0

    def test_missing_from_weights_uses_default(self):
        """Test uses default when config key not in weights dict."""
        from app.training.data_loader_factory import compute_curriculum_file_weights

        paths = ['/data/hex8_2p.npz', '/data/square8_3p.npz']
        weights = {'hex8_2p': 1.5}  # square8_3p not in weights

        result = compute_curriculum_file_weights(paths, weights, default_weight=0.5)

        assert result['/data/hex8_2p.npz'] == 1.5
        assert result['/data/square8_3p.npz'] == 0.5


# =============================================================================
# Test collect_data_paths
# =============================================================================

class TestCollectDataPaths:
    """Tests for collect_data_paths function."""

    def test_single_path(self):
        """Test collects single path."""
        from app.training.data_loader_factory import collect_data_paths

        result = collect_data_paths('/data/file.npz', None)
        assert result == ['/data/file.npz']

    def test_list_of_paths(self):
        """Test collects list of paths."""
        from app.training.data_loader_factory import collect_data_paths

        paths = ['/data/a.npz', '/data/b.npz']
        result = collect_data_paths(paths, None)
        assert result == paths

    def test_data_dir(self):
        """Test collects from data directory."""
        from app.training.data_loader_factory import collect_data_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .npz files
            for name in ['file_a.npz', 'file_b.npz', 'file_c.npz']:
                Path(tmpdir, name).touch()

            result = collect_data_paths(None, tmpdir)

            assert len(result) == 3
            assert all(p.endswith('.npz') for p in result)

    def test_removes_duplicates(self):
        """Test removes duplicate paths while preserving order."""
        from app.training.data_loader_factory import collect_data_paths

        paths = ['/data/a.npz', '/data/b.npz', '/data/a.npz']
        result = collect_data_paths(paths, None)
        assert result == ['/data/a.npz', '/data/b.npz']

    def test_empty_inputs(self):
        """Test returns empty list for no inputs."""
        from app.training.data_loader_factory import collect_data_paths

        assert collect_data_paths(None, None) == []
        assert collect_data_paths('', None) == []


# =============================================================================
# Test validate_dataset_metadata
# =============================================================================

class TestValidateDatasetMetadata:
    """Tests for validate_dataset_metadata function."""

    def test_nonexistent_file(self):
        """Test returns empty dict for nonexistent file."""
        from app.training.data_loader_factory import validate_dataset_metadata

        result = validate_dataset_metadata(
            '/nonexistent/path.npz',
            config_history_length=8,
            config_feature_version=2,
        )
        assert result == {}

    def test_extracts_in_channels(self):
        """Test extracts in_channels from features shape."""
        from app.training.data_loader_factory import validate_dataset_metadata

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name

        try:
            # Create NPZ with features array
            features = np.zeros((100, 64, 8, 8), dtype=np.float32)
            np.savez(temp_path, features=features)

            result = validate_dataset_metadata(
                temp_path,
                config_history_length=8,
                config_feature_version=2,
            )

            assert result['in_channels'] == 64
        finally:
            os.unlink(temp_path)

    def test_validates_history_length_mismatch(self):
        """Test raises error on history_length mismatch."""
        from app.training.data_loader_factory import validate_dataset_metadata

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name

        try:
            np.savez(temp_path, history_length=np.array(8))

            with pytest.raises(ValueError, match="history_length"):
                validate_dataset_metadata(
                    temp_path,
                    config_history_length=16,  # Mismatch!
                    config_feature_version=2,
                )
        finally:
            os.unlink(temp_path)

    def test_validates_feature_version_mismatch(self):
        """Test raises error on feature_version mismatch."""
        from app.training.data_loader_factory import validate_dataset_metadata

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name

        try:
            np.savez(temp_path, feature_version=np.array(2))

            with pytest.raises(ValueError, match="feature_version"):
                validate_dataset_metadata(
                    temp_path,
                    config_history_length=8,
                    config_feature_version=3,  # Mismatch!
                )
        finally:
            os.unlink(temp_path)

    def test_validates_globals_dimension(self):
        """Test raises error on wrong globals dimension."""
        from app.training.data_loader_factory import validate_dataset_metadata

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name

        try:
            globals_arr = np.zeros((100, 30), dtype=np.float32)  # Wrong dim!
            np.savez(temp_path, globals=globals_arr)

            with pytest.raises(ValueError, match="globals dimension"):
                validate_dataset_metadata(
                    temp_path,
                    config_history_length=8,
                    config_feature_version=2,
                )
        finally:
            os.unlink(temp_path)

    def test_validates_legacy_policy_encoding(self):
        """Test raises error for legacy policy encoding with v3/v4."""
        from app.training.data_loader_factory import validate_dataset_metadata

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name

        try:
            np.savez(temp_path, policy_encoding=np.array('legacy_max_n'))

            with pytest.raises(ValueError, match="legacy MAX_N"):
                validate_dataset_metadata(
                    temp_path,
                    config_history_length=8,
                    config_feature_version=2,
                    model_version='v3',
                )
        finally:
            os.unlink(temp_path)


# =============================================================================
# Test create_streaming_loaders
# =============================================================================

class TestCreateStreamingLoaders:
    """Tests for create_streaming_loaders function."""

    def test_empty_paths_returns_empty_result(self):
        """Test returns empty result for empty paths."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            create_streaming_loaders,
        )

        config = DataLoaderConfig()
        result = create_streaming_loaders(config, [])

        assert result.train_loader is None
        assert result.val_loader is None
        assert result.use_streaming is True

    @patch('app.training.data_loader_factory.get_sample_count')
    def test_no_samples_returns_empty(self, mock_count):
        """Test returns empty result when no samples found."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            create_streaming_loaders,
        )

        mock_count.return_value = 0

        config = DataLoaderConfig()
        result = create_streaming_loaders(config, ['/data.npz'])

        assert result.train_loader is None
        assert result.val_loader is None

    @patch('app.training.data_loader_factory.os.path.exists')
    @patch('app.training.data_loader_factory.get_sample_count')
    @patch('app.training.data_loader_factory.StreamingDataLoader')
    def test_creates_streaming_loaders(self, mock_loader_cls, mock_count, mock_exists):
        """Test creates streaming loaders with proper config."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            create_streaming_loaders,
        )

        mock_exists.return_value = True
        mock_count.return_value = 10000
        mock_loader = MagicMock()
        mock_loader.has_multi_player_values = False
        mock_loader_cls.return_value = mock_loader

        config = DataLoaderConfig(
            batch_size=256,
            seed=42,
            policy_size=512,
        )
        result = create_streaming_loaders(config, ['/data.npz'])

        assert result.train_size == 8000  # 80% of 10000
        assert result.val_size == 2000  # 20% of 10000
        assert result.use_streaming is True

    @patch('app.training.data_loader_factory.os.path.exists')
    @patch('app.training.data_loader_factory.get_sample_count')
    @patch('app.training.data_loader_factory.WeightedStreamingDataLoader')
    @patch('app.training.data_loader_factory.StreamingDataLoader')
    def test_uses_weighted_loader_for_quality(
        self, mock_base_loader, mock_weighted_loader, mock_count, mock_exists
    ):
        """Test uses WeightedStreamingDataLoader for non-uniform sampling."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            create_streaming_loaders,
        )

        mock_exists.return_value = True
        mock_count.return_value = 10000
        mock_loader = MagicMock()
        mock_loader.has_multi_player_values = False
        mock_weighted_loader.return_value = mock_loader
        mock_base_loader.return_value = mock_loader

        config = DataLoaderConfig(
            sampling_weights='quality',
        )
        result = create_streaming_loaders(config, ['/data.npz'])

        # WeightedStreamingDataLoader should be used for train
        mock_weighted_loader.assert_called()
        assert result.train_loader is not None


# =============================================================================
# Test create_standard_loaders
# =============================================================================

class TestCreateStandardLoaders:
    """Tests for create_standard_loaders function."""

    @patch('app.training.data_loader_factory.RingRiftDataset')
    def test_creates_dataset(self, mock_dataset_cls):
        """Test creates RingRiftDataset with proper config."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            create_standard_loaders,
        )

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1000)
        mock_dataset.has_multi_player_values = False
        mock_dataset.spatial_shape = (8, 8)
        mock_dataset_cls.return_value = mock_dataset

        config = DataLoaderConfig(
            board_type=BoardType.SQUARE8,
            augment_hex_symmetry=False,
            multi_player=False,
            filter_empty_policies=True,
        )

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, features=np.zeros((100, 64, 8, 8)))
            temp_path = f.name

        try:
            result = create_standard_loaders(config, temp_path)

            mock_dataset_cls.assert_called_once()
            assert result.train_size == 800  # 80% of 1000
            assert result.val_size == 200  # 20% of 1000
            assert result.use_streaming is False
        finally:
            os.unlink(temp_path)

    @patch('app.training.data_loader_factory.WeightedRingRiftDataset')
    def test_uses_weighted_dataset(self, mock_weighted_cls):
        """Test uses WeightedRingRiftDataset for non-uniform sampling."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            create_standard_loaders,
        )

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1000)
        mock_dataset.has_multi_player_values = False
        mock_dataset.sample_weights = np.ones(1000)
        mock_dataset.spatial_shape = (8, 8)  # Add spatial_shape to avoid unpacking error
        mock_weighted_cls.return_value = mock_dataset

        config = DataLoaderConfig(
            sampling_weights='quality',
        )

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, features=np.zeros((100, 64, 8, 8)))
            temp_path = f.name

        try:
            result = create_standard_loaders(config, temp_path)

            mock_weighted_cls.assert_called_once()
            # Should use weighted sampling
            assert result.train_sampler is not None or result.train_loader is not None
        finally:
            os.unlink(temp_path)

    @patch('app.training.data_loader_factory.RingRiftDataset')
    def test_empty_dataset_returns_empty_result(self, mock_dataset_cls):
        """Test returns empty result for empty dataset."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            create_standard_loaders,
        )

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        mock_dataset_cls.return_value = mock_dataset

        config = DataLoaderConfig()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, features=np.zeros((0, 64, 8, 8)))
            temp_path = f.name

        try:
            result = create_standard_loaders(config, temp_path)

            assert result.train_loader is None
            assert result.val_loader is None
        finally:
            os.unlink(temp_path)


# =============================================================================
# Test create_data_loaders
# =============================================================================

class TestCreateDataLoaders:
    """Tests for create_data_loaders main entry point."""

    @patch('app.training.data_loader_factory.create_standard_loaders')
    @patch('app.training.data_loader_factory.should_use_streaming')
    def test_routes_to_standard_for_small_data(
        self, mock_should_stream, mock_create_standard
    ):
        """Test routes to standard loaders for small data."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            DataLoaderResult,
            create_data_loaders,
        )

        mock_should_stream.return_value = False
        mock_create_standard.return_value = DataLoaderResult(
            train_size=800,
            val_size=200,
        )

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, features=np.zeros((100, 64, 8, 8)))
            temp_path = f.name

        try:
            config = DataLoaderConfig(data_path=temp_path)
            result = create_data_loaders(config)

            mock_create_standard.assert_called_once()
            assert result.train_size == 800
        finally:
            os.unlink(temp_path)

    @patch('app.training.data_loader_factory.create_streaming_loaders')
    @patch('app.training.data_loader_factory.collect_data_paths')
    def test_routes_to_streaming_when_forced(
        self, mock_collect, mock_create_streaming
    ):
        """Test routes to streaming loaders when use_streaming=True."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            DataLoaderResult,
            create_data_loaders,
        )

        mock_collect.return_value = ['/data.npz']
        mock_create_streaming.return_value = DataLoaderResult(
            train_size=8000,
            val_size=2000,
            use_streaming=True,
        )

        config = DataLoaderConfig(
            data_path='/data.npz',
            use_streaming=True,
        )
        result = create_data_loaders(config)

        mock_create_streaming.assert_called_once()
        assert result.use_streaming is True

    def test_missing_data_path_returns_empty(self):
        """Test returns empty result for missing data path."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            create_data_loaders,
        )

        config = DataLoaderConfig(data_path='/nonexistent/path.npz')
        result = create_data_loaders(config)

        assert result.train_loader is None
        assert result.val_loader is None

    @patch('app.training.data_loader_factory.create_standard_loaders')
    @patch('app.training.data_loader_factory.should_use_streaming')
    def test_uses_first_path_from_list_for_standard(
        self, mock_should_stream, mock_create_standard
    ):
        """Test uses first path from list for standard loaders."""
        from app.training.data_loader_factory import (
            DataLoaderConfig,
            DataLoaderResult,
            create_data_loaders,
        )

        mock_should_stream.return_value = False
        mock_create_standard.return_value = DataLoaderResult()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, features=np.zeros((100, 64, 8, 8)))
            temp_path = f.name

        try:
            config = DataLoaderConfig(data_path=[temp_path, '/other.npz'])
            create_data_loaders(config)

            # Should pass the first path to create_standard_loaders
            call_args = mock_create_standard.call_args
            assert call_args[0][1] == temp_path
        finally:
            os.unlink(temp_path)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_corrupt_npz_file(self):
        """Test handles corrupt NPZ file gracefully."""
        from app.training.data_loader_factory import validate_dataset_metadata

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            f.write(b'not a valid npz file')
            temp_path = f.name

        try:
            # Should not crash, just return empty metadata
            result = validate_dataset_metadata(
                temp_path,
                config_history_length=8,
                config_feature_version=2,
            )
            assert result == {}
        finally:
            os.unlink(temp_path)

    def test_curriculum_weights_with_empty_dict(self):
        """Test curriculum weights with empty dict."""
        from app.training.data_loader_factory import compute_curriculum_file_weights

        result = compute_curriculum_file_weights(
            ['/data/hex8_2p.npz'],
            {},  # Empty weights dict
            default_weight=1.0,
        )

        # Should use default weight
        assert result['/data/hex8_2p.npz'] == 1.0

    def test_infer_config_with_unusual_path(self):
        """Test config inference with unusual path patterns."""
        from app.training.data_loader_factory import infer_config_key_from_path

        # Should handle paths without directory
        assert infer_config_key_from_path('hex8_2p.npz') == 'hex8_2p'

        # Should handle Windows-style paths
        assert infer_config_key_from_path('C:\\data\\hex8_2p.npz') == 'hex8_2p'

        # Should handle multiple underscores
        assert infer_config_key_from_path('/a/b/data_hex8_2p_v2.npz') == 'hex8_2p'
