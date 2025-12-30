"""Unit tests for unified distributed training module.

Tests distributed training infrastructure including:
- Configuration dataclasses
- Gradient compression
- Async SGD
- Distributed trainer lifecycle

This module is critical for multi-GPU/multi-node training (used by 15 files).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestUnifiedDistributedConfig:
    """Tests for UnifiedDistributedConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from app.training.distributed_unified import UnifiedDistributedConfig

        config = UnifiedDistributedConfig()
        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.backend == "nccl"
        assert config.master_addr == "localhost"
        assert config.master_port == 29500
        assert config.init_method == "env://"

    def test_ddp_settings(self):
        """Test DDP-related settings."""
        from app.training.distributed_unified import UnifiedDistributedConfig

        config = UnifiedDistributedConfig()
        assert config.gradient_sync_every == 1
        assert config.use_sync_batchnorm is True
        assert config.find_unused_parameters is False
        assert config.broadcast_buffers is True
        assert config.bucket_cap_mb == 25

    def test_compression_settings(self):
        """Test gradient compression settings."""
        from app.training.distributed_unified import UnifiedDistributedConfig

        config = UnifiedDistributedConfig()
        assert config.compress_gradients is False
        assert config.compression_ratio == 0.01
        assert config.compression_warmup_steps == 100

    def test_async_sgd_settings(self):
        """Test async SGD settings."""
        from app.training.distributed_unified import UnifiedDistributedConfig

        config = UnifiedDistributedConfig()
        assert config.async_sgd is False
        assert config.max_staleness == 3
        assert config.staleness_lr_decay == 0.9

    def test_fault_tolerance_settings(self):
        """Test fault tolerance settings."""
        from app.training.distributed_unified import UnifiedDistributedConfig

        config = UnifiedDistributedConfig()
        assert config.checkpoint_dir == "data/distributed_checkpoints"
        assert config.checkpoint_interval == 1000
        assert config.auto_resume is True
        assert config.elastic_training is True

    def test_amp_settings(self):
        """Test mixed precision settings."""
        from app.training.distributed_unified import UnifiedDistributedConfig

        config = UnifiedDistributedConfig()
        assert config.use_amp is True
        assert config.amp_dtype == "bfloat16"

    def test_custom_config(self):
        """Test creating config with custom values."""
        from app.training.distributed_unified import UnifiedDistributedConfig

        config = UnifiedDistributedConfig(
            world_size=4,
            rank=2,
            local_rank=2,
            backend="gloo",
            master_addr="192.168.1.1",
            master_port=29501,
            compress_gradients=True,
            use_amp=False,
        )
        assert config.world_size == 4
        assert config.rank == 2
        assert config.backend == "gloo"
        assert config.master_addr == "192.168.1.1"
        assert config.master_port == 29501
        assert config.compress_gradients is True
        assert config.use_amp is False

    def test_auto_detect_backend_empty_string(self):
        """Test backend auto-detection with empty string."""
        from app.training.distributed_unified import UnifiedDistributedConfig

        # Mock torch.cuda.is_available to return False for CPU backend
        with patch("app.training.distributed_unified.get_torch") as mock_torch:
            mock_torch.return_value.cuda.is_available.return_value = False
            config = UnifiedDistributedConfig(backend="")
            assert config.backend == "gloo"

    def test_backend_env_override(self):
        """Test backend override via environment variable."""
        from app.training.distributed_unified import _auto_detect_backend

        with patch.dict(os.environ, {"RINGRIFT_DISTRIBUTED_BACKEND": "nccl"}):
            backend = _auto_detect_backend()
            assert backend == "nccl"


class TestNodeInfo:
    """Tests for NodeInfo dataclass."""

    def test_create_node_info(self):
        """Test creating node info."""
        from app.training.distributed_unified import NodeInfo

        node = NodeInfo(
            node_id="node_0",
            hostname="worker-1",
            rank=0,
            local_rank=0,
            device="cuda:0",
        )
        assert node.node_id == "node_0"
        assert node.hostname == "worker-1"
        assert node.rank == 0
        assert node.device == "cuda:0"
        assert node.status == "active"
        assert node.last_heartbeat == 0.0
        assert node.gradient_delay_ms == 0.0
        assert node.throughput_samples_sec == 0.0

    def test_node_info_with_metrics(self):
        """Test node info with performance metrics."""
        from app.training.distributed_unified import NodeInfo

        node = NodeInfo(
            node_id="node_1",
            hostname="worker-2",
            rank=1,
            local_rank=0,
            device="cuda:0",
            status="active",
            last_heartbeat=1234567890.0,
            gradient_delay_ms=5.5,
            throughput_samples_sec=1000.0,
        )
        assert node.last_heartbeat == 1234567890.0
        assert node.gradient_delay_ms == 5.5
        assert node.throughput_samples_sec == 1000.0


class TestGradientCompressor:
    """Tests for GradientCompressor class."""

    @pytest.fixture
    def mock_torch(self):
        """Create mock torch module."""
        torch_mock = MagicMock()
        torch_mock.topk = MagicMock()
        torch_mock.zeros = MagicMock()
        torch_mock.arange = MagicMock()
        return torch_mock

    def test_create_compressor(self):
        """Test creating gradient compressor."""
        from app.training.distributed_unified import GradientCompressor

        compressor = GradientCompressor(
            compression_ratio=0.01,
            warmup_steps=100,
        )
        assert compressor.compression_ratio == 0.01
        assert compressor.warmup_steps == 100
        assert compressor._step == 0
        assert compressor._error_buffers == {}

    def test_compressor_default_values(self):
        """Test compressor default values."""
        from app.training.distributed_unified import GradientCompressor

        compressor = GradientCompressor()
        assert compressor.compression_ratio == 0.01
        assert compressor.warmup_steps == 100

    def test_compressor_reset(self):
        """Test resetting error buffers."""
        from app.training.distributed_unified import GradientCompressor

        compressor = GradientCompressor()
        compressor._error_buffers = {"param1": MagicMock(), "param2": MagicMock()}
        compressor.reset()
        assert compressor._error_buffers == {}

    def test_compress_during_warmup(self):
        """Test that compression is skipped during warmup."""
        from app.training.distributed_unified import GradientCompressor

        compressor = GradientCompressor(warmup_steps=100)
        assert compressor._step == 0

        # During warmup, compress should return full gradients
        with patch("app.training.distributed_unified._get_torch_distributed") as mock_dist:
            mock_torch = MagicMock()
            mock_torch.arange.return_value = MagicMock()

            # Create mock gradient
            mock_grad = MagicMock()
            mock_grad.flatten.return_value = mock_grad
            mock_grad.numel.return_value = 1000

            mock_dist.return_value = (mock_torch, None, None)
            gradients = {"param1": mock_grad}

            compressed = compressor.compress(gradients)

            # Should have called arange (full gradient)
            mock_torch.arange.assert_called()
            assert compressor._step == 1


class TestAsyncSGD:
    """Tests for AsyncSGD class."""

    def test_create_async_sgd(self):
        """Test creating async SGD."""
        from app.training.distributed_unified import AsyncSGD

        async_sgd = AsyncSGD(
            max_staleness=3,
            learning_rate_decay=0.9,
        )
        assert async_sgd.max_staleness == 3
        assert async_sgd.lr_decay == 0.9
        assert async_sgd._current_step == 0
        assert async_sgd._discarded_count == 0
        assert async_sgd.pending_updates == 0

    def test_async_sgd_default_values(self):
        """Test async SGD default values."""
        from app.training.distributed_unified import AsyncSGD

        async_sgd = AsyncSGD()
        assert async_sgd.max_staleness == 3
        assert async_sgd.lr_decay == 0.9

    def test_push_gradients(self):
        """Test pushing gradients to queue."""
        from app.training.distributed_unified import AsyncSGD

        async_sgd = AsyncSGD()
        gradients = {"param1": MagicMock()}

        async_sgd.push_gradients(gradients, step=0)
        assert async_sgd.pending_updates == 1

        async_sgd.push_gradients(gradients, step=1)
        assert async_sgd.pending_updates == 2

    def test_get_update_empty_queue(self):
        """Test getting update from empty queue."""
        from app.training.distributed_unified import AsyncSGD

        async_sgd = AsyncSGD()
        result = async_sgd.get_update()
        assert result is None

    def test_discarded_count_property(self):
        """Test discarded count property."""
        from app.training.distributed_unified import AsyncSGD

        async_sgd = AsyncSGD()
        assert async_sgd.discarded_count == 0
        async_sgd._discarded_count = 5
        assert async_sgd.discarded_count == 5


class TestUnifiedDistributedTrainer:
    """Tests for UnifiedDistributedTrainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock PyTorch model."""
        model = MagicMock()
        model.state_dict.return_value = {"layer1.weight": MagicMock()}
        model.to.return_value = model
        return model

    @pytest.fixture
    def trainer_config(self):
        """Create test trainer configuration."""
        from app.training.distributed_unified import UnifiedDistributedConfig

        return UnifiedDistributedConfig(
            world_size=1,
            rank=0,
            backend="gloo",
            use_amp=False,
            auto_resume=False,
            checkpoint_dir="/tmp/test_checkpoints",
        )

    def test_create_trainer(self, mock_model, trainer_config):
        """Test creating distributed trainer."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)
        assert trainer.config == trainer_config
        assert trainer._original_model == mock_model
        assert trainer._initialized is False
        assert trainer._step == 0
        assert trainer._epoch == 0

    def test_trainer_default_config(self, mock_model):
        """Test trainer with default config."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model)
        assert trainer.config is not None
        assert trainer.config.world_size == 1

    def test_is_main_process(self, mock_model, trainer_config):
        """Test is_main_process property."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)
        assert trainer.is_main_process is True

        trainer_config.rank = 1
        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)
        assert trainer.is_main_process is False

    def test_step_property(self, mock_model, trainer_config):
        """Test step property."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)
        assert trainer.step == 0
        trainer._step = 100
        assert trainer.step == 100

    def test_model_property(self, mock_model, trainer_config):
        """Test model property."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)
        assert trainer.model == mock_model

    def test_nodes_property(self, mock_model, trainer_config):
        """Test nodes property."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)
        assert trainer.nodes == {}

    def test_get_metrics(self, mock_model, trainer_config):
        """Test get_metrics method."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)
        metrics = trainer.get_metrics()

        assert "step" in metrics
        assert "epoch" in metrics
        assert "rank" in metrics
        assert "world_size" in metrics
        assert "initialized" in metrics
        assert metrics["step"] == 0
        assert metrics["rank"] == 0
        assert metrics["initialized"] is False

    def test_set_optimizer(self, mock_model, trainer_config):
        """Test set_optimizer method."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)
        assert trainer._optimizer is None

        mock_optimizer = MagicMock()
        trainer.set_optimizer(mock_optimizer)
        assert trainer._optimizer == mock_optimizer

    def test_train_step_without_setup_raises(self, mock_model, trainer_config):
        """Test that train_step raises without setup."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            trainer.train_step((MagicMock(), MagicMock()), lambda x, y: x)

    def test_cleanup_not_initialized(self, mock_model, trainer_config):
        """Test cleanup when not initialized."""
        from app.training.distributed_unified import UnifiedDistributedTrainer

        trainer = UnifiedDistributedTrainer(mock_model, trainer_config)
        # Should not raise
        with patch("app.training.distributed_unified._get_torch_distributed") as mock_dist:
            mock_dist_module = MagicMock()
            mock_dist_module.is_initialized.return_value = False
            mock_dist.return_value = (MagicMock(), mock_dist_module, None)
            trainer.cleanup()


class TestCreateDistributedTrainer:
    """Tests for create_distributed_trainer factory function."""

    def test_create_with_defaults(self):
        """Test factory with default values."""
        from app.training.distributed_unified import create_distributed_trainer

        mock_model = MagicMock()
        trainer = create_distributed_trainer(mock_model)

        assert trainer.config.world_size == 1
        assert trainer.config.rank == 0
        assert trainer.config.backend == "nccl"
        assert trainer.config.compress_gradients is False
        assert trainer.config.use_amp is True

    def test_create_with_custom_values(self):
        """Test factory with custom values."""
        from app.training.distributed_unified import create_distributed_trainer

        mock_model = MagicMock()
        trainer = create_distributed_trainer(
            mock_model,
            world_size=4,
            rank=2,
            backend="gloo",
            compress_gradients=True,
            use_amp=False,
        )

        assert trainer.config.world_size == 4
        assert trainer.config.rank == 2
        assert trainer.config.backend == "gloo"
        assert trainer.config.compress_gradients is True
        assert trainer.config.use_amp is False


class TestBackwardsCompatibilityAliases:
    """Tests for backwards compatibility aliases."""

    def test_distributed_config_alias(self):
        """Test DistributedConfig alias."""
        from app.training.distributed_unified import (
            DistributedConfig,
            UnifiedDistributedConfig,
        )

        assert DistributedConfig is UnifiedDistributedConfig

    def test_distributed_trainer_alias(self):
        """Test DistributedTrainer alias."""
        from app.training.distributed_unified import (
            DistributedTrainer,
            UnifiedDistributedTrainer,
        )

        assert DistributedTrainer is UnifiedDistributedTrainer


class TestAutoDetectBackend:
    """Tests for _auto_detect_backend function."""

    def test_auto_detect_with_cuda(self):
        """Test auto-detection with CUDA available."""
        from app.training.distributed_unified import _auto_detect_backend

        with patch.dict(os.environ, {}, clear=True):
            # Remove RINGRIFT_DISTRIBUTED_BACKEND if present
            os.environ.pop("RINGRIFT_DISTRIBUTED_BACKEND", None)

            with patch("app.training.distributed_unified.get_torch") as mock_torch:
                mock_torch.return_value.cuda.is_available.return_value = True
                backend = _auto_detect_backend()
                assert backend == "nccl"

    def test_auto_detect_without_cuda(self):
        """Test auto-detection without CUDA."""
        from app.training.distributed_unified import _auto_detect_backend

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_DISTRIBUTED_BACKEND", None)

            with patch("app.training.distributed_unified.get_torch") as mock_torch:
                mock_torch.return_value.cuda.is_available.return_value = False
                backend = _auto_detect_backend()
                assert backend == "gloo"

    def test_auto_detect_env_override(self):
        """Test env variable override."""
        from app.training.distributed_unified import _auto_detect_backend

        with patch.dict(os.environ, {"RINGRIFT_DISTRIBUTED_BACKEND": "custom_backend"}):
            backend = _auto_detect_backend()
            assert backend == "custom_backend"


class TestTrainerContextManager:
    """Tests for trainer context manager support."""

    def test_context_manager_calls_setup_and_cleanup(self):
        """Test that context manager calls setup and cleanup."""
        from app.training.distributed_unified import (
            UnifiedDistributedConfig,
            UnifiedDistributedTrainer,
        )

        mock_model = MagicMock()
        config = UnifiedDistributedConfig(
            world_size=1,
            rank=0,
            backend="gloo",
            auto_resume=False,
        )

        trainer = UnifiedDistributedTrainer(mock_model, config)

        with patch.object(trainer, "setup") as mock_setup, patch.object(
            trainer, "cleanup"
        ) as mock_cleanup:
            mock_setup.return_value = True
            with trainer:
                mock_setup.assert_called_once()

            mock_cleanup.assert_called_once()


class TestGradientCompressorIntegration:
    """Integration tests for gradient compression."""

    def test_compressor_step_increment_during_warmup(self):
        """Test that compressor step increments during warmup phase."""
        from app.training.distributed_unified import GradientCompressor

        compressor = GradientCompressor(warmup_steps=100)  # Long warmup so we stay in warmup

        with patch("app.training.distributed_unified._get_torch_distributed") as mock_dist:
            mock_torch = MagicMock()
            mock_arange_result = MagicMock()
            mock_torch.arange.return_value = mock_arange_result

            mock_grad = MagicMock()
            mock_flat_grad = MagicMock()
            mock_grad.flatten.return_value = mock_flat_grad
            mock_flat_grad.numel.return_value = 100
            mock_grad.device = "cpu"

            mock_dist.return_value = (mock_torch, None, None)
            gradients = {"param1": mock_grad}

            # Run 5 iterations (all within warmup)
            for i in range(5):
                compressor.compress(gradients)
                assert compressor._step == i + 1


class TestAsyncSGDIntegration:
    """Integration tests for async SGD."""

    def test_gradient_queue_max_size(self):
        """Test that gradient queue respects max size."""
        from app.training.distributed_unified import AsyncSGD

        async_sgd = AsyncSGD(max_staleness=2)
        gradients = {"param1": MagicMock()}

        # Push more gradients than max_staleness + 1
        for i in range(10):
            async_sgd.push_gradients(gradients, step=i)

        # Queue should be capped at max_staleness + 1 = 3
        assert async_sgd.pending_updates <= 3
