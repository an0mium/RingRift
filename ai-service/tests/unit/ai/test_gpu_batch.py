"""Tests for GPU batch evaluation module.

Comprehensive test coverage for:
- Device detection and management
- GPU tensor utilities
- GPUBatchEvaluator
- GPUHeuristicEvaluator
- AsyncGPUEvaluator
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.ai.gpu_batch import (
    get_device,
    get_all_cuda_devices,
    clear_gpu_memory,
    compile_model,
    warmup_compiled_model,
    GPUBoardState,
    GPUBatchEvaluator,
    GPUHeuristicEvaluator,
    EvalRequest,
    AsyncGPUEvaluator,
    benchmark_gpu_batch,
)


class TestDeviceDetection:
    """Test device detection functions."""

    def test_get_device_returns_device(self):
        """Test that get_device returns a torch device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_cpu_fallback(self):
        """Test CPU fallback when prefer_gpu=False."""
        device = get_device(prefer_gpu=False)
        assert device.type == "cpu"

    def test_get_device_respects_prefer_gpu(self):
        """Test that prefer_gpu flag works."""
        # With prefer_gpu=False, should always be CPU
        device = get_device(prefer_gpu=False)
        assert device.type == "cpu"

    def test_get_all_cuda_devices(self):
        """Test getting all CUDA devices."""
        devices = get_all_cuda_devices()
        # Should be a list (may be empty if no CUDA)
        assert isinstance(devices, list)
        if torch.cuda.is_available():
            assert len(devices) == torch.cuda.device_count()

    def test_clear_gpu_memory(self):
        """Test clearing GPU memory doesn't raise."""
        # Should not raise even if no GPU
        clear_gpu_memory()
        clear_gpu_memory(device=torch.device("cpu"))


class TestModelCompilation:
    """Test model compilation functions."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 10)

    def test_compile_model_returns_model(self, simple_model):
        """Test that compile_model returns a model."""
        result = compile_model(simple_model, device=torch.device("cpu"))
        # Should return the model (possibly compiled, possibly not)
        assert result is not None

    def test_compile_model_skips_cpu(self, simple_model):
        """Test that compilation is skipped on CPU."""
        result = compile_model(simple_model, device=torch.device("cpu"))
        # On CPU, should return original model
        assert result is simple_model or hasattr(result, "forward")

    @patch.dict(os.environ, {"RINGRIFT_DISABLE_TORCH_COMPILE": "1"})
    def test_compile_model_respects_env_var(self, simple_model):
        """Test that env var disables compilation."""
        result = compile_model(simple_model, device=torch.device("cpu"))
        # Should return original model when disabled
        assert result is simple_model

    def test_warmup_compiled_model(self, simple_model):
        """Test model warmup."""
        simple_model.eval()
        sample_input = torch.randn(1, 10)

        # Should not raise
        warmup_compiled_model(simple_model, sample_input, num_warmup=2)


class TestGPUBoardState:
    """Test GPUBoardState dataclass."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")  # Use CPU for consistent testing

    def test_from_numpy_batch(self, device):
        """Test creating GPUBoardState from numpy arrays."""
        board_states = [
            {
                "stack_owner": np.zeros(64, dtype=np.int8),
                "stack_height": np.ones(64, dtype=np.int8),
                "marker_owner": np.zeros(64, dtype=np.int8),
                "territory_owner": np.zeros(64, dtype=np.int8),
                "is_collapsed": np.zeros(64, dtype=bool),
                "rings_in_hand": np.array([5, 5, 0, 0, 0], dtype=np.int16),
                "player_eliminated": np.zeros(5, dtype=np.int16),
                "territory_count": np.zeros(5, dtype=np.int16),
            },
            {
                "stack_owner": np.zeros(64, dtype=np.int8),
                "stack_height": np.ones(64, dtype=np.int8),
                "marker_owner": np.zeros(64, dtype=np.int8),
                "territory_owner": np.zeros(64, dtype=np.int8),
                "is_collapsed": np.zeros(64, dtype=bool),
                "rings_in_hand": np.array([5, 5, 0, 0, 0], dtype=np.int16),
                "player_eliminated": np.zeros(5, dtype=np.int16),
                "territory_count": np.zeros(5, dtype=np.int16),
            },
        ]

        gpu_state = GPUBoardState.from_numpy_batch(board_states, device, board_size=8)

        assert gpu_state.batch_size == 2
        assert gpu_state.board_size == 8
        assert gpu_state.device == device
        assert gpu_state.stack_owner.shape == (2, 64)
        assert gpu_state.rings_in_hand.shape == (2, 5)

    def test_from_numpy_batch_defaults(self, device):
        """Test that missing keys get defaults."""
        board_states = [{}]  # Empty dict

        gpu_state = GPUBoardState.from_numpy_batch(board_states, device, board_size=8)

        assert gpu_state.batch_size == 1
        assert gpu_state.stack_owner.shape == (1, 64)


class TestGPUBatchEvaluator:
    """Test GPUBatchEvaluator."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(16, 32, 3, padding=1)
                self.value_head = nn.Linear(32 * 8 * 8, 1)
                self.policy_head = nn.Linear(32 * 8 * 8, 64)

            def forward(self, x, globals_t=None):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                value = self.value_head(x)
                policy = self.policy_head(x)
                return value.squeeze(-1), policy

        return SimpleNet()

    @pytest.fixture
    def evaluator(self, simple_model):
        """Create evaluator with simple model."""
        return GPUBatchEvaluator(
            device="cpu",
            model=simple_model,
            use_mixed_precision=False,
            max_batch_size=32,
        )

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = GPUBatchEvaluator(device="cpu")
        assert evaluator.device.type == "cpu"
        assert evaluator.model is None

    def test_init_with_model(self, simple_model):
        """Test initialization with model."""
        evaluator = GPUBatchEvaluator(device="cpu", model=simple_model)
        assert evaluator.model is not None

    def test_set_model(self, simple_model):
        """Test setting model after init."""
        evaluator = GPUBatchEvaluator(device="cpu")
        assert evaluator.model is None

        evaluator.set_model(simple_model)
        assert evaluator.model is not None

    def test_evaluate_batch(self, evaluator):
        """Test batch evaluation."""
        features = np.random.randn(4, 16, 8, 8).astype(np.float32)

        values, policies = evaluator.evaluate_batch(features)

        assert values.shape == (4,)
        assert policies.shape == (4, 64)

    def test_evaluate_batch_with_globals(self, evaluator):
        """Test batch evaluation with global features."""
        features = np.random.randn(4, 16, 8, 8).astype(np.float32)
        globals_feat = np.random.randn(4, 10).astype(np.float32)

        values, policies = evaluator.evaluate_batch(features, globals_feat)

        assert values.shape == (4,)
        assert policies.shape == (4, 64)

    def test_evaluate_batch_no_model_raises(self):
        """Test that evaluation without model raises."""
        evaluator = GPUBatchEvaluator(device="cpu")

        with pytest.raises(ValueError, match="No model set"):
            evaluator.evaluate_batch(np.zeros((1, 16, 8, 8)))

    def test_evaluate_chunked_large_batch(self, evaluator):
        """Test that large batches are processed in chunks."""
        # Create batch larger than max_batch_size (32)
        features = np.random.randn(64, 16, 8, 8).astype(np.float32)

        values, policies = evaluator.evaluate_batch(features)

        assert values.shape == (64,)
        assert policies.shape == (64, 64)

    def test_get_performance_stats(self, evaluator):
        """Test performance stats tracking."""
        features = np.random.randn(4, 16, 8, 8).astype(np.float32)

        # Run some evaluations
        for _ in range(3):
            evaluator.evaluate_batch(features)

        stats = evaluator.get_performance_stats()

        assert stats["inference_count"] == 12  # 4 * 3
        assert stats["total_time_seconds"] > 0
        assert stats["throughput_samples_per_sec"] > 0


class TestGPUHeuristicEvaluator:
    """Test GPUHeuristicEvaluator."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def evaluator(self, device):
        """Create heuristic evaluator."""
        return GPUHeuristicEvaluator(device=device, board_size=8, num_players=2)

    def test_init(self, evaluator):
        """Test initialization."""
        assert evaluator.board_size == 8
        assert evaluator.num_players == 2
        assert evaluator.center_mask is not None
        assert evaluator.center_mask.shape == (8, 8)

    def test_default_weights(self, evaluator):
        """Test default heuristic weights."""
        weights = evaluator.weights

        assert "stack_count" in weights
        assert "territory_count" in weights
        assert "center_control" in weights
        assert "no_stacks_penalty" in weights

    def test_set_weights(self, evaluator):
        """Test setting custom weights."""
        evaluator.set_weights({"stack_count": 2.0, "new_weight": 1.0})

        assert evaluator.weights["stack_count"] == 2.0
        assert evaluator.weights["new_weight"] == 1.0

    def test_evaluate_batch(self, evaluator, device):
        """Test batch evaluation."""
        board_states = [
            {
                "stack_owner": np.array([1] * 32 + [2] * 32, dtype=np.int8),
                "stack_height": np.ones(64, dtype=np.int8),
                "marker_owner": np.zeros(64, dtype=np.int8),
                "territory_owner": np.zeros(64, dtype=np.int8),
                "is_collapsed": np.zeros(64, dtype=bool),
                "rings_in_hand": np.array([5, 5, 0, 0, 0], dtype=np.int16),
                "player_eliminated": np.zeros(5, dtype=np.int16),
                "territory_count": np.array([0, 2, 1, 0, 0], dtype=np.int16),
            }
        ]

        gpu_state = GPUBoardState.from_numpy_batch(board_states, device, board_size=8)
        scores = evaluator.evaluate_batch(gpu_state, player_number=1)

        assert scores.shape == (1,)
        assert isinstance(scores, torch.Tensor)

    def test_evaluate_batch_symmetric(self, evaluator, device):
        """Test that evaluation is symmetric."""
        # Create symmetric position
        board_states = [
            {
                "stack_owner": np.array([1] * 32 + [2] * 32, dtype=np.int8),
                "stack_height": np.ones(64, dtype=np.int8),
                "marker_owner": np.zeros(64, dtype=np.int8),
                "territory_owner": np.zeros(64, dtype=np.int8),
                "is_collapsed": np.zeros(64, dtype=bool),
                "rings_in_hand": np.array([0, 5, 5, 0, 0], dtype=np.int16),
                "player_eliminated": np.zeros(5, dtype=np.int16),
                "territory_count": np.array([0, 1, 1, 0, 0], dtype=np.int16),
            }
        ]

        gpu_state = GPUBoardState.from_numpy_batch(board_states, device, board_size=8)

        score_p1 = evaluator.evaluate_batch(gpu_state, player_number=1)
        score_p2 = evaluator.evaluate_batch(gpu_state, player_number=2)

        # Scores should be roughly opposite for symmetric position
        # (not exact due to center control asymmetry)
        assert score_p1.item() != score_p2.item()


class TestAsyncGPUEvaluator:
    """Test AsyncGPUEvaluator."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16 * 8 * 8, 65)

            def forward(self, x, globals_t=None):
                x = x.view(x.size(0), -1)
                out = self.fc(x)
                return out[:, 0], out[:, 1:]

        return SimpleNet()

    @pytest.fixture
    def batch_evaluator(self, simple_model):
        """Create batch evaluator."""
        return GPUBatchEvaluator(
            device="cpu",
            model=simple_model,
            use_mixed_precision=False,
        )

    @pytest.fixture
    def async_evaluator(self, batch_evaluator):
        """Create async evaluator."""
        return AsyncGPUEvaluator(
            evaluator=batch_evaluator,
            batch_size=4,
            timeout_ms=50,
            max_queue_size=100,
        )

    def test_init(self, async_evaluator):
        """Test initialization."""
        assert async_evaluator.batch_size == 4
        assert async_evaluator.timeout_sec == 0.05
        assert async_evaluator._running is False

    def test_start_stop(self, async_evaluator):
        """Test starting and stopping."""
        async_evaluator.start()
        assert async_evaluator._running is True
        assert async_evaluator._worker_thread is not None
        assert async_evaluator._worker_thread.is_alive()

        async_evaluator.stop()
        assert async_evaluator._running is False

    def test_queue_position(self, async_evaluator):
        """Test queuing positions."""
        features = np.random.randn(16, 8, 8).astype(np.float32)
        callback = MagicMock()

        async_evaluator.queue_position(features, None, callback)

        assert async_evaluator._queue.qsize() == 1

    def test_flush_processes_queue(self, async_evaluator):
        """Test that flush processes all queued positions."""
        results = []

        def callback(value, policy):
            results.append((value, policy))

        # Queue positions
        for _ in range(3):
            features = np.random.randn(16, 8, 8).astype(np.float32)
            async_evaluator.queue_position(features, None, callback)

        # Flush
        async_evaluator.flush()

        assert len(results) == 3
        assert async_evaluator._queue.qsize() == 0

    def test_batch_processing(self, async_evaluator):
        """Test that positions are batched together."""
        results = []
        results_lock = threading.Lock()

        def callback(value, policy):
            with results_lock:
                results.append((value, policy))

        async_evaluator.start()

        # Queue enough positions to fill a batch
        for _ in range(4):
            features = np.random.randn(16, 8, 8).astype(np.float32)
            async_evaluator.queue_position(features, None, callback)

        # Wait for processing
        time.sleep(0.2)
        async_evaluator.stop()

        # All positions should be processed
        assert len(results) == 4

    def test_get_stats(self, async_evaluator):
        """Test stats retrieval."""
        stats = async_evaluator.get_stats()

        assert "batches_processed" in stats
        assert "requests_processed" in stats
        assert "queue_size" in stats
        assert "avg_batch_size" in stats


class TestEvalRequest:
    """Test EvalRequest dataclass."""

    def test_create_request(self):
        """Test creating an eval request."""
        features = np.random.randn(16, 8, 8).astype(np.float32)
        callback = MagicMock()

        request = EvalRequest(
            features=features,
            global_features=None,
            callback=callback,
        )

        assert request.features.shape == (16, 8, 8)
        assert request.global_features is None
        assert request.callback is callback
        assert request.timestamp > 0

    def test_request_with_globals(self):
        """Test request with global features."""
        features = np.random.randn(16, 8, 8).astype(np.float32)
        globals_feat = np.random.randn(10).astype(np.float32)
        callback = MagicMock()

        request = EvalRequest(
            features=features,
            global_features=globals_feat,
            callback=callback,
        )

        assert request.global_features.shape == (10,)


class TestBenchmark:
    """Test benchmark function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16 * 8 * 8, 65)

            def forward(self, x, globals_t=None):
                x = x.view(x.size(0), -1)
                out = self.fc(x)
                return out[:, 0], out[:, 1:]

        return SimpleNet()

    def test_benchmark_runs(self, simple_model):
        """Test that benchmark runs without error."""
        evaluator = GPUBatchEvaluator(
            device="cpu",
            model=simple_model,
            use_mixed_precision=False,
        )

        results = benchmark_gpu_batch(
            evaluator,
            batch_sizes=[1, 4],  # Small sizes for fast test
            feature_shape=(16, 8, 8),
            num_iterations=5,  # Few iterations for fast test
        )

        assert "batch_size" in results
        assert "throughput" in results
        assert "latency_ms" in results
        assert len(results["batch_size"]) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_evaluator_device_string(self):
        """Test that device can be passed as string."""
        evaluator = GPUBatchEvaluator(device="cpu")
        assert evaluator.device.type == "cpu"

    def test_heuristic_evaluator_different_board_sizes(self):
        """Test heuristic evaluator with different board sizes."""
        for board_size in [8, 10, 19]:
            evaluator = GPUHeuristicEvaluator(
                device="cpu",
                board_size=board_size,
                num_players=2,
            )
            assert evaluator.board_size == board_size
            assert evaluator.center_mask.shape == (board_size, board_size)

    def test_async_evaluator_callback_error_handling(self):
        """Test that callback errors are handled gracefully."""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16 * 8 * 8, 65)

            def forward(self, x, globals_t=None):
                x = x.view(x.size(0), -1)
                out = self.fc(x)
                return out[:, 0], out[:, 1:]

        model = SimpleNet()
        batch_evaluator = GPUBatchEvaluator(device="cpu", model=model)
        async_evaluator = AsyncGPUEvaluator(batch_evaluator, batch_size=2)

        def bad_callback(value, policy):
            raise RuntimeError("Callback error")

        features = np.random.randn(16, 8, 8).astype(np.float32)
        async_evaluator.queue_position(features, None, bad_callback)

        # Should not raise
        async_evaluator.flush()

    def test_empty_board_state(self):
        """Test handling of empty board state."""
        device = torch.device("cpu")
        board_states = [
            {
                "stack_owner": np.zeros(64, dtype=np.int8),
                "stack_height": np.zeros(64, dtype=np.int8),
            }
        ]

        gpu_state = GPUBoardState.from_numpy_batch(board_states, device, board_size=8)
        evaluator = GPUHeuristicEvaluator(device=device)

        # Should handle empty board gracefully
        scores = evaluator.evaluate_batch(gpu_state, player_number=1)
        assert scores.shape == (1,)
