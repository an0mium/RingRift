"""Comprehensive tests for dynamic_thresholds.py.

Tests DynamicThreshold, ThresholdManager, and global threshold utilities.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.coordination.dynamic_thresholds import (
    AdjustmentStrategy,
    DynamicThreshold,
    ThresholdManager,
    ThresholdObservation,
    get_threshold_manager,
    reset_threshold_manager,
)


class TestAdjustmentStrategy:
    """Tests for AdjustmentStrategy enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert AdjustmentStrategy.LINEAR.value == "linear"
        assert AdjustmentStrategy.EXPONENTIAL.value == "exponential"
        assert AdjustmentStrategy.ADAPTIVE.value == "adaptive"

    def test_enum_members(self):
        """Test enum has exactly 3 members."""
        assert len(AdjustmentStrategy) == 3


class TestThresholdObservation:
    """Tests for ThresholdObservation dataclass."""

    def test_creation_minimal(self):
        """Test observation with required fields only."""
        obs = ThresholdObservation(timestamp=1000.0, success=True)
        assert obs.timestamp == 1000.0
        assert obs.success is True
        assert obs.measured_value is None
        assert obs.metadata == {}

    def test_creation_full(self):
        """Test observation with all fields."""
        obs = ThresholdObservation(
            timestamp=1000.0,
            success=False,
            measured_value=45.5,
            metadata={"node": "worker-1", "retry": True},
        )
        assert obs.timestamp == 1000.0
        assert obs.success is False
        assert obs.measured_value == 45.5
        assert obs.metadata == {"node": "worker-1", "retry": True}


class TestDynamicThreshold:
    """Tests for DynamicThreshold class."""

    def test_initialization(self):
        """Test threshold initialization with defaults."""
        threshold = DynamicThreshold(
            name="test_threshold",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )
        assert threshold.name == "test_threshold"
        assert threshold.value == 30.0
        assert threshold.min_value == 5.0
        assert threshold.max_value == 100.0
        assert threshold.target_success_rate == 0.95
        assert threshold.adjustment_strategy == AdjustmentStrategy.ADAPTIVE
        assert threshold.window_size == 100

    def test_initialization_custom(self):
        """Test threshold initialization with custom values."""
        threshold = DynamicThreshold(
            name="custom",
            initial_value=50.0,
            min_value=10.0,
            max_value=200.0,
            target_success_rate=0.90,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
            adjustment_factor=0.2,
            window_size=50,
            cooldown_seconds=30.0,
            higher_is_more_permissive=False,
        )
        assert threshold.target_success_rate == 0.90
        assert threshold.adjustment_strategy == AdjustmentStrategy.LINEAR
        assert threshold.adjustment_factor == 0.2
        assert threshold.window_size == 50
        assert threshold.cooldown_seconds == 30.0
        assert threshold.higher_is_more_permissive is False

    def test_value_property(self):
        """Test value property returns current threshold."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=25.0,
            min_value=5.0,
            max_value=100.0,
        )
        assert threshold.value == 25.0

    def test_success_rate_empty(self):
        """Test success rate when no observations."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )
        # Default to 1.0 when no observations
        assert threshold.success_rate == 1.0

    def test_success_rate_calculated(self):
        """Test success rate calculation."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )
        # Add 8 successes and 2 failures
        for _ in range(8):
            threshold.record_outcome(success=True)
        for _ in range(2):
            threshold.record_outcome(success=False)

        assert threshold.success_rate == 0.8

    def test_record_outcome_basic(self):
        """Test recording outcomes updates observation count."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )
        threshold.record_outcome(success=True, measured_value=25.0)
        threshold.record_outcome(success=False, measured_value=45.0)

        stats = threshold.get_stats()
        assert stats["observations_in_window"] == 2
        assert stats["total_observations"] == 2

    def test_record_outcome_with_metadata(self):
        """Test recording outcomes with metadata."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )
        threshold.record_outcome(success=True, measured_value=25.0, node="worker-1")

        # Metadata should be stored in observation
        assert len(threshold._observations) == 1
        assert threshold._observations[0].metadata.get("node") == "worker-1"

    def test_window_eviction(self):
        """Test old observations are evicted from window."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
            window_size=10,
        )
        # Add 15 observations
        for i in range(15):
            threshold.record_outcome(success=True)

        # Should only keep last 10
        assert len(threshold._observations) == 10
        assert threshold.get_stats()["total_observations"] == 15

    def test_cooldown_prevents_rapid_adjustment(self):
        """Test cooldown prevents adjustments too close together."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
            window_size=10,
            cooldown_seconds=60.0,  # 60 second cooldown
            target_success_rate=0.95,
        )
        initial_value = threshold.value

        # Fill window with failures to trigger adjustment
        for _ in range(10):
            threshold.record_outcome(success=False)

        # Value should have adjusted
        first_adjustment = threshold.value
        assert first_adjustment != initial_value

        # Add more failures - should not adjust due to cooldown
        for _ in range(10):
            threshold.record_outcome(success=False)

        assert threshold.value == first_adjustment

    def test_adjustment_respects_bounds(self):
        """Test adjustments stay within min/max bounds."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=10.0,  # Close to min
            min_value=5.0,
            max_value=100.0,
            window_size=10,
            cooldown_seconds=0.0,  # No cooldown for testing
            adjustment_factor=0.5,  # Large adjustment
        )

        # Force many successes to try to decrease below min
        with patch.object(threshold, '_last_adjustment_time', 0):
            for _ in range(20):
                threshold.record_outcome(success=True)
                threshold._last_adjustment_time = 0  # Reset cooldown

        # Should not go below min
        assert threshold.value >= threshold.min_value

    def test_linear_strategy_increase(self):
        """Test LINEAR strategy increases threshold on low success."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=0.0,
            max_value=100.0,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
            adjustment_factor=0.1,
            window_size=10,
            cooldown_seconds=0.0,
            target_success_rate=0.95,
            higher_is_more_permissive=True,
        )
        initial = threshold.value

        # Low success rate should increase threshold
        for _ in range(10):
            threshold.record_outcome(success=False)
            threshold._last_adjustment_time = 0

        assert threshold.value > initial

    def test_linear_strategy_decrease(self):
        """Test LINEAR strategy decreases threshold on high success."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=0.0,
            max_value=100.0,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
            adjustment_factor=0.1,
            window_size=10,
            cooldown_seconds=0.0,
            target_success_rate=0.50,  # Target 50%, so 100% is too high
            higher_is_more_permissive=True,
        )
        initial = threshold.value

        # High success rate should decrease threshold
        for _ in range(10):
            threshold.record_outcome(success=True)
            threshold._last_adjustment_time = 0

        assert threshold.value < initial

    def test_exponential_strategy(self):
        """Test EXPONENTIAL strategy adjusts proportionally."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=0.0,
            max_value=100.0,
            adjustment_strategy=AdjustmentStrategy.EXPONENTIAL,
            adjustment_factor=0.1,
            window_size=10,
            cooldown_seconds=0.0,
            target_success_rate=0.95,
        )
        initial = threshold.value

        # Low success should increase
        for _ in range(10):
            threshold.record_outcome(success=False)
            threshold._last_adjustment_time = 0

        # Exponential: should be initial * (1 + factor)
        assert threshold.value > initial

    def test_adaptive_strategy_proportional(self):
        """Test ADAPTIVE strategy adjusts based on deviation magnitude."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=0.0,
            max_value=100.0,
            adjustment_strategy=AdjustmentStrategy.ADAPTIVE,
            adjustment_factor=0.1,
            window_size=10,
            cooldown_seconds=0.0,
            target_success_rate=0.95,
        )

        # Low success should increase threshold
        for _ in range(10):
            threshold.record_outcome(success=False)
            threshold._last_adjustment_time = 0

        assert threshold.value > 50.0

    def test_higher_is_less_permissive(self):
        """Test behavior when higher_is_more_permissive=False."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=0.0,
            max_value=100.0,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
            adjustment_factor=0.1,
            window_size=10,
            cooldown_seconds=0.0,
            target_success_rate=0.95,
            higher_is_more_permissive=False,  # Inverted logic
        )
        initial = threshold.value

        # Low success should decrease threshold (inverted)
        for _ in range(10):
            threshold.record_outcome(success=False)
            threshold._last_adjustment_time = 0

        assert threshold.value < initial

    def test_reset(self):
        """Test reset clears observations."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )

        # Add some observations
        for _ in range(10):
            threshold.record_outcome(success=True)

        threshold.reset()

        assert len(threshold._observations) == 0
        assert threshold._last_adjustment_time == 0.0
        assert threshold.success_rate == 1.0  # Default when empty

    def test_get_stats(self):
        """Test get_stats returns comprehensive statistics."""
        threshold = DynamicThreshold(
            name="test_stats",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
            target_success_rate=0.90,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
        )

        for i in range(10):
            threshold.record_outcome(success=i < 8, measured_value=float(i * 5))

        stats = threshold.get_stats()

        assert stats["name"] == "test_stats"
        assert stats["current_value"] == 30.0
        assert stats["min_value"] == 5.0
        assert stats["max_value"] == 100.0
        assert stats["target_success_rate"] == 0.90
        assert stats["observations_in_window"] == 10
        assert stats["total_observations"] == 10
        assert stats["strategy"] == "linear"
        assert stats["success_rate"] == 0.8  # 8/10
        assert "measured_mean" in stats

    def test_stats_without_measured_values(self):
        """Test stats when no measured values recorded."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )
        threshold.record_outcome(success=True)  # No measured_value

        stats = threshold.get_stats()
        assert stats["measured_mean"] is None
        assert stats["measured_p95"] is None


class TestThresholdManager:
    """Tests for ThresholdManager class."""

    def test_initialization(self):
        """Test manager initializes empty."""
        manager = ThresholdManager()
        assert len(manager._thresholds) == 0

    def test_register_threshold(self):
        """Test registering a threshold."""
        manager = ThresholdManager()
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )

        manager.register(threshold)

        assert "test" in manager._thresholds
        assert manager._thresholds["test"] is threshold

    def test_get_existing(self):
        """Test getting an existing threshold."""
        manager = ThresholdManager()
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )
        manager.register(threshold)

        result = manager.get("test")
        assert result is threshold

    def test_get_nonexistent(self):
        """Test getting a non-existent threshold."""
        manager = ThresholdManager()
        result = manager.get("nonexistent")
        assert result is None

    def test_get_value_existing(self):
        """Test getting value of existing threshold."""
        manager = ThresholdManager()
        threshold = DynamicThreshold(
            name="test",
            initial_value=42.0,
            min_value=5.0,
            max_value=100.0,
        )
        manager.register(threshold)

        value = manager.get_value("test")
        assert value == 42.0

    def test_get_value_nonexistent_with_default(self):
        """Test getting value of non-existent threshold with default."""
        manager = ThresholdManager()
        value = manager.get_value("nonexistent", default=99.0)
        assert value == 99.0

    def test_get_value_nonexistent_no_default(self):
        """Test getting value of non-existent threshold without default."""
        manager = ThresholdManager()
        value = manager.get_value("nonexistent")
        assert value is None

    def test_record_existing(self):
        """Test recording to existing threshold."""
        manager = ThresholdManager()
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
        )
        manager.register(threshold)

        result = manager.record("test", success=True, measured_value=25.0)

        assert result is True
        assert len(threshold._observations) == 1

    def test_record_nonexistent(self):
        """Test recording to non-existent threshold."""
        manager = ThresholdManager()
        result = manager.record("nonexistent", success=True)
        assert result is False

    def test_reset_all(self):
        """Test resetting all thresholds."""
        manager = ThresholdManager()

        for name in ["a", "b", "c"]:
            threshold = DynamicThreshold(
                name=name,
                initial_value=30.0,
                min_value=5.0,
                max_value=100.0,
            )
            manager.register(threshold)
            threshold.record_outcome(success=True)

        # All should have observations
        for name in ["a", "b", "c"]:
            assert len(manager.get(name)._observations) == 1

        manager.reset_all()

        # All should be cleared
        for name in ["a", "b", "c"]:
            assert len(manager.get(name)._observations) == 0

    def test_get_all_stats(self):
        """Test getting stats for all thresholds."""
        manager = ThresholdManager()

        for name in ["threshold_a", "threshold_b"]:
            threshold = DynamicThreshold(
                name=name,
                initial_value=30.0,
                min_value=5.0,
                max_value=100.0,
            )
            manager.register(threshold)

        stats = manager.get_all_stats()

        assert "threshold_a" in stats
        assert "threshold_b" in stats
        assert stats["threshold_a"]["name"] == "threshold_a"
        assert stats["threshold_b"]["name"] == "threshold_b"

    def test_get_health_all_healthy(self):
        """Test health when all thresholds are healthy."""
        manager = ThresholdManager()
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
            target_success_rate=0.90,
        )
        manager.register(threshold)

        # Add 90% successes (meets target)
        for i in range(10):
            threshold.record_outcome(success=i < 9)

        health = manager.get_health()

        assert health["all_healthy"] is True
        assert health["unhealthy_thresholds"] == []
        assert health["threshold_count"] == 1

    def test_get_health_some_unhealthy(self):
        """Test health when some thresholds are unhealthy."""
        manager = ThresholdManager()

        # Healthy threshold
        healthy = DynamicThreshold(
            name="healthy",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
            target_success_rate=0.90,
        )
        manager.register(healthy)
        for i in range(10):
            healthy.record_outcome(success=i < 9)

        # Unhealthy threshold (below target - 0.1)
        unhealthy = DynamicThreshold(
            name="unhealthy",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
            target_success_rate=0.90,
        )
        manager.register(unhealthy)
        for i in range(10):
            unhealthy.record_outcome(success=i < 5)  # Only 50%

        health = manager.get_health()

        assert health["all_healthy"] is False
        assert "unhealthy" in health["unhealthy_thresholds"]
        assert "healthy" not in health["unhealthy_thresholds"]

    def test_health_check_healthy(self):
        """Test health_check returns healthy result."""
        manager = ThresholdManager()
        threshold = DynamicThreshold(
            name="test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
            target_success_rate=0.80,
        )
        manager.register(threshold)

        # Add 90% successes (above target)
        for i in range(10):
            threshold.record_outcome(success=i < 9)

        result = manager.health_check()

        assert result.healthy is True
        assert result.status.value == "running"

    def test_health_check_degraded(self):
        """Test health_check returns degraded when some thresholds unhealthy."""
        manager = ThresholdManager()

        # Add 3 thresholds, 1 unhealthy
        for i in range(3):
            threshold = DynamicThreshold(
                name=f"test_{i}",
                initial_value=30.0,
                min_value=5.0,
                max_value=100.0,
                target_success_rate=0.90,
            )
            manager.register(threshold)

            if i == 0:
                # Make first one unhealthy
                for j in range(10):
                    threshold.record_outcome(success=j < 5)
            else:
                for _ in range(10):
                    threshold.record_outcome(success=True)

        result = manager.health_check()

        assert result.status.value == "degraded"
        assert "test_0" in result.message

    def test_health_check_error(self):
        """Test health_check returns error when majority unhealthy."""
        manager = ThresholdManager()

        # Add 3 thresholds, 2 unhealthy
        for i in range(3):
            threshold = DynamicThreshold(
                name=f"test_{i}",
                initial_value=30.0,
                min_value=5.0,
                max_value=100.0,
                target_success_rate=0.90,
            )
            manager.register(threshold)

            if i < 2:
                # Make first two unhealthy
                for j in range(10):
                    threshold.record_outcome(success=j < 5)
            else:
                for _ in range(10):
                    threshold.record_outcome(success=True)

        result = manager.health_check()

        assert result.status.value == "error"


class TestGlobalFunctions:
    """Tests for global threshold functions."""

    def test_get_threshold_manager(self):
        """Test getting global threshold manager."""
        reset_threshold_manager()

        manager = get_threshold_manager()
        assert isinstance(manager, ThresholdManager)

        # Should return same instance
        manager2 = get_threshold_manager()
        assert manager is manager2

    def test_reset_threshold_manager(self):
        """Test resetting global threshold manager."""
        manager1 = get_threshold_manager()
        reset_threshold_manager()
        manager2 = get_threshold_manager()

        assert manager1 is not manager2

    def test_default_thresholds_initialized(self):
        """Test default thresholds are created."""
        reset_threshold_manager()
        manager = get_threshold_manager()

        # Check default thresholds exist
        assert manager.get("handler_timeout") is not None
        assert manager.get("heartbeat_threshold") is not None
        assert manager.get("plateau_window") is not None
        assert manager.get("memory_warning") is not None

    def test_handler_timeout_defaults(self):
        """Test handler_timeout has correct defaults."""
        reset_threshold_manager()
        manager = get_threshold_manager()

        threshold = manager.get("handler_timeout")
        assert threshold.value == 30.0
        assert threshold.min_value == 5.0
        assert threshold.max_value == 120.0
        assert threshold.target_success_rate == 0.95

    def test_heartbeat_threshold_defaults(self):
        """Test heartbeat_threshold has correct defaults."""
        reset_threshold_manager()
        manager = get_threshold_manager()

        threshold = manager.get("heartbeat_threshold")
        assert threshold.value == 60.0
        assert threshold.target_success_rate == 0.99

    def test_memory_warning_defaults(self):
        """Test memory_warning has correct defaults."""
        reset_threshold_manager()
        manager = get_threshold_manager()

        threshold = manager.get("memory_warning")
        assert threshold.value == 0.8
        assert threshold.min_value == 0.5
        assert threshold.max_value == 0.95


class TestThresholdIntegration:
    """Integration tests combining multiple threshold features."""

    def test_manager_with_multiple_strategies(self):
        """Test manager handling thresholds with different strategies."""
        manager = ThresholdManager()

        manager.register(DynamicThreshold(
            name="linear",
            initial_value=50.0,
            min_value=0.0,
            max_value=100.0,
            adjustment_strategy=AdjustmentStrategy.LINEAR,
        ))
        manager.register(DynamicThreshold(
            name="exponential",
            initial_value=50.0,
            min_value=0.0,
            max_value=100.0,
            adjustment_strategy=AdjustmentStrategy.EXPONENTIAL,
        ))
        manager.register(DynamicThreshold(
            name="adaptive",
            initial_value=50.0,
            min_value=0.0,
            max_value=100.0,
            adjustment_strategy=AdjustmentStrategy.ADAPTIVE,
        ))

        # Record to all
        for name in ["linear", "exponential", "adaptive"]:
            manager.record(name, success=True, measured_value=25.0)

        stats = manager.get_all_stats()
        assert len(stats) == 3

    def test_threshold_adjustment_over_time(self):
        """Test threshold adjusts correctly over many observations."""
        threshold = DynamicThreshold(
            name="test",
            initial_value=50.0,
            min_value=10.0,
            max_value=90.0,
            window_size=20,
            cooldown_seconds=0.0,
            target_success_rate=0.80,
            adjustment_strategy=AdjustmentStrategy.ADAPTIVE,
        )
        initial = threshold.value

        # Simulate many failures - should increase threshold
        for _ in range(50):
            threshold.record_outcome(success=False)
            threshold._last_adjustment_time = 0  # Reset cooldown

        # Should have increased toward max
        assert threshold.value > initial

        # Now simulate many successes - should decrease threshold
        for _ in range(50):
            threshold.record_outcome(success=True)
            threshold._last_adjustment_time = 0

        # Should have decreased (may not go back to initial if bounds apply)
        assert threshold.value < threshold.max_value

    def test_stats_reflect_observations(self):
        """Test statistics accurately reflect recorded observations."""
        threshold = DynamicThreshold(
            name="accuracy_test",
            initial_value=30.0,
            min_value=5.0,
            max_value=100.0,
            window_size=100,
        )

        # Record specific pattern
        for i in range(75):
            threshold.record_outcome(success=True, measured_value=float(i))
        for i in range(25):
            threshold.record_outcome(success=False, measured_value=float(i + 75))

        stats = threshold.get_stats()

        assert stats["total_observations"] == 100
        assert stats["observations_in_window"] == 100
        assert stats["success_rate"] == 0.75
