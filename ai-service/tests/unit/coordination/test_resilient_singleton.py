"""Tests for ResilientSingletonMixin (December 2025).

Tests the health-checked singleton pattern that supports automatic
recreation when singletons become unhealthy.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from app.coordination.singleton_mixin import (
    ResilientSingletonMixin,
    SingletonMixin,
    with_singleton_health_check,
)


class TestResilientSingletonMixin:
    """Tests for ResilientSingletonMixin."""

    def setup_method(self):
        """Clear singleton state before each test."""
        SingletonMixin._instances.clear()
        ResilientSingletonMixin._recreation_attempts.clear()
        ResilientSingletonMixin._last_recreation.clear()

    def test_get_healthy_instance_creates_new(self):
        """get_healthy_instance() creates instance if none exists."""

        class TestDaemon(ResilientSingletonMixin):
            def __init__(self):
                self._running = True

            def health_check(self):
                return True

        instance = TestDaemon.get_healthy_instance()
        assert instance is not None
        assert isinstance(instance, TestDaemon)

    def test_get_healthy_instance_returns_existing_healthy(self):
        """get_healthy_instance() returns existing healthy instance."""

        class TestDaemon(ResilientSingletonMixin):
            def __init__(self):
                self._running = True
                self.creation_time = time.time()

            def health_check(self):
                return True

        instance1 = TestDaemon.get_healthy_instance()
        instance2 = TestDaemon.get_healthy_instance()
        assert instance1 is instance2

    def test_recreates_unhealthy_instance(self):
        """get_healthy_instance() recreates unhealthy instances."""

        class TestDaemon(ResilientSingletonMixin):
            creation_count = 0

            def __init__(self):
                TestDaemon.creation_count += 1
                self._healthy = True

            def health_check(self):
                return self._healthy

        # Create initial instance
        instance1 = TestDaemon.get_healthy_instance()
        assert TestDaemon.creation_count == 1

        # Mark as unhealthy
        instance1._healthy = False

        # Get healthy instance should recreate
        instance2 = TestDaemon.get_healthy_instance()
        assert TestDaemon.creation_count == 2
        assert instance1 is not instance2

    def test_respects_recreation_cooldown(self):
        """get_healthy_instance() respects cooldown between recreations."""

        class TestDaemon(ResilientSingletonMixin):
            creation_count = 0

            def __init__(self):
                TestDaemon.creation_count += 1
                self._healthy = False  # Always unhealthy

            def health_check(self):
                return self._healthy

        # Set short cooldown for test
        TestDaemon._recreation_cooldown = 1.0

        # Create initial instance
        instance1 = TestDaemon.get_healthy_instance()
        assert TestDaemon.creation_count == 1

        # Immediately try to get again - should hit cooldown
        instance2 = TestDaemon.get_healthy_instance()
        assert TestDaemon.creation_count == 2  # First recreation

        # Immediately try again - should hit cooldown
        instance3 = TestDaemon.get_healthy_instance()
        assert TestDaemon.creation_count == 2  # No recreation due to cooldown

    def test_respects_max_recreations(self):
        """get_healthy_instance() stops after max recreations."""

        class TestDaemon(ResilientSingletonMixin):
            creation_count = 0

            def __init__(self):
                TestDaemon.creation_count += 1
                self._healthy = False  # Always unhealthy

            def health_check(self):
                return self._healthy

        # Set very short cooldown and low max
        TestDaemon._recreation_cooldown = 0.01
        TestDaemon._max_recreations = 3

        # Force multiple recreations
        for i in range(10):
            TestDaemon.get_healthy_instance()
            time.sleep(0.02)  # Wait for cooldown

        # Should stop at max recreations (initial + 3 recreations)
        assert TestDaemon.creation_count <= 4

    def test_check_health_dict_result(self):
        """_check_instance_health handles dict health check result."""

        class TestDaemon(ResilientSingletonMixin):
            def health_check(self):
                return {"healthy": True, "message": "OK"}

        instance = TestDaemon()
        assert TestDaemon._check_instance_health(instance) is True

        # Test unhealthy dict
        class UnhealthyDaemon(ResilientSingletonMixin):
            def health_check(self):
                return {"healthy": False, "message": "Failed"}

        unhealthy = UnhealthyDaemon()
        assert UnhealthyDaemon._check_instance_health(unhealthy) is False

    def test_check_health_object_result(self):
        """_check_instance_health handles object with .healthy attribute."""

        class HealthResult:
            def __init__(self, healthy: bool):
                self.healthy = healthy

        class TestDaemon(ResilientSingletonMixin):
            def __init__(self, healthy: bool):
                self._is_healthy = healthy

            def health_check(self):
                return HealthResult(self._is_healthy)

        healthy_instance = TestDaemon(True)
        assert TestDaemon._check_instance_health(healthy_instance) is True

        unhealthy_instance = TestDaemon(False)
        assert TestDaemon._check_instance_health(unhealthy_instance) is False

    def test_check_health_bool_result(self):
        """_check_instance_health handles bool health check result."""

        class TestDaemon(ResilientSingletonMixin):
            def __init__(self, healthy: bool):
                self._healthy = healthy

            def health_check(self):
                return self._healthy

        healthy = TestDaemon(True)
        assert TestDaemon._check_instance_health(healthy) is True

        unhealthy = TestDaemon(False)
        assert TestDaemon._check_instance_health(unhealthy) is False

    def test_check_health_is_healthy_method(self):
        """_check_instance_health uses is_healthy() if no health_check()."""

        class TestDaemon(ResilientSingletonMixin):
            def __init__(self, healthy: bool):
                self._healthy = healthy

            def is_healthy(self):
                return self._healthy

        healthy = TestDaemon(True)
        assert TestDaemon._check_instance_health(healthy) is True

    def test_check_health_running_attribute(self):
        """_check_instance_health uses _running if no methods."""

        class TestDaemon(ResilientSingletonMixin):
            def __init__(self, running: bool):
                self._running = running

        running = TestDaemon(True)
        assert TestDaemon._check_instance_health(running) is True

        stopped = TestDaemon(False)
        assert TestDaemon._check_instance_health(stopped) is False

    def test_check_health_no_method_assumes_healthy(self):
        """_check_instance_health assumes healthy if no method/attribute."""

        class TestDaemon(ResilientSingletonMixin):
            pass

        instance = TestDaemon()
        assert TestDaemon._check_instance_health(instance) is True

    def test_check_health_exception_returns_false(self):
        """_check_instance_health returns False if health_check raises."""

        class TestDaemon(ResilientSingletonMixin):
            def health_check(self):
                raise RuntimeError("Health check failed")

        instance = TestDaemon()
        assert TestDaemon._check_instance_health(instance) is False

    def test_recreate_instance_calls_cleanup(self):
        """_recreate_instance calls cleanup methods."""
        cleanup_called = []

        class TestDaemon(ResilientSingletonMixin):
            def __init__(self):
                self._healthy = True

            def health_check(self):
                return self._healthy

            def stop(self):
                cleanup_called.append("stop")

        # Create initial instance
        instance1 = TestDaemon.get_healthy_instance()
        instance1._healthy = False

        # Recreate - should call stop
        TestDaemon.recreate_instance()
        assert "stop" in cleanup_called

    def test_get_recreation_stats(self):
        """get_recreation_stats returns correct metrics."""

        class TestDaemon(ResilientSingletonMixin):
            def __init__(self):
                self._healthy = False

            def health_check(self):
                return self._healthy

        # Set short cooldown
        TestDaemon._recreation_cooldown = 0.01

        # Force some recreations
        TestDaemon.get_healthy_instance()
        time.sleep(0.02)
        TestDaemon.get_healthy_instance()

        stats = TestDaemon.get_recreation_stats()
        assert stats["class"] == "TestDaemon"
        assert stats["recreation_attempts"] >= 1
        assert stats["max_recreations"] == 5
        assert stats["cooldown"] == 0.01

    def test_skip_health_check(self):
        """get_healthy_instance(check_health=False) skips health check."""

        class TestDaemon(ResilientSingletonMixin):
            def __init__(self):
                self._healthy = False

            def health_check(self):
                return self._healthy

        # Create unhealthy instance
        instance1 = TestDaemon.get_healthy_instance(check_health=False)

        # Should return same instance without recreating
        instance2 = TestDaemon.get_healthy_instance(check_health=False)
        assert instance1 is instance2

    def test_disable_recreation(self):
        """get_healthy_instance(recreate_if_unhealthy=False) disables recreation."""

        class TestDaemon(ResilientSingletonMixin):
            creation_count = 0

            def __init__(self):
                TestDaemon.creation_count += 1
                self._healthy = False

            def health_check(self):
                return self._healthy

        # Create unhealthy instance
        TestDaemon.get_healthy_instance()
        assert TestDaemon.creation_count == 1

        # Should not recreate
        TestDaemon.get_healthy_instance(recreate_if_unhealthy=False)
        assert TestDaemon.creation_count == 1


class TestWithSingletonHealthCheck:
    """Tests for @with_singleton_health_check decorator."""

    def setup_method(self):
        """Clear singleton state before each test."""
        SingletonMixin._instances.clear()

    def test_decorator_adds_methods(self):
        """Decorator adds health check methods to class."""

        @with_singleton_health_check
        class TestDaemon(SingletonMixin):
            def __init__(self):
                self._running = True

            def health_check(self):
                return True

        assert hasattr(TestDaemon, "get_healthy_instance")
        assert hasattr(TestDaemon, "recreate_instance")
        assert hasattr(TestDaemon, "get_recreation_stats")

    def test_decorated_class_works(self):
        """Decorated class can use health-checked access."""

        @with_singleton_health_check
        class TestDaemon(SingletonMixin):
            creation_count = 0

            def __init__(self):
                TestDaemon.creation_count += 1
                self._healthy = True

            def health_check(self):
                return self._healthy

        # Use health-checked access
        instance = TestDaemon.get_healthy_instance()
        assert instance is not None
        assert TestDaemon.creation_count == 1

        # Same instance returned
        instance2 = TestDaemon.get_healthy_instance()
        assert instance is instance2
