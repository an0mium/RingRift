"""Unit tests for app/coordination/singleton_registry.py.

Tests for:
- SingletonRegistry class static methods
- get_all_singletons, get_singleton_count, has_singleton
- reset_all_sync, reset_all_async
- get_singletons_by_category, get_running_daemons
- stop_all_daemons_sync, stop_all_daemons_async
- Convenience functions

Created: December 28, 2025
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.singleton_registry import (
    SingletonRegistry,
    get_singleton_count,
    reset_all_singletons,
    reset_all_singletons_async,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clean_singletons():
    """Ensure singletons are reset before and after each test."""
    from app.coordination.singleton_mixin import SingletonMixin

    # Store original state
    original = dict(SingletonMixin._instances)

    # Clear for test
    SingletonMixin._instances.clear()

    yield

    # Restore original state
    SingletonMixin._instances.clear()
    SingletonMixin._instances.update(original)


@pytest.fixture
def mock_singleton_class():
    """Create a mock singleton class with proper structure."""
    from app.coordination.singleton_mixin import SingletonMixin

    class MockSingleton(SingletonMixin):
        def __init__(self):
            self._running = False
            self.stopped = False

        def stop(self):
            self.stopped = True
            self._running = False

        @classmethod
        def reset_instance(cls):
            if cls in SingletonMixin._instances:
                del SingletonMixin._instances[cls]

    return MockSingleton


# =============================================================================
# SingletonRegistry.get_all_singletons Tests
# =============================================================================


class TestGetAllSingletons:
    """Tests for SingletonRegistry.get_all_singletons()."""

    def test_returns_dict(self, clean_singletons):
        """Test that get_all_singletons returns a dict."""
        result = SingletonRegistry.get_all_singletons()
        assert isinstance(result, dict)

    def test_returns_copy(self, clean_singletons):
        """Test that get_all_singletons returns a copy, not the original."""
        from app.coordination.singleton_mixin import SingletonMixin

        result1 = SingletonRegistry.get_all_singletons()
        result2 = SingletonRegistry.get_all_singletons()

        # Should be equal but not the same object
        assert result1 == result2
        assert result1 is not result2
        assert result1 is not SingletonMixin._instances

    def test_empty_when_no_singletons(self, clean_singletons):
        """Test that get_all_singletons returns empty dict when no singletons."""
        result = SingletonRegistry.get_all_singletons()
        assert result == {}


# =============================================================================
# SingletonRegistry.get_singleton_count Tests
# =============================================================================


class TestGetSingletonCount:
    """Tests for SingletonRegistry.get_singleton_count()."""

    def test_returns_zero_when_empty(self, clean_singletons):
        """Test count is 0 when no singletons exist."""
        assert SingletonRegistry.get_singleton_count() == 0

    def test_counts_singletons(self, clean_singletons, mock_singleton_class):
        """Test that count reflects number of singletons."""
        from app.coordination.singleton_mixin import SingletonMixin

        # Add mock singleton
        instance = object()
        SingletonMixin._instances[mock_singleton_class] = instance

        assert SingletonRegistry.get_singleton_count() == 1


# =============================================================================
# SingletonRegistry.has_singleton Tests
# =============================================================================


class TestHasSingleton:
    """Tests for SingletonRegistry.has_singleton()."""

    def test_returns_false_when_not_present(self, clean_singletons, mock_singleton_class):
        """Test has_singleton returns False when class not registered."""
        assert SingletonRegistry.has_singleton(mock_singleton_class) is False

    def test_returns_true_when_present(self, clean_singletons, mock_singleton_class):
        """Test has_singleton returns True when class is registered."""
        from app.coordination.singleton_mixin import SingletonMixin

        SingletonMixin._instances[mock_singleton_class] = object()
        assert SingletonRegistry.has_singleton(mock_singleton_class) is True


# =============================================================================
# SingletonRegistry.reset_all_sync Tests
# =============================================================================


class TestResetAllSync:
    """Tests for SingletonRegistry.reset_all_sync()."""

    def test_returns_count(self, clean_singletons):
        """Test that reset_all_sync returns count of reset singletons."""
        count = SingletonRegistry.reset_all_sync()
        assert isinstance(count, int)
        assert count >= 0

    def test_resets_singletons(self, clean_singletons, mock_singleton_class):
        """Test that reset_all_sync clears all singletons."""
        from app.coordination.singleton_mixin import SingletonMixin

        # Add mock singletons
        SingletonMixin._instances[mock_singleton_class] = mock_singleton_class()

        assert SingletonRegistry.get_singleton_count() == 1

        count = SingletonRegistry.reset_all_sync()
        assert count == 1
        assert SingletonRegistry.get_singleton_count() == 0

    def test_handles_errors_gracefully(self, clean_singletons):
        """Test that reset_all_sync handles errors without raising."""
        from app.coordination.singleton_mixin import SingletonMixin

        # Create a class that raises on reset
        class BadSingleton:
            @classmethod
            def reset_instance(cls):
                raise RuntimeError("intentional error")

            @classmethod
            def _get_lock(cls):
                from threading import RLock

                return RLock()

        SingletonMixin._instances[BadSingleton] = object()

        # Should not raise
        count = SingletonRegistry.reset_all_sync()
        # May or may not have reset depending on fallback path


# =============================================================================
# SingletonRegistry.reset_all_async Tests
# =============================================================================


class TestResetAllAsync:
    """Tests for SingletonRegistry.reset_all_async()."""

    @pytest.mark.asyncio
    async def test_returns_count(self, clean_singletons):
        """Test that reset_all_async returns count."""
        count = await SingletonRegistry.reset_all_async()
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.asyncio
    async def test_resets_singletons(self, clean_singletons, mock_singleton_class):
        """Test that reset_all_async clears all singletons."""
        from app.coordination.singleton_mixin import SingletonMixin

        SingletonMixin._instances[mock_singleton_class] = mock_singleton_class()
        assert SingletonRegistry.get_singleton_count() == 1

        count = await SingletonRegistry.reset_all_async()
        assert count == 1
        assert SingletonRegistry.get_singleton_count() == 0

    @pytest.mark.asyncio
    async def test_calls_async_stop(self, clean_singletons):
        """Test that reset_all_async calls async stop methods."""
        from app.coordination.singleton_mixin import SingletonMixin

        class AsyncDaemon:
            def __init__(self):
                self.stopped = False

            async def stop(self):
                self.stopped = True

            @classmethod
            def reset_instance(cls):
                if cls in SingletonMixin._instances:
                    del SingletonMixin._instances[cls]

            @classmethod
            def _get_lock(cls):
                from threading import RLock

                return RLock()

        instance = AsyncDaemon()
        SingletonMixin._instances[AsyncDaemon] = instance

        await SingletonRegistry.reset_all_async()
        assert instance.stopped is True

    @pytest.mark.asyncio
    async def test_handles_stop_timeout(self, clean_singletons):
        """Test that reset_all_async handles stop timeout gracefully."""
        from app.coordination.singleton_mixin import SingletonMixin

        class SlowDaemon:
            async def stop(self):
                await asyncio.sleep(10)  # Will timeout

            @classmethod
            def reset_instance(cls):
                if cls in SingletonMixin._instances:
                    del SingletonMixin._instances[cls]

            @classmethod
            def _get_lock(cls):
                from threading import RLock

                return RLock()

        SingletonMixin._instances[SlowDaemon] = SlowDaemon()

        # Should not hang forever
        count = await asyncio.wait_for(
            SingletonRegistry.reset_all_async(), timeout=10.0
        )
        assert count >= 0


# =============================================================================
# SingletonRegistry.get_singletons_by_category Tests
# =============================================================================


class TestGetSingletonsByCategory:
    """Tests for SingletonRegistry.get_singletons_by_category()."""

    def test_returns_dict_with_categories(self, clean_singletons):
        """Test that get_singletons_by_category returns expected categories."""
        result = SingletonRegistry.get_singletons_by_category()
        assert isinstance(result, dict)
        assert "coordination" in result
        assert "training" in result
        assert "ai" in result
        assert "distributed" in result
        assert "core" in result
        assert "other" in result

    def test_categories_are_lists(self, clean_singletons):
        """Test that category values are lists."""
        result = SingletonRegistry.get_singletons_by_category()
        for category, classes in result.items():
            assert isinstance(classes, list)


# =============================================================================
# SingletonRegistry.get_running_daemons Tests
# =============================================================================


class TestGetRunningDaemons:
    """Tests for SingletonRegistry.get_running_daemons()."""

    def test_returns_list(self, clean_singletons):
        """Test that get_running_daemons returns a list."""
        result = SingletonRegistry.get_running_daemons()
        assert isinstance(result, list)

    def test_empty_when_no_running(self, clean_singletons):
        """Test that returns empty list when no daemons running."""
        result = SingletonRegistry.get_running_daemons()
        assert result == []

    def test_finds_running_daemons(self, clean_singletons, mock_singleton_class):
        """Test that finds daemons with _running=True."""
        from app.coordination.singleton_mixin import SingletonMixin

        instance = mock_singleton_class()
        instance._running = True
        SingletonMixin._instances[mock_singleton_class] = instance

        result = SingletonRegistry.get_running_daemons()
        assert len(result) == 1
        assert result[0][0] is mock_singleton_class
        assert result[0][1] is instance

    def test_ignores_stopped_daemons(self, clean_singletons, mock_singleton_class):
        """Test that ignores daemons with _running=False."""
        from app.coordination.singleton_mixin import SingletonMixin

        instance = mock_singleton_class()
        instance._running = False
        SingletonMixin._instances[mock_singleton_class] = instance

        result = SingletonRegistry.get_running_daemons()
        assert result == []


# =============================================================================
# SingletonRegistry.stop_all_daemons_sync Tests
# =============================================================================


class TestStopAllDaemonsSync:
    """Tests for SingletonRegistry.stop_all_daemons_sync()."""

    def test_returns_count(self, clean_singletons):
        """Test that stop_all_daemons_sync returns count."""
        count = SingletonRegistry.stop_all_daemons_sync()
        assert isinstance(count, int)
        assert count >= 0

    def test_stops_running_daemons(self, clean_singletons, mock_singleton_class):
        """Test that stops daemons with _running=True."""
        from app.coordination.singleton_mixin import SingletonMixin

        instance = mock_singleton_class()
        instance._running = True
        SingletonMixin._instances[mock_singleton_class] = instance

        count = SingletonRegistry.stop_all_daemons_sync()
        assert count == 1
        assert instance.stopped is True


# =============================================================================
# SingletonRegistry.stop_all_daemons_async Tests
# =============================================================================


class TestStopAllDaemonsAsync:
    """Tests for SingletonRegistry.stop_all_daemons_async()."""

    @pytest.mark.asyncio
    async def test_returns_count(self, clean_singletons):
        """Test that stop_all_daemons_async returns count."""
        count = await SingletonRegistry.stop_all_daemons_async()
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.asyncio
    async def test_stops_async_daemons(self, clean_singletons):
        """Test that stops daemons with async stop method."""
        from app.coordination.singleton_mixin import SingletonMixin

        class AsyncStopDaemon:
            def __init__(self):
                self._running = True
                self.stopped = False

            async def stop(self):
                self.stopped = True
                self._running = False

        instance = AsyncStopDaemon()
        SingletonMixin._instances[AsyncStopDaemon] = instance

        count = await SingletonRegistry.stop_all_daemons_async()
        assert count == 1
        assert instance.stopped is True


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_singleton_count_function(self, clean_singletons):
        """Test get_singleton_count convenience function."""
        count = get_singleton_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_reset_all_singletons_function(self, clean_singletons):
        """Test reset_all_singletons convenience function."""
        count = reset_all_singletons()
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.asyncio
    async def test_reset_all_singletons_async_function(self, clean_singletons):
        """Test reset_all_singletons_async convenience function."""
        count = await reset_all_singletons_async()
        assert isinstance(count, int)
        assert count >= 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for SingletonRegistry."""

    def test_concurrent_modifications(self, clean_singletons):
        """Test handling of concurrent modifications during iteration."""
        from app.coordination.singleton_mixin import SingletonMixin

        # This tests the copy pattern used to avoid modification during iteration
        class DynamicSingleton:
            @classmethod
            def reset_instance(cls):
                # Simulate modification during reset
                if cls in SingletonMixin._instances:
                    del SingletonMixin._instances[cls]

            @classmethod
            def _get_lock(cls):
                from threading import RLock

                return RLock()

        SingletonMixin._instances[DynamicSingleton] = object()

        # Should not raise even with modifications
        count = SingletonRegistry.reset_all_sync()
        assert count >= 0

    def test_none_instance(self, clean_singletons):
        """Test handling of None instance in registry."""
        from app.coordination.singleton_mixin import SingletonMixin

        class NullSingleton:
            @classmethod
            def reset_instance(cls):
                if cls in SingletonMixin._instances:
                    del SingletonMixin._instances[cls]

            @classmethod
            def _get_lock(cls):
                from threading import RLock

                return RLock()

        # Register None as instance (edge case)
        SingletonMixin._instances[NullSingleton] = None

        # Should handle gracefully
        result = SingletonRegistry.get_all_singletons()
        assert NullSingleton in result

    def test_module_path_categorization(self, clean_singletons):
        """Test that module path categorization works correctly."""
        # Test the categorization logic
        categories = SingletonRegistry.get_singletons_by_category()

        # All category lists should exist
        assert len(categories) == 6  # coordination, training, ai, distributed, core, other
