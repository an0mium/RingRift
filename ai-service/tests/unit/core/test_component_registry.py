"""Tests for app.core.component_registry module.

Tests the centralized component registry for singleton management:
- ComponentInfo dataclass
- ComponentRegistry class
- Factory and class registration
- Lazy initialization
- Health tracking
- Reset functionality

December 2025 - Phase 2 architecture cleanup.
"""

import threading
import time

import pytest

from app.core.component_registry import (
    ComponentInfo,
    ComponentRegistry,
    component,
    get_registry,
    reset_registry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset the global registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def registry():
    """Create a fresh registry for testing."""
    return ComponentRegistry()


# =============================================================================
# ComponentInfo Tests
# =============================================================================


class TestComponentInfo:
    """Tests for ComponentInfo dataclass."""

    def test_create_basic(self):
        """Should create info with name."""
        info = ComponentInfo(name="test")
        assert info.name == "test"
        assert info.instance is None
        assert not info.is_instantiated

    def test_is_instantiated(self):
        """Should report instantiation status."""
        info = ComponentInfo(name="test")
        assert not info.is_instantiated

        info.instance = object()
        assert info.is_instantiated

    def test_default_healthy(self):
        """Should default to healthy."""
        info = ComponentInfo(name="test")
        assert info.is_healthy
        assert info.health_message == ""


# =============================================================================
# ComponentRegistry Basic Tests
# =============================================================================


class TestComponentRegistryBasic:
    """Basic tests for ComponentRegistry."""

    def test_register_instance(self, registry):
        """Should register pre-created instance."""
        obj = {"value": 42}
        registry.register("test", instance=obj)

        result = registry.get("test")
        assert result is obj

    def test_register_factory(self, registry):
        """Should register factory function."""
        created = []

        def factory():
            obj = {"id": len(created)}
            created.append(obj)
            return obj

        registry.register_factory("test", factory)

        # First get creates the instance
        result1 = registry.get("test")
        assert result1["id"] == 0

        # Second get returns same instance
        result2 = registry.get("test")
        assert result2 is result1
        assert len(created) == 1

    def test_register_class(self, registry):
        """Should register class for instantiation."""
        class TestClass:
            def __init__(self):
                self.value = 123

        registry.register_class("test", TestClass)

        result = registry.get("test")
        assert isinstance(result, TestClass)
        assert result.value == 123

    def test_register_requires_something(self, registry):
        """Should require instance, factory, or class."""
        with pytest.raises(ValueError, match="must have"):
            registry.register("test")

    def test_get_unknown_returns_default(self, registry):
        """Should return default for unknown component."""
        assert registry.get("unknown") is None
        assert registry.get("unknown", "default") == "default"

    def test_has(self, registry):
        """Should check if component is registered."""
        assert not registry.has("test")
        registry.register("test", instance={})
        assert registry.has("test")

    def test_is_instantiated(self, registry):
        """Should check if component is instantiated."""
        registry.register_factory("test", dict)
        assert not registry.is_instantiated("test")

        registry.get("test")
        assert registry.is_instantiated("test")


# =============================================================================
# ComponentRegistry Lazy Initialization Tests
# =============================================================================


class TestComponentRegistryLazy:
    """Tests for lazy initialization."""

    def test_factory_not_called_until_get(self, registry):
        """Should not call factory until get."""
        called = []

        def factory():
            called.append(True)
            return {}

        registry.register_factory("test", factory)
        assert len(called) == 0

        registry.get("test")
        assert len(called) == 1

    def test_class_not_instantiated_until_get(self, registry):
        """Should not instantiate class until get."""
        instantiated = []

        class TestClass:
            def __init__(self):
                instantiated.append(True)

        registry.register_class("test", TestClass)
        assert len(instantiated) == 0

        registry.get("test")
        assert len(instantiated) == 1

    def test_get_or_create(self, registry):
        """Should register and create if not found."""
        result = registry.get_or_create("test", lambda: {"value": 42})
        assert result["value"] == 42

        # Second call returns same instance
        result2 = registry.get_or_create("test", lambda: {"value": 99})
        assert result2 is result


# =============================================================================
# ComponentRegistry Reset Tests
# =============================================================================


class TestComponentRegistryReset:
    """Tests for reset functionality."""

    def test_reset_single(self, registry):
        """Should reset single component."""
        registry.register_factory("test", dict)
        instance1 = registry.get("test")

        registry.reset("test")
        assert not registry.is_instantiated("test")

        instance2 = registry.get("test")
        assert instance2 is not instance1

    def test_reset_all(self, registry):
        """Should reset all components."""
        registry.register_factory("test1", dict)
        registry.register_factory("test2", dict)

        registry.get("test1")
        registry.get("test2")
        assert registry.is_instantiated("test1")
        assert registry.is_instantiated("test2")

        count = registry.reset_all()
        assert count == 2
        assert not registry.is_instantiated("test1")
        assert not registry.is_instantiated("test2")

    def test_reset_keeps_registration(self, registry):
        """Should keep registration after reset."""
        registry.register_factory("test", dict)
        registry.get("test")
        registry.reset("test")

        # Should still be registered
        assert registry.has("test")
        # Should be able to get new instance
        assert registry.get("test") is not None

    def test_unregister(self, registry):
        """Should completely remove registration."""
        registry.register_factory("test", dict)
        registry.get("test")

        registry.unregister("test")
        assert not registry.has("test")
        assert registry.get("test") is None


# =============================================================================
# ComponentRegistry Health Tests
# =============================================================================


class TestComponentRegistryHealth:
    """Tests for health tracking."""

    def test_health_status_empty(self, registry):
        """Should report empty status."""
        status = registry.get_health_status()
        assert status["total_registered"] == 0
        assert status["instantiated"] == 0
        assert status["healthy"] == 0

    def test_health_status_with_components(self, registry):
        """Should report component status."""
        registry.register_factory("test1", dict)
        registry.register_factory("test2", dict)
        registry.get("test1")  # Instantiate only one

        status = registry.get_health_status()
        assert status["total_registered"] == 2
        assert status["instantiated"] == 1
        assert status["healthy"] == 2

    def test_mark_unhealthy(self, registry):
        """Should mark component unhealthy."""
        registry.register_factory("test", dict)

        registry.mark_unhealthy("test", "connection failed")
        info = registry.get_info("test")
        assert not info.is_healthy
        assert info.health_message == "connection failed"

    def test_mark_healthy(self, registry):
        """Should mark component healthy."""
        registry.register_factory("test", dict)
        registry.mark_unhealthy("test", "error")
        registry.mark_healthy("test")

        info = registry.get_info("test")
        assert info.is_healthy
        assert info.health_message == ""

    def test_factory_error_marks_unhealthy(self, registry):
        """Should mark unhealthy if factory fails."""
        def bad_factory():
            raise RuntimeError("creation failed")

        registry.register_factory("test", bad_factory)
        result = registry.get("test")

        assert result is None
        info = registry.get_info("test")
        assert not info.is_healthy
        assert "creation failed" in info.health_message


# =============================================================================
# ComponentRegistry Access Tracking Tests
# =============================================================================


class TestComponentRegistryTracking:
    """Tests for access tracking."""

    def test_tracks_access_count(self, registry):
        """Should track access count."""
        registry.register_factory("test", dict)

        for _ in range(5):
            registry.get("test")

        info = registry.get_info("test")
        assert info.access_count == 5

    def test_tracks_last_accessed(self, registry):
        """Should track last accessed time."""
        registry.register_factory("test", dict)

        before = time.time()
        registry.get("test")
        after = time.time()

        info = registry.get_info("test")
        assert before <= info.last_accessed <= after


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_singleton(self):
        """Should return singleton registry."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_reset_registry(self):
        """Should reset global registry."""
        reg1 = get_registry()
        reg1.register_factory("test", dict)
        reg1.get("test")

        reset_registry()

        reg2 = get_registry()
        assert reg2 is not reg1
        assert not reg2.has("test")


# =============================================================================
# Decorator Tests
# =============================================================================


class TestComponentDecorator:
    """Tests for @component decorator."""

    def test_decorator_registers_class(self):
        """Should register decorated class."""
        @component("decorated_test")
        class MyService:
            def __init__(self):
                self.value = 42

        registry = get_registry()
        assert registry.has("decorated_test")

        instance = registry.get("decorated_test")
        assert isinstance(instance, MyService)
        assert instance.value == 42

    def test_decorator_with_metadata(self):
        """Should pass metadata."""
        @component("meta_test", metadata={"version": "1.0"})
        class MetaService:
            pass

        registry = get_registry()
        info = registry.get_info("meta_test")
        assert info.metadata["version"] == "1.0"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestComponentRegistryThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_get(self, registry):
        """Should handle concurrent gets."""
        create_count = []
        lock = threading.Lock()

        def factory():
            with lock:
                create_count.append(1)
            return {"id": len(create_count)}

        registry.register_factory("test", factory)

        results = []
        threads = []

        def get_component():
            result = registry.get("test")
            results.append(result)

        for _ in range(10):
            t = threading.Thread(target=get_component)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should get same instance
        assert len(results) == 10
        assert all(r is results[0] for r in results)
        # Factory should only be called once
        assert len(create_count) == 1
