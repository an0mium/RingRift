"""
Thread-safe singleton patterns for RingRift.

This module provides standardized singleton implementations to replace
duplicate patterns across the codebase. Supports:
- Simple singletons (no initialization parameters)
- Parameterized singletons (with init args)
- Thread-safe access
- Reset capability for testing

Usage:
    # Method 1: Metaclass (preferred for new classes)
    class MyService(metaclass=SingletonMeta):
        def __init__(self, config: str = "default"):
            self.config = config

    # Method 2: Mixin (for existing classes)
    class MyRegistry(SingletonMixin):
        _instance: MyRegistry | None = None

        @classmethod
        def get_instance(cls) -> MyRegistry:
            return cls._get_or_create_instance()

    # Method 3: Decorator (simplest)
    @singleton
    class MyCache:
        pass

All patterns support:
    - MyClass.get_instance() - get or create singleton
    - MyClass.reset_instance() - clear singleton (for testing)
    - MyClass.has_instance() - check if instance exists
"""

from __future__ import annotations

import functools
import threading
from typing import Any, TypeVar, Generic

__all__ = [
    "SingletonMeta",
    "SingletonMixin",
    "singleton",
    "ThreadSafeSingletonMixin",
]

T = TypeVar("T")


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass.

    Creates exactly one instance per class. Supports initialization
    with arguments (first call's args are used).

    Example:
        class DatabaseConnection(metaclass=SingletonMeta):
            def __init__(self, host: str = "localhost"):
                self.host = host

        conn1 = DatabaseConnection("db.example.com")
        conn2 = DatabaseConnection()  # Returns same instance
        assert conn1 is conn2
        assert conn1.host == "db.example.com"

    Testing:
        DatabaseConnection.reset_instance()  # Clear for next test
    """

    _instances: dict[type, Any] = {}
    _locks: dict[type, threading.RLock] = {}
    _global_lock = threading.RLock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create or return singleton instance."""
        # Ensure we have a lock for this class
        if cls not in cls._locks:
            with cls._global_lock:
                if cls not in cls._locks:
                    cls._locks[cls] = threading.RLock()

        # Double-checked locking for thread safety
        if cls not in cls._instances:
            with cls._locks[cls]:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    def get_instance(cls, *args: Any, **kwargs: Any) -> Any:
        """Get or create the singleton instance.

        Alias for __call__ for explicit singleton access.
        """
        return cls(*args, **kwargs)

    def reset_instance(cls) -> None:
        """Clear the singleton instance.

        Primarily for testing. After reset, next access creates new instance.
        """
        with cls._global_lock:
            if cls in cls._instances:
                # Allow cleanup if instance has a _cleanup method
                instance = cls._instances[cls]
                if hasattr(instance, "_cleanup"):
                    try:
                        instance._cleanup()
                    except (AttributeError, RuntimeError):
                        pass  # Don't fail reset on cleanup errors
                del cls._instances[cls]

    def has_instance(cls) -> bool:
        """Check if singleton instance exists."""
        return cls in cls._instances


class SingletonMixin:
    """
    Mixin class for singleton pattern.

    Use when you can't change the metaclass (e.g., already using another).
    Subclasses must:
    1. Define _instance class variable
    2. Call _get_or_create_instance() in get_instance()

    Example:
        class MyService(SomeBaseClass, SingletonMixin):
            _instance: MyService | None = None
            _lock: threading.RLock = threading.RLock()

            @classmethod
            def get_instance(cls, config: str = "default") -> MyService:
                return cls._get_or_create_instance(config=config)

            def __init__(self, config: str = "default"):
                self.config = config
    """

    _instance: Any = None
    _lock: threading.RLock

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure each subclass has its own instance and lock."""
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._lock = threading.RLock()

    @classmethod
    def _get_or_create_instance(cls, *args: Any, **kwargs: Any) -> Any:
        """Thread-safe singleton instance access.

        Call this from your get_instance() method.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(*args, **kwargs)
        return cls._instance

    @classmethod
    def get_instance(cls) -> Any:
        """Get or create the singleton instance.

        Override this in subclasses to add parameters.
        """
        return cls._get_or_create_instance()

    @classmethod
    def reset_instance(cls) -> None:
        """Clear the singleton instance for testing."""
        with cls._lock:
            if cls._instance is not None:
                # Allow cleanup
                if hasattr(cls._instance, "_cleanup"):
                    try:
                        cls._instance._cleanup()
                    except (AttributeError, RuntimeError):
                        pass
                # Allow state saving (for coordinators)
                if hasattr(cls._instance, "_save_state"):
                    try:
                        cls._instance._save_state()
                    except (AttributeError, RuntimeError, OSError):
                        pass
                cls._instance = None

    @classmethod
    def has_instance(cls) -> bool:
        """Check if singleton instance exists."""
        return cls._instance is not None


class ThreadSafeSingletonMixin(SingletonMixin):
    """
    Enhanced mixin for singletons requiring explicit thread safety.

    Adds additional safeguards for highly concurrent access patterns.
    Use for registries, caches, and coordinators accessed from many threads.

    Example:
        class MetricRegistry(ThreadSafeSingletonMixin):
            def __init__(self):
                self._metrics: dict[str, Metric] = {}
                self._metrics_lock = threading.RLock()

            def register(self, name: str, metric: Metric) -> None:
                with self._metrics_lock:
                    self._metrics[name] = metric
    """

    @classmethod
    def _get_or_create_instance(cls, *args: Any, **kwargs: Any) -> Any:
        """Double-checked locking with memory barrier."""
        # First check without lock (fast path)
        instance = cls._instance
        if instance is not None:
            return instance

        # Slow path with lock
        with cls._lock:
            # Check again inside lock
            if cls._instance is None:
                cls._instance = cls(*args, **kwargs)
            return cls._instance


def singleton(cls: type[T]) -> type[T]:
    """
    Decorator to make a class a singleton.

    Simplest way to create a singleton. The decorated class will
    return the same instance on every instantiation.

    Example:
        @singleton
        class AppConfig:
            def __init__(self):
                self.debug = False

        config1 = AppConfig()
        config2 = AppConfig()
        assert config1 is config2

    Note: Does not support reset_instance(). Use SingletonMeta or
    SingletonMixin if you need testing support.
    """
    _instance: T | None = None
    _lock = threading.RLock()
    _original_new = cls.__new__
    _original_init = cls.__init__

    @functools.wraps(cls.__new__)
    def __new__(inner_cls: type[T], *args: Any, **kwargs: Any) -> T:
        nonlocal _instance
        if _instance is None:
            with _lock:
                if _instance is None:
                    if _original_new is object.__new__:
                        _instance = object.__new__(inner_cls)
                    else:
                        _instance = _original_new(inner_cls, *args, **kwargs)
        return _instance  # type: ignore

    _initialized = False

    @functools.wraps(cls.__init__)
    def __init__(self: T, *args: Any, **kwargs: Any) -> None:
        nonlocal _initialized
        if not _initialized:
            with _lock:
                if not _initialized:
                    _original_init(self, *args, **kwargs)
                    _initialized = True

    cls.__new__ = __new__  # type: ignore
    cls.__init__ = __init__  # type: ignore

    # Add helper methods
    @classmethod
    def get_instance(inner_cls: type[T]) -> T:
        return inner_cls()

    @classmethod
    def has_instance(inner_cls: type[T]) -> bool:
        return _instance is not None

    @classmethod
    def reset_instance(inner_cls: type[T]) -> None:
        nonlocal _instance, _initialized
        with _lock:
            _instance = None
            _initialized = False

    cls.get_instance = get_instance  # type: ignore
    cls.has_instance = has_instance  # type: ignore
    cls.reset_instance = reset_instance  # type: ignore

    return cls
