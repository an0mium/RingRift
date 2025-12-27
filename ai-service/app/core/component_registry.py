"""ComponentRegistry - Centralized singleton management for cluster components.

This module provides a centralized registry for managing singleton instances
across the codebase. Instead of 170+ scattered `_instance: T | None = None`
patterns, components can register themselves with this registry.

Benefits:
- Single point of control for all singletons
- Easy reset for testing
- Dependency injection support
- Health tracking across all components
- Lazy initialization

Usage:
    from app.core.component_registry import get_registry, component

    # Option 1: Decorator-based registration (preferred)
    @component("backpressure_monitor")
    class BackpressureMonitor:
        def __init__(self):
            self.config = BackpressureConfig()

    # Get instance (lazy-created on first access)
    monitor = get_registry().get("backpressure_monitor")

    # Option 2: Factory-based registration
    def create_sync_router():
        return SyncRouter(config=load_config())

    get_registry().register_factory("sync_router", create_sync_router)
    router = get_registry().get("sync_router")

    # Reset for testing
    get_registry().reset("backpressure_monitor")  # Reset single component
    get_registry().reset_all()  # Reset all components

December 2025 - Phase 2 architecture cleanup.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ComponentInfo:
    """Information about a registered component."""

    name: str
    instance: Any | None = None
    factory: Callable[[], Any] | None = None
    cls: type | None = None
    created_at: float | None = None
    access_count: int = 0
    last_accessed: float | None = None
    is_healthy: bool = True
    health_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_instantiated(self) -> bool:
        """Check if the component has been instantiated."""
        return self.instance is not None


class ComponentRegistry:
    """Centralized registry for managing singleton components.

    Thread-safe singleton management with lazy initialization,
    factory support, and health tracking.
    """

    def __init__(self) -> None:
        self._components: dict[str, ComponentInfo] = {}
        self._lock = threading.RLock()
        self._initialized_at = time.time()

    def register(
        self,
        name: str,
        instance: Any | None = None,
        factory: Callable[[], Any] | None = None,
        cls: type | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a component.

        Args:
            name: Unique component name
            instance: Pre-created instance (optional)
            factory: Factory function to create instance (optional)
            cls: Class to instantiate (optional, requires no-arg constructor)
            metadata: Additional metadata about the component

        At least one of instance, factory, or cls must be provided.
        """
        if instance is None and factory is None and cls is None:
            raise ValueError(
                f"Component '{name}' must have instance, factory, or cls"
            )

        with self._lock:
            if name in self._components:
                logger.warning(f"[ComponentRegistry] Overwriting component: {name}")

            info = ComponentInfo(
                name=name,
                instance=instance,
                factory=factory,
                cls=cls,
                created_at=time.time() if instance else None,
                metadata=metadata or {},
            )
            self._components[name] = info
            logger.debug(f"[ComponentRegistry] Registered: {name}")

    def register_factory(
        self,
        name: str,
        factory: Callable[[], Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a component with a factory function.

        Args:
            name: Unique component name
            factory: Function that creates the component instance
            metadata: Additional metadata
        """
        self.register(name, factory=factory, metadata=metadata)

    def register_class(
        self,
        name: str,
        cls: type,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a component class (must have no-arg constructor).

        Args:
            name: Unique component name
            cls: Class to instantiate
            metadata: Additional metadata
        """
        self.register(name, cls=cls, metadata=metadata)

    def get(self, name: str, default: Any = None) -> Any:
        """Get a component instance, creating it if needed.

        Args:
            name: Component name
            default: Default value if component not registered

        Returns:
            Component instance or default
        """
        with self._lock:
            info = self._components.get(name)
            if info is None:
                return default

            # Lazy initialization
            if info.instance is None:
                try:
                    if info.factory:
                        info.instance = info.factory()
                    elif info.cls:
                        info.instance = info.cls()
                    info.created_at = time.time()
                    logger.debug(f"[ComponentRegistry] Created: {name}")
                except Exception as e:
                    logger.error(f"[ComponentRegistry] Failed to create {name}: {e}")
                    info.is_healthy = False
                    info.health_message = str(e)
                    return default

            # Update access tracking
            info.access_count += 1
            info.last_accessed = time.time()

            return info.instance

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], Any],
    ) -> Any:
        """Get a component, registering and creating it if not found.

        Args:
            name: Component name
            factory: Factory to use if component not registered

        Returns:
            Component instance
        """
        with self._lock:
            if name not in self._components:
                self.register_factory(name, factory)
            return self.get(name)

    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components

    def is_instantiated(self, name: str) -> bool:
        """Check if a component has been instantiated."""
        info = self._components.get(name)
        return info is not None and info.is_instantiated

    def reset(self, name: str) -> bool:
        """Reset a component instance (keeps registration).

        Args:
            name: Component name

        Returns:
            True if component was reset
        """
        with self._lock:
            info = self._components.get(name)
            if info is None:
                return False

            info.instance = None
            info.created_at = None
            info.access_count = 0
            info.last_accessed = None
            info.is_healthy = True
            info.health_message = ""
            logger.debug(f"[ComponentRegistry] Reset: {name}")
            return True

    def reset_all(self) -> int:
        """Reset all component instances.

        Returns:
            Number of components reset
        """
        with self._lock:
            count = 0
            for name in list(self._components.keys()):
                if self.reset(name):
                    count += 1
            logger.info(f"[ComponentRegistry] Reset {count} components")
            return count

    def unregister(self, name: str) -> bool:
        """Completely remove a component registration.

        Args:
            name: Component name

        Returns:
            True if component was removed
        """
        with self._lock:
            if name in self._components:
                del self._components[name]
                logger.debug(f"[ComponentRegistry] Unregistered: {name}")
                return True
            return False

    def list_components(self) -> list[str]:
        """List all registered component names."""
        return list(self._components.keys())

    def get_info(self, name: str) -> ComponentInfo | None:
        """Get information about a component."""
        return self._components.get(name)

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of all components.

        Returns:
            Dict with component health information
        """
        with self._lock:
            components = {}
            healthy_count = 0
            unhealthy_count = 0
            instantiated_count = 0

            for name, info in self._components.items():
                if info.is_instantiated:
                    instantiated_count += 1
                if info.is_healthy:
                    healthy_count += 1
                else:
                    unhealthy_count += 1

                components[name] = {
                    "is_instantiated": info.is_instantiated,
                    "is_healthy": info.is_healthy,
                    "health_message": info.health_message,
                    "access_count": info.access_count,
                    "created_at": info.created_at,
                    "last_accessed": info.last_accessed,
                }

            return {
                "total_registered": len(self._components),
                "instantiated": instantiated_count,
                "healthy": healthy_count,
                "unhealthy": unhealthy_count,
                "uptime_seconds": time.time() - self._initialized_at,
                "components": components,
            }

    def mark_unhealthy(
        self,
        name: str,
        message: str = "",
    ) -> bool:
        """Mark a component as unhealthy.

        Args:
            name: Component name
            message: Health message

        Returns:
            True if component was found
        """
        with self._lock:
            info = self._components.get(name)
            if info:
                info.is_healthy = False
                info.health_message = message
                return True
            return False

    def mark_healthy(self, name: str) -> bool:
        """Mark a component as healthy.

        Args:
            name: Component name

        Returns:
            True if component was found
        """
        with self._lock:
            info = self._components.get(name)
            if info:
                info.is_healthy = True
                info.health_message = ""
                return True
            return False


# Module-level singleton
_registry: ComponentRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ComponentRegistry:
    """Get the global component registry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ComponentRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.reset_all()
        _registry = None


def component(
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Callable[[type], type]:
    """Decorator to register a class as a component.

    Usage:
        @component("my_service")
        class MyService:
            def __init__(self):
                pass

        # Get instance
        service = get_registry().get("my_service")

    Args:
        name: Unique component name
        metadata: Additional metadata

    Returns:
        Class decorator
    """
    def decorator(cls: type) -> type:
        get_registry().register_class(name, cls, metadata=metadata)
        return cls
    return decorator


__all__ = [
    "ComponentInfo",
    "ComponentRegistry",
    "component",
    "get_registry",
    "reset_registry",
]
