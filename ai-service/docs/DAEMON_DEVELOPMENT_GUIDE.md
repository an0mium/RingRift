# Daemon Development Guide

> **Target Audience**: Developers adding new daemons to the RingRift coordination infrastructure.

This guide walks through the process of adding a new daemon to the `DaemonManager` system. All daemons follow a consistent pattern for lifecycle management, health checking, and event integration.

## Quick Start (5 Steps)

1. **Define the DaemonType** in `app/coordination/daemon_types.py`
2. **Create the daemon class** in a new module
3. **Create a runner function** in `app/coordination/daemon_runners.py`
4. **Register in the daemon registry** in `app/coordination/daemon_registry.py`
5. **Add to startup order** if needed

---

## Step 1: Define the DaemonType

Add your new daemon type to the `DaemonType` enum in `app/coordination/daemon_types.py`:

```python
class DaemonType(Enum):
    # ... existing daemons ...

    # Your new daemon (add at end of appropriate category)
    MY_NEW_DAEMON = "my_new_daemon"  # Brief description
```

**Naming Conventions**:

- Use UPPER_SNAKE_CASE for enum names
- Use lower_snake_case for values
- Group with related daemons (sync, health, event, pipeline, etc.)

---

## Step 2: Create the Daemon Class

Create a new module in `app/coordination/`. Your daemon should inherit from `BaseDaemon` or implement the daemon protocol.

### Using BaseDaemon (Recommended)

```python
# app/coordination/my_new_daemon.py
"""My New Daemon - Brief description of what it does."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.coordination.base_daemon import BaseDaemon
from app.coordination.protocols import HealthCheckResult

logger = logging.getLogger(__name__)


@dataclass
class MyNewDaemonConfig:
    """Configuration for MyNewDaemon."""

    check_interval_seconds: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "MyNewDaemonConfig":
        """Load configuration from environment variables."""
        import os

        return cls(
            check_interval_seconds=float(
                os.environ.get("RINGRIFT_MY_DAEMON_INTERVAL", "60.0")
            ),
            max_retries=int(
                os.environ.get("RINGRIFT_MY_DAEMON_MAX_RETRIES", "3")
            ),
        )


class MyNewDaemon(BaseDaemon):
    """My New Daemon - does something useful."""

    def __init__(self, config: MyNewDaemonConfig | None = None):
        """Initialize the daemon.

        Args:
            config: Optional configuration. Uses environment defaults if not provided.
        """
        super().__init__(daemon_name="my_new_daemon")
        self.config = config or MyNewDaemonConfig.from_env()
        self._last_run_time: float = 0.0
        self._cycle_count: int = 0
        self._errors: int = 0

    async def _setup(self) -> None:
        """Called once before the main loop starts."""
        logger.info(f"MyNewDaemon starting with interval={self.config.check_interval_seconds}s")
        # Initialize resources, subscribe to events, etc.
        self._subscribe_to_events()

    async def _run_cycle(self) -> None:
        """Called repeatedly in the main loop."""
        import time

        self._last_run_time = time.time()
        self._cycle_count += 1

        try:
            # Do the actual work
            await self._do_work()
        except Exception as e:
            self._errors += 1
            logger.error(f"MyNewDaemon cycle failed: {e}")
            if self._errors > self.config.max_retries:
                raise  # Let DaemonManager handle restart

    async def _do_work(self) -> None:
        """Main work logic - override in subclasses or modify here."""
        # Your daemon logic here
        pass

    async def _cleanup(self) -> None:
        """Called once when the daemon is stopping."""
        logger.info(f"MyNewDaemon stopping after {self._cycle_count} cycles")
        # Clean up resources, unsubscribe from events, etc.

    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            bus.subscribe("MY_EVENT_TYPE", self._on_my_event)
        except ImportError:
            logger.debug("Event bus not available, skipping subscription")

    async def _on_my_event(self, event: dict) -> None:
        """Handle MY_EVENT_TYPE events."""
        payload = event.get("payload", {})
        logger.debug(f"Received event: {payload}")

    def health_check(self) -> HealthCheckResult:
        """Return current health status.

        Required for DaemonManager integration.
        """
        import time

        is_healthy = self._errors < self.config.max_retries
        stale = time.time() - self._last_run_time > self.config.check_interval_seconds * 3

        if stale:
            is_healthy = False

        return HealthCheckResult(
            healthy=is_healthy,
            message="MyNewDaemon running" if is_healthy else "MyNewDaemon degraded",
            details={
                "cycle_count": self._cycle_count,
                "errors": self._errors,
                "last_run_time": self._last_run_time,
                "stale": stale,
            },
        )

    @property
    def cycle_interval(self) -> float:
        """Return the interval between cycles."""
        return self.config.check_interval_seconds


# Singleton pattern (optional but recommended)
_instance: MyNewDaemon | None = None


def get_my_new_daemon() -> MyNewDaemon:
    """Get or create the singleton instance."""
    global _instance
    if _instance is None:
        _instance = MyNewDaemon()
    return _instance


def reset_my_new_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
```

### Key Requirements

1. **`health_check()`**: Must return `HealthCheckResult` for DaemonManager monitoring
2. **`cycle_interval`**: Property returning seconds between cycles
3. **`_setup()`**: One-time initialization before main loop
4. **`_run_cycle()`**: Called repeatedly at `cycle_interval`
5. **`_cleanup()`**: Clean up when daemon stops

---

## Step 3: Create a Runner Function

Add a runner function to `app/coordination/daemon_runners.py`:

```python
async def create_my_new_daemon() -> None:
    """Start MyNewDaemon for background processing.

    This daemon does [brief description].
    """
    try:
        from app.coordination.my_new_daemon import MyNewDaemon, MyNewDaemonConfig

        config = MyNewDaemonConfig.from_env()
        daemon = MyNewDaemon(config)
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"MyNewDaemon not available: {e}")
        raise
```

**Pattern Notes**:

- Use lazy imports to avoid circular dependencies
- Load config from environment
- Call `await daemon.start()` then `await _wait_for_daemon(daemon)`
- Handle ImportError gracefully

---

## Step 4: Register in Daemon Registry

Add a `DaemonSpec` to `DAEMON_REGISTRY` in `app/coordination/daemon_registry.py`:

```python
from app.coordination.daemon_types import DaemonType

DAEMON_REGISTRY: dict[DaemonType, DaemonSpec] = {
    # ... existing entries ...

    DaemonType.MY_NEW_DAEMON: DaemonSpec(
        runner_name="create_my_new_daemon",
        depends_on=(DaemonType.EVENT_ROUTER,),  # Dependencies
        category="pipeline",  # Category for grouping
        health_check_interval=30.0,  # Custom interval (optional)
        auto_restart=True,  # Restart on failure
        max_restarts=5,  # Max restart attempts
    ),
}
```

**DaemonSpec Fields**:

| Field                   | Type                   | Default  | Description                                       |
| ----------------------- | ---------------------- | -------- | ------------------------------------------------- |
| `runner_name`           | str                    | required | Function name in daemon_runners.py                |
| `depends_on`            | tuple[DaemonType, ...] | ()       | Must be running first                             |
| `category`              | str                    | "misc"   | Grouping: sync, event, health, pipeline, resource |
| `health_check_interval` | float                  | None     | Custom interval (seconds)                         |
| `auto_restart`          | bool                   | True     | Restart on failure                                |
| `max_restarts`          | int                    | 5        | Max restart attempts                              |
| `deprecated`            | bool                   | False    | Mark as deprecated                                |
| `deprecated_message`    | str                    | ""       | Migration guidance                                |

---

## Step 5: Add to Startup Order (If Needed)

If your daemon has specific ordering requirements, update `DAEMON_STARTUP_ORDER` in `daemon_types.py`:

```python
DAEMON_STARTUP_ORDER: list[DaemonType] = [
    # Core infrastructure (must be first)
    DaemonType.EVENT_ROUTER,

    # Event consumers (before event producers)
    DaemonType.DATA_PIPELINE,
    DaemonType.FEEDBACK_LOOP,
    DaemonType.MY_NEW_DAEMON,  # Add here if it consumes events

    # Event producers (after consumers)
    DaemonType.AUTO_SYNC,

    # ... rest of order ...
]
```

**Ordering Rules**:

1. `EVENT_ROUTER` is always first
2. Event consumers before event producers
3. Dependencies before dependents
4. Critical daemons before non-critical

---

## Testing Your Daemon

### Unit Tests

Create `tests/unit/coordination/test_my_new_daemon.py`:

```python
"""Tests for MyNewDaemon."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.my_new_daemon import (
    MyNewDaemon,
    MyNewDaemonConfig,
    get_my_new_daemon,
    reset_my_new_daemon,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_my_new_daemon()
    yield
    reset_my_new_daemon()


class TestMyNewDaemonConfig:
    """Tests for configuration."""

    def test_default_config(self):
        config = MyNewDaemonConfig()
        assert config.check_interval_seconds == 60.0
        assert config.max_retries == 3

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("RINGRIFT_MY_DAEMON_INTERVAL", "30.0")
        config = MyNewDaemonConfig.from_env()
        assert config.check_interval_seconds == 30.0


class TestMyNewDaemon:
    """Tests for daemon lifecycle."""

    def test_init(self):
        daemon = MyNewDaemon()
        assert daemon.config is not None
        assert daemon._cycle_count == 0

    @pytest.mark.asyncio
    async def test_start_stop(self):
        daemon = MyNewDaemon()
        await daemon.start()
        assert daemon.is_running
        await daemon.stop()
        assert not daemon.is_running

    def test_health_check_healthy(self):
        daemon = MyNewDaemon()
        daemon._cycle_count = 10
        daemon._errors = 0
        daemon._last_run_time = time.time()

        health = daemon.health_check()
        assert health.healthy
        assert health.details["cycle_count"] == 10

    def test_health_check_degraded(self):
        daemon = MyNewDaemon()
        daemon._errors = 10  # Above max_retries

        health = daemon.health_check()
        assert not health.healthy


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_my_new_daemon_singleton(self):
        daemon1 = get_my_new_daemon()
        daemon2 = get_my_new_daemon()
        assert daemon1 is daemon2

    def test_reset_my_new_daemon(self):
        daemon1 = get_my_new_daemon()
        reset_my_new_daemon()
        daemon2 = get_my_new_daemon()
        assert daemon1 is not daemon2
```

### Integration Tests

Create `tests/integration/coordination/test_my_new_daemon_integration.py`:

```python
"""Integration tests for MyNewDaemon."""

import pytest

from app.coordination.daemon_runners import get_runner
from app.coordination.daemon_types import DaemonType
from app.coordination.daemon_registry import DAEMON_REGISTRY, validate_registry


class TestDaemonRegistration:
    """Tests for daemon registry integration."""

    def test_daemon_in_registry(self):
        """Verify daemon is registered."""
        assert DaemonType.MY_NEW_DAEMON in DAEMON_REGISTRY

    def test_runner_exists(self):
        """Verify runner function exists."""
        runner = get_runner(DaemonType.MY_NEW_DAEMON)
        assert runner is not None
        assert callable(runner)

    def test_registry_validates(self):
        """Verify registry passes validation."""
        errors = validate_registry()
        assert not errors, f"Registry validation failed: {errors}"
```

---

## Health Check Protocol

All daemons must implement `health_check()` returning `HealthCheckResult`:

```python
from app.coordination.protocols import HealthCheckResult

def health_check(self) -> HealthCheckResult:
    return HealthCheckResult(
        healthy=True,  # or False
        message="Human-readable status message",
        details={
            "key": "value",  # Arbitrary details
            "metrics": {...},
        },
    )
```

**Health Status Levels**:

- `healthy=True`: Daemon operating normally
- `healthy=False`: Daemon degraded or failing

The DaemonManager monitors health and may restart unhealthy daemons.

---

## Event Integration

### Subscribing to Events

```python
from app.coordination.event_router import get_event_bus

bus = get_event_bus()
bus.subscribe("TRAINING_COMPLETED", self._on_training_complete)
bus.subscribe("DATA_SYNC_COMPLETED", self._on_sync_complete)
```

### Emitting Events

```python
from app.coordination.event_emitters import emit_training_complete

# Typed emitter (preferred)
emit_training_complete(
    config_key="hex8_2p",
    model_path="models/canonical_hex8_2p.pth",
)

# Or via EventBus
from app.distributed.data_events import DataEventType, DataEvent

bus.publish(DataEvent(
    event_type=DataEventType.MY_NEW_EVENT,
    payload={"key": "value"},
    source="MyNewDaemon",
))
```

---

## Configuration Best Practices

### Environment Variables

Use the `RINGRIFT_` prefix for all config:

```python
import os

interval = float(os.environ.get("RINGRIFT_MY_DAEMON_INTERVAL", "60.0"))
enabled = os.environ.get("RINGRIFT_MY_DAEMON_ENABLED", "true").lower() == "true"
```

### Typed Configuration

Use dataclasses with `from_env()`:

```python
@dataclass
class MyDaemonConfig:
    interval: float = 60.0
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "MyDaemonConfig":
        return cls(
            interval=float(os.environ.get("RINGRIFT_MY_DAEMON_INTERVAL", "60.0")),
            enabled=os.environ.get("RINGRIFT_MY_DAEMON_ENABLED", "true").lower() == "true",
        )
```

---

## Common Patterns

### Backpressure Handling

```python
async def _run_cycle(self) -> None:
    # Check backpressure before doing work
    if self._is_under_backpressure():
        logger.debug("Backpressure active, skipping cycle")
        return

    await self._do_work()
```

### Graceful Degradation

```python
def health_check(self) -> HealthCheckResult:
    if self._critical_component_failed:
        return HealthCheckResult(
            healthy=False,
            message="Critical component unavailable",
            details={"component": "database"},
        )

    if self._optional_component_failed:
        return HealthCheckResult(
            healthy=True,  # Still healthy, just degraded
            message="Optional component unavailable",
            details={"degraded": True},
        )

    return HealthCheckResult(healthy=True, message="Running normally")
```

### Event Deduplication

```python
from app.coordination.sync_bloom_filter import create_event_dedup_filter

class MyDaemon:
    def __init__(self):
        self._event_filter = create_event_dedup_filter()

    async def _on_event(self, event: dict) -> None:
        event_id = event.get("event_id", "")
        if self._event_filter.contains(event_id):
            return  # Already processed
        self._event_filter.add(event_id)
        # Process event...
```

---

## Checklist

Before submitting your daemon:

- [ ] `DaemonType` added to enum
- [ ] Daemon class implements `health_check()`
- [ ] Daemon class has `cycle_interval` property
- [ ] Runner function in `daemon_runners.py`
- [ ] `DaemonSpec` in `DAEMON_REGISTRY`
- [ ] Dependencies specified in `depends_on`
- [ ] Unit tests written and passing
- [ ] Integration test verifies registration
- [ ] Environment variables documented
- [ ] Docstrings and logging added

---

## See Also

- `app/coordination/daemon_manager.py` - DaemonManager implementation
- `app/coordination/daemon_registry.py` - Registry specification
- `app/coordination/daemon_runners.py` - Runner functions
- `app/coordination/base_daemon.py` - Base class
- `app/coordination/protocols.py` - HealthCheckResult and protocols
- `docs/runbooks/DAEMON_FAILURE_RECOVERY.md` - Troubleshooting guide
