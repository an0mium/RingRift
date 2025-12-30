# HandlerBase API Reference

**Last Updated:** December 29, 2025
**Location:** `app/coordination/handler_base.py`
**Test File:** `tests/unit/coordination/test_handler_base.py` (45 tests)

## Overview

`HandlerBase` is the unified base class for daemon handlers in the RingRift coordination infrastructure. It consolidates patterns from 15+ daemon files to reduce duplication and ensure consistent behavior.

## Key Features

| Feature                  | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| **Singleton Pattern**    | Thread-safe singleton management via `get_instance()`             |
| **Event Subscription**   | Automatic subscription to events via `_get_event_subscriptions()` |
| **Event Deduplication**  | SHA256-based deduplication with configurable TTL                  |
| **Health Checks**        | Standardized `HealthCheckResult` format                           |
| **Error Tracking**       | Bounded error log with recent errors                              |
| **Lifecycle Management** | `start()` / `stop()` with hooks                                   |

---

## Quick Start

```python
from app.coordination.handler_base import HandlerBase, HealthCheckResult

class MyDaemon(HandlerBase):
    def __init__(self):
        super().__init__(name="my_daemon", cycle_interval=60.0)

    def _get_event_subscriptions(self) -> dict:
        return {
            "training_completed": self._on_training_completed,
            "evaluation_completed": self._on_evaluation_completed,
        }

    async def _run_cycle(self) -> None:
        """Main work loop - runs every cycle_interval seconds."""
        # Do periodic work here
        pass

    async def _on_training_completed(self, event: dict) -> None:
        """Handle training_completed event."""
        if self._is_duplicate_event(event):
            return  # Skip duplicate

        # Process event
        config_key = event.get("config_key")
        self._mark_event_processed(event)

# Usage
daemon = MyDaemon.get_instance()
await daemon.start()
# ... later ...
await daemon.stop()
```

---

## Constructor

```python
def __init__(
    self,
    name: str,
    config: Any | None = None,
    *,
    cycle_interval: float = 60.0,
    dedup_enabled: bool = True,
)
```

| Parameter        | Type    | Default  | Description                          |
| ---------------- | ------- | -------- | ------------------------------------ |
| `name`           | `str`   | Required | Handler name for logging             |
| `config`         | `Any`   | `None`   | Optional configuration object        |
| `cycle_interval` | `float` | `60.0`   | Seconds between `_run_cycle()` calls |
| `dedup_enabled`  | `bool`  | `True`   | Enable event deduplication           |

---

## Abstract Methods

These methods **must** or **should** be implemented by subclasses:

### `_run_cycle()` (Required)

```python
@abstractmethod
async def _run_cycle(self) -> None:
    """Main work loop iteration. Called every cycle_interval seconds."""
    pass
```

### `_get_event_subscriptions()` (Recommended)

```python
def _get_event_subscriptions(self) -> dict[str, Callable]:
    """Return event_type -> handler mapping for automatic subscription."""
    return {
        "training_completed": self._on_training_completed,
    }
```

---

## Lifecycle Methods

### `start()`

```python
async def start(self) -> None
```

Starts the handler:

1. Subscribes to all events from `_get_event_subscriptions()`
2. Calls `_on_start()` hook
3. Starts main loop task

### `stop()`

```python
async def stop(self) -> None
```

Stops the handler gracefully:

1. Cancels main loop task (5s timeout)
2. Unsubscribes from all events
3. Calls `_on_stop()` hook

### Lifecycle Hooks

```python
async def _on_start(self) -> None:
    """Called before main loop starts. Override in subclass."""
    pass

async def _on_stop(self) -> None:
    """Called after main loop stops. Override in subclass."""
    pass
```

---

## Singleton Management

### `get_instance()`

```python
@classmethod
def get_instance(cls, *args, **kwargs) -> "HandlerBase"
```

Gets or creates the singleton instance. Thread-safe.

```python
daemon = MyDaemon.get_instance()
daemon2 = MyDaemon.get_instance()  # Same instance
assert daemon is daemon2
```

### `reset_instance()`

```python
@classmethod
def reset_instance(cls) -> None
```

Resets the singleton instance. Useful for testing.

```python
@pytest.fixture(autouse=True)
def reset_singletons():
    yield
    MyDaemon.reset_instance()
```

### `has_instance()`

```python
@classmethod
def has_instance(cls) -> bool
```

Checks if a singleton instance exists.

---

## Event Deduplication

### `_is_duplicate_event()`

```python
def _is_duplicate_event(
    self,
    event: dict[str, Any],
    key_fields: list[str] | None = None
) -> bool
```

Checks if an event is a duplicate based on SHA256 hash.

| Parameter    | Description                                           |
| ------------ | ----------------------------------------------------- |
| `event`      | Event payload dictionary                              |
| `key_fields` | Optional list of fields to hash (default: all fields) |

**Returns:** `True` if duplicate (skip), `False` if new event.

```python
async def _on_training_completed(self, event: dict) -> None:
    if self._is_duplicate_event(event, key_fields=["config_key", "model_path"]):
        return  # Skip duplicate

    # Process event...
    self._mark_event_processed(event)
```

### Deduplication Configuration

```python
class MyDaemon(HandlerBase):
    DEDUP_TTL_SECONDS = 600.0  # 10 minutes (default: 300)
    DEDUP_MAX_SIZE = 2000      # Max cached hashes (default: 1000)
```

---

## Health Check

### `health_check()`

```python
def health_check(self) -> HealthCheckResult
```

Returns standardized health status:

```python
HealthCheckResult(
    healthy=True,           # Overall health
    status=CoordinatorStatus.RUNNING,
    message="Operating normally",
    details={...}           # Custom details
)
```

**Health Thresholds:**

- Error rate > 50%: `healthy=False`, status=`ERROR`
- Error rate > 20%: `healthy=True`, status=`DEGRADED`
- Otherwise: `healthy=True`, status=`RUNNING`

### Custom Health Check

```python
class MyDaemon(HandlerBase):
    def health_check(self) -> HealthCheckResult:
        # Call parent for base checks
        base_health = super().health_check()

        # Add custom checks
        if self._some_critical_condition():
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="Critical condition detected",
                details=self._get_health_details(),
            )

        return base_health
```

### `get_status()`

```python
def get_status(self) -> dict[str, Any]
```

Returns comprehensive status dictionary:

```python
{
    "name": "my_daemon",
    "running": True,
    "status": "running",
    "health": {...},
    "stats": {
        "events_processed": 42,
        "events_deduplicated": 3,
        "cycles_completed": 100,
        "errors_count": 2,
    }
}
```

---

## Error Tracking

### `_record_error()`

```python
def _record_error(self, error: str, exc: Exception | None = None) -> None
```

Records an error with bounded logging:

- Increments `errors_count`
- Stores in `_error_log` (max 50 entries)
- Updates `last_error` and `last_error_time`

```python
try:
    await self._do_something()
except ValueError as e:
    self._record_error(f"Failed to process: {e}", exc=e)
```

### `get_recent_errors()`

```python
def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]
```

Returns recent errors from the log:

```python
[
    {
        "timestamp": 1735500000.0,
        "error": "Connection failed",
        "exception": "ConnectionError: timeout"
    },
    ...
]
```

---

## Statistics

### `HandlerStats` Dataclass

```python
@dataclass
class HandlerStats:
    events_processed: int = 0
    events_deduplicated: int = 0
    success_count: int = 0
    errors_count: int = 0
    last_error: str = ""
    last_error_time: float = 0.0
    cycles_completed: int = 0
    started_at: float = 0.0
    last_activity: float = 0.0
    subscribed: bool = False
    custom_stats: dict = field(default_factory=dict)
```

### Properties

| Property         | Type           | Description                |
| ---------------- | -------------- | -------------------------- |
| `stats`          | `HandlerStats` | Handler statistics         |
| `name`           | `str`          | Handler name               |
| `is_running`     | `bool`         | Whether handler is running |
| `uptime_seconds` | `float`        | Seconds since start        |

---

## Backward Compatibility

### Legacy API (base_handler.py)

These methods maintain compatibility with older code:

| Method                        | Replacement                          |
| ----------------------------- | ------------------------------------ |
| `subscribe()`                 | `start()` handles subscription       |
| `unsubscribe()`               | `stop()` handles unsubscription      |
| `get_stats()`                 | `stats.to_dict()`                    |
| `reset()`                     | `reset_instance()`                   |
| `add_custom_stat(key, value)` | Access `stats.custom_stats` directly |

### Aliases

```python
# All point to HandlerBase
BaseEventHandler = HandlerBase
BaseSingletonHandler = HandlerBase
MultiEventHandler = HandlerBase
```

### Legacy Event Subscriptions

The old `_get_subscriptions()` method is supported with deprecation warning:

```python
# Legacy (deprecated - emits warning)
def _get_subscriptions(self) -> dict:
    return {"event": self.handler}

# New (canonical)
def _get_event_subscriptions(self) -> dict:
    return {"event": self.handler}
```

---

## Best Practices

### 1. Always Use Singleton Pattern

```python
# Good
daemon = MyDaemon.get_instance()

# Bad - creates multiple instances
daemon1 = MyDaemon()
daemon2 = MyDaemon()
```

### 2. Check for Duplicates in Event Handlers

```python
async def _on_event(self, event: dict) -> None:
    if self._is_duplicate_event(event):
        return
    # Process...
```

### 3. Mark Events as Processed

```python
async def _on_event(self, event: dict) -> None:
    try:
        await self._process_event(event)
        self._mark_event_processed(event)
    except Exception as e:
        self._record_error(f"Failed: {e}", exc=e)
```

### 4. Reset Singletons in Tests

```python
@pytest.fixture(autouse=True)
def reset_singletons():
    yield
    MyDaemon.reset_instance()
```

### 5. Override `health_check()` for Custom Metrics

```python
def health_check(self) -> HealthCheckResult:
    base = super().health_check()
    if not base.healthy:
        return base

    # Add custom checks
    if self._queue_size > 1000:
        return HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.DEGRADED,
            message=f"Queue backlog: {self._queue_size}",
        )
    return base
```

---

## Related Classes

| Class             | Purpose                                       |
| ----------------- | --------------------------------------------- |
| `MonitorBase`     | Extended base for health monitoring daemons   |
| `DaemonAdapter`   | Wrapper for daemons in DaemonManager          |
| `CoordinatorBase` | Base for coordinators with SQLite persistence |

---

## See Also

- `DAEMON_REGISTRY.md` - Full daemon reference
- `EVENT_SYSTEM_REFERENCE.md` - Event system documentation
- `app/coordination/monitor_base.py` - Extended monitoring base class
