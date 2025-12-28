# Daemon Development Guide

A step-by-step guide for adding new daemons to the RingRift AI training infrastructure.

**Created**: December 28, 2025
**Last Updated**: December 28, 2025

---

## Overview

RingRift uses a daemon-based architecture for background processing. There are currently 66+ daemon types handling:

- Data synchronization
- Training pipeline orchestration
- Health monitoring
- Model distribution
- Resource management

This guide covers how to add new daemons using the declarative registry pattern.

---

## Quick Start (5 Steps)

### Step 1: Add DaemonType Enum Value

Edit `app/coordination/daemon_types.py`:

```python
class DaemonType(Enum):
    # ... existing types ...
    MY_NEW_DAEMON = "my_new_daemon"
```

### Step 2: Create the Daemon Class

Create `app/coordination/my_new_daemon.py`:

```python
"""My New Daemon for RingRift.

December 2025 - Brief description of what it does.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus

logger = logging.getLogger(__name__)


class MyNewDaemon(HandlerBase):
    """Daemon that does X, Y, and Z.

    Features:
        - Feature 1
        - Feature 2
        - Feature 3
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(
            name="my_new_daemon",
            cycle_interval=60.0,  # Run every 60 seconds
        )
        self._config = config or {}
        self._processed_count = 0

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Subscribe to events that trigger work."""
        return {
            "SOME_EVENT": self._on_some_event,
            "ANOTHER_EVENT": self._on_another_event,
        }

    async def _run_cycle(self) -> None:
        """Main work loop - called every cycle_interval seconds."""
        self._processed_count += 1
        logger.info(f"Cycle {self._processed_count} completed")

    async def _on_some_event(self, event: dict[str, Any]) -> None:
        """Handle SOME_EVENT."""
        if self._is_duplicate_event(event):
            return
        payload = event.get("payload", {})
        logger.info(f"Processing event: {payload}")

    async def _on_another_event(self, event: dict[str, Any]) -> None:
        """Handle ANOTHER_EVENT."""
        if self._is_duplicate_event(event):
            return
        # Process event...

    def health_check(self) -> HealthCheckResult:
        """Report daemon health for monitoring."""
        if not self._is_running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Daemon not running",
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="OK",
            details={
                "processed_count": self._processed_count,
                "cycle_interval": self._cycle_interval,
                "uptime_seconds": self._get_uptime(),
            },
        )


# Module-level singleton accessor
_instance: MyNewDaemon | None = None


def get_my_new_daemon() -> MyNewDaemon:
    """Get or create singleton instance."""
    global _instance
    if _instance is None:
        _instance = MyNewDaemon()
    return _instance


def reset_my_new_daemon() -> None:
    """Reset singleton (for testing)."""
    global _instance
    _instance = None
```

### Step 3: Create the Runner Function

Edit `app/coordination/daemon_runners.py`:

```python
async def create_my_new_daemon() -> None:
    """Run the MyNewDaemon until shutdown.

    December 2025 - Brief description.
    """
    from app.coordination.my_new_daemon import get_my_new_daemon

    daemon = get_my_new_daemon()
    await daemon.start()

    # Wait for shutdown signal
    while daemon.is_running:
        await asyncio.sleep(1)
```

### Step 4: Register in Daemon Registry

Edit `app/coordination/daemon_registry.py`:

```python
DAEMON_REGISTRY: dict[DaemonType, DaemonSpec] = {
    # ... existing entries ...

    DaemonType.MY_NEW_DAEMON: DaemonSpec(
        runner_name="create_my_new_daemon",
        depends_on=(DaemonType.EVENT_ROUTER,),  # Dependencies
        category="misc",  # sync, event, health, pipeline, resource, etc.
        auto_restart=True,
        max_restarts=5,
        health_check_interval=30.0,
    ),
}
```

### Step 5: Add Unit Tests

Create `tests/unit/coordination/test_my_new_daemon.py`:

```python
"""Unit tests for MyNewDaemon."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.coordination.my_new_daemon import (
    MyNewDaemon,
    get_my_new_daemon,
    reset_my_new_daemon,
)


class TestMyNewDaemon:
    """Tests for MyNewDaemon class."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_my_new_daemon()

    def test_initialization(self):
        """Test daemon initializes correctly."""
        daemon = MyNewDaemon()
        assert daemon.name == "my_new_daemon"
        assert daemon._cycle_interval == 60.0

    def test_health_check_not_running(self):
        """Test health check when not running."""
        daemon = MyNewDaemon()
        health = daemon.health_check()
        assert not health.healthy
        assert health.message == "Daemon not running"

    @pytest.mark.asyncio
    async def test_run_cycle(self):
        """Test single cycle execution."""
        daemon = MyNewDaemon()
        await daemon._run_cycle()
        assert daemon._processed_count == 1

    def test_singleton(self):
        """Test singleton pattern."""
        d1 = get_my_new_daemon()
        d2 = get_my_new_daemon()
        assert d1 is d2

    def test_event_subscriptions(self):
        """Test event subscription mapping."""
        daemon = MyNewDaemon()
        subs = daemon._get_event_subscriptions()
        assert "SOME_EVENT" in subs
        assert "ANOTHER_EVENT" in subs

    @pytest.mark.asyncio
    async def test_on_some_event(self):
        """Test event handling."""
        daemon = MyNewDaemon()
        event = {"payload": {"key": "value"}}
        await daemon._on_some_event(event)
        # Assert expected behavior
```

---

## Architecture Patterns

### HandlerBase Class

All daemons should inherit from `HandlerBase` which provides:

| Feature              | Description                                             |
| -------------------- | ------------------------------------------------------- |
| Singleton pattern    | Thread-safe singleton access via `get_instance()`       |
| Event subscription   | Automatic subscription via `_get_event_subscriptions()` |
| Event deduplication  | Hash-based dedup with TTL (5 min default)               |
| Health check         | Standardized `HealthCheckResult` format                 |
| Error tracking       | Bounded error log with timestamps                       |
| Lifecycle management | Async `start()`, `stop()`, `shutdown()`                 |

### Required Methods

| Method                       | Required | Description                                   |
| ---------------------------- | -------- | --------------------------------------------- |
| `_run_cycle()`               | Yes      | Main work loop, called every `cycle_interval` |
| `_get_event_subscriptions()` | No       | Return dict of event_type → handler           |
| `health_check()`             | No       | Return `HealthCheckResult` for monitoring     |
| `_on_start()`                | No       | Called before main loop starts                |
| `_on_stop()`                 | No       | Called after main loop stops                  |

### Dependency Injection

Daemons should accept dependencies as constructor parameters for testability:

```python
class MyDaemon(HandlerBase):
    def __init__(
        self,
        get_data_fn: Callable[[], dict] | None = None,
        process_fn: Callable[[dict], None] | None = None,
        config: dict | None = None,
    ):
        super().__init__(name="my_daemon")
        self._get_data = get_data_fn or self._default_get_data
        self._process = process_fn or self._default_process
```

---

## Registry Categories

Daemons are organized by category in the registry:

| Category       | Description             | Examples                              |
| -------------- | ----------------------- | ------------------------------------- |
| `sync`         | Data synchronization    | AUTO_SYNC, ELO_SYNC                   |
| `event`        | Event processing        | EVENT_ROUTER, DLQ_RETRY               |
| `health`       | Monitoring              | CLUSTER_MONITOR, QUALITY_MONITOR      |
| `pipeline`     | Training pipeline       | DATA_PIPELINE, TRAINING_TRIGGER       |
| `evaluation`   | Model evaluation        | EVALUATION, AUTO_PROMOTION            |
| `distribution` | Model/data distribution | MODEL_DISTRIBUTION                    |
| `resource`     | Resource management     | IDLE_RESOURCE, NODE_RECOVERY          |
| `provider`     | Cloud provider specific | MULTI_PROVIDER                        |
| `queue`        | Job queue management    | QUEUE_POPULATOR, JOB_SCHEDULER        |
| `feedback`     | Training feedback       | FEEDBACK_LOOP, CURRICULUM_INTEGRATION |
| `recovery`     | Error recovery          | RECOVERY_ORCHESTRATOR, MAINTENANCE    |
| `misc`         | Everything else         | S3_BACKUP, DISTILLATION               |

---

## Dependency Management

### Declaring Dependencies

Dependencies ensure daemons start in correct order:

```python
DaemonType.MY_DAEMON: DaemonSpec(
    runner_name="create_my_daemon",
    depends_on=(
        DaemonType.EVENT_ROUTER,     # Almost always needed
        DaemonType.DATA_PIPELINE,    # If emitting pipeline events
        DaemonType.CLUSTER_MONITOR,  # If using cluster info
    ),
    category="misc",
)
```

### Common Dependencies

| Daemon                 | When to Depend On                                         |
| ---------------------- | --------------------------------------------------------- |
| `EVENT_ROUTER`         | Almost always (required for event emission/subscription)  |
| `DATA_PIPELINE`        | If emitting DATA*SYNC*_, NEW*GAMES*_, TRAINING\_\* events |
| `FEEDBACK_LOOP`        | If events should trigger feedback adjustments             |
| `CLUSTER_MONITOR`      | If using cluster node information                         |
| `SELFPLAY_COORDINATOR` | If scheduling selfplay work                               |

### Validating Dependencies

Run registry validation to catch dependency issues:

```python
from app.coordination.daemon_registry import validate_registry

errors = validate_registry()
if errors:
    for error in errors:
        print(f"  - {error}")
```

---

## Event Handling

### Subscribing to Events

Daemons receive events by implementing `_get_event_subscriptions()`:

```python
def _get_event_subscriptions(self) -> dict[str, Callable]:
    return {
        "TRAINING_COMPLETED": self._on_training_completed,
        "DATA_SYNC_COMPLETED": self._on_sync_completed,
        "NODE_RECOVERED": self._on_node_recovered,
    }
```

### Event Deduplication

Prevent duplicate processing with built-in deduplication:

```python
async def _on_training_completed(self, event: dict) -> None:
    if self._is_duplicate_event(event):
        logger.debug("Skipping duplicate event")
        return

    # Process event...
    self._mark_event_processed(event)
```

### Emitting Events

Use typed emitters from `event_emitters.py`:

```python
from app.coordination.event_emitters import emit_training_complete

emit_training_complete(
    config_key="hex8_2p",
    model_path="models/canonical_hex8_2p.pth",
    epochs_completed=50,
)
```

### Common Event Types

See `CLAUDE.md` for the full event reference. Key events:

| Event                  | Description              |
| ---------------------- | ------------------------ |
| `TRAINING_COMPLETED`   | Training job finished    |
| `DATA_SYNC_COMPLETED`  | Sync operation finished  |
| `MODEL_PROMOTED`       | Model passed gauntlet    |
| `EVALUATION_COMPLETED` | Gauntlet evaluation done |
| `NEW_GAMES_AVAILABLE`  | New selfplay games ready |

---

## Health Checks

### HealthCheckResult Format

All daemons should return `HealthCheckResult`:

```python
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

def health_check(self) -> HealthCheckResult:
    # Check for errors
    if self._stats.errors_count > 10:
        return HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.ERROR,
            message=f"Too many errors: {self._stats.errors_count}",
            details={"last_error": self._stats.last_error},
        )

    # Check for startup grace period
    uptime = time.time() - self._stats.started_at
    if uptime < 30.0:
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.STARTING,
            message=f"Starting (uptime: {uptime:.1f}s)",
        )

    # Normal operation
    return HealthCheckResult(
        healthy=True,
        status=CoordinatorStatus.RUNNING,
        message="OK",
        details={
            "cycles": self._stats.cycles_completed,
            "events": self._stats.events_processed,
        },
    )
```

### Status Values

| Status     | Meaning                 |
| ---------- | ----------------------- |
| `RUNNING`  | Normal operation        |
| `STARTING` | In startup grace period |
| `DEGRADED` | Reduced functionality   |
| `ERROR`    | Error state             |
| `STOPPED`  | Not running             |

---

## Testing Guidelines

### Unit Test Structure

```python
class TestMyDaemon:
    """Tests for MyDaemon."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_my_daemon()

    # 1. Initialization tests
    def test_initialization(self): ...
    def test_config_loading(self): ...

    # 2. Singleton tests
    def test_singleton_pattern(self): ...
    def test_reset_singleton(self): ...

    # 3. Health check tests
    def test_health_check_not_running(self): ...
    def test_health_check_running(self): ...
    def test_health_check_with_errors(self): ...

    # 4. Event handling tests
    @pytest.mark.asyncio
    async def test_on_some_event(self): ...
    @pytest.mark.asyncio
    async def test_event_deduplication(self): ...

    # 5. Cycle tests
    @pytest.mark.asyncio
    async def test_run_cycle(self): ...
    @pytest.mark.asyncio
    async def test_run_cycle_error_handling(self): ...
```

### Mocking Patterns

```python
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_daemon_with_mocks():
    # Mock external dependencies
    with patch('app.coordination.my_daemon.get_cluster_config') as mock_config:
        mock_config.return_value = {"key": "value"}

        daemon = MyDaemon()
        await daemon._run_cycle()

        mock_config.assert_called_once()

@pytest.mark.asyncio
async def test_daemon_with_dependency_injection():
    # Use dependency injection for testing
    mock_get_data = MagicMock(return_value={"key": "value"})
    mock_process = AsyncMock()

    daemon = MyDaemon(
        get_data_fn=mock_get_data,
        process_fn=mock_process,
    )

    await daemon._run_cycle()

    mock_get_data.assert_called()
    mock_process.assert_called()
```

### Running Tests

```bash
# Run daemon tests
pytest tests/unit/coordination/test_my_daemon.py -v

# Run with coverage
pytest tests/unit/coordination/test_my_daemon.py --cov=app/coordination/my_daemon

# Run all coordination tests
pytest tests/unit/coordination/ -v --tb=short
```

---

## Lifecycle Management

### Starting Daemons

Daemons are started by DaemonManager:

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()
await dm.start(DaemonType.MY_NEW_DAEMON)
```

### Stopping Daemons

```python
await dm.stop(DaemonType.MY_NEW_DAEMON)
```

### Auto-Restart

Daemons with `auto_restart=True` automatically restart on failure:

- Default: 5 restart attempts
- Exponential backoff: 1s → 2s → 4s → 8s → 16s
- Counter resets after 5 minutes of successful operation

---

## Debugging

### Logging

Use structured logging with the daemon name:

```python
logger = logging.getLogger(__name__)

class MyDaemon(HandlerBase):
    def _log_info(self, msg: str) -> None:
        logger.info(f"[{self.name}] {msg}")

    def _log_error(self, msg: str, exc_info: bool = False) -> None:
        logger.error(f"[{self.name}] {msg}", exc_info=exc_info)
```

### Health Endpoints

Query daemon health via HTTP:

```bash
# All daemon health
curl http://localhost:8790/health | jq

# Specific daemon
python -c "
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType
import asyncio

async def check():
    dm = get_daemon_manager()
    health = await dm.get_daemon_health(DaemonType.MY_NEW_DAEMON)
    print(health)

asyncio.run(check())
"
```

### Daemon Status CLI

```bash
# Check all daemon status
python scripts/launch_daemons.py --status

# Watch daemon health
python scripts/launch_daemons.py --watch
```

---

## Best Practices

### Do

- Inherit from `HandlerBase` for consistency
- Implement `health_check()` for monitoring
- Use event deduplication for idempotency
- Add proper logging with daemon name prefix
- Write unit tests for all event handlers
- Document dependencies in registry

### Don't

- Don't use bare `except Exception:` - narrow to specific types
- Don't import at module level if optional - use lazy imports
- Don't hold locks during I/O operations
- Don't emit events in `__init__` - wait for `start()`
- Don't skip registry validation

### Performance

- Keep `_run_cycle()` fast (<1 second ideally)
- Use appropriate cycle intervals (don't poll too frequently)
- Batch operations when possible
- Use async for I/O operations

---

## Example: Complete Daemon

Here's a complete example of a well-structured daemon:

```python
"""Example daemon demonstrating best practices.

December 2025 - Monitors data quality and emits alerts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus
from app.coordination.event_emitters import emit_data_quality_alert

logger = logging.getLogger(__name__)


@dataclass
class DataQualityConfig:
    """Configuration for data quality daemon."""

    check_interval_seconds: float = 300.0  # 5 minutes
    quality_threshold: float = 0.8
    max_alerts_per_hour: int = 10


class DataQualityDaemon(HandlerBase):
    """Monitors data quality and emits alerts when below threshold."""

    def __init__(self, config: DataQualityConfig | None = None):
        self._config = config or DataQualityConfig()
        super().__init__(
            name="data_quality",
            cycle_interval=self._config.check_interval_seconds,
        )
        self._checks_performed = 0
        self._alerts_emitted = 0

    def _get_event_subscriptions(self) -> dict[str, Any]:
        return {
            "NEW_GAMES_AVAILABLE": self._on_new_games,
            "TRAINING_COMPLETED": self._on_training_completed,
        }

    async def _run_cycle(self) -> None:
        """Check data quality periodically."""
        quality_score = await self._compute_quality_score()
        self._checks_performed += 1

        if quality_score < self._config.quality_threshold:
            if self._alerts_emitted < self._config.max_alerts_per_hour:
                emit_data_quality_alert(
                    quality_score=quality_score,
                    threshold=self._config.quality_threshold,
                )
                self._alerts_emitted += 1
                logger.warning(f"Quality below threshold: {quality_score:.2f}")

    async def _on_new_games(self, event: dict) -> None:
        """Check quality of new games."""
        if self._is_duplicate_event(event):
            return
        # Trigger quality check
        await self._run_cycle()

    async def _on_training_completed(self, event: dict) -> None:
        """Reset alert counter after training."""
        if self._is_duplicate_event(event):
            return
        self._alerts_emitted = 0

    async def _compute_quality_score(self) -> float:
        """Compute current data quality score."""
        # Implementation would check database quality
        return 0.85

    def health_check(self) -> HealthCheckResult:
        if not self._is_running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Not running",
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="OK",
            details={
                "checks_performed": self._checks_performed,
                "alerts_emitted": self._alerts_emitted,
                "quality_threshold": self._config.quality_threshold,
            },
        )


# Singleton pattern
_instance: DataQualityDaemon | None = None


def get_data_quality_daemon() -> DataQualityDaemon:
    global _instance
    if _instance is None:
        _instance = DataQualityDaemon()
    return _instance


def reset_data_quality_daemon() -> None:
    global _instance
    _instance = None
```

---

## Related Documentation

- `docs/architecture/DAEMON_LIFECYCLE.md` - Daemon lifecycle details
- `docs/architecture/DAEMON_SYSTEM_ARCHITECTURE.md` - System overview
- `docs/runbooks/DAEMON_FAILURE_RECOVERY.md` - Troubleshooting guide
- `app/coordination/daemon_registry.py` - Full registry
- `CLAUDE.md` - Event system reference
