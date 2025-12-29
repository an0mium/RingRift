# Coordination Bootstrap Failures Runbook

This runbook covers troubleshooting failures during coordination system bootstrap.

## Overview

The coordination bootstrap (`app/coordination/coordination_bootstrap.py`) initializes coordinators and delegates event wiring via `event_subscription_registry`. Failures here can prevent the training pipeline from starting.

## Quick Diagnostics

```bash
# Check if bootstrap completed
python3 -c "
from app.coordination.coordination_bootstrap import get_bootstrap_status
status = get_bootstrap_status()
print(f'Bootstrap complete: {status.is_complete}')
print(f'Coordinators initialized: {len(status.initialized_coordinators)}')
if status.errors:
    print(f'Errors: {status.errors}')
"

# Check daemon status
python scripts/launch_daemons.py --status

# Check event subscriptions
python3 -c "
from app.coordination.event_router import get_event_bus
bus = get_event_bus()
print(f'Subscriptions: {len(bus.subscriptions)}')
for event_type, handlers in bus.subscriptions.items():
    print(f'  {event_type}: {len(handlers)} handlers')
"
```

## Common Failures

### 1. Import Error - Missing Module

**Symptom:**

```
ImportError: No module named 'app.coordination.some_module'
```

**Cause:** Module was renamed, moved, or hasn't been created yet.

**Solution:**

1. Check if module exists: `ls -la app/coordination/some_module.py`
2. Check for typos in import path
3. Verify module is in `COORDINATOR_REGISTRY` in `coordination_bootstrap.py`

```python
# Find the registry entry
grep -n "some_module" app/coordination/coordination_bootstrap.py
```

### 2. Circular Import

**Symptom:**

```
ImportError: cannot import name 'X' from partially initialized module 'Y'
```

**Cause:** Module A imports B, and B imports A at module level.

**Solution:**

1. Use lazy imports inside functions:

```python
def my_function():
    from app.coordination.other_module import X  # Lazy import
    return X()
```

2. Or use TYPE_CHECKING guard:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.other_module import X
```

### 3. Missing Dependency Coordinator

**Symptom:**

```
ValueError: Coordinator 'X' depends on 'Y' which is not initialized
```

**Cause:** Coordinator X listed Y as dependency but Y failed to initialize.

**Solution:**

1. Check Y's initialization first - it's the root cause
2. Verify Y is in `COORDINATOR_REGISTRY` (or the special-case `pipeline_orchestrator`)
3. Check Y's own dependencies

```python
# Inspect registry ordering
grep -n "COORDINATOR_REGISTRY" app/coordination/coordination_bootstrap.py
```

### 4. Event Subscription Failed

**Symptom:**

```
Warning: Failed to subscribe X to event Y: ...
```

**Cause:** Event type doesn't exist, handler signature wrong, or delegated wiring spec is stale.

**Solution:**

1. Verify event type exists:

```python
from app.distributed.data_events import DataEventType
print([e.name for e in DataEventType])
```

2. Verify handler signature matches:

```python
# Correct signature
async def handler(event: dict) -> None:
    pass
```

3. If the wiring comes from the delegation registry, verify the spec:

```python
from app.coordination.event_subscription_registry import DELEGATION_REGISTRY
print([spec.name for spec in DELEGATION_REGISTRY])
```

### 5. Database Connection Failed

**Symptom:**

```
sqlite3.OperationalError: unable to open database file
```

**Cause:** Database directory doesn't exist or no write permission.

**Solution:**

```bash
# Check directory exists
mkdir -p data/coordination

# Check permissions
ls -la data/coordination/

# Fix permissions if needed
chmod 755 data/coordination/
```

### 6. DaemonManager Not Started

**Symptom:**

```
RuntimeError: DaemonManager not initialized
```

**Cause:** Bootstrap was called but DaemonManager wasn't started.

**Solution:**

```python
from app.coordination.daemon_manager import get_daemon_manager

dm = get_daemon_manager()
await dm.start_all()  # Or start individual daemons
```

### 7. Registry Validation Failed

**Symptom:**

```
ValueError: Daemon registry validation failed:
  - X: DaemonType exists but has no DAEMON_REGISTRY entry
```

**Cause:** New DaemonType added but not registered in `daemon_registry.py`.

**Solution:**
Add entry to `DAEMON_REGISTRY` in `app/coordination/daemon_registry.py`:

```python
DaemonType.NEW_TYPE: DaemonSpec(
    runner_name="create_new_type",
    depends_on=(DaemonType.EVENT_ROUTER,),
    category="misc",
),
```

And create runner in `daemon_runners.py`:

```python
async def create_new_type() -> None:
    from app.coordination.new_type import NewType
    instance = NewType()
    await instance.start()
    await instance.wait_for_shutdown()
```

## Startup Order Issues

The correct daemon startup order is critical. Key constraints:

1. **EVENT_ROUTER must be first** - All other daemons emit events
2. **DATA_PIPELINE before AUTO_SYNC** - Sync events need handlers ready
3. **FEEDBACK_LOOP before AUTO_SYNC** - Same reason
4. **EVALUATION before AUTO_PROMOTION** - Promotion needs eval results

Check startup order:

```python
from app.coordination.daemon_types import DAEMON_STARTUP_ORDER
for i, dt in enumerate(DAEMON_STARTUP_ORDER):
    print(f"{i+1}. {dt.name}")
```

Validate order consistency:

```python
from app.coordination.daemon_types import validate_startup_order_or_raise
validate_startup_order_or_raise()  # Raises if inconsistent
```

## Recovery Procedures

### Full Bootstrap Reset

If bootstrap is in a bad state:

```python
from app.coordination.coordination_bootstrap import reset_bootstrap
reset_bootstrap()

# Then reinitialize
from app.coordination.coordination_bootstrap import bootstrap_coordination
await bootstrap_coordination()
```

### Reinitialize Single Coordinator

```python
from app.coordination.coordination_bootstrap import reinitialize_coordinator
await reinitialize_coordinator("data_pipeline_orchestrator")
```

### Rewire Event Subscriptions

```python
from app.coordination.coordination_bootstrap import rewire_event_subscriptions
rewire_event_subscriptions()
```

## Logging

Enable verbose bootstrap logging:

```bash
export RINGRIFT_LOG_LEVEL=DEBUG
export RINGRIFT_BOOTSTRAP_VERBOSE=true

python scripts/master_loop.py
```

Check bootstrap logs:

```bash
grep -i "bootstrap\|coordinator\|initialized" logs/coordination.log | tail -50
```

## Prevention

1. **Run registry validation on startup** - Enable `strict_registry_validation: True`
2. **Add tests for new coordinators** - Test import and initialization
3. **Document dependencies** - Add to `DAEMON_DEPENDENCIES` and `depends_on`
4. **Use lazy imports** - Avoid circular dependencies

## Related Documentation

- [DAEMON_FAILURE_RECOVERY.md](DAEMON_FAILURE_RECOVERY.md) - Daemon-specific issues
- [COORDINATION_EVENT_SYSTEM.md](COORDINATION_EVENT_SYSTEM.md) - Event system details
- [EVENT_WIRING_VERIFICATION.md](EVENT_WIRING_VERIFICATION.md) - Verify event wiring
