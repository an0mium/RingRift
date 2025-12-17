# Resource Management Architecture

This document describes the resource management system for RingRift AI, ensuring consistent 80% utilization limits across all components.

## Overview

The resource management system prevents CPU, GPU, RAM, and disk overloading by enforcing consistent limits across the codebase. All long-running operations must check resource availability before proceeding.

## Resource Limits

| Resource | Max Utilization | Warning Threshold | Reason |
|----------|-----------------|-------------------|--------|
| CPU      | 80%             | 70%               | Standard operating margin |
| Memory   | 80%             | 70%               | Standard operating margin |
| GPU      | 80%             | 70%               | Standard operating margin |
| Disk     | 70%             | 65%               | Tighter limit because cleanup takes time |

## Core Module: `app/utils/resource_guard.py`

The unified resource guard provides all resource checking functionality:

```python
from app.utils.resource_guard import (
    # Simple checks
    check_disk_space,      # Check disk availability
    check_memory,          # Check RAM availability
    check_cpu,             # Check CPU load
    check_gpu_memory,      # Check GPU VRAM

    # Combined check
    can_proceed,           # Check all resources at once

    # Blocking wait
    wait_for_resources,    # Wait until resources available
    require_resources,     # Require resources or fail

    # Status reporting
    get_resource_status,   # Get full resource status dict
    print_resource_status, # Print formatted status

    # Context manager
    ResourceGuard,         # Context manager for resource-safe ops

    # Async support
    AsyncResourceLimiter,  # Async limiter with backoff

    # Decorator
    respect_resource_limits, # Decorator for resource-aware functions

    # Constants
    LIMITS,                # ResourceLimits dataclass
)
```

## Usage Patterns

### 1. Simple Check Before Operation

```python
from app.utils.resource_guard import check_disk_space, check_memory

def save_training_data():
    if not check_disk_space(required_gb=5.0):
        logger.warning("Insufficient disk space, skipping save")
        return

    if not check_memory(required_gb=2.0):
        logger.warning("Insufficient memory, skipping save")
        return

    # Proceed with save
    ...
```

### 2. Combined Check

```python
from app.utils.resource_guard import can_proceed

def run_selfplay():
    if not can_proceed(check_disk=True, check_mem=True, check_gpu=True):
        logger.warning("Resources not available")
        return

    # Proceed with selfplay
    ...
```

### 3. Wait for Resources

```python
from app.utils.resource_guard import wait_for_resources

def start_training():
    # Wait up to 5 minutes for resources
    if not wait_for_resources(timeout=300):
        raise RuntimeError("Resources not available after 5 minutes")

    # Proceed with training
    ...
```

### 4. Context Manager

```python
from app.utils.resource_guard import ResourceGuard

def process_games():
    with ResourceGuard(disk_required_gb=5.0, mem_required_gb=4.0) as guard:
        if not guard.ok:
            logger.warning("Resources not available")
            return

        # Proceed with processing
        ...
```

### 5. Async Support

```python
from app.utils.resource_guard import AsyncResourceLimiter

async def distributed_training():
    limiter = AsyncResourceLimiter(
        disk_required_gb=10.0,
        mem_required_gb=8.0,
        gpu_required_gb=4.0,
    )

    async with limiter.acquire("training"):
        # Resources guaranteed available
        await train_model()
```

### 6. Decorator

```python
from app.utils.resource_guard import respect_resource_limits

@respect_resource_limits(disk_gb=5.0, mem_gb=4.0, gpu_gb=2.0)
async def train_model():
    # Automatically waits for resources before executing
    ...
```

## Integration with Scripts

All major scripts should import and use resource_guard:

```python
# Unified resource guard - 80% utilization limits
try:
    from app.utils.resource_guard import (
        can_proceed as resource_can_proceed,
        check_disk_space,
        check_memory,
        check_gpu_memory,
        require_resources,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    resource_can_proceed = lambda **kwargs: True
    check_disk_space = lambda *args, **kwargs: True
    check_memory = lambda *args, **kwargs: True
    check_gpu_memory = lambda *args, **kwargs: True
    require_resources = lambda *args, **kwargs: True
    RESOURCE_LIMITS = None
```

## Related Modules

### `app/coordination/resource_targets.py`
Provides tier-specific utilization targets for different host types:
- HIGH_END (96GB+ RAM): Target 65% utilization
- MID_TIER (32-64GB): Target 60% utilization
- LOW_TIER (16-32GB): Target 55% utilization
- CPU_ONLY: Target 50% utilization

### `app/coordination/safeguards.py`
Provides circuit breakers and backpressure mechanisms:
- Circuit breaker: Prevents spawning after repeated failures
- Spawn rate tracking: Limits new process creation rate
- Resource thresholds: Enforces 80% limits

### `app/config/config_validator.py`
Validates configuration files and ensures resource limits are consistent:
- Validates unified_loop.yaml
- Checks resource limit consistency
- Reports warnings for unsafe configurations

## Testing

Tests are located in `tests/test_resource_guard.py`:

```bash
PYTHONPATH=. pytest tests/test_resource_guard.py -v
```

## Configuration Validation

Run validation to check resource limit consistency:

```bash
PYTHONPATH=. python -c "from app.config.config_validator import validate_all_configs; print(validate_all_configs())"
```

## Enforcement Date

Resource limits were unified and enforced starting 2025-12-16:
- CPU: 80% max
- Memory: 80% max
- GPU: 80% max
- Disk: 70% max

All scripts were updated to use the unified resource_guard module.
