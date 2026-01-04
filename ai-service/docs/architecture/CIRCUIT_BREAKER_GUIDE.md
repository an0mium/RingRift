# Circuit Breaker Architecture Guide

**Last Updated**: January 3, 2026 (Sprint 15)

This guide documents the circuit breaker patterns used in the RingRift AI-service coordination layer.

## Overview

Circuit breakers prevent cascading failures by temporarily blocking operations to unhealthy components. When failures exceed a threshold, the circuit "opens" and requests are blocked until the component recovers.

## Circuit Breaker Hierarchy (10 Types)

RingRift uses multiple circuit breaker types serving different failure domains. These are intentionally separate - do NOT consolidate them further.

| Type                         | Location                         | Failure Domain        | Purpose                        |
| ---------------------------- | -------------------------------- | --------------------- | ------------------------------ |
| `CircuitBreakerBase`         | `circuit_breaker_base.py`        | Abstract              | Base class for custom breakers |
| `OperationCircuitBreaker`    | `circuit_breaker_base.py`        | Per-operation         | Generic operation isolation    |
| `NodeCircuitBreaker`         | `node_circuit_breaker.py`        | Per-node              | Node health isolation          |
| `ClusterCircuitBreaker`      | `node_circuit_breaker.py`        | Cluster-wide          | Global cluster protection      |
| `DaemonStatusCircuitBreaker` | `daemon_manager.py`              | Per-daemon            | Daemon restart throttling      |
| `PipelineCircuitBreaker`     | `data_pipeline_orchestrator.py`  | Per-pipeline-stage    | Pipeline stage protection      |
| Per-transport CB             | `cluster_transport.py`           | Per-(node, transport) | Transport failover             |
| `CircuitBreaker`             | `distributed/circuit_breaker.py` | Registry              | Centralized CB registry        |
| `GlobalCircuitBreaker`       | `transport_cascade.py`           | Cross-transport       | Transport cascade protection   |
| Component CB                 | `safeguards.py`                  | Per-component         | Pipeline component isolation   |

### Why Different Types?

Each CB type protects a different failure domain:

- **Per-node** (`NodeCircuitBreaker`): Isolates failing nodes without affecting healthy ones
- **Per-transport** (cluster_transport.py): Allows SSH to fail while HTTP continues
- **Per-daemon** (`DaemonStatusCircuitBreaker`): Prevents restart storms for specific daemons
- **Cluster-wide** (`ClusterCircuitBreaker`): Emergency brake for catastrophic failures
- **Per-pipeline-stage** (`PipelineCircuitBreaker`): Allows export to continue when training fails

## Circuit Breaker Types

### 1. Node Circuit Breaker (`node_circuit_breaker.py`)

**Purpose**: Per-node failure isolation for health checks and sync operations.

**States**:

- `CLOSED` - Normal operation, all requests allowed
- `OPEN` - Too many failures, requests blocked
- `HALF_OPEN` - Testing recovery with limited requests

**Configuration** (`NodeCircuitConfig`):

```python
from app.coordination.node_circuit_breaker import (
    NodeCircuitBreaker,
    NodeCircuitConfig,
    get_node_circuit_breaker,
)

# Default configuration
config = NodeCircuitConfig(
    failure_threshold=5,       # Opens after 5 failures
    recovery_timeout=300.0,    # 5 minutes before half-open
    half_open_max_calls=3,     # 3 test calls in half-open
    success_threshold=2,       # 2 successes to close
)

# Get registry (singleton)
registry = get_node_circuit_breaker()

# Check before operation
if registry.can_check("gpu-node-1"):
    try:
        result = await check_node_health("gpu-node-1")
        registry.record_success("gpu-node-1")
    except (ConnectionError, TimeoutError):
        registry.record_failure("gpu-node-1")
```

**Used by**:

- `health_check_orchestrator.py` - Node health checks
- `sync_bandwidth.py` - Bandwidth allocation
- `cluster_transport.py` - File transfers
- `database_sync_manager.py` - Database sync

### 2. Component Circuit Breaker (`safeguards.py`)

**Purpose**: Per-component failure isolation for pipeline operations.

**Configuration**:

```python
from app.coordination.safeguards import (
    CircuitBreaker,
    get_circuit_breaker,
)

# Get breaker for component
breaker = get_circuit_breaker("training")

# Check state
if breaker.is_closed():
    try:
        await run_training()
        breaker.record_success()
    except TrainingError:
        breaker.record_failure("Training failed")
```

**Used by**:

- `pipeline_actions.py` - Pipeline stage execution
- `training_coordinator.py` - Training operations
- `auto_sync_daemon.py` - Sync operations

### 3. Transport Circuit Breaker (`transport_base.py`)

**Purpose**: Per-transport failure isolation for file transfers.

**Configuration** (`CircuitBreakerConfig`):

```python
from app.coordination.transport_base import CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=3,       # Opens after 3 failures
    recovery_timeout=120.0,    # 2 minutes
    half_open_success=1,       # 1 success to close
)
```

**Used by**:

- `cluster_transport.py` - Multi-transport failover
- `transport_manager.py` - Transport selection

## Configuration Reference

### Environment Variables

| Variable                             | Default | Description              |
| ------------------------------------ | ------- | ------------------------ |
| `RINGRIFT_CIRCUIT_FAILURE_THRESHOLD` | 5       | Failures before opening  |
| `RINGRIFT_CIRCUIT_RECOVERY_TIMEOUT`  | 300     | Seconds before half-open |
| `RINGRIFT_CIRCUIT_HALF_OPEN_CALLS`   | 3       | Test calls in half-open  |

### Per-Module Thresholds

| Module                    | Failure Threshold | Recovery Timeout | Notes                   |
| ------------------------- | ----------------- | ---------------- | ----------------------- |
| `node_circuit_breaker.py` | 5                 | 300s             | Per-node isolation      |
| `safeguards.py`           | 10-20 (dynamic)   | 60s              | Based on cluster health |
| `transport_base.py`       | 3                 | 120s             | Per-transport           |
| `sync_bandwidth.py`       | 5                 | 300s             | Uses NodeCircuitBreaker |

## Best Practices

### 1. Use Per-Node Breakers for Cluster Operations

```python
# GOOD: Per-node isolation
from app.coordination.node_circuit_breaker import get_node_circuit_breaker

breaker = get_node_circuit_breaker()
for node in nodes:
    if breaker.can_check(node):
        await check_node(node)

# BAD: Single breaker for all nodes
if global_breaker.is_closed():
    for node in nodes:  # All blocked if one fails
        await check_node(node)
```

### 2. Record Both Success and Failure

```python
# GOOD: Always record outcome
try:
    result = await operation()
    breaker.record_success(node_id)
except Exception:
    breaker.record_failure(node_id)

# BAD: Only record failures
try:
    result = await operation()
except Exception:
    breaker.record_failure(node_id)  # Success not tracked
```

### 3. Use Appropriate Recovery Timeouts

- **Fast operations** (health checks): 60-120s
- **Medium operations** (sync): 300s
- **Slow operations** (training): 600s

### 4. Monitor Circuit States

```python
from app.coordination.node_circuit_breaker import get_node_circuit_breaker

registry = get_node_circuit_breaker()
summary = registry.get_summary()

# Log open circuits
for status in summary["nodes"]:
    if status["state"] == "open":
        logger.warning(f"Circuit open for {status['node_id']}")
```

## Events

Circuit breakers emit events for monitoring:

| Event               | Description                        |
| ------------------- | ---------------------------------- |
| `CIRCUIT_OPENED`    | Circuit transitioned to open state |
| `CIRCUIT_CLOSED`    | Circuit recovered and closed       |
| `CIRCUIT_HALF_OPEN` | Circuit testing recovery           |

Subscribe via event router:

```python
from app.coordination.event_router import get_router
from app.distributed.data_events import DataEventType

router = get_router()
router.subscribe(DataEventType.CIRCUIT_OPENED, on_circuit_opened)
```

## Troubleshooting

### Circuit Won't Close

1. Check recovery timeout hasn't elapsed
2. Verify success threshold is reachable
3. Check for ongoing failures blocking half-open tests

### False Positives (Circuit Opens Incorrectly)

1. Increase `failure_threshold`
2. Check for transient network issues
3. Verify timeout settings match operation duration

### Cascade Failures

1. Ensure using per-node breakers (not global)
2. Check `RINGRIFT_CIRCUIT_FAILURE_THRESHOLD` isn't too low
3. Review logs for root cause node

## See Also

- `node_circuit_breaker.py` - Per-node implementation
- `safeguards.py` - Component-level breakers
- `transport_base.py` - Transport breakers
- `EVENT_SUBSCRIPTION_MATRIX.md` - Event documentation
