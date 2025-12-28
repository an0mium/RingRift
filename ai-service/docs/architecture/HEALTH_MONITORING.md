# Health Monitoring Architecture

This document describes the health monitoring infrastructure for the RingRift AI service cluster.

## Overview

The health monitoring system provides multi-layer observability across the distributed training cluster:

1. **Node-level health**: GPU utilization, memory, disk, process health
2. **Daemon-level health**: Individual daemon status, restart counts, error rates
3. **Cluster-level health**: Peer connectivity, quorum status, leader election
4. **Pipeline-level health**: Training progress, sync status, data freshness

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Health Check Endpoints                          │
│                    /health  /ready  /metrics  /status                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────┐
│                        UnifiedHealthManager                             │
│  - System health score (0-100)                                          │
│  - Pipeline state tracking                                               │
│  - Health level determination (HEALTHY/DEGRADED/UNHEALTHY)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌───────────────────┬───────────────┴───────────────┬─────────────────────┐
│ DaemonManager     │ HealthCheckOrchestrator       │ ClusterMonitor      │
│ - Daemon health   │ - Node health aggregation     │ - Peer status       │
│ - Auto-restart    │ - Health check scheduling     │ - Job tracking      │
│ - Lifecycle       │ - Alert generation            │ - Resource metrics  │
└───────────────────┴───────────────────────────────┴─────────────────────┘
```

## Key Components

### HealthCheckMixin (`app/coordination/mixins/health_check_mixin.py`)

Base mixin providing standardized health check methods for 76+ coordinator classes:

```python
from app.coordination.mixins.health_check_mixin import HealthCheckMixin

class MyCoordinator(HealthCheckMixin):
    def health_check(self) -> HealthCheckResult:
        status = "healthy"
        details = {}

        # Check error rate
        if self.get_error_rate() > 0.5:
            status = "degraded"
            details["error_rate"] = self.get_error_rate()

        return HealthCheckResult(
            status=status,
            details=details,
            timestamp=time.time()
        )
```

### UnifiedHealthManager (`app/coordination/unified_health_manager.py`)

Central health scoring and pipeline state management:

```python
from app.coordination.unified_health_manager import (
    get_system_health_score,
    get_system_health_level,
    SystemHealthLevel,
)

# Get overall health score (0-100)
score = get_system_health_score()

# Get health level
level = get_system_health_level()  # HEALTHY, DEGRADED, UNHEALTHY, CRITICAL
```

**Health Score Components:**

- Node availability: 40%
- Circuit health: 25%
- Error rate: 20%
- Recovery status: 15%

### HealthCheckOrchestrator (`app/coordination/health_check_orchestrator.py`)

Coordinates health checks across all nodes:

```python
from app.coordination.health_check_orchestrator import get_health_orchestrator

orchestrator = get_health_orchestrator()
node_health = orchestrator.get_node_health("runpod-h100")
cluster_health = orchestrator.get_cluster_health_summary()
```

### DaemonManager (`app/coordination/daemon_manager.py`)

Manages daemon lifecycle and health monitoring:

```python
from app.coordination.daemon_manager import get_daemon_manager

dm = get_daemon_manager()

# Check individual daemon health
health = await dm.get_daemon_health(DaemonType.AUTO_SYNC)

# Liveness probe (for container health checks)
if dm.liveness_probe():
    print("All critical daemons healthy")
```

## HTTP Health Endpoints

The HealthServer daemon exposes these endpoints on port 8790:

| Endpoint   | Purpose            | Response                      |
| ---------- | ------------------ | ----------------------------- |
| `/health`  | Liveness check     | 200 OK if alive               |
| `/ready`   | Readiness check    | 200 OK if ready for traffic   |
| `/metrics` | Prometheus metrics | OpenMetrics format            |
| `/status`  | Detailed status    | JSON with full health details |

## Health Check Result Protocol

All health checks return a standardized result:

```python
@dataclass
class HealthCheckResult:
    status: str  # "healthy", "degraded", "unhealthy"
    message: str = ""
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"
```

## Event-Driven Health Updates

Health events enable reactive monitoring:

| Event                   | Emitter                 | Purpose                      |
| ----------------------- | ----------------------- | ---------------------------- |
| `HEALTH_CHECK_PASSED`   | HealthCheckOrchestrator | Node passed health check     |
| `HEALTH_CHECK_FAILED`   | HealthCheckOrchestrator | Node failed health check     |
| `COORDINATOR_HEALTHY`   | DaemonManager           | Coordinator became healthy   |
| `COORDINATOR_UNHEALTHY` | DaemonManager           | Coordinator became unhealthy |
| `NODE_RECOVERED`        | NodeRecoveryDaemon      | Node recovered from failure  |

## Configuration

Environment variables for health monitoring:

| Variable                         | Default | Description                              |
| -------------------------------- | ------- | ---------------------------------------- |
| `RINGRIFT_HEALTH_CHECK_INTERVAL` | 60      | Seconds between health checks            |
| `RINGRIFT_HEALTH_CHECK_TIMEOUT`  | 5       | Timeout for health check RPC             |
| `RINGRIFT_HEALTH_SERVER_PORT`    | 8790    | Port for health endpoints                |
| `RINGRIFT_ERROR_RATE_THRESHOLD`  | 0.5     | Error rate threshold for degraded status |

## Centralized Configuration

Health thresholds are defined in `app/config/coordination_defaults.py`:

```python
from app.config.coordination_defaults import HealthThresholds

# Access thresholds
error_rate_warning = HealthThresholds.ERROR_RATE_WARNING  # 0.3
error_rate_critical = HealthThresholds.ERROR_RATE_CRITICAL  # 0.5
```

## Best Practices

1. **Implement health_check() on all coordinators** - Use HealthCheckMixin as base
2. **Include key metrics in details** - Error rates, latencies, queue depths
3. **Use appropriate status levels** - healthy/degraded/unhealthy
4. **Subscribe to health events** - React to status changes
5. **Configure alerting thresholds** - Don't alert on transient issues

## Related Documentation

- [DAEMON_LIFECYCLE.md](DAEMON_LIFECYCLE.md) - Daemon lifecycle management
- [DAEMON_FAILURE_RECOVERY.md](../runbooks/DAEMON_FAILURE_RECOVERY.md) - Troubleshooting guide
- [P2P_ORCHESTRATOR_ARCHITECTURE.md](P2P_ORCHESTRATOR_ARCHITECTURE.md) - P2P cluster health
