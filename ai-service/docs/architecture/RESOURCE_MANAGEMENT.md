# Resource Management Architecture

This document describes the resource management infrastructure for GPU/CPU utilization, memory, and disk space across the RingRift training cluster.

## Overview

Resource management ensures optimal utilization of expensive GPU compute while preventing resource exhaustion:

1. **GPU Utilization**: Target 60-80% utilization through dynamic job scheduling
2. **Memory Management**: Prevent OOM by monitoring and limiting concurrent jobs
3. **Disk Space**: Proactive cleanup before thresholds are reached
4. **Job Scheduling**: Intelligent job placement based on node capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ResourceTargetManager                            │
│  - Target utilization ranges (min/max)                                   │
│  - PID controller for adaptive scaling                                   │
│  - Backpressure signaling                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌───────────────────┬───────────────┴───────────────┬─────────────────────┐
│ IdleResourceDaemon│ SelfplayScheduler             │ DiskSpaceManager    │
│ - GPU idle detect │ - Priority scheduling         │ - Proactive cleanup │
│ - Job spawning    │ - Curriculum weights          │ - Sync before clean │
│                   │ - Elo velocity boost          │ - Threshold alerts  │
└───────────────────┴───────────────────────────────┴─────────────────────┘
```

## Key Components

### ResourceTargetManager (`app/coordination/resource_targets.py`)

Central resource utilization targets with PID controller:

```python
from app.coordination.resource_targets import get_resource_targets

targets = get_resource_targets()
print(f"GPU target: {targets.gpu_util_min}% - {targets.gpu_util_max}%")
print(f"Memory limit: {targets.memory_limit_gb}GB")
print(f"Disk threshold: {targets.disk_threshold_percent}%")
```

**Default Targets:**

- GPU Utilization: 60-80%
- Memory: 80% max
- Disk: 70% max

### IdleResourceDaemon (`app/coordination/idle_resource_daemon.py`)

Monitors idle GPUs and spawns selfplay jobs:

```python
from app.coordination.idle_resource_daemon import get_idle_resource_daemon

daemon = get_idle_resource_daemon()
await daemon.start()

# Check idle status
idle_gpus = daemon.get_idle_gpus()
for gpu in idle_gpus:
    print(f"GPU {gpu.index}: {gpu.utilization}% (idle for {gpu.idle_seconds}s)")
```

### SelfplayScheduler (`scripts/p2p/managers/selfplay_scheduler.py`)

Priority-based selfplay job scheduling:

```python
from scripts.p2p.managers.selfplay_scheduler import SelfplayScheduler

scheduler = SelfplayScheduler(orchestrator)

# Get next config to run based on:
# - Curriculum weights
# - Elo velocity (faster improvement = more resources)
# - Data freshness
# - Quality signals
config = scheduler.pick_weighted_config()
```

**Scheduling Factors:**

1. Curriculum weights (from training feedback)
2. Elo velocity (configs improving fastest get priority)
3. Data age (stale configs get boost)
4. Quality feedback (high-quality generation gets more resources)

### DiskSpaceManagerDaemon (`app/coordination/disk_space_manager_daemon.py`)

Proactive disk space management:

```python
from app.coordination.disk_space_manager_daemon import get_disk_space_daemon

daemon = get_disk_space_daemon()

# Check current usage
usage = daemon.get_disk_usage()
print(f"Disk usage: {usage.percent}%")

# Trigger cleanup if needed
if usage.percent > 60:
    await daemon.run_cleanup()
```

**Cleanup Priority (oldest first):**

1. Old log files (>7 days)
2. Empty databases
3. Stale checkpoints
4. Redundant training data (after sync)

### CoordinatorDiskManager (`app/coordination/disk_space_manager_daemon.py`)

Specialized disk management for coordinator nodes:

```python
from app.coordination.disk_space_manager_daemon import get_coordinator_disk_daemon

daemon = get_coordinator_disk_daemon()
await daemon.start()

# Syncs to OWC before cleanup
# Uses lower thresholds (50% vs 60%)
```

## GPU Power Rankings

Nodes are ranked by GPU capability for training priority:

```python
from app.p2p.constants import GPU_POWER_RANKINGS

# Higher score = more powerful = higher training priority
rankings = {
    "H200": 2500,
    "H100": 2000,
    "A100": 624,
    "5090": 419,
    "L40": 362,
    "4090": 330,
    # ...
}
```

## Resource Events

| Event                    | Emitter               | Purpose                    |
| ------------------------ | --------------------- | -------------------------- |
| `IDLE_RESOURCE_DETECTED` | IdleResourceDaemon    | GPU/CPU became idle        |
| `RESOURCE_CONSTRAINT`    | ResourceTargetManager | Resource limit reached     |
| `NODE_OVERLOADED`        | ResourceTargetManager | Node under heavy load      |
| `DISK_SPACE_LOW`         | DiskSpaceManager      | Disk usage above threshold |
| `BACKPRESSURE_ACTIVATED` | BackpressureMonitor   | Need to slow down          |
| `BACKPRESSURE_RELEASED`  | BackpressureMonitor   | Can resume normal rate     |

## Configuration

Environment variables for resource management:

| Variable                       | Default | Description                        |
| ------------------------------ | ------- | ---------------------------------- |
| `RINGRIFT_TARGET_UTIL_MIN`     | 60      | Minimum GPU utilization target (%) |
| `RINGRIFT_TARGET_UTIL_MAX`     | 80      | Maximum GPU utilization target (%) |
| `RINGRIFT_MAX_DISK_PERCENT`    | 70      | Disk cleanup threshold (%)         |
| `RINGRIFT_MEMORY_LIMIT_GB`     | 80      | Per-process memory limit (GB)      |
| `RINGRIFT_IDLE_CHECK_INTERVAL` | 60      | Seconds between idle checks        |
| `RINGRIFT_IDLE_GPU_THRESHOLD`  | 5       | GPU util % below which is "idle"   |
| `RINGRIFT_IDLE_GRACE_PERIOD`   | 300     | Seconds idle before spawning job   |

## Disk Space Thresholds

```
0%  ─────────────────────────────────────────────────────── 100%
    │                    │              │              │
    │                    │              │              │
  Normal              Warning       Cleanup        Critical
                      (60%)         (65%)          (70%)
                                                      │
                                                      ▼
                                              Training blocked
```

## PID Controller for Utilization

The resource system uses a PID controller for adaptive scaling:

```python
from app.config.env import env

# PID gains (from environment)
kp = env.pid_kp  # Proportional: 0.1
ki = env.pid_ki  # Integral: 0.01
kd = env.pid_kd  # Derivative: 0.05

# Controller adjusts job spawn rate to maintain target utilization
```

## Best Practices

1. **Set appropriate thresholds per provider** - Vast.ai nodes have less disk than Lambda
2. **Use backpressure events** - React to resource constraints immediately
3. **Implement cleanup before sync** - Never delete without backing up first
4. **Monitor GPU idle time** - Underutilized GPUs waste money
5. **Respect memory limits** - OOM kills can corrupt training state

## Related Documentation

- [P2P_ORCHESTRATOR_ARCHITECTURE.md](P2P_ORCHESTRATOR_ARCHITECTURE.md) - Job scheduling details
- [DISK_SPACE_MANAGEMENT.md](../runbooks/DISK_SPACE_MANAGEMENT.md) - Operational runbook
- [TRAINING_LOOP_STALLED.md](../runbooks/TRAINING_LOOP_STALLED.md) - Troubleshooting resource issues
