# Coordination Resources Package

Resource management and optimization utilities for the RingRift training cluster.

## Overview

This package provides abstractions for managing cluster resources including:

- Bandwidth allocation and limits
- Dynamic threshold management
- Resource optimization with PID control

## Modules

### `manager.py` - ResourceOptimizer

Adaptive resource allocation using PID-controlled scaling:

```python
from app.coordination.resources import ResourceOptimizer

optimizer = ResourceOptimizer()
allocation = optimizer.get_allocation(
    gpu_utilization=0.75,
    memory_usage=0.60,
    queue_depth=150
)
```

### `bandwidth.py` - BandwidthManager

Per-host bandwidth limits for data sync:

```python
from app.coordination.resources import BandwidthManager

manager = BandwidthManager()
limit = manager.get_limit("lambda-gh200-a")  # Returns MB/s limit
```

**Default limits** (from `distributed_hosts.yaml`):

- Lambda GH200: 100 MB/s
- RunPod: 50 MB/s
- Vast.ai: 50 MB/s (ephemeral)
- Hetzner CPU: 20 MB/s

### `thresholds.py` - DynamicThreshold

Adaptive thresholds based on system load:

```python
from app.coordination.resources import DynamicThreshold

threshold = DynamicThreshold(
    base=0.7,
    min_value=0.5,
    max_value=0.9
)
current = threshold.get(load_factor=0.8)
```

## Integration

Used by:

- `AutoSyncDaemon` - Bandwidth-limited data sync
- `SyncRouter` - Capacity-aware routing decisions
- `IdleResourceDaemon` - GPU utilization detection
- `WorkDistributor` - Job scheduling based on capacity

## Configuration

Settings loaded from `config/distributed_hosts.yaml`:

```yaml
sync_routing:
  max_disk_usage_percent: 70.0
  target_disk_usage_percent: 60.0

auto_sync:
  bandwidth_limit_mbps: 100
  max_concurrent_syncs: 4
```

## See Also

- `../sync_bandwidth.py` - Bandwidth-coordinated rsync
- `../sync_router.py` - Intelligent sync routing
- `../../distributed/cluster_manifest.py` - Capacity tracking
