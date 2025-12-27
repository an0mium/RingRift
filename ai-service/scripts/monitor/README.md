# Monitoring Scripts

Python modules for cluster health monitoring and dashboards.

## Modules

| Module                    | Purpose                        |
| ------------------------- | ------------------------------ |
| `dashboard.py`            | Real-time monitoring dashboard |
| `health.py`               | Health check utilities         |
| `health_metrics_smoke.py` | Smoke tests for health metrics |

## Usage

```bash
# Run as a module
python -m scripts.monitor

# Or import directly
from scripts.monitor import dashboard, health
```

## See Also

- `app/monitoring/` - Core monitoring infrastructure
- `app/distributed/cluster_monitor.py` - Cluster-wide monitoring
- `scripts/cluster/cluster_health.sh` - Shell-based health checks
