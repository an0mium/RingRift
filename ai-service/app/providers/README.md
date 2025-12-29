# Cloud Provider Managers

This module contains cloud provider-specific management code for the distributed training cluster.

## Modules

| Module                 | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `base.py`              | Base class `CloudProviderManager` defining common interface |
| `aws_manager.py`       | AWS EC2 instance management                                 |
| `lambda_manager.py`    | Lambda Labs GPU instance management                         |
| `hetzner_manager.py`   | Hetzner Cloud server management                             |
| `tailscale_manager.py` | Tailscale VPN mesh management                               |

## Usage

```python
from app.providers.lambda_manager import LambdaManager

manager = LambdaManager()
instances = await manager.list_instances()
```

## Common Interface

All managers implement:

- `list_instances()` - List active instances
- `get_instance_status(instance_id)` - Get instance health
- `terminate_instance(instance_id)` - Terminate instance
- `get_ssh_config(instance_id)` - Get SSH connection config

## Integration

Used by:

- `daemon_manager.py` for multi-provider orchestration
- `node_recovery.py` for auto-recovery
- `unified_idle_shutdown_daemon.py` for cost optimization across all providers

## Status Updates (December 2025)

**Lambda Account Status**: Lambda Labs account restored December 28, 2025.
6 GH200 nodes (96GB each) are now available for training workloads. The `lambda_manager.py`
module is active again for Lambda node management.

**Idle Daemon Consolidation**: `lambda_idle_daemon.py` and `vast_idle_daemon.py` have been
consolidated into `unified_idle_shutdown_daemon.py` which provides provider-agnostic idle detection
and shutdown functionality. See `app/coordination/unified_idle_shutdown_daemon.py`.
