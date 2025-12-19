# Scripts Library (`scripts/lib/`)

Shared utilities for RingRift AI training cluster operations, data processing, and monitoring.

## Overview

This library provides common functionality used across training scripts, reducing code duplication and ensuring consistent patterns for:

- **Cluster operations**: SSH, file transfers, node management
- **Data processing**: Quality scoring, file formats, validation
- **Monitoring**: Health checks, alerts, metrics collection
- **Configuration**: Hosts, training params, paths
- **Process management**: Signals, locks, retries

## Modules

### Core Infrastructure

| Module   | Purpose                    | Key Exports                                                     |
| -------- | -------------------------- | --------------------------------------------------------------- |
| `ssh`    | SSH command execution      | `SSHConfig`, `run_ssh_command`, `run_ssh_command_async`         |
| `hosts`  | Cluster host configuration | `get_hosts`, `get_host`, `HostConfig`, `get_active_hosts`       |
| `paths`  | Standard project paths     | `AI_SERVICE_ROOT`, `DATA_DIR`, `MODELS_DIR`, `get_game_db_path` |
| `config` | Training configuration     | `TrainingConfig`, `get_config`, `ConfigManager`                 |

### Operations

| Module     | Purpose            | Key Exports                                        |
| ---------- | ------------------ | -------------------------------------------------- |
| `cluster`  | Node management    | `ClusterManager`, `ClusterNode`, `get_cluster`     |
| `transfer` | File transfers     | `scp_push`, `scp_pull`, `rsync_push`, `rsync_pull` |
| `process`  | Process management | `SingletonLock`, `SignalHandler`, `run_command`    |
| `retry`    | Retry with backoff | `@retry`, `@retry_on_exception`, `RetryConfig`     |

### Data & Quality

| Module         | Purpose              | Key Exports                                                 |
| -------------- | -------------------- | ----------------------------------------------------------- |
| `data_quality` | Game quality scoring | `GameQualityScorer`, `QualityFilter`, `QualityStats`        |
| `database`     | Database utilities   | `safe_transaction`, `get_elo_db_path`, `count_games`        |
| `file_formats` | JSONL/JSON handling  | `read_jsonl_lines`, `load_json`, `save_json`                |
| `validation`   | Data validation      | `validate_npz_file`, `validate_jsonl_file`, `DataValidator` |

### Monitoring & Alerts

| Module           | Purpose               | Key Exports                                                  |
| ---------------- | --------------------- | ------------------------------------------------------------ |
| `health`         | Health checks         | `check_system_health`, `check_http_health`, `SystemHealth`   |
| `alerts`         | Alert infrastructure  | `Alert`, `AlertManager`, `AlertSeverity`, `check_disk_alert` |
| `metrics`        | Statistics collection | `TimingStats`, `RateCalculator`, `ProgressTracker`           |
| `logging_config` | Logging setup         | `setup_script_logging`, `get_logger`, `JsonFormatter`        |

### Utilities

| Module           | Purpose              | Key Exports                                           |
| ---------------- | -------------------- | ----------------------------------------------------- |
| `cli`            | CLI argument helpers | `add_common_args`, `add_board_args`, `get_config_key` |
| `datetime_utils` | Timestamp utilities  | `format_elapsed_time`, `timestamp_id`, `ElapsedTimer` |
| `state_manager`  | State persistence    | `StateManager`, `load_json_state`, `save_json_state`  |

## Quick Start

### Basic Imports

```python
# Logging setup (do this first)
from scripts.lib import setup_script_logging, get_logger
setup_script_logging("my_script")
logger = get_logger(__name__)

# Common patterns
from scripts.lib import (
    get_hosts,           # Cluster host configuration
    run_ssh_command,     # SSH execution
    safe_transaction,    # Database operations
    retry,               # Retry decorator
)
```

### SSH Operations

```python
from scripts.lib.ssh import SSHConfig, run_ssh_command, run_ssh_command_async

# Simple SSH command
success, output = run_ssh_command("10.0.0.1", "nvidia-smi", user="ubuntu")

# With full configuration
config = SSHConfig(
    host="10.0.0.1",
    port=22,
    user="ubuntu",
    ssh_key="~/.ssh/id_rsa",
    timeout=30,
)
success, output = run_ssh_command_with_config(config, "ls -la")

# Async version
success, output = await run_ssh_command_async("10.0.0.1", "df -h")
```

### Cluster Host Configuration

```python
from scripts.lib.hosts import get_hosts, get_host, get_active_hosts

# Get all configured hosts
hosts = get_hosts()
for name, config in hosts.items():
    print(f"{name}: {config.ssh_host}")

# Get specific host
host = get_host("lambda-h100")
print(f"SSH: {host.ssh_user}@{host.ssh_host}:{host.ssh_port}")

# Get only active (non-disabled) hosts
active = get_active_hosts()
```

### File Transfers

```python
from scripts.lib.transfer import scp_push, rsync_pull, TransferConfig

# Push file to remote
result = scp_push(
    local_path="/path/to/data.npz",
    remote_path="/home/ubuntu/data/",
    host="10.0.0.1",
    user="ubuntu",
)

# Rsync with resume
result = rsync_pull(
    remote_path="/data/games/",
    local_path="./games/",
    host="10.0.0.1",
    user="ubuntu",
    compress=True,
)
```

### Database Operations

```python
from scripts.lib.database import safe_transaction, get_elo_db_path, count_games

# Safe transaction context
with safe_transaction(get_elo_db_path()) as conn:
    cursor = conn.execute("SELECT * FROM models WHERE elo > ?", (1600,))
    results = cursor.fetchall()

# Count games in selfplay database
total = count_games("hex_2p")  # Returns game count
```

### Retry Pattern

```python
from scripts.lib.retry import retry, RetryConfig

@retry(max_attempts=3, delay=1.0, backoff=2.0)
def flaky_network_call():
    response = requests.get("http://api.example.com/data")
    response.raise_for_status()
    return response.json()

# With custom config
config = RetryConfig(
    max_attempts=5,
    delay=0.5,
    backoff=2.0,
    max_delay=30.0,
    exceptions=(ConnectionError, TimeoutError),
)

@retry(config=config)
def resilient_operation():
    ...
```

### Health Checks

```python
from scripts.lib.health import (
    check_system_health,
    check_http_health,
    check_disk_space,
)

# Full system health
health = check_system_health()
print(f"CPU: {health.cpu.percent}%")
print(f"Memory: {health.memory.percent}%")
print(f"Disk: {health.disk.percent}%")

# HTTP endpoint health
service = check_http_health("http://localhost:8000/health")
if not service.healthy:
    print(f"Service unhealthy: {service.error}")

# Disk space check
disk = check_disk_space("/data")
if disk.percent > 90:
    print(f"Warning: Disk at {disk.percent}%")
```

### Metrics & Timing

```python
from scripts.lib.metrics import TimingStats, ProgressTracker, RateCalculator

# Timing statistics
timing = TimingStats()
with timing.measure("operation_name"):
    do_something()
print(f"Avg time: {timing.avg_ms:.2f}ms")

# Progress tracking with ETA
progress = ProgressTracker(total=1000)
for item in items:
    process(item)
    progress.update(1)
    print(f"Progress: {progress.percent:.1f}% | ETA: {progress.eta_str}")

# Rate calculation
rate = RateCalculator()
for batch in batches:
    rate.add(len(batch))
print(f"Throughput: {rate.rate:.1f} items/sec")
```

### Alerts

```python
from scripts.lib.alerts import AlertManager, AlertSeverity, AlertType

manager = AlertManager(webhook_url="https://hooks.slack.com/...")

# Check thresholds and alert
from scripts.lib.health import check_disk_space
disk = check_disk_space("/data")
if disk.percent > 90:
    manager.send_alert(
        severity=AlertSeverity.WARNING,
        alert_type=AlertType.DISK,
        message=f"Disk usage critical: {disk.percent:.1f}%",
        node="gpu-server-1",
    )
```

### Process Management

```python
from scripts.lib.process import SingletonLock, SignalHandler

# Prevent duplicate processes
with SingletonLock("my_daemon"):
    # Only one instance can run
    run_daemon()

# Graceful shutdown handling
handler = SignalHandler()
while handler.running:
    do_work()
print("Shutting down gracefully...")
```

## Configuration Files

The library reads configuration from:

- `config/distributed_hosts.yaml` - Cluster host definitions
- `config/cluster.yaml` - Legacy cluster config (fallback)
- `config/training_config.yaml` - Training hyperparameters

## Testing

Run library tests:

```bash
python -m pytest tests/test_lib_*.py -v
```

## Adding New Modules

1. Create `scripts/lib/my_module.py` with docstring
2. Add exports to `scripts/lib/__init__.py`
3. Add tests in `tests/test_lib_my_module.py`
4. Update this README

## Module Dependencies

```
                 ┌─────────────────────────────────────────┐
                 │              logging_config             │
                 └─────────────────┬───────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        v                          v                          v
   ┌─────────┐              ┌─────────────┐            ┌───────────┐
   │  paths  │              │   config    │            │   retry   │
   └────┬────┘              └──────┬──────┘            └─────┬─────┘
        │                          │                         │
        v                          v                         v
   ┌─────────┐              ┌─────────────┐            ┌───────────┐
   │   ssh   │              │    hosts    │            │  process  │
   └────┬────┘              └──────┬──────┘            └───────────┘
        │                          │
        v                          v
   ┌─────────┐              ┌─────────────┐
   │ cluster │─────────────>│  transfer   │
   └─────────┘              └─────────────┘
```

Low-level modules (paths, logging_config, retry) have no intra-library dependencies.
Higher-level modules (cluster, transfer) build on foundational utilities.
