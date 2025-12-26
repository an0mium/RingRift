# Master Loop Guard - Ensuring Full Automation

**December 2025** - Makes the master loop mandatory for full automation.

## Overview

The Master Loop Guard ensures that critical coordination operations only run when the master loop (`scripts/master_loop.py`) is active. This prevents fragmented automation where daemons run independently without unified orchestration.

## Usage

### Starting the Master Loop

For full automation, always use the master loop:

```bash
# Full automation mode (recommended)
python scripts/master_loop.py

# With specific configs
python scripts/master_loop.py --configs hex8_2p,square8_2p

# Dry run (preview without executing)
python scripts/master_loop.py --dry-run
```

### Checking if Master Loop is Running

```python
from app.coordination.master_loop_guard import is_master_loop_running

if is_master_loop_running():
    print("Master loop is active - full automation enabled")
else:
    print("Master loop is NOT running - limited automation")
```

### Enforcing Master Loop Requirement

```python
from app.coordination.master_loop_guard import ensure_master_loop_running

# This will raise RuntimeError if master loop is not running
ensure_master_loop_running(
    require_for_automation=True,
    operation_name="critical coordination task"
)
```

### Warning-Only Check

```python
from app.coordination.master_loop_guard import check_or_warn

# Logs a warning but doesn't raise
is_running = check_or_warn("daemon startup")
if not is_running:
    # Handle degraded automation mode
    pass
```

## Integration Points

### Coordination Bootstrap

The coordination bootstrap can enforce master loop requirement:

```python
from app.coordination.coordination_bootstrap import bootstrap_coordination

# Require master loop for full automation
bootstrap_coordination(
    enable_integrations=True,
    pipeline_auto_trigger=True,
    require_master_loop=True,  # Will raise if master loop not running
)
```

### Daemon Manager

The daemon manager automatically warns when starting daemons without the master loop:

```python
from app.coordination.daemon_manager import get_daemon_manager

manager = get_daemon_manager()
# This will log a warning if master loop is not running:
# "[DaemonManager] Master loop is not running for daemon management."
# "[DaemonManager] For full automation, use: python scripts/master_loop.py"
await manager.start_all()
```

## Environment Variable Override

To skip the master loop check (e.g., for testing), set:

```bash
export RINGRIFT_SKIP_MASTER_LOOP_CHECK=1
python scripts/your_script.py
```

## Implementation Details

### PID File Location

The master loop creates a PID file at:

```
data/coordination/master_loop.pid
```

This file contains the process ID of the running master loop. The guard checks:

1. Does the PID file exist?
2. Is the process with that PID still running?
3. If not, clean up the stale PID file.

### Process Detection

The guard uses `os.kill(pid, 0)` to check if a process exists:

- Signal 0 doesn't kill the process, just checks existence
- Returns successfully if process exists
- Raises OSError if process doesn't exist

### Automatic Cleanup

If a PID file exists but the process is dead (e.g., master loop crashed), the guard automatically removes the stale PID file.

## Master Loop Methods

The `MasterLoopController` class provides these static methods:

```python
from scripts.master_loop import MasterLoopController

# Check if running
is_running = MasterLoopController.is_running()

# Check health (uses heartbeat from state DB)
health = MasterLoopController.check_health()
if health["healthy"]:
    print(f"Loop iteration: {health['loop_iteration']}")
    print(f"Last beat: {health['age_seconds']:.1f} seconds ago")
```

## Benefits

1. **Unified Control**: All automation runs through a single entry point
2. **Resource Coordination**: Master loop prevents resource conflicts
3. **Clear State**: Easy to check if full automation is active
4. **Graceful Degradation**: Can detect and warn about degraded automation mode
5. **Prevents Fragmentation**: Ensures daemons don't run independently

## When to Use

**Always use master loop for:**

- Production cluster automation
- Long-running training pipelines
- Multi-node coordination
- Resource-intensive operations

**Can skip master loop for:**

- Testing individual components
- Single-script operations
- Manual debugging
- Development mode

## Troubleshooting

### Master Loop Not Detected

If `is_master_loop_running()` returns `False` but you've started it:

1. Check PID file exists: `ls data/coordination/master_loop.pid`
2. Verify process is running: `ps aux | grep master_loop`
3. Check for stale PID file: manually remove if process crashed

### Unwanted Enforcement

If you're getting errors about master loop not running:

1. Set `RINGRIFT_SKIP_MASTER_LOOP_CHECK=1` to bypass
2. Or start the master loop with `--skip-daemons` for testing

## Examples

### Full Automation Setup

```bash
# Start master loop in background
nohup python scripts/master_loop.py > logs/master_loop.log 2>&1 &

# Check it's running
python -c "
from app.coordination.master_loop_guard import is_master_loop_running
print('Master loop running:', is_master_loop_running())
"

# Now run coordination bootstrap
python -c "
from app.coordination.coordination_bootstrap import bootstrap_coordination
bootstrap_coordination(
    enable_integrations=True,
    pipeline_auto_trigger=True,
    require_master_loop=True,
)
"
```

### Testing Without Master Loop

```bash
# Skip master loop check for testing
RINGRIFT_SKIP_MASTER_LOOP_CHECK=1 python scripts/my_test.py
```

### Manual Daemon Start (with warning)

```bash
# This will warn but not fail
python scripts/launch_daemons.py --all
# Warning: "[DaemonManager] Master loop is not running for daemon management."
```

## See Also

- `scripts/master_loop.py` - Main automation controller
- `app/coordination/master_loop_guard.py` - Guard implementation
- `app/coordination/coordination_bootstrap.py` - Bootstrap integration
- `app/coordination/daemon_manager.py` - Daemon management
