# Master Loop Enforcement - Implementation Summary

**Date**: December 26, 2025
**Purpose**: Make the master loop mandatory for full automation

## Changes Made

### 1. PID File Management (`scripts/master_loop.py`)

**Location**: `ai-service/scripts/master_loop.py`

**Changes**:

- Added `PID_FILE_PATH` constant pointing to `data/coordination/master_loop.pid`
- Added `_create_pid_file()` method - creates PID file on startup
- Added `_remove_pid_file()` method - removes PID file on shutdown
- Added `is_running()` static method - checks if master loop process is active
- Modified `start()` to create PID file
- Modified `stop()` to remove PID file

**PID File Contents**: Simple text file containing the process ID (PID) of the running master loop.

### 2. Master Loop Guard Module (`app/coordination/master_loop_guard.py`)

**Location**: `ai-service/app/coordination/master_loop_guard.py`

**New utility module providing**:

- `is_master_loop_running()` - Check if master loop is active
- `ensure_master_loop_running()` - Enforce master loop requirement (raises RuntimeError)
- `check_or_warn()` - Check and log warning if master loop not running
- `PID_FILE_PATH` - Shared constant for PID file location

**Features**:

- Process existence check using `os.kill(pid, 0)`
- Automatic stale PID file cleanup
- Configurable error messages
- Optional enforcement

### 3. Coordination Bootstrap Integration (`app/coordination/coordination_bootstrap.py`)

**Location**: `ai-service/app/coordination/coordination_bootstrap.py`

**Changes**:

- Added `require_master_loop` parameter to `bootstrap_coordination()`
- Added master loop check at start of bootstrap (before any coordinator initialization)
- Respects `RINGRIFT_SKIP_MASTER_LOOP_CHECK=1` environment variable
- Raises `RuntimeError` with helpful message if master loop not running

**Integration Point**:

```python
bootstrap_coordination(
    enable_integrations=True,
    pipeline_auto_trigger=True,
    require_master_loop=True,  # New parameter
)
```

### 4. Daemon Manager Warning (`app/coordination/daemon_manager.py`)

**Location**: `ai-service/app/coordination/daemon_manager.py`

**Changes**:

- Added warning in `start_all()` method when master loop not running
- Uses `check_or_warn()` for non-blocking notification
- Provides guidance to use master loop for full automation

**Warning Message**:

```
[DaemonManager] Master loop is not running for daemon management.
[DaemonManager] For full automation, use: python scripts/master_loop.py
```

### 5. Documentation

**Files Created**:

- `ai-service/docs/MASTER_LOOP_GUARD.md` - User guide
- `ai-service/docs/MASTER_LOOP_ENFORCEMENT.md` - This file

**Files Updated**:

- None (new feature, no existing docs to update)

### 6. Tests

**File**: `ai-service/tests/test_master_loop_guard.py`

**Test Coverage**:

- `test_is_master_loop_running_no_file` - No PID file returns False
- `test_is_master_loop_running_valid_process` - Valid process returns True
- `test_is_master_loop_running_stale_pid` - Stale PID file cleanup
- `test_is_master_loop_running_invalid_pid` - Invalid PID handling
- `test_ensure_master_loop_running_no_check` - Disabled requirement
- `test_ensure_master_loop_running_raises` - Raises when not running
- `test_ensure_master_loop_running_passes` - Passes when running
- `test_check_or_warn_returns_bool` - Warning mode

**Test Results**: ✓ 8/8 passed

## Usage Examples

### 1. Starting Master Loop (Recommended)

```bash
# Full automation with all configs
python scripts/master_loop.py

# Specific configs only
python scripts/master_loop.py --configs hex8_2p,square8_2p

# Dry run (preview without executing)
python scripts/master_loop.py --dry-run
```

### 2. Checking if Master Loop is Running

```python
from app.coordination.master_loop_guard import is_master_loop_running

if is_master_loop_running():
    print("Full automation enabled")
else:
    print("Manual/limited automation mode")
```

### 3. Enforcing Master Loop Requirement

```python
from app.coordination.master_loop_guard import ensure_master_loop_running

# Raises RuntimeError if master loop not running
ensure_master_loop_running(
    require_for_automation=True,
    operation_name="critical coordination"
)
```

### 4. Bootstrap with Requirement

```python
from app.coordination.coordination_bootstrap import bootstrap_coordination

bootstrap_coordination(
    enable_integrations=True,
    pipeline_auto_trigger=True,
    require_master_loop=True,  # Enforce master loop
)
```

### 5. Skip Check (Testing/Development)

```bash
# Set env var to skip check
export RINGRIFT_SKIP_MASTER_LOOP_CHECK=1
python scripts/your_script.py
```

## Implementation Details

### PID File Location

```
ai-service/data/coordination/master_loop.pid
```

### Process Detection Algorithm

1. Check if PID file exists
2. Read PID from file
3. Use `os.kill(pid, 0)` to check if process exists
   - Signal 0 doesn't kill, just checks existence
   - Returns success if process alive
   - Raises OSError if process dead
4. If process dead, remove stale PID file
5. Return True/False based on process existence

### Stale PID Cleanup

The guard automatically removes stale PID files:

- Master loop crashed without cleanup
- PID file from previous boot
- Process terminated externally

This ensures `is_master_loop_running()` is always accurate.

### Environment Variable Override

`RINGRIFT_SKIP_MASTER_LOOP_CHECK=1` bypasses all checks:

- Useful for testing individual components
- Development/debugging scenarios
- Single-script operations
- Should NOT be used in production

## Design Decisions

### 1. Why PID File Instead of Process Name?

**Chosen**: PID file
**Rejected**: Process name matching

**Reasoning**:

- More reliable - exact process identification
- No false positives from similar process names
- Works across different shells/environments
- Standard Unix practice

### 2. Why Static Methods on MasterLoopController?

**Chosen**: Static methods for checking
**Rejected**: Separate utility module only

**Reasoning**:

- Keeps logic close to implementation
- Allows access to internal constants
- Provides both instance and static APIs
- Guard module can use static methods

### 3. Why RuntimeError Instead of Warning?

**Chosen**: RuntimeError when `require_master_loop=True`
**Rejected**: Warning-only approach

**Reasoning**:

- Explicit enforcement for critical operations
- Prevents silent failures
- Still allows warnings via `check_or_warn()`
- Can be bypassed with env var

### 4. Why Optional Instead of Mandatory?

**Chosen**: Optional via `require_master_loop` parameter
**Rejected**: Always enforce

**Reasoning**:

- Backward compatibility
- Gradual rollout
- Testing flexibility
- Development convenience

## Benefits

1. **Unified Control**: All automation runs through master loop
2. **Resource Coordination**: Prevents daemon conflicts
3. **Clear State**: Easy to verify automation status
4. **Graceful Degradation**: Detects partial automation
5. **Better Debugging**: Single process to monitor
6. **Prevents Fragmentation**: No orphaned daemons

## Rollout Strategy

### Phase 1: Warning Only (Current)

- Daemon manager warns if master loop not running
- No breaking changes
- Users become aware of requirement

### Phase 2: Opt-In Enforcement

- Add `require_master_loop=True` to critical paths
- Automated training pipelines
- Production deployments

### Phase 3: Default Enforcement

- Make `require_master_loop=True` the default
- Require explicit opt-out
- Only for testing/development

### Phase 4: Mandatory

- Remove `require_master_loop` parameter
- Always enforce (unless `RINGRIFT_SKIP_MASTER_LOOP_CHECK=1`)
- Production-only mode

## Current Status

**Phase**: 1 (Warning Only)
**Enforcement**: Optional via `require_master_loop=True`
**Default**: False (backward compatible)
**Recommendation**: Use master loop for all automation

## Testing Verification

```bash
# Run unit tests
python -m pytest tests/test_master_loop_guard.py -v

# Test integration
python -c "
from app.coordination.coordination_bootstrap import bootstrap_coordination
from app.coordination.master_loop_guard import is_master_loop_running

print('Master loop running:', is_master_loop_running())

# This should fail
try:
    bootstrap_coordination(require_master_loop=True)
    print('ERROR: Should have raised')
except RuntimeError as e:
    print('SUCCESS: Correctly raised:', e)
"
```

## See Also

- `scripts/master_loop.py` - Main automation controller
- `app/coordination/master_loop_guard.py` - Guard implementation
- `app/coordination/coordination_bootstrap.py` - Bootstrap integration
- `MASTER_LOOP_GUARD.md` - User guide
