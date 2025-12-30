# Auto-Cascade NPZ Export

Automatic NPZ export triggering when selfplay completes.

## Overview

The `AutoExportDaemon` automatically triggers NPZ dataset export when enough games have been generated from selfplay. This eliminates the manual step of running `export_replay_dataset.py` and enables fully automated training pipelines.

## How It Works

### Event Flow

```
SELFPLAY_COMPLETE event
         ↓
AutoExportDaemon records games
         ↓
Game count >= threshold (100 games)?
         ↓
Trigger NPZ export subprocess
         ↓
NPZ_EXPORT_COMPLETE event
         ↓
Training can begin automatically
```

### Key Features

1. **Event-Driven**: Listens for `SELFPLAY_COMPLETE` and `DATA_SYNC_COMPLETED` events
2. **Threshold-Based**: Only exports when game count reaches threshold (default: 100 games)
3. **Cooldown Protection**: Prevents rapid re-exports (default: 5 minutes)
4. **Concurrent Exports**: Supports multiple configs exporting in parallel (max: 2)
5. **State Persistence**: Survives daemon restarts without losing pending game counts
6. **Incremental Export**: Uses `--use-cache` flag for faster subsequent exports

## Configuration

Default configuration (from `AutoExportConfig`):

```python
min_games_threshold: int = 100          # Minimum games before export
export_cooldown_seconds: int = 300      # 5 minutes between exports
max_concurrent_exports: int = 2         # Parallel exports limit
export_timeout_seconds: int = 3600      # 1 hour timeout
use_incremental_export: bool = True     # Enable --use-cache
require_completed_games: bool = True    # Only export finished games
min_moves: int = 10                     # Minimum moves per game
output_dir: Path = "data/training"      # NPZ output directory
```

## Usage

### Automatic Startup

The daemon is automatically started by `DaemonManager` when enabled:

```python
from app.coordination.daemon_manager import DaemonManager

manager = DaemonManager()
await manager.start_daemon(DaemonType.AUTO_EXPORT)
```

### Manual Control

```python
from app.coordination.auto_export_daemon import get_auto_export_daemon

# Get singleton instance
daemon = get_auto_export_daemon()

# Start daemon
await daemon.start()

# Check status
status = daemon.get_status()
print(f"Tracking {status['configs_tracked']} configurations")
print(f"States: {status['states']}")

# Stop daemon
await daemon.stop()
```

### Custom Configuration

```python
from app.coordination.auto_export_daemon import AutoExportDaemon, AutoExportConfig
from pathlib import Path

config = AutoExportConfig(
    min_games_threshold=200,           # Higher threshold
    export_cooldown_seconds=600,       # 10 minute cooldown
    output_dir=Path("custom/output"),  # Custom output directory
)

daemon = AutoExportDaemon(config)
await daemon.start()
```

## Events Emitted

### NPZ_EXPORT_STARTED

Emitted when export subprocess begins:

```python
{
    "event": "npz_export_started",
    "config": "hex8_2p",
    "games_pending": 125,
    "timestamp": "2025-12-26T10:30:00"
}
```

### NPZ_EXPORT_COMPLETE

Emitted when export subprocess completes successfully:

```python
{
    "event": "npz_export_complete",
    "config": "hex8_2p",
    "output_path": "data/training/hex8_2p.npz",
    "samples": 5432,
    "games_exported": 125,
    "duration_seconds": 12.5,
    "timestamp": "2025-12-26T10:30:12"
}
```

## Integration with Training Pipeline

The `NPZ_EXPORT_COMPLETE` event can trigger downstream training:

```python
from app.coordination.event_router import get_stage_event_bus, StageEvent

bus = get_stage_event_bus()

async def on_export_complete(result):
    """Trigger training when export completes."""
    config = result.metadata["config"]
    npz_path = result.metadata["output_path"]

    # Start training with the new dataset
    await start_training(config, npz_path)

bus.subscribe(StageEvent.NPZ_EXPORT_COMPLETE, on_export_complete)
```

## State Persistence

The daemon persists state to SQLite (`data/export_daemon_state.db`) to survive crashes:

```sql
CREATE TABLE export_state (
    config_key TEXT PRIMARY KEY,
    board_type TEXT NOT NULL,
    num_players INTEGER NOT NULL,
    games_since_last_export INTEGER DEFAULT 0,
    last_export_time REAL DEFAULT 0,
    last_export_games INTEGER DEFAULT 0,
    total_exported_samples INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    updated_at REAL DEFAULT 0
);
```

## Monitoring

### Via Status API

```python
daemon = get_auto_export_daemon()
status = daemon.get_status()

for config, state in status["states"].items():
    print(f"{config}:")
    print(f"  Pending games: {state['games_pending']}")
    print(f"  Last export: {state['last_export']} seconds ago")
    print(f"  Total samples: {state['total_samples']}")
    print(f"  In progress: {state['in_progress']}")
    print(f"  Failures: {state['failures']}")
```

### Via Logs

The daemon logs export activity:

```
[AutoExportDaemon] hex8_2p: +50 games, total pending: 75
[AutoExportDaemon] hex8_2p: +30 games, total pending: 105
[AutoExportDaemon] Starting export for hex8_2p (105 games pending)
[AutoExportDaemon] Export complete for hex8_2p: 105 games, 4523 samples, 11.2s
[AutoExportDaemon] Emitted NPZ_EXPORT_COMPLETE for hex8_2p
```

## Error Handling

### Export Failures

- Consecutive failures are tracked per config
- Failed exports don't reset the pending game count
- Logs include stderr output for debugging

### Timeout Protection

- Exports timeout after 1 hour (configurable)
- Timed-out processes are killed cleanly
- State is persisted even on timeout

### Cooldown Bypass

The cooldown can be bypassed by directly calling:

```python
daemon = get_auto_export_daemon()
await daemon._run_export("hex8_2p")  # Force export
```

## Performance Characteristics

- **Export time**: ~10-15 seconds for 100 games (varies by board size)
- **Memory overhead**: Minimal (~10MB state tracking)
- **Disk I/O**: Sequential reads from game DBs
- **CPU usage**: Export subprocess only (not the daemon itself)

## Comparison to Manual Export

### Before (Manual)

```bash
# After selfplay completes
python scripts/selfplay.py --board hex8 --num-players 2 --num-games 100

# Manually export (easy to forget!)
python scripts/export_replay_dataset.py \
  --use-discovery --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# Manually start training
python -m app.training.train --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz
```

### After (Automatic)

```bash
# Selfplay completes
python scripts/selfplay.py --board hex8 --num-players 2 --num-games 100

# AutoExportDaemon automatically:
# 1. Detects SELFPLAY_COMPLETE event
# 2. Checks threshold (100 games >= 100)
# 3. Triggers export subprocess
# 4. Emits NPZ_EXPORT_COMPLETE

# Training coordinator automatically:
# 1. Detects NPZ_EXPORT_COMPLETE event
# 2. Starts training with new dataset
```

## Testing

Run the test suite:

```bash
pytest tests/unit/coordination/test_auto_export_daemon.py -v
```

Test coverage includes:

- Event handler registration
- Game count accumulation
- Threshold triggering
- Cooldown prevention
- Concurrent export limiting
- State persistence
- Sample count parsing

## See Also

- `ai-service/app/coordination/auto_export_daemon.py` - Implementation
- `ai-service/app/coordination/daemon_manager.py` - Daemon lifecycle
- `ai-service/scripts/export_replay_dataset.py` - Export script
- `EVENT_CATALOG.md` - Event types reference
