# AI Service Scripts

This directory contains scripts for the RingRift AI training and improvement infrastructure.

## Canonical Entry Points

### Primary Orchestrator

**`unified_ai_loop.py`** - The canonical self-improvement orchestrator. This is the main entry point for the AI improvement loop.

```bash
# Start the unified loop
python scripts/unified_ai_loop.py --start

# Run in foreground with verbose output
python scripts/unified_ai_loop.py --foreground --verbose

# Check status
python scripts/unified_ai_loop.py --status
```

Features:

- Streaming data collection from distributed nodes (60s sync)
- Shadow tournament evaluation (15min lightweight)
- Training scheduler with data quality gates
- Model promotion with Elo thresholds
- Adaptive curriculum weighting
- Value calibration analysis
- Temperature scheduling for exploration control

## Key Categories

### Cluster Management

- `cluster_orchestrator.py` - Distributed cluster coordination
- `cluster_manager.py` - Cluster node management
- `cluster_worker.py` - Worker node implementation
- `cluster_control.py` - Cluster control commands
- `cluster_health_check.py` - Health monitoring

### Training

- `curriculum_training.py` - Generation-based curriculum training
- `run_self_play_soak.py` - Self-play data generation
- `run_hybrid_selfplay.py` - Hybrid self-play modes

### Evaluation

- `run_model_elo_tournament.py` - Model Elo tournaments
- `run_diverse_tournaments.py` - Multi-configuration tournaments
- `elo_promotion_gate.py` - Elo-based model promotion

### Data Management

- `cluster_sync_coordinator.py` - **Cluster-wide sync orchestrator** (coordinates all sync utilities)
  - Full sync: `python scripts/cluster_sync_coordinator.py --mode full`
  - Models only: `python scripts/cluster_sync_coordinator.py --mode models`
  - Games only: `python scripts/cluster_sync_coordinator.py --mode games`
  - ELO only: `python scripts/cluster_sync_coordinator.py --mode elo`
  - Status: `python scripts/cluster_sync_coordinator.py --status`
  - Uses aria2/tailscale/cloudflare for hard-to-reach nodes
- `unified_data_sync.py` - **Unified data sync service** (replaces deprecated scripts below)
  - Run as daemon: `python scripts/unified_data_sync.py`
  - With watchdog: `python scripts/unified_data_sync.py --watchdog`
  - One-shot sync: `python scripts/unified_data_sync.py --once`
- `streaming_data_collector.py` - _(DEPRECATED)_ Incremental game data sync - use `unified_data_sync.py`
- `collector_watchdog.py` - _(DEPRECATED)_ Collector health monitoring - use `unified_data_sync.py --watchdog`
- `sync_all_data.py` - _(DEPRECATED)_ Batch data sync - use `unified_data_sync.py --once`
- `build_canonical_training_pool_db.py` - Training data pooling
- `aggregate_jsonl_to_db.py` - JSONL to SQLite conversion
- `elo_db_sync.py` - ELO database synchronization across cluster

### Model Management

- `sync_models.py` - Model synchronization across cluster
- `prune_models.py` - Old model cleanup
- `model_promotion_manager.py` - Automated model promotion

### Analysis

- `analyze_game_statistics.py` - Game statistics analysis
- `check_ts_python_replay_parity.py` - TS/Python parity validation
- `track_elo_improvement.py` - Elo trend tracking

## Module Dependencies

### Canonical Service Interfaces

| Module                                | Purpose                | Usage                                        |
| ------------------------------------- | ---------------------- | -------------------------------------------- |
| `app.training.elo_service`            | Elo rating operations  | `get_elo_service()` singleton                |
| `app.training.curriculum`             | Curriculum training    | `CurriculumTrainer`, `CurriculumConfig`      |
| `app.training.value_calibration`      | Value head calibration | `ValueCalibrator`, `CalibrationTracker`      |
| `app.training.temperature_scheduling` | Exploration control    | `TemperatureScheduler`, `create_scheduler()` |

### Supporting Modules

| Module                                | Purpose                     |
| ------------------------------------- | --------------------------- |
| `app.tournament.elo`                  | Elo calculation utilities   |
| `app.training.elo_reconciliation`     | Distributed Elo consistency |
| `app.distributed.cluster_coordinator` | Cluster coordination        |
| `app.integration.pipeline_feedback`   | Training feedback loops     |

## Archived Scripts

The `archive/` subdirectory contains deprecated scripts that have been superseded:

| Script                                   | Superseded By                              |
| ---------------------------------------- | ------------------------------------------ |
| `master_self_improvement.py`             | `unified_ai_loop.py`                       |
| `unified_improvement_controller.py`      | `unified_ai_loop.py`                       |
| `integrated_self_improvement.py`         | `unified_ai_loop.py`                       |
| `export_replay_dataset.py`               | Direct DB queries                          |
| `validate_canonical_training_sources.py` | Data quality gates in `unified_ai_loop.py` |

## Resource Management

All scripts enforce **80% maximum resource utilization** to prevent overloading:

### Resource Limits (enforced 2025-12-16)
| Resource | Warning | Critical | Notes |
| -------- | ------- | -------- | ----- |
| Disk | 65% | 70% | Tighter limit - cleanup takes time |
| Memory | 70% | 80% | Hard stop when exceeded |
| CPU | 70% | 80% | Hard stop when exceeded |
| GPU | 70% | 80% | CUDA memory safety |
| Load Avg | - | 1.5x CPUs | System overload detection |

### Using Resource Guard
```python
from app.utils.resource_guard import (
    check_disk_space, check_memory, check_gpu_memory,
    can_proceed, wait_for_resources, ResourceGuard
)

# Pre-flight check before heavy operations
if not can_proceed(disk_required_gb=5.0, mem_required_gb=2.0):
    logger.error("Resource limits exceeded")
    sys.exit(1)

# Context manager for resource-safe operations
with ResourceGuard(disk_required_gb=5.0, mem_required_gb=2.0) as guard:
    if not guard.ok:
        return  # Resources not available
    # ... do work ...

# Periodic check in long-running loops
for i in range(num_games):
    if i % 50 == 0 and not check_memory():
        logger.warning("Memory pressure, stopping early")
        break
```

### Key Files
- `app/utils/resource_guard.py` - Unified resource checking utilities
- `app/coordination/safeguards.py` - Circuit breakers and backpressure
- `app/coordination/resource_targets.py` - Utilization targets for scaling
- `app/coordination/resource_optimizer.py` - PID-based workload adjustment
- `scripts/disk_monitor.py` - Disk cleanup automation

## Environment Variables

| Variable                         | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `RINGRIFT_DISABLE_LOCAL_TASKS`   | Skip local training/eval (coordinator mode) |
| `RINGRIFT_TRACE_DEBUG`           | Enable detailed tracing                     |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | Skip shadow contract validation             |

## Configuration

The unified loop reads configuration from `config/unified_loop.yaml`. Key settings:

- Data sync intervals
- Training thresholds
- Evaluation frequencies
- Cluster coordination options
