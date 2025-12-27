# Unified AI Self-Improvement Loop

> **Doc Status (2025-12-26): LEGACY**
>
> **Canonical Entrypoint:** `scripts/master_loop.py` (use this)
> **Legacy Entrypoint:** `scripts/unified_ai_loop.py` (deprecated, requires `RINGRIFT_UNIFIED_LOOP_LEGACY=1`)
>
> **Migration Note:** The monolithic unified_ai_loop.py is deprecated. Use master_loop.py with
> the DaemonManager for better isolation, profile-based startup, and EventRouter integration.
> Key features from unified_loop (data collection, local selfplay, pruning, holdout promotion)
> are being ported to dedicated daemons.

The unified loop architecture is now driven by `scripts/master_loop.py` plus the daemon stack. The
legacy monolithic loop (`scripts/unified_ai_loop.py`) still exists for compatibility but defaults
to redirecting to `master_loop.py`.

## Overview

The unified loop coordinates five major subsystems:

| Component                     | Interval          | Purpose                                  |
| ----------------------------- | ----------------- | ---------------------------------------- |
| **Streaming Data Collection** | 30s               | Incremental rsync from all remote hosts  |
| **Shadow Tournament Service** | 5min              | Lightweight evaluation (15 games/config) |
| **Training Scheduler**        | Threshold-based   | Auto-trigger when data thresholds met    |
| **Model Promoter**            | After tournaments | Auto-deploy on Elo threshold             |
| **Adaptive Curriculum**       | 1 hour            | Elo-weighted training focus              |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Unified AI Loop Daemon                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Data       │  │   Shadow     │  │   Training   │       │
│  │   Collector  │──│   Tournament │──│   Scheduler  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                │                  │                │
│         ▼                ▼                  ▼                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Model      │  │   Adaptive   │  │   Metrics    │       │
│  │   Promoter   │──│   Curriculum │──│   Exporter   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Command Line

```bash
# Start the master loop (foreground)
python scripts/master_loop.py --config config/unified_loop.yaml

# Watch live status (does not start the loop)
python scripts/master_loop.py --watch

# Check status
python scripts/master_loop.py --status

# Limit to specific configs
python scripts/master_loop.py --configs square8_2p,square19_2p

# Minimal profile (sync + health)
python scripts/master_loop.py --profile minimal

# Skip daemons (testing)
python scripts/master_loop.py --skip-daemons

# Dry run (simulate without changes)
python scripts/master_loop.py --dry-run
```

### CLI Arguments Reference

| Argument         | Description                                 | Default                    |
| ---------------- | ------------------------------------------- | -------------------------- |
| `--config`       | Path to config YAML                         | `config/unified_loop.yaml` |
| `--configs`      | Comma-separated configs (override YAML)     | All configs                |
| `--profile`      | Daemon profile (minimal/standard/full)      | `standard`                 |
| `--dry-run`      | Simulate without changes                    | False                      |
| `--skip-daemons` | Don't start/stop daemons (testing)          | False                      |
| `--watch`        | Watch live status (does not start the loop) | False                      |
| `--interval`     | Watch mode refresh interval (seconds)       | 10                         |
| `--status`       | Show current status and exit                | False                      |

### Emergency Halt (Legacy Unified Loop Only)

The legacy unified loop supports an emergency halt mechanism for safely stopping all operations:

```bash
# Trigger emergency halt
RINGRIFT_UNIFIED_LOOP_LEGACY=1 python scripts/unified_ai_loop.py --halt

# Check if halt is active (shown in --status output)
RINGRIFT_UNIFIED_LOOP_LEGACY=1 python scripts/unified_ai_loop.py --status

# Clear halt flag to allow restart
RINGRIFT_UNIFIED_LOOP_LEGACY=1 python scripts/unified_ai_loop.py --resume
```

The halt flag is stored at `data/coordination/EMERGENCY_HALT`. When set:

- Running loops will stop at the next health check interval (every 5 minutes)
- New instances will refuse to start
- The flag persists across restarts until explicitly cleared

### Systemd Service

```bash
# Install the service
sudo cp deploy/systemd/master-loop.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable master-loop
sudo systemctl start master-loop

# Check status
sudo systemctl status master-loop
journalctl -u master-loop -f
```

The `unified-ai-loop.service` unit still exists for legacy installs, but the canonical
service name is `master-loop.service`.

## Configuration

Configuration is specified in `config/unified_loop.yaml`:

```yaml
# Data ingestion from remote hosts
data_ingestion:
  poll_interval_seconds: 30 # Check every 30s (optimized for fast feedback)
  sync_method: 'incremental' # "incremental" (rsync append) or "full"
  deduplication: true # Deduplicate games by ID
  min_games_per_sync: 5 # Sync smaller batches more frequently
  use_external_sync: true # Use unified_data_sync.py with P2P fallback

# Automatic training triggers
training:
  trigger_threshold_games: 300 # Start training when this many new games (optimized)
  min_interval_seconds: 1200 # 20 min between training runs (optimized)
  max_concurrent_jobs: 1 # Only one training job at a time
  prefer_gpu_hosts: true # Schedule training on GPU hosts

# Continuous evaluation
evaluation:
  shadow_interval_seconds: 300 # 5 min between shadow evals (optimized: 3x faster parallel)
  shadow_games_per_config: 15 # Games per shadow tournament (increased for lower variance)
  full_tournament_interval_seconds: 3600 # 1 hour between full tournaments

# Automatic model promotion
promotion:
  auto_promote: true # Enable automatic promotion
  elo_threshold: 20 # Must beat current best by this many Elo
  min_games: 40 # Minimum games before promotion eligible (Wilson CI provides safety)
  significance_level: 0.05 # Statistical significance requirement
  cooldown_seconds: 900 # 15 min cooldown between promotions (optimized)

# Adaptive curriculum (Elo-weighted training)
curriculum:
  adaptive: true # Enable adaptive curriculum
  rebalance_interval_seconds: 3600
  max_weight_multiplier: 1.5 # Reduced to avoid over-rotation

# Board/player configurations
configurations:
  - board_type: 'square8'
    num_players: [2, 3, 4]
  - board_type: 'square19'
    num_players: [2, 3, 4]
  - board_type: 'hexagonal'
    num_players: [2, 3, 4]
```

> **Note:** The above values are optimized defaults. See `config/unified_loop.yaml` for the complete configuration with all advanced options.

## Related Services

The unified loop can also be run as separate services if needed:

| Service        | Script                               | Systemd Unit                |
| -------------- | ------------------------------------ | --------------------------- |
| Master Loop    | `scripts/master_loop.py`             | `master-loop.service`       |
| Data Collector | `scripts/unified_data_sync.py`       | `unified-data-sync.service` |
| Model Promoter | `scripts/model_promotion_manager.py` | `model-promoter.service`    |

> **Note:** `streaming_data_collector.py` was removed. Use `scripts/unified_data_sync.py` instead.

## Prometheus Metrics

When the Prometheus client is installed, the loop exports metrics on port 9090:

- `ringrift_games_synced_total` - Total games synced per host
- `ringrift_sync_duration_seconds` - Sync duration histogram
- `ringrift_training_runs_total` - Training runs counter
- `ringrift_elo_rating` - Current Elo ratings by model
- `ringrift_promotion_total` - Model promotions counter

## Data Flow

1. **Collection**: Every 30s, rsync pulls new games from all configured hosts
2. **Validation**: Games are validated against canonical gates before ingestion
3. **Tournament**: Every 5min, shadow tournaments evaluate model strength (15 games/config)
4. **Training**: When 300+ new games accumulated, training is auto-triggered
5. **Promotion**: If new model beats current by 20+ Elo, it's deployed (15min cooldown)
6. **Curriculum**: Training weights are adjusted based on Elo performance

## Coordinator-Only Mode

For machines that should only orchestrate the cluster without performing local compute-intensive tasks (selfplay, training, tournaments), set the `RINGRIFT_DISABLE_LOCAL_TASKS` environment variable:

```bash
# Enable coordinator-only mode
export RINGRIFT_DISABLE_LOCAL_TASKS=true

# Start the master loop
python scripts/master_loop.py --config config/unified_loop.yaml
```

### What Coordinator-Only Mode Disables

| Component              | Behavior in Coordinator Mode                      |
| ---------------------- | ------------------------------------------------- |
| **Local Selfplay**     | Skipped - games generated on cluster nodes only   |
| **Local Training**     | Skipped - training delegated to GPU nodes         |
| **Local Tournaments**  | Skipped - tournaments run on remote hosts         |
| **Data Collection**    | Active - syncs games from cluster nodes           |
| **Model Distribution** | Active - pushes models to cluster nodes           |
| **Metrics Export**     | Active - Prometheus metrics still available       |
| **Tournament Service** | Active - orchestrates remote tournament execution |

### Setting Up Persistent Coordinator Mode

Add to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
# RingRift: Run this machine as coordinator only
export RINGRIFT_DISABLE_LOCAL_TASKS=true
```

### Startup Message (Legacy Unified Loop)

When coordinator-only mode is enabled, the legacy unified loop displays:

```
[UnifiedLoop] ════════════════════════════════════════════════════════════
[UnifiedLoop] COORDINATOR-ONLY MODE (RINGRIFT_DISABLE_LOCAL_TASKS=true)
[UnifiedLoop] Local selfplay, training, and tournaments will be delegated to cluster
[UnifiedLoop] ════════════════════════════════════════════════════════════
```

### Low-Memory Machines (Legacy Unified Loop)

On machines with less than 32GB RAM, the legacy unified loop suggests coordinator-only mode if not already set. This prevents OOM kills during memory-intensive operations.

## Related Documentation

- [Training Features](TRAINING_FEATURES.md) - Training configuration options
- [Training Triggers](TRAINING_TRIGGERS.md) - 3-signal trigger system
- [Training Internals](TRAINING_INTERNALS.md) - Internal training modules
- [Curriculum Feedback](CURRICULUM_FEEDBACK.md) - Adaptive curriculum weights
- [Training Optimizations](TRAINING_OPTIMIZATIONS.md) - Pipeline optimizations
- [Coordination System](../architecture/COORDINATION_SYSTEM.md) - Task coordination and resource management
- [Distributed Selfplay](DISTRIBUTED_SELFPLAY.md) - Remote host configuration
- [Pipeline Orchestrator](../infrastructure/PIPELINE_ORCHESTRATOR.md) - ⚠️ _Archived; replaced by unified loop_

---

_Last updated: 2025-12-17_
