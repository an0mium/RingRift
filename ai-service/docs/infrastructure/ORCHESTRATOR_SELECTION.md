# Orchestrator Selection Guide

## Quick Reference

| Use Case              | Script                       | When to Use                                          |
| --------------------- | ---------------------------- | ---------------------------------------------------- |
| **Full AI Loop**      | `master_loop.py`             | Production deployment, continuous improvement        |
| **P2P Cluster**       | `p2p_orchestrator.py`        | Multi-node P2P coordination, distributed selfplay    |
| **Slurm HPC**         | `master_loop.py`             | Stable Slurm cluster with shared filesystem          |
| **Sync Operations**   | `unified_data_sync.py`       | CLI entry point for data/model sync                  |
| **Multi-Board Train** | `master_loop.py`             | Multi-board training via unified loop config         |
| **Elo Tournament**    | `run_model_elo_tournament.py`| Scheduled Elo evaluation and leaderboard updates     |
| **Model Promotion**   | `model_promotion_manager.py` | Manual promotion, Elo testing, rollback              |

---

## Script Details

### master_loop.py (Canonical - Use This)

**Purpose**: Complete AI self-improvement feedback loop

**Features**:

- Selfplay generation with curriculum learning
- Automatic data sync to training pool
- Training trigger on data thresholds
- Model evaluation via tournaments
- Promotion with regression testing
- Health monitoring with Prometheus metrics
- Emergency halt mechanism

**CLI**:

```bash
# Start (foreground)
python scripts/master_loop.py --config config/unified_loop.yaml

# Watch status (does not start the loop)
python scripts/master_loop.py --watch

# Check status
python scripts/master_loop.py --status

# Legacy emergency controls (unified_ai_loop only)
RINGRIFT_UNIFIED_LOOP_LEGACY=1 python scripts/unified_ai_loop.py --halt
RINGRIFT_UNIFIED_LOOP_LEGACY=1 python scripts/unified_ai_loop.py --resume
```

**Config**: `config/unified_loop.yaml`

---

### p2p_orchestrator.py

**Purpose**: Distributed P2P cluster coordination and selfplay orchestration

**Features**:

- Self-healing compute cluster with leader election
- Peer discovery and resource monitoring
- Auto-starts selfplay/training jobs across nodes
- Vast.ai and Runpod instance integration
- Supports all board types: square8, hex8, square19, hexagonal
- Keepalive and unretire management for cloud instances

**CLI**:

```bash
# Start as node in P2P cluster (replace with your node ID and coordinator URLs)
PYTHONPATH=. venv/bin/python scripts/p2p_orchestrator.py --node-id gpu-node-1 --port 8770 --peers <coordinator_urls>

# View cluster status
curl http://localhost:8770/status
```

**When to use**:

- Production distributed training across 3+ nodes
- Vast.ai or Runpod GPU instances
- Self-healing cluster with automatic recovery

---

### Slurm Backend (Optional)

**Purpose**: Run the unified AI loop on a stable Slurm-managed HPC cluster.

**When to use**:

- You have a shared filesystem mounted on all nodes.
- You want queue-based scheduling, accounting, and fair-share.

**Notes**:

- Use `master_loop.py` with `execution_backend: "slurm"` (or `auto` + `slurm.enabled: true`).
- See `docs/infrastructure/SLURM_BACKEND_DESIGN.md` for details.

---

### Sync Facade (Module)

**Purpose**: Unified programmatic entry point for data/model sync operations.

**When to use**:

- You want a single API that routes to AutoSyncDaemon / SyncRouter / DistributedSyncCoordinator.
- You need sync operations from code without wiring a full orchestrator.

**Usage**:

```python
from app.coordination.sync_facade import sync

result = await sync("all")
```

**Legacy Note**: `app/distributed/sync_orchestrator.py` remains available but is pending deprecation.

---

### ~~pipeline_orchestrator.py~~ (DEPRECATED)

> ⚠️ **Deprecated**: This script has been removed. Use `master_loop.py` instead.
> **Archive location**: Removed (see git history)

**Original Purpose**: CI/CD pipeline for automated testing and validation

The functionality has been integrated into `master_loop.py` with:

- Regression detection via the unified loop (see `run_strength_regression_gate.py` for a standalone gate)
- Automated model quality validation
- Shadow tournament validation every 5 minutes

**Migration**: Replace `pipeline_orchestrator.py` calls with `master_loop.py`:

```bash
# Old (deprecated)
# python pipeline_orchestrator.py (removed) --pr 123 --validate

# New (use master loop with validation)
python scripts/master_loop.py --config config/unified_loop.yaml
```

---

### model_promotion_manager.py

**Purpose**: Direct model promotion with Elo validation

**Features**:

- Elo threshold validation
- Statistical significance testing
- Cluster sync via SSH
- Rollback on regression
- Daemon mode for continuous promotion

**CLI**:

```bash
# Single promotion check
python scripts/model_promotion_manager.py --candidate new_model.pt

# Continuous daemon mode
python scripts/model_promotion_manager.py --daemon

# Rollback
python scripts/model_promotion_manager.py --rollback
```

**When to use instead of master_loop.py**:

- Manual promotion control
- Testing specific models
- Debugging promotion issues

---

## Deprecated Scripts (Do Not Use)

| Script                             | Replacement                                 |
| ---------------------------------- | ------------------------------------------- |
| `continuous_improvement_daemon.py` | `master_loop.py`                            |
| `improvement_cycle_manager.py`     | `master_loop.py`                            |
| `auto_promote_best_models.py`      | Archived (use `model_promotion_manager.py`) |
| `auto_promote_weights.py`          | Archived (use `model_promotion_manager.py`) |

---

## Decision Tree

```
Do you need continuous AI improvement?
├─ Yes → master_loop.py
│        ├─ Have a stable Slurm cluster? → Use Slurm backend
│        └─ Need distributed across 3+ nodes? → Also use p2p_orchestrator.py
│
├─ No, just need model promotion
│  └─ model_promotion_manager.py
│
├─ No, need CI/CD validation
│  └─ run_strength_regression_gate.py (CI gate)
│
├─ No, need multi-board/multi-player training
│  └─ master_loop.py (multi-board config)
│
└─ No, need distributed P2P selfplay
   └─ p2p_orchestrator.py
```

---

## Configuration Files

| Script                        | Config File                                     |
| ----------------------------- | ----------------------------------------------- |
| `master_loop.py`              | `config/unified_loop.yaml`                      |
| `p2p_orchestrator.py`         | `config/unified_loop.yaml` (p2p section)        |
| `model_promotion_manager.py`  | Uses CLI args or `config/promotion_daemon.yaml` |
| `run_model_elo_tournament.py` | Uses CLI args                                   |
