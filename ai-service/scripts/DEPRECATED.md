# Scripts Deprecation Manifest

**Created**: 2025-12-19
**Purpose**: Track which scripts are obsolete and superseded by the P2P orchestrator system.

## Overview

The `p2p_orchestrator.py` (27,560 lines) is the production system that handles:

- Cluster coordination and leader election
- Self-play job scheduling and distribution
- Training job management
- Model deployment and synchronization
- Health monitoring and auto-recovery
- Gauntlet/tournament execution

Many standalone scripts are now redundant. This manifest tracks what to keep vs deprecate.

---

## KEEP - Core Scripts Used by P2P Orchestrator

These scripts are actively called by `p2p_orchestrator.py`:

| Script                              | Purpose                       |
| ----------------------------------- | ----------------------------- |
| `p2p_orchestrator.py`               | Main production orchestrator  |
| `p2p_watchdog.py`                   | Watchdog for P2P daemon       |
| `p2p_model_sync.py`                 | Model synchronization         |
| `p2p_model_distribution.py`         | Model distribution            |
| `run_gpu_selfplay.py`               | GPU-accelerated self-play     |
| `run_hybrid_selfplay.py`            | Hybrid self-play mode         |
| `run_diverse_selfplay.py`           | Diverse self-play generation  |
| `run_self_play_soak.py`             | Self-play soak testing        |
| `run_nn_training_baseline.py`       | Neural network training       |
| `run_ssh_distributed_tournament.py` | Distributed tournaments       |
| `run_improvement_eval.py`           | Model improvement evaluation  |
| `export_replay_dataset.py`          | Export training data          |
| `import_gpu_selfplay_to_db.py`      | Import selfplay to database   |
| `generate_canonical_selfplay.py`    | Canonical selfplay generation |
| `auto_deploy_models.py`             | Auto model deployment         |
| `jsonl_to_npz.py`                   | Data format conversion        |
| `generate_gpu_training_data.py`     | GPU training data generation  |
| `train_nnue.py`                     | NNUE training script          |

---

## KEEP - Useful Utilities

| Script                       | Purpose             |
| ---------------------------- | ------------------- |
| `analyze_training_run.py`    | Training analysis   |
| `analyze_game_statistics.py` | Game statistics     |
| `benchmark_*.py`             | Various benchmarks  |
| `compare_models_elo.py`      | ELO comparison      |
| `debug_*.py`                 | Debugging utilities |

---

## DEPRECATE - Superseded by P2P Orchestrator

### Cluster Management (47 scripts -> 1)

All superseded by `p2p_orchestrator.py`:

| Script                        | Replacement      |
| ----------------------------- | ---------------- |
| `cluster_automation.py`       | P2P orchestrator |
| `cluster_control.py`          | P2P orchestrator |
| `cluster_manager.py`          | P2P orchestrator |
| `cluster_monitor.py`          | P2P orchestrator |
| `cluster_health_check.py`     | P2P orchestrator |
| `cluster_health_monitor.py`   | P2P orchestrator |
| `cluster_auto_recovery.py`    | P2P orchestrator |
| `cluster_file_sync.py`        | P2P orchestrator |
| `cluster_sync_coordinator.py` | P2P orchestrator |
| `cluster_worker.py`           | P2P orchestrator |
| `cluster_ssh_init.py`         | P2P orchestrator |
| `cluster_*.sh` (all)          | P2P orchestrator |

### Monitoring (24 scripts -> P2P status endpoint)

| Script                    | Replacement                  |
| ------------------------- | ---------------------------- |
| `monitor_*.sh`            | `curl localhost:8770/status` |
| `disk_monitor.py`         | P2P orchestrator (built-in)  |
| `elo_monitor.py`          | P2P orchestrator             |
| `data_quality_monitor.py` | P2P orchestrator             |

### Training Orchestration

| Script                        | Replacement               |
| ----------------------------- | ------------------------- |
| `auto_training_pipeline.py`   | P2P orchestrator          |
| `auto_training_trigger.py`    | P2P orchestrator          |
| `continuous_training_loop.py` | P2P orchestrator          |
| `curriculum_training.py`      | P2P orchestrator          |
| `training_orchestrator.py`    | P2P orchestrator (DELETE) |
| `job_scheduler.py`            | P2P orchestrator (DELETE) |

### Data Collection

| Script         | Replacement         |
| -------------- | ------------------- |
| `collect_*.sh` | P2P orchestrator    |
| `cron_*.sh`    | P2P systemd service |

---

## NEWLY CREATED - Review Status

Scripts created in this session that may conflict:

| Script                     | Status             | Action             |
| -------------------------- | ------------------ | ------------------ |
| `training_orchestrator.py` | Conflicts with P2P | DELETE             |
| `job_scheduler.py`         | Conflicts with P2P | DELETE             |
| `gpu_cluster_manager.py`   | CLI status tool    | KEEP (lightweight) |
| `setup_cluster_ssh.sh`     | SSH key helper     | KEEP               |

---

## SAFE TO DELETE

These can be deleted immediately (no dependencies):

```bash
# Duplicate monitoring scripts
rm scripts/cluster_monitor.py
rm scripts/cluster_monitor.sh
rm scripts/cluster_monitor_daemon.sh
rm scripts/cluster_monitor_unified.sh
rm scripts/cluster_monitoring.sh
rm scripts/cluster_health_monitor.sh
rm scripts/monitor_10h.sh
rm scripts/monitor_10h_enhanced.sh
rm scripts/monitor_all_jobs.sh
rm scripts/monitor_and_test.sh

# Duplicate cluster management
rm scripts/cluster_automation.py
rm scripts/cluster_control.py
rm scripts/cluster_manager.py
rm scripts/cluster_health_check.py
rm scripts/cluster_health_check.sh

# Conflicting new scripts
rm scripts/training_orchestrator.py
rm scripts/job_scheduler.py
```

---

## Migration Notes

1. **Config**: P2P orchestrator should read from `config/cluster.yaml`
2. **Alerts**: P2P orchestrator should use unified alerting (Slack/Discord)
3. **Cleanup**: P2P orchestrator should auto-kill stale processes
4. **Dashboard**: Single web dashboard at `localhost:8770/dashboard`

---

## Timeline

- **Week 1**: Create this manifest, integrate config.yaml
- **Week 2**: Add auto-cleanup to P2P
- **Week 3**: Consolidate monitoring
- **Week 4**: Delete deprecated scripts
