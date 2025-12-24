# Configuration System Guide

**Created**: December 2025
**Purpose**: Document the unified configuration architecture

## Overview

The RingRift AI service uses a hierarchical configuration system with `UnifiedConfig` as the **single source of truth**. This guide documents the configuration classes and their relationships.

## Primary Configuration

### UnifiedConfig (`unified_config.py`)

The master configuration class that consolidates all subsystems:

```python
from app.config.unified_config import get_config, UnifiedConfig

config = get_config()  # Loads from config/unified_loop.yaml

# Access any subsystem
threshold = config.training.trigger_threshold_games
shadow_games = config.evaluation.shadow_games_per_config
```

#### Sub-configurations (28 total)

| Config Class                   | Purpose                    | Key Settings                        |
| ------------------------------ | -------------------------- | ----------------------------------- |
| `DataIngestionConfig`          | Data sync from cluster     | poll_interval, sync_method          |
| `TrainingConfig`               | Training triggers          | trigger_threshold_games, batch_size |
| `EvaluationConfig`             | Model evaluation           | shadow_games, elo_k_factor          |
| `PromotionConfig`              | Auto-promotion             | elo_threshold, min_games            |
| `CurriculumConfig`             | Adaptive curriculum        | adaptive, rebalance_interval        |
| `SafeguardsConfig`             | Process limits             | max_python_processes, max_selfplay  |
| `SafetyConfig`                 | Training safety            | overfit_threshold, max_failures     |
| `ClusterConfig`                | Cluster orchestration      | target_selfplay_rate, health_check  |
| `SSHConfig`                    | SSH execution              | max_retries, timeout                |
| `SlurmConfig`                  | HPC execution              | partition, resources                |
| `SelfplayDefaults`             | Selfplay settings          | mcts_simulations, temperature       |
| `TournamentConfig`             | Tournament settings        | games_per_matchup, k_factor         |
| `DistributedConfig`            | P2P/distributed            | p2p_port, gossip_port               |
| `IntegratedEnhancementsConfig` | Training enhancements      | auxiliary_tasks, batch_scheduling   |
| `PBTConfig`                    | Population-based training  | population_size, exploit_interval   |
| `NASConfig`                    | Neural architecture search | strategy, population_size           |
| `PERConfig`                    | Prioritized replay         | alpha, beta, capacity               |
| `FeedbackConfig`               | Pipeline feedback          | plateau_detection thresholds        |
| `P2PClusterConfig`             | P2P integration            | base_url, model_sync                |
| `ModelPruningConfig`           | Model lifecycle            | threshold, archive_models           |
| `StorageConfig`                | Provider-specific paths    | lambda\_, vast, mac paths           |
| `DataLoadingConfig`            | Data pipeline              | batch_size, num_workers             |
| `QualityConfig`                | Quality scoring            | weights, thresholds                 |
| `HealthConfig`                 | Health monitoring          | check_interval, auto_restart        |
| `ReplayBufferConfig`           | Experience replay          | priority_alpha, capacity            |
| `PlateauDetectionConfig`       | Training plateaus          | elo_threshold, lookback             |
| `AlertingConfig`               | Alerting thresholds        | sync_failure, elo_drop              |
| `RegressionConfig`             | Regression testing         | hard_block, timeout                 |

## Specialized Configurations

These files provide domain-specific configuration:

### Keep (Specialized Scope)

| File                  | Purpose                   | Status                         |
| --------------------- | ------------------------- | ------------------------------ |
| `selfplay_config.py`  | Per-run selfplay settings | **KEEP** - runtime dataclass   |
| `logging_config.py`   | Logging setup             | **KEEP** - framework config    |
| `thresholds.py`       | Canonical constants       | **KEEP** - imported by unified |
| `config_validator.py` | Validation utilities      | **KEEP** - infrastructure      |

### Consolidate or Deprecate

| File                     | Purpose                 | Recommendation                         |
| ------------------------ | ----------------------- | -------------------------------------- |
| `training_config.py`     | Legacy training config  | Use `UnifiedConfig.training`           |
| `coordinator_config.py`  | Coordinator settings    | Use `UnifiedConfig` via `get_config()` |
| `tier_eval_config.py`    | Tier evaluation         | Migrate to `UnifiedConfig.evaluation`  |
| `notification_config.py` | Alert thresholds        | Migrate to `UnifiedConfig.alerting`    |
| `diverse_ai_config.py`   | AI diversity settings   | Migrate to `UnifiedConfig.selfplay`    |
| `high_tier_config.py`    | High-tier settings      | Migrate to `UnifiedConfig.training`    |
| `ladder_config.py`       | Ladder/ranking settings | Migrate to `UnifiedConfig.tournament`  |

## Usage Patterns

### Loading Configuration

```python
# Standard usage - singleton pattern
from app.config.unified_config import get_config

config = get_config()  # Loads once, returns cached instance

# Force reload (e.g., after config file change)
config = get_config(force_reload=True)

# Custom config path
config = get_config(config_path="config/custom.yaml")
```

### Environment Variable Overrides

```bash
# Override config file path
export RINGRIFT_CONFIG_PATH=config/dev.yaml

# Override training threshold
export RINGRIFT_TRAINING_THRESHOLD=1000

# Override Elo database
export RINGRIFT_ELO_DB=data/custom_elo.db
```

### Convenience Functions

```python
from app.config.unified_config import (
    get_training_threshold,      # Training trigger threshold
    get_elo_db_path,             # Path to Elo database
    get_min_elo_improvement,     # Promotion Elo threshold
    get_target_selfplay_rate,    # Target games/hour
    get_storage_paths,           # Provider-specific paths
    detect_storage_provider,     # Auto-detect provider
)

# Example: Get threshold
threshold = get_training_threshold()  # Returns int

# Example: Get paths for current provider
paths = get_storage_paths()  # Auto-detects lambda/vast/mac
selfplay_dir = paths.selfplay_games
```

### Per-Run Configuration

For runtime configuration that varies per execution, use dedicated dataclasses:

```python
from app.training.selfplay_config import SelfplayConfig

# Per-run selfplay configuration
config = SelfplayConfig(
    board_type="hex8",
    num_players=2,
    mcts_simulations=200,
    temperature=0.5,
)
```

## Configuration File

The YAML configuration file (`config/unified_loop.yaml`):

```yaml
version: '1.2'
execution_backend: 'auto'

training:
  trigger_threshold_games: 500
  min_interval_seconds: 1200
  max_concurrent_jobs: 1
  prefer_gpu_hosts: true

evaluation:
  shadow_interval_seconds: 900
  shadow_games_per_config: 15
  baseline_models:
    - random
    - heuristic
    - mcts_100
    - mcts_500

promotion:
  auto_promote: true
  elo_threshold: 25
  min_games: 50

# ... (see config/unified_loop.yaml for full schema)
```

## Validation

Configuration is validated on load:

```python
config = get_config()
errors = config.validate()  # Returns list of errors

# Or raise on invalid
config.validate_or_raise()  # Raises ValueError if invalid
```

## Migration Guide

### From Hardcoded Constants

```python
# Before (scattered hardcoded values)
MIN_GAMES = 500
ELO_THRESHOLD = 25
K_FACTOR = 32

# After (unified config)
from app.config.unified_config import get_config

config = get_config()
min_games = config.training.trigger_threshold_games
elo_threshold = config.promotion.elo_threshold
k_factor = config.evaluation.elo_k_factor
```

### From Legacy Config Classes

```python
# Before (legacy class)
from app.training.training_config import TrainingConfig
config = TrainingConfig(batch_size=512)

# After (unified config)
from app.config.unified_config import get_config
config = get_config()
batch_size = config.data_loading.batch_size
```

### From Environment Variables

```python
# Before (direct env access)
threshold = int(os.environ.get("MIN_GAMES", 500))

# After (unified with fallback)
from app.config.unified_config import get_training_threshold
threshold = get_training_threshold()  # Handles env override internally
```

## Storage Provider Detection

The config system auto-detects storage provider:

```python
from app.config.unified_config import (
    detect_storage_provider,
    get_storage_paths,
    get_selfplay_dir,
    should_use_nfs_sync,
)

# Auto-detect provider (lambda, vast, mac, default)
provider = detect_storage_provider()

# Get provider-specific paths
paths = get_storage_paths(provider)
print(paths.selfplay_games)   # /lambda/nfs/RingRift/selfplay
print(paths.model_checkpoints)  # /lambda/nfs/RingRift/models

# Check if NFS sync should be used
if should_use_nfs_sync():
    print("Using NFS - no rsync needed")
```

## Factory Functions

Create pre-configured managers:

```python
from app.config.unified_config import create_training_manager

# Create IntegratedTrainingManager from config
manager = create_training_manager(
    model=my_model,
    board_type="square8",
)

if manager:
    manager.initialize_all()
    batch_size = manager.get_batch_size()
```

## Best Practices

1. **Always use `get_config()`** - Never instantiate `UnifiedConfig` directly
2. **Use convenience functions** - `get_training_threshold()` instead of `get_config().training.trigger_threshold_games`
3. **Environment overrides** - Use env vars for temporary changes, YAML for permanent
4. **Validate on startup** - Call `config.validate_or_raise()` in application entry points
5. **Don't duplicate settings** - If a value exists in UnifiedConfig, use it

## Related Documentation

- `CONSOLIDATION_ROADMAP.md` - Overall consolidation plan
- `ORCHESTRATOR_GUIDE.md` - Training orchestrator consolidation
- `COORDINATOR_GUIDE.md` - Coordinator/manager hierarchy
- `config/unified_loop.yaml` - Default configuration file
