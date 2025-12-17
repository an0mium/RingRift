# Developer Guide

This guide covers the key patterns and canonical sources used throughout the RingRift AI service codebase. Following these patterns ensures consistency and makes the codebase easier to maintain.

## Canonical Sources

The codebase uses a "single source of truth" pattern for configuration, logging, and resource management. Always use these canonical modules instead of inline implementations.

### Configuration

**Canonical Source:** `app/config/unified_config.py`

```python
from app.config.unified_config import (
    get_config,           # Get the singleton config
    get_training_threshold,  # Training game threshold
    get_elo_db_path,      # Elo database path
    get_min_elo_improvement,  # Promotion threshold
)

# Access config values
config = get_config()
threshold = config.training.trigger_threshold_games

# Or use convenience functions
threshold = get_training_threshold()
```

**Do NOT:**

- Hardcode values like `MIN_GAMES_FOR_TRAINING = 1000`
- Create duplicate config classes
- Load config from environment variables directly (use the config module)

### Logging

**Canonical Source:** `app/core/logging_config.py`

```python
from app.core.logging_config import setup_logging, get_logger

# In script entry points
logger = setup_logging("my_script", log_dir="logs")

# In modules
logger = get_logger(__name__)
```

**Do NOT:**

- Use `logging.basicConfig()` directly
- Create custom log formatters (use `format_style` parameter)

### Resource Checking

**Canonical Source:** `app/utils/resource_guard.py`

```python
from app.utils.resource_guard import (
    check_disk_space,
    check_memory,
    can_proceed,
    wait_for_resources,
    LIMITS,
)

# Before disk operations
if not check_disk_space(required_gb=2.0):
    return

# Check all resources
if not can_proceed():
    wait_for_resources(timeout=300)
```

**Do NOT:**

- Use `shutil.disk_usage()` directly
- Use `psutil.virtual_memory()` directly
- Use `psutil.cpu_percent()` directly
- Hardcode utilization thresholds

### Host Configuration

**Canonical Source:** `config/distributed_hosts.yaml`

All cluster host information should be loaded from this file, not hardcoded:

```python
import yaml
from pathlib import Path

def _load_hosts():
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        return []  # Graceful fallback

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    return list(config.get("hosts", {}).keys())
```

**Do NOT:**

- Hardcode IP addresses
- Hardcode hostnames
- Commit cluster-specific configuration

## Directory Structure

```
ai-service/
├── app/                    # Core application code
│   ├── ai/                 # AI implementations (random, heuristic, minimax, mcts, descent)
│   ├── config/             # Configuration modules (unified_config.py)
│   ├── core/               # Core utilities (logging_config.py)
│   ├── distributed/        # P2P and cluster coordination
│   ├── training/           # Training pipeline
│   └── utils/              # Shared utilities (resource_guard.py)
├── scripts/                # CLI tools and daemons
│   ├── unified_ai_loop.py  # Main training orchestrator
│   ├── unified_loop/       # Unified loop submodules
│   └── ...
├── config/                 # Configuration files (gitignored if sensitive)
│   ├── distributed_hosts.yaml       # Cluster hosts (gitignored)
│   ├── distributed_hosts.yaml.example  # Template
│   └── unified_loop.yaml            # Loop configuration
├── models/                 # Neural network checkpoints (gitignored)
├── data/                   # Training data and databases (gitignored)
└── docs/                   # Documentation
```

## Key Patterns

### Graceful Fallback

All config loading should work even if config files are missing:

```python
def _load_config():
    config_path = Path("config/distributed_hosts.yaml")
    if not config_path.exists():
        logger.warning("Config not found, using defaults")
        return {}  # Return empty, don't crash

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Config load failed: {e}")
        return {}
```

### Import Guards

For optional dependencies, use import guards:

```python
try:
    from app.core.logging_config import setup_logging
    HAS_LOGGING_CONFIG = True
except ImportError:
    HAS_LOGGING_CONFIG = False
    setup_logging = None

# Later in code
if HAS_LOGGING_CONFIG:
    logger = setup_logging("my_script")
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
```

### Resource Limits

The codebase enforces 80% maximum utilization to prevent system overload:

| Resource | Warning | Critical |
| -------- | ------- | -------- |
| CPU      | 70%     | 80%      |
| GPU      | 70%     | 80%      |
| Memory   | 80%     | 90%      |
| Disk     | 75%     | 80%      |

These are defined in `app/utils/resource_guard.py`.

## AI Factory Pattern

AI creation should always use the factory:

```python
from app.ai.factory import create_ai, get_canonical_profile

# Create AI by difficulty
ai = create_ai(difficulty=5, board_type="square8")

# Get difficulty profile
profile = get_canonical_profile(difficulty=5)
print(f"AI type: {profile.ai_type}, simulations: {profile.simulations}")
```

**Do NOT:**

- Instantiate AI classes directly (unless testing)
- Hardcode AI parameters

## Training Pipeline

The training system uses a unified loop with event-driven components:

1. **Data Collection** — Streams games from cluster nodes
2. **Training Trigger** — Fires when `trigger_threshold_games` is reached
3. **Model Training** — Neural network training with checkpointing
4. **Evaluation** — Shadow tournaments to measure Elo
5. **Promotion** — Auto-promote models that improve by `elo_threshold`

All thresholds come from `unified_config.py`.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_resource_guard.py -v
```

## Common Tasks

### Adding a New Script

1. Use `setup_logging()` for logging
2. Load hosts from `distributed_hosts.yaml`
3. Use `get_training_threshold()` for training thresholds
4. Use `resource_guard` for resource checks
5. Add graceful fallbacks for missing config

### Modifying Configuration

1. Add new fields to `app/config/unified_config.py`
2. Update `config/unified_loop.yaml.example` if needed
3. Add convenience getter if commonly used
4. Document the new field

### Adding Training Features

1. Add config options to `TrainingConfig` in `unified_config.py`
2. Implement in `app/training/`
3. Integrate with `unified_ai_loop.py`
4. Add tests

## Migration Status

### Scripts Using Unified Logging (20+ migrated)

The following scripts have been migrated to use `app.core.logging_config.setup_logging()`:

**Core Orchestration:**

- `unified_ai_loop.py` (via submodules)
- `unified_data_sync.py`
- `unified_promotion_daemon.py`
- `p2p_orchestrator.py`

**Training & Evaluation:**

- `train_nnue.py`
- `run_gauntlet.py`
- `run_tournament.py`
- `hex8_training_pipeline.py`
- `auto_training_pipeline.py`

**Cluster Management:**

- `cluster_manager.py`
- `cluster_automation.py`
- `cluster_sync_coordinator.py`
- `cluster_worker.py`
- `node_resilience.py`

**Data Processing:**

- `aggregate_jsonl_to_db.py`
- `aria2_data_sync.py`
- `auto_export_training_data.py`
- `model_sync_aria2.py`

**Infrastructure:**

- `health_alerting.py`
- `vast_autoscaler.py`
- `vast_keepalive.py`
- `vast_lifecycle.py`
- `run_distributed_selfplay.py`
- `auto_model_promotion.py`

### Scripts Still Using basicConfig

Some utility scripts still use `logging.basicConfig()`. These can be migrated incrementally using the same pattern:

```python
# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("script_name", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
```

### Config Integration

The `scripts/unified_loop/config.py` module now includes integration functions:

```python
from scripts.unified_loop.config import (
    sync_with_unified_config,      # Sync defaults from app.config.unified_config
    get_canonical_training_threshold,  # Get threshold from canonical source
)

# Ensure config uses canonical values
config = UnifiedLoopConfig.from_yaml(config_path)
config = sync_with_unified_config(config)
```
