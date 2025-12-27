# Train.py Decomposition Plan

**Date:** December 27, 2025
**Target:** Reduce `app/training/train.py` from 5,471 LOC to ~600 LOC

---

## Executive Summary

The `train_model()` function (lines 547-5385, ~4,838 LOC) contains multiple distinct logical layers that should be decomposed into 7 separate modules. The current monolithic structure makes it difficult to maintain, test, and reuse individual components.

---

## Proposed Module Structure

| Module                  | Est. LOC | Purpose                                          |
| ----------------------- | -------- | ------------------------------------------------ |
| `train_distributed.py`  | ~450     | DDP setup, device selection, distributed metrics |
| `train_init.py`         | ~500     | Validation, model initialization, board params   |
| `train_data.py`         | ~750     | Data loading, streaming, samplers                |
| `train_optimization.py` | ~650     | Optimizer, LR scheduling, feedback handlers      |
| `train_enhancements.py` | ~650     | Curriculum, mining, augmentation                 |
| `train_checkpoints.py`  | ~550     | Save/load, resume, checkpoint averaging          |
| `train_loop_core.py`    | ~1,600   | Core training/validation loop, epoch iteration   |
| `train.py` (facade)     | ~600     | Orchestration only                               |

---

## Part 1: train_distributed.py (~450 LOC)

**Source lines:** 724-744, 1020-1041, 2184-2191, 3174-3187

**Functions to create:**

```python
def setup_device(distributed: bool, local_rank: int) -> torch.device:
    """Device selection with MPS/CUDA/CPU fallback."""

def wrap_model_distributed(model: nn.Module, device: torch.device, distributed: bool) -> nn.Module:
    """DDP wrapping if distributed."""

def setup_heartbeat_monitoring(heartbeat_file: str, interval: float) -> HeartbeatMonitor | None:
    """Heartbeat setup for fault tolerance."""

def initialize_distributed_metrics(distributed: bool) -> DistributedMetrics | None:
    """Initialize metrics for distributed training."""
```

**Dependencies:**

- `app.training.distributed` - setup_distributed, get_world_size, is_main_process
- `app.training.fault_tolerance` - HeartbeatMonitor

---

## Part 2: train_init.py (~500 LOC)

**Source lines:** 446-545, 747-923, 1220-1322, 1910-2099

**Functions to create:**

```python
def validate_training_compatibility(model, dataset, config) -> None:
    """Validate model-dataset compatibility."""

def check_data_freshness(config, num_players, ...) -> FreshnessResult | None:
    """Check data freshness with event emission."""

def validate_npz_structure_and_content(data_paths, config) -> bool:
    """Validate NPZ structure and content."""

def select_model_and_board_params(config, board_type, model_version) -> tuple:
    """Returns (use_hex, policy_size, board_size, ...)"""

def initialize_model(config, board_size, policy_size, ...) -> nn.Module:
    """Create model based on version (v2, v3, v4, v5, v5-heavy)."""
```

**Dependencies:**

- `app.training.train_validation`
- `app.coordination.training_freshness`
- `app.ai.neural_net` - model classes

---

## Part 3: train_data.py (~750 LOC)

**Source lines:** 2327-3128

**Classes/Functions to create:**

```python
class DataLoaderFactory:
    def create_train_val_loaders(data_path, config, device) -> tuple:
        """Create train/val loaders. Returns (train_loader, val_loader, train_size, val_size)"""

def prepare_data_sources(data_path, data_dir, discover_synced) -> list[str]:
    """Collect data paths for streaming using DataCatalog."""

def should_use_streaming(total_size: int) -> bool:
    """Auto-detection threshold for streaming mode."""

def create_weighted_sampler(dataset, sampling_weights) -> WeightedRandomSampler:
    """Position sampling for non-streaming mode."""
```

**Dependencies:**

- `app.training.data_loader` - StreamingDataLoader
- `app.distributed.data_catalog` - DataCatalog
- `app.training.datasets` - RingRiftDataset

---

## Part 4: train_optimization.py (~650 LOC)

**Source lines:** 2230-2283, 3462-3577, 4532-4564

**Classes/Functions to create:**

```python
class OptimizerFactory:
    def create_optimizer(model_params, config) -> Optimizer:
        """Create optimizer with freeze_policy support."""

class SchedulerFactory:
    def create_epoch_scheduler(...) -> Scheduler | None:
        """Create LR scheduler (step, cosine, cosine-warm-restarts)."""

def apply_gauntlet_feedback(optimizer, pending_updates) -> bool:
    """Apply learning rate, quality threshold updates from gauntlet."""

def apply_improvement_optimizer_feedback(optimizer, config_label) -> bool:
    """Check improvement optimizer every 5 epochs."""
```

**Dependencies:**

- `app.training.schedulers`
- `app.training.training_enhancements` - EarlyStopping, EvaluationFeedbackHandler

---

## Part 5: train_enhancements.py (~650 LOC)

**Source lines:** 1063-1218, 3755-3805, 4046-4131

**Classes/Functions to create:**

```python
class EnhancementsFactory:
    def create_hot_buffer(...) -> HotDataBuffer | None
    def create_quality_bridge(...) -> QualityBridge | None
    def create_hard_example_miner(...) -> HardExampleMiner | None
    def create_training_facade(...) -> TrainingEnhancementsFacade | None
    def create_quality_trainer(...) -> QualityWeightedTrainer | None

def apply_data_augmentation(features, policy_targets, mgr) -> tuple:
    """Apply augmentation with graceful error handling."""

def compute_enhanced_loss(policy_pred, policy_targets, value_pred, value_targets, ...) -> dict:
    """Combine quality weighting, outcome weighting, hard mining."""

def apply_hard_example_mining(facade, miner, per_sample_losses) -> torch.Tensor:
    """Unified mining interface."""
```

**Dependencies:**

- `app.training.hot_data_buffer`
- `app.training.quality_bridge`
- `app.training.enhancements.hard_example_mining`
- `app.training.enhancements.training_facade`

---

## Part 6: train_checkpoints.py (~550 LOC)

**Source lines:** 2286-2324, 4992-5200, 5230-5360

**Classes/Functions to create:**

```python
class CheckpointManager:
    def load_resume_checkpoint(...) -> tuple[int, dict]:
        """Load checkpoint for resume. Returns (start_epoch, state_dict)."""

    def save_checkpoint_at_interval(epoch, model, optimizer, ...) -> None:
        """Periodic checkpoint saving."""

    def save_best_checkpoint(...) -> None:
        """Save best model checkpoint."""

    def cleanup_old_checkpoints(checkpoint_dir, keep_n=5) -> None:
        """Remove old checkpoints."""

class CheckpointAveragingManager:
    def collect_final_checkpoints(...) -> dict:
        """Collect and average final checkpoints."""

def setup_emergency_checkpoint_handler(...) -> GracefulShutdownHandler:
    """Emergency checkpoint on SIGTERM."""
```

**Dependencies:**

- `app.training.checkpoint_unified`
- `app.training.training_enhancements` - CheckpointAverager

---

## Part 7: train_loop_core.py (~1,600 LOC)

**Source lines:** 3378-4800, 5000-5200

**Functions to create:**

```python
def run_training_epoch(epoch, model, train_loader, optimizer, device, config, ...) -> dict:
    """Core training loop for one epoch.

    Returns: {avg_loss, grad_norm, anomalies_detected}
    """

def run_validation_epoch(epoch, model, val_loader, device, config, ...) -> dict:
    """Validation loop for one epoch.

    Returns: {avg_loss, policy_accuracy, calibration_metrics}
    """

def save_epoch_checkpoint(checkpoint_dir, model, optimizer, epoch, loss, ...) -> str:
    """Save checkpoint at epoch boundary."""
```

**Dependencies:**

- `train_optimization.py` - optimizer step
- `train_enhancements.py` - augmentation, mining
- `train_distributed.py` - distributed synchronization
- `train_checkpoints.py` - checkpoint saving

---

## Dependency Graph

```
train_model() [Main Orchestrator in train.py]
├── train_init.py
│   ├── validate_training_compatibility()
│   ├── check_data_freshness()
│   ├── validate_npz_structure_and_content()
│   └── initialize_model()
│
├── train_distributed.py (no dependencies on other train_* modules)
│   ├── setup_device()
│   ├── wrap_model_distributed()
│   └── setup_heartbeat_monitoring()
│
├── train_data.py
│   ├── DataLoaderFactory.create_train_val_loaders()
│   └── prepare_data_sources()
│
├── train_optimization.py
│   ├── OptimizerFactory.create_optimizer()
│   ├── SchedulerFactory.create_*_scheduler()
│   └── apply_gauntlet_feedback()
│
├── train_enhancements.py
│   ├── EnhancementsFactory.create_*()
│   ├── apply_data_augmentation()
│   └── compute_enhanced_loss()
│
├── train_checkpoints.py
│   ├── CheckpointManager.*()
│   └── setup_emergency_checkpoint_handler()
│
└── train_loop_core.py (depends on all above)
    ├── run_training_epoch()
    └── run_validation_epoch()
```

---

## Extraction Order (Recommended)

1. **train_distributed.py** - No dependencies on other new modules
2. **train_init.py** - Minimal dependencies
3. **train_data.py** - Depends on train_distributed (device awareness)
4. **train_optimization.py** - Independent
5. **train_enhancements.py** - Depends on train_optimization
6. **train_checkpoints.py** - Depends on train_enhancements
7. **train_loop_core.py** - Final, depends on all others

---

## Migration Checklist

- [ ] Extract train_distributed.py (~450 LOC)
- [ ] Extract train_init.py (~500 LOC)
- [ ] Extract train_data.py (~750 LOC)
- [ ] Extract train_optimization.py (~650 LOC)
- [ ] Extract train_enhancements.py (~650 LOC)
- [ ] Extract train_checkpoints.py (~550 LOC)
- [ ] Extract train_loop_core.py (~1,600 LOC)
- [ ] Update train.py to be facade only (~600 LOC)
- [ ] Add unit tests for each module
- [ ] Run integration tests
- [ ] Update documentation

---

## Expected Outcome

| Metric            | Before    | After    |
| ----------------- | --------- | -------- |
| train.py LOC      | 5,471     | ~600     |
| Largest function  | 4,838 LOC | ~300 LOC |
| Number of modules | 1         | 8        |
| Testability       | Low       | High     |
| Reusability       | Low       | High     |

---

## Benefits

1. **Testability**: Each module can be unit tested independently
2. **Reusability**: Training loop components usable in other contexts
3. **Maintainability**: Clear separation of concerns
4. **Extensibility**: Easy to add enhancements without touching core loop
5. **Debugging**: Smaller functions easier to debug and profile

---

_Plan created: December 27, 2025_
