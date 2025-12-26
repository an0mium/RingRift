# Coordination Training Package

Training orchestration and scheduling for the RingRift AI training pipeline.

## Overview

This package manages the training phase of the self-improvement loop:

- Job scheduling with priority queues
- ELO-based curriculum weighting
- Training slot coordination across cluster

## Modules

### `orchestrator.py` - TrainingOrchestrator

Coordinates training jobs across the cluster:

```python
from app.coordination.training import TrainingOrchestrator

orchestrator = TrainingOrchestrator()
await orchestrator.start()

# Submit training job
job_id = await orchestrator.submit_training(
    board_type="hex8",
    num_players=2,
    npz_path="data/training/hex8_2p.npz"
)
```

**Features**:

- NFS lock coordination for shared storage
- Circuit breaker protection
- Integration with `DataPipelineOrchestrator` events

### `scheduler.py` - TrainingScheduler

Priority-based job scheduling:

```python
from app.coordination.training import TrainingScheduler

scheduler = TrainingScheduler()

# Get next config to train based on priorities
next_config = scheduler.get_next_training_config(
    curriculum_weights={
        "hex8_2p": 1.5,
        "square8_2p": 1.0,
        "square19_2p": 0.5
    }
)
```

**Priority factors**:

1. Curriculum weights (from `CurriculumFeedback`)
2. Time since last training
3. Data freshness (newer data = higher priority)
4. ELO velocity (configs with faster ELO growth get priority)

## Integration

### Event Subscriptions

The orchestrator subscribes to:

- `NPZ_EXPORT_COMPLETE` - Trigger training after export
- `TRAINING_BLOCKED_BY_QUALITY` - Handle quality gate blocks
- `CLUSTER_HEALTH_CHANGED` - Adjust for node availability

### Event Emissions

Emits:

- `TRAINING_STARTED` - Training job began
- `TRAINING_COMPLETE` - Training finished successfully
- `TRAINING_FAILED` - Training job failed

## Configuration

From `config/distributed_hosts.yaml`:

```yaml
training:
  default_epochs: 50
  batch_size: 512
  learning_rate: 0.001
  early_stopping_patience: 5
```

## See Also

- `../data_pipeline_orchestrator.py` - Pipeline stage coordination
- `../training_coordinator.py` - Cluster-wide training coordination
- `../../training/train.py` - Actual training implementation
