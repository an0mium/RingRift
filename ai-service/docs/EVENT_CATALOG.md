# Event Catalog

This document catalogs the key events in the RingRift AI-Service event system.

**Last Updated**: December 2025

## Overview

Events are defined in `app/distributed/data_events.py` as `DataEventType` enum values.
The unified event router (`app/coordination/event_router.py`) handles publishing and subscribing.

## Critical Events (Pipeline Flow)

These events form the main training pipeline:

| Event                  | Emitters                             | Subscribers                                | Effect                                    |
| ---------------------- | ------------------------------------ | ------------------------------------------ | ----------------------------------------- |
| `SELFPLAY_COMPLETE`    | P2P orchestrator, SelfplayRunner     | SelfplayScheduler, QueuePopulator          | Updates game counts, triggers sync        |
| `DATA_SYNC_COMPLETED`  | AutoSyncDaemon, P2P orchestrator     | TrainingCoordinator, PipelineOrchestrator  | Enables training on fresh data            |
| `TRAINING_COMPLETED`   | TrainingCoordinator                  | SelfplayScheduler, EvaluationDaemon        | Triggers model evaluation                 |
| `EVALUATION_COMPLETED` | GauntletRunner, AutoEvaluationDaemon | PromotionController, SelfplayScheduler     | Enables model promotion                   |
| `MODEL_PROMOTED`       | PromotionController                  | ModelDistributionDaemon, SelfplayScheduler | Distributes new model, updates curriculum |

## Selfplay Events

| Event                     | Description              | Payload Fields                                   |
| ------------------------- | ------------------------ | ------------------------------------------------ |
| `SELFPLAY_COMPLETE`       | Selfplay batch finished  | `config_key`, `games_completed`, `quality_score` |
| `SELFPLAY_TARGET_UPDATED` | Request more/fewer games | `config_key`, `new_target`, `reason`             |
| `SELFPLAY_RATE_CHANGED`   | Rate multiplier changed  | `config_key`, `multiplier`, `reason`             |

## Training Events

| Event                | Description           | Payload Fields                           |
| -------------------- | --------------------- | ---------------------------------------- |
| `TRAINING_STARTED`   | Training job started  | `config_key`, `model_version`, `node_id` |
| `TRAINING_PROGRESS`  | Epoch progress update | `epoch`, `loss`, `val_loss`, `lr`        |
| `TRAINING_COMPLETED` | Training finished     | `config_key`, `model_path`, `final_loss` |
| `TRAINING_FAILED`    | Training failed       | `config_key`, `error`, `traceback`       |

## Quality & Feedback Events

| Event                  | Description                     | Payload Fields                         |
| ---------------------- | ------------------------------- | -------------------------------------- |
| `QUALITY_DEGRADED`     | Quality dropped below threshold | `config_key`, `score`, `threshold`     |
| `ELO_VELOCITY_CHANGED` | Elo improvement rate changed    | `config_key`, `velocity`, `delta`      |
| `EXPLORATION_BOOST`    | Request higher exploration      | `config_key`, `boost_factor`, `reason` |
| `PLATEAU_DETECTED`     | Training plateau detected       | `config_key`, `epochs_stalled`         |

## Regression Events

| Event                 | Severity | Action                         |
| --------------------- | -------- | ------------------------------ |
| `REGRESSION_MINOR`    | Low      | Log warning, continue training |
| `REGRESSION_MODERATE` | Medium   | Alert, consider rollback       |
| `REGRESSION_SEVERE`   | High     | Pause training, investigate    |
| `REGRESSION_CRITICAL` | Critical | Auto-rollback recommended      |

## Node/Cluster Events

| Event            | Description          | Payload Fields                        |
| ---------------- | -------------------- | ------------------------------------- |
| `HOST_ONLINE`    | Node joined cluster  | `node_id`, `hostname`, `capabilities` |
| `HOST_OFFLINE`   | Node left cluster    | `node_id`, `reason`                   |
| `NODE_UNHEALTHY` | Health check failed  | `node_id`, `health_score`, `issues`   |
| `NODE_RECOVERED` | Node health restored | `node_id`                             |

## Subscribing to Events

```python
from app.coordination.event_router import subscribe, get_router

# Method 1: Direct subscription
def on_training_complete(event):
    payload = event.payload if hasattr(event, 'payload') else event
    config_key = payload.get('config_key')
    print(f"Training completed for {config_key}")

subscribe("TRAINING_COMPLETED", on_training_complete)

# Method 2: Via router instance
router = get_router()
if router:
    router.subscribe("SELFPLAY_COMPLETE", handler)
```

## Emitting Events

```python
from app.coordination.event_router import publish_async, get_router

# Method 1: Convenience function (async)
await publish_async("SELFPLAY_COMPLETE", {
    "config_key": "hex8_2p",
    "games_completed": 100,
    "quality_score": 0.85,
})

# Method 2: Via router instance
router = get_router()
if router:
    await router.publish_async("MODEL_PROMOTED", {
        "config_key": "hex8_2p",
        "model_path": "models/canonical_hex8_2p.pth",
    })
```

## Event System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ SelfplayRunner  │────▶│  EventRouter    │────▶│ SelfplayScheduler│
│                 │     │  (event_router) │     │                  │
│ emits:          │     │                 │     │ subscribes:      │
│ SELFPLAY_COMPLETE     │ deduplication   │     │ SELFPLAY_COMPLETE│
└─────────────────┘     │ SHA256 hash     │     │ ELO_VELOCITY_CHG │
                        └─────────────────┘     │ QUALITY_DEGRADED │
                               │                └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │ DataEventBus    │
                        │ (data_events)   │
                        │                 │
                        │ cross-process   │
                        │ SQLite-backed   │
                        └─────────────────┘
```

## Adding a New Event Type

1. Add to `DataEventType` enum in `app/distributed/data_events.py`
2. Document in this catalog
3. Add emitter(s) at appropriate locations
4. Add subscriber(s) in relevant coordinators
5. Test with `bootstrap_coordination()` smoke test

## See Also

- `app/distributed/data_events.py` - Event type definitions
- `app/coordination/event_router.py` - Unified event router
- `app/coordination/coordination_bootstrap.py` - Event wiring at startup
