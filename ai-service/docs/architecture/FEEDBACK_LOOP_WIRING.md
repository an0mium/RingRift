# Feedback Loop Wiring Guide

**Last Updated**: January 3, 2026 (Sprint 15)

This document describes the 5 primary feedback loops in the RingRift training infrastructure, their event wiring, and how they affect training behavior.

## Overview

The training infrastructure uses closed-loop feedback to dynamically adjust training parameters based on observed performance. All 5 feedback loops are fully operational (verified Sprint 15).

## Feedback Loop Summary

| Loop                       | Trigger Event           | Effect                         | Controller               |
| -------------------------- | ----------------------- | ------------------------------ | ------------------------ |
| Quality → Training         | `QUALITY_ASSESSMENT`    | Adjusts training intensity     | `FeedbackLoopController` |
| Elo Velocity → Selfplay    | `ELO_UPDATED`           | Reallocates selfplay resources | `SelfplayScheduler`      |
| Regression → Curriculum    | `REGRESSION_DETECTED`   | Rolls back curriculum weights  | `CurriculumIntegration`  |
| Loss Anomaly → Exploration | `LOSS_ANOMALY_DETECTED` | Boosts exploration temperature | `FeedbackLoopController` |
| Curriculum → Weights       | `CURRICULUM_REBALANCED` | Updates selfplay allocation    | `SelfplayScheduler`      |

## Loop 1: Quality → Training Intensity

**Purpose**: Adjust training intensity based on data quality scores.

**Event Flow**:

```
QualityMonitorDaemon ──QUALITY_ASSESSMENT──▶ FeedbackLoopController
                                                      │
                                                      ▼
                                           TrainingTriggerDaemon
                                           (adjusts training trigger threshold)
```

**Emitter**: `app/coordination/quality_monitor_daemon.py`

- Emits `QUALITY_ASSESSMENT` with quality_score, sample_count, config_key

**Subscriber**: `app/coordination/feedback_loop_controller.py:_on_quality_assessment()`

- Computes quality-weighted training intensity
- High quality (>0.8): Normal training intensity
- Low quality (<0.5): Reduced intensity or skip

**Effect**:

- Configs with low-quality selfplay data get reduced training epochs
- Prevents wasting compute on poor data

## Loop 2: Elo Velocity → Selfplay Allocation

**Purpose**: Allocate more selfplay resources to configs with high Elo improvement rate.

**Event Flow**:

```
EloService ──ELO_UPDATED──▶ SelfplayScheduler
                                   │
                                   ▼
                           _elo_velocity dict updated
                           Priority weights recalculated
```

**Emitter**: `app/training/elo_service.py`

- Emits `ELO_UPDATED` with config_key, new_elo, delta, velocity

**Subscriber**: `app/coordination/selfplay_scheduler.py:_on_elo_updated()`

- Tracks Elo velocity per config in `_elo_velocity` dict
- High velocity configs get higher priority in allocation

**Effect**:

- Fast-improving configs receive more selfplay games
- Stalled configs gradually reduce allocation
- Weight: `RINGRIFT_ELO_VELOCITY_WEIGHT` (default 0.10)

## Loop 3: Regression → Curriculum Rollback

**Purpose**: Roll back curriculum weights when model performance regresses.

**Event Flow**:

```
RegressionDetector ──REGRESSION_DETECTED──▶ FeedbackLoopController
                                                      │
                                                      ▼
                                           CurriculumIntegration
                                           (restores previous weights)
                                                      │
                                                      ▼
                                           ──CURRICULUM_EMERGENCY_UPDATE──▶
```

**Emitter**: `app/coordination/regression_detector.py`

- Emits `REGRESSION_DETECTED` with config_key, severity, elo_drop

**Subscriber**: `app/coordination/feedback_loop_controller.py:_on_regression_detected()`

- Severity-based response:
  - MILD: Reduce training intensity 10%
  - MODERATE: Boost exploration, reduce intensity 25%
  - SEVERE: Rollback curriculum weights, pause training

**Subscriber**: `app/coordination/curriculum_integration.py:_on_regression_detected()`

- Reverts curriculum weights to last-known-good snapshot
- Emits `CURRICULUM_EMERGENCY_UPDATE`

**Effect**:

- Prevents training on bad data from degrading model
- Automatic recovery without manual intervention

## Loop 4: Loss Anomaly → Exploration Boost

**Purpose**: Increase exploration when training loss becomes anomalous.

**Event Flow**:

```
TrainingCoordinator ──LOSS_ANOMALY_DETECTED──▶ FeedbackLoopController
                                                        │
                                                        ▼
                                               ──EXPLORATION_BOOST──▶
                                                        │
                                                        ▼
                                               SelfplayRunner
                                               (increases temperature)
```

**Emitter**: `app/coordination/training_coordinator.py`

- Emits `LOSS_ANOMALY_DETECTED` with config_key, loss_value, expected_range

**Subscriber**: `app/coordination/feedback_loop_controller.py:_on_loss_anomaly()`

- Computes boost magnitude based on anomaly severity
- Emits `EXPLORATION_BOOST` event

**Subscriber**: `app/training/selfplay_runner.py:_on_exploration_boost()`

- Temporarily increases temperature parameter
- Exploration decays adaptively based on Elo improvement

**Effect**:

- Training plateaus trigger automatic exploration
- Helps escape local minima in policy space

## Loop 5: Curriculum → Selfplay Weights

**Purpose**: Apply curriculum weight changes to selfplay allocation.

**Event Flow**:

```
CurriculumIntegration ──CURRICULUM_REBALANCED──▶ SelfplayScheduler
                                                        │
                                                        ▼
                                               _curriculum_weights updated
                                               Allocation recalculated
```

**Emitter**: `app/coordination/curriculum_integration.py`

- Emits `CURRICULUM_REBALANCED` with weights dict, trigger_reason

**Subscriber**: `app/coordination/selfplay_scheduler.py:_on_curriculum_rebalanced()`

- Updates internal `_curriculum_weights` dict
- Recalculates priority for each config
- Weight: `RINGRIFT_CURRICULUM_WEIGHT` (default 0.20)

**Effect**:

- Curriculum changes propagate to selfplay immediately
- No lag between curriculum decision and resource allocation

## Verification

To verify all feedback loops are wired correctly:

```bash
# Check emitters exist for all loop events
for event in QUALITY_ASSESSMENT ELO_UPDATED REGRESSION_DETECTED LOSS_ANOMALY_DETECTED CURRICULUM_REBALANCED; do
  echo "=== $event ==="
  grep -rn "emit.*$event\|emit_event.*$event" app/coordination/ scripts/p2p/ | head -3
done

# Check subscribers exist
for event in QUALITY_ASSESSMENT ELO_UPDATED REGRESSION_DETECTED LOSS_ANOMALY_DETECTED CURRICULUM_REBALANCED; do
  echo "=== $event ==="
  grep -rn "subscribe.*$event\|_on_.*$event" app/coordination/ | head -3
done
```

## Environment Variables

| Variable                       | Default | Description                     |
| ------------------------------ | ------- | ------------------------------- |
| `RINGRIFT_QUALITY_WEIGHT`      | 0.15    | Weight of quality in allocation |
| `RINGRIFT_ELO_VELOCITY_WEIGHT` | 0.10    | Weight of Elo velocity          |
| `RINGRIFT_CURRICULUM_WEIGHT`   | 0.20    | Weight of curriculum            |
| `RINGRIFT_STALENESS_WEIGHT`    | 0.15    | Weight of data staleness        |
| `RINGRIFT_DIVERSITY_WEIGHT`    | 0.10    | Weight of config diversity      |

## See Also

- [EVENT_SUBSCRIPTION_MATRIX.md](EVENT_SUBSCRIPTION_MATRIX.md) - Full event wiring matrix
- [COORDINATION_SYSTEM.md](COORDINATION_SYSTEM.md) - Overall coordination architecture
- `app/coordination/feedback_loop_controller.py` - Central feedback controller
- `app/coordination/selfplay_scheduler.py` - Selfplay allocation
