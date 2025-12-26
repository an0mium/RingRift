# ADR-004: Quality Gate and Feedback Loop Architecture

**Status**: Accepted
**Date**: December 2025
**Author**: RingRift AI Team

## Context

Training on low-quality data produces weak models. The training pipeline needed quality control that:

1. Prevents training on bad data
2. Automatically regenerates data when quality is low
3. Adapts training parameters based on evaluation results
4. Rolls back models when regressions are detected

## Decision

Implement a multi-layered **quality gate and feedback loop** system:

### Layer 1: Quality Scoring

`UnifiedQualityScorer` (`app/quality/unified_quality.py`) scores each game:

- Game length (too short = random, too long = stuck)
- Move diversity (low = repetitive play)
- Outcome balance (all one-sided = poor matchup)
- ELO weight (higher opponent ELO = higher weight)

### Layer 2: Quality Gate

Before training, `DataPipelineOrchestrator` checks quality:

```python
if quality_score < 0.6:
    emit(TRAINING_BLOCKED_BY_QUALITY, {config_key, reason, quality_score})
    return  # Don't proceed to training
```

### Layer 3: Data Regeneration

When `TRAINING_BLOCKED_BY_QUALITY` is emitted:

1. `SelfplayScheduler` boosts exploration (1.5x) for that config
2. `QueuePopulator` adds 3 priority selfplay work items (150 games)
3. Fresh data generation begins immediately

### Layer 4: Gauntlet Feedback

After training, `GauntletFeedbackController` evaluates and adjusts:

| Win Rate             | Action                                      |
| -------------------- | ------------------------------------------- |
| vs RANDOM < 70%      | Trigger extra selfplay, extend epochs       |
| vs HEURISTIC > 80%   | Reduce temperature, raise quality threshold |
| ELO plateau detected | Advance curriculum stage                    |
| Regression detected  | Emit REGRESSION_CRITICAL → auto-rollback    |

### Layer 5: Auto-Rollback

`AutoRollbackHandler` subscribes to `REGRESSION_CRITICAL`:

- CRITICAL severity: Immediate auto-rollback to previous model
- SEVERE severity: Log pending rollback (manual approval required)

## Consequences

### Positive

- Self-healing pipeline: low quality triggers regeneration
- Adaptive training: parameters adjust based on results
- Protection against regressions: bad models automatically reverted
- Complete feedback loop: evaluation results feed back to selfplay

### Negative

- Complexity: Multiple event types and handlers to maintain
- Latency: Quality checks add time before training starts
- Threshold tuning: Quality thresholds need periodic adjustment

## Implementation Notes

- Quality gate enabled by default (`auto_trigger=True`)
- Minimum quality score: 0.6 (configurable)
- Phase 7 fixed quality gate deadlock by wiring QueuePopulator to TRAINING_BLOCKED_BY_QUALITY
- Phase 7 wired AutoRollbackHandler in DaemonManager for daemon-level rollback

## Event Flow

```
SELFPLAY_COMPLETE → Quality Check
  ├─ Pass → TRAINING_STARTED → TRAINING_COMPLETED → EVALUATION_COMPLETE
  │                                                        ↓
  │                                            GauntletFeedbackController
  │                                                        ↓
  │                            ┌─────────────────────────────────────────┐
  │                            │ HYPERPARAMETER_UPDATED (adjust LR/batch)│
  │                            │ CURRICULUM_ADVANCED (harder opponents)  │
  │                            │ ADAPTIVE_PARAMS_CHANGED (exploration)   │
  │                            │ REGRESSION_CRITICAL → AutoRollback      │
  │                            └─────────────────────────────────────────┘
  │
  └─ Fail → TRAINING_BLOCKED_BY_QUALITY
                   ↓
       ┌───────────────────────────────────┐
       │ SelfplayScheduler: boost priority │
       │ QueuePopulator: add selfplay items│
       └───────────────────────────────────┘
                   ↓
           Fresh data generated → Retry quality check
```

## Related ADRs

- ADR-001: Event-Driven Architecture (quality events)
- ADR-003: PFSP Opponent Selection (quality affects opponent selection)
