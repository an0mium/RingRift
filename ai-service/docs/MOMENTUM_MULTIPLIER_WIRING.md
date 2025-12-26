# Momentum Multiplier Wiring Documentation

## Overview

This document describes how the selfplay rate multiplier from `FeedbackAccelerator` is wired to `SelfplayScheduler` to implement Elo momentum → Selfplay rate coupling.

## Architecture

```
FeedbackAccelerator.get_selfplay_multiplier(config_key)
    ↓
SelfplayScheduler._get_momentum_multipliers()
    ↓
priority.momentum_multiplier (stored in ConfigPriority)
    ↓
SelfplayScheduler._compute_priority_score()
    ↓
Selfplay allocation (more/less games based on momentum)
```

## Implementation Details

### 1. FeedbackAccelerator.get_selfplay_multiplier()

**Location:** `/ai-service/app/training/feedback_accelerator.py:787`

**Purpose:** Returns a multiplier (0.5 - 1.5) based on Elo momentum state.

**Multiplier Values:**

- **ACCELERATING**: 1.5x (capitalize on positive momentum)
- **IMPROVING**: 1.25x (boost for continued improvement)
- **STABLE**: 1.0x (normal rate)
- **PLATEAU**: 1.1x (slight boost to try to break plateau)
- **REGRESSING**: 0.75x (reduce noise, focus on quality)

**Code:**

```python
def get_selfplay_multiplier(self, config_key: str) -> float:
    momentum = self._configs.get(config_key)
    if not momentum:
        return 1.0

    multiplier_map = {
        MomentumState.ACCELERATING: 1.5,
        MomentumState.IMPROVING: 1.25,
        MomentumState.STABLE: 1.0,
        MomentumState.PLATEAU: 1.1,
        MomentumState.REGRESSING: 0.75,
    }

    base_multiplier = multiplier_map.get(momentum.momentum_state, 1.0)

    # Additional boost for consecutive improvements
    if momentum.consecutive_improvements >= 3:
        base_multiplier = min(base_multiplier * 1.1, 1.5)

    # Limit during consecutive plateaus
    if momentum.consecutive_plateaus >= 3:
        base_multiplier = max(base_multiplier * 0.9, 0.5)

    return base_multiplier
```

### 2. SelfplayScheduler.\_get_momentum_multipliers()

**Location:** `/ai-service/app/coordination/selfplay_scheduler.py:631`

**Purpose:** Fetches momentum multipliers from FeedbackAccelerator for all configs.

**Code:**

```python
def _get_momentum_multipliers(self) -> dict[str, float]:
    result: dict[str, float] = {}

    try:
        from app.training.feedback_accelerator import get_feedback_accelerator

        accelerator = get_feedback_accelerator()

        for config_key in ALL_CONFIGS:
            multiplier = accelerator.get_selfplay_multiplier(config_key)
            if multiplier != 1.0:  # Only log non-default values
                result[config_key] = multiplier
                logger.debug(
                    f"[SelfplayScheduler] Momentum multiplier for {config_key}: "
                    f"{multiplier:.2f}x"
                )
            else:
                result[config_key] = multiplier

    except ImportError:
        logger.debug("[SelfplayScheduler] feedback_accelerator not available")
    except Exception as e:
        logger.debug(f"[SelfplayScheduler] Error getting momentum multipliers: {e}")

    return result
```

### 3. SelfplayScheduler.\_update_priorities()

**Location:** `/ai-service/app/coordination/selfplay_scheduler.py:307`

**Purpose:** Updates all config priorities, including momentum multipliers.

**Code (excerpt):**

```python
async def _update_priorities(self) -> None:
    # ... get other priority factors ...

    # Get momentum multipliers (Phase 19)
    momentum_data = self._get_momentum_multipliers()

    # Update each config
    for config_key, priority in self._config_priorities.items():
        # ... update other factors ...

        # Update momentum multiplier (Phase 19)
        if config_key in momentum_data:
            priority.momentum_multiplier = momentum_data[config_key]

        # Compute priority score
        priority.priority_score = self._compute_priority_score(priority)
```

### 4. SelfplayScheduler.\_compute_priority_score()

**Location:** `/ai-service/app/coordination/selfplay_scheduler.py:401`

**Purpose:** Computes final priority score with momentum multiplier applied.

**Code (excerpt):**

```python
def _compute_priority_score(self, priority: ConfigPriority) -> float:
    # Base factors
    staleness = priority.staleness_factor * STALENESS_WEIGHT
    velocity = priority.velocity_factor * ELO_VELOCITY_WEIGHT
    # ... other factors ...

    # Combine factors
    score = staleness + velocity + training + exploration + ...

    # Apply exploration boost as multiplier
    score *= priority.exploration_boost

    # Phase 19: Apply momentum multiplier from FeedbackAccelerator
    score_before_momentum = score
    score *= priority.momentum_multiplier

    # Log when momentum multiplier significantly affects priority (>10% change)
    if abs(priority.momentum_multiplier - 1.0) > 0.1:
        logger.info(
            f"[SelfplayScheduler] Momentum multiplier applied to {priority.config_key}: "
            f"{priority.momentum_multiplier:.2f}x (score: {score_before_momentum:.3f} → {score:.3f})"
        )

    # Apply priority override from config
    score *= override_multiplier

    return score
```

### 5. ConfigPriority Data Structure

**Location:** `/ai-service/app/coordination/selfplay_scheduler.py:130`

**Field:** `momentum_multiplier: float = 1.0`

**Purpose:** Stores the momentum multiplier for each config.

## Logging

### Debug-Level Logging (Line 657)

When momentum multipliers are retrieved:

```
[SelfplayScheduler] Momentum multiplier for hex8_2p: 1.50x
```

### Info-Level Logging (Line 451)

When momentum significantly affects priority score (>10% change):

```
[SelfplayScheduler] Momentum multiplier applied to hex8_2p: 1.50x (score: 0.450 → 0.675)
```

## Testing the Wiring

To verify the wiring is working:

1. **Check logs during priority updates:**

   ```bash
   tail -f logs/selfplay_scheduler.log | grep "Momentum multiplier"
   ```

2. **Inspect config priorities programmatically:**

   ```python
   from app.coordination.selfplay_scheduler import get_selfplay_scheduler

   scheduler = get_selfplay_scheduler()
   priority = scheduler.get_config_priority("hex8_2p")
   print(f"Momentum multiplier: {priority.momentum_multiplier:.2f}x")
   print(f"Priority score: {priority.priority_score:.3f}")
   ```

3. **Monitor FeedbackAccelerator momentum states:**

   ```python
   from app.training.feedback_accelerator import get_feedback_accelerator

   accelerator = get_feedback_accelerator()
   momentum = accelerator.get_config_momentum("hex8_2p")
   print(f"Momentum state: {momentum.momentum_state.value}")
   print(f"Selfplay multiplier: {accelerator.get_selfplay_multiplier('hex8_2p'):.2f}x")
   ```

## Expected Behavior

### Accelerating Models (1.5x multiplier)

- Strong Elo improvement (+25 Elo over last 5 updates)
- Gets 50% more selfplay games to capitalize on momentum
- Example: Base allocation 500 games → 750 games

### Improving Models (1.25x multiplier)

- Moderate Elo improvement (+12 Elo over last 5 updates)
- Gets 25% more selfplay games
- Example: Base allocation 500 games → 625 games

### Stable Models (1.0x multiplier)

- No significant change in Elo
- Normal selfplay allocation
- Example: Base allocation 500 games → 500 games

### Plateau Models (1.1x multiplier)

- Stuck at same Elo (<5 Elo change)
- Slight boost to try breaking plateau
- Example: Base allocation 500 games → 550 games

### Regressing Models (0.75x multiplier)

- Declining Elo (-12 Elo over last 5 updates)
- Reduced selfplay to focus on quality
- Example: Base allocation 500 games → 375 games

## Maintenance Notes

- The wiring is **fully implemented** as of December 2025
- All components are in production use
- No additional changes needed - the system is operational
- The momentum multiplier integrates with other priority factors (curriculum weights, improvement boosts, data deficit)

## Related Documentation

- `app/training/feedback_accelerator.py` - Momentum tracking and multiplier calculation
- `app/coordination/selfplay_scheduler.py` - Priority-based selfplay scheduling
- `app/coordination/README.md` - Coordination infrastructure overview
