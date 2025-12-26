# ADR-003: PFSP (Prioritized Fictitious Self-Play) Opponent Selection

**Status**: Accepted
**Date**: December 2025
**Author**: RingRift AI Team

## Context

Self-play generates training data by having the current model play against opponents. The quality of training data depends heavily on opponent selection:

- **Too weak opponents**: Easy wins, no learning signal
- **Too strong opponents**: Constant losses, no learning signal
- **Optimal opponents**: ~50% win rate provides maximum learning signal

AlphaZero and similar systems use PFSP to select opponents that maximize learning.

## Decision

Implement **PFSPOpponentSelector** (`app/training/pfsp_opponent_selector.py`) with:

### Opponent Prioritization

Priority calculated as:

```python
priority = 1.0 / (abs(win_rate - 0.5) + 0.1)
```

Opponents with ~50% win rate against current model get highest priority.

### Opponent Pool Management

1. **Bootstrap**: On startup, discover canonical models in `models/` directory
2. **Dynamic Updates**: Subscribe to `MODEL_PROMOTED` events to add new models
3. **Result Recording**: `record_pfsp_result()` called after each game to update win rates

### Integration with Selfplay

All selfplay runners (`HeuristicSelfplayRunner`, `GumbelMCTSSelfplayRunner`, `GNNSelfplayRunner`) integrate PFSP via:

```python
# In _init_pfsp()
self._pfsp_selector = get_pfsp_selector()
wire_pfsp_events()  # Subscribe to MODEL_PROMOTED
bootstrap_pfsp_opponents()  # Pre-populate with canonical models

# In run_game()
opponent_context = self._get_pfsp_context()  # Select opponent
# ... play game ...
self.record_pfsp_result(...)  # Update win rate
```

### Cold-Start Solution

Phase 7 added `bootstrap_pfsp_opponents()` to solve the cold-start problem:

- Before: No opponents until MODEL_PROMOTED events (new models only)
- After: Existing canonical models discovered and registered on startup

## Consequences

### Positive

- Maximum learning signal per game
- Automatic difficulty progression
- Historical model versions maintained for diverse training

### Negative

- Memory overhead for opponent pool history
- Requires model versioning/registry
- Win rate estimates noisy for new opponents

## Implementation Notes

- Per-config opponent pools: `hex8_2p` has different opponents than `square8_4p`
- ELO-based opponent weighting available as alternative strategy
- `OPPONENT_SELECTED` events emitted for analysis/debugging
- Default win rate for new opponents: 0.5 (neutral prior)

## Related ADRs

- ADR-001: Event-Driven Architecture (PFSP subscribes to MODEL_PROMOTED)
- ADR-002: Daemon Lifecycle Management (PFSP initialized during selfplay daemon startup)
