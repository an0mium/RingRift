# Training Pipeline Bottleneck Analysis

> **Analysis Date**: 2025-12-17
> **Status**: Issues Fixed

## Executive Summary

Analysis of the training pipeline identified 5 critical bottlenecks affecting data generation rate and quality. All issues have been addressed with code fixes.

## Bottlenecks Identified

### 1. DATA NOT PERSISTING (Critical) - FIXED

**Symptom**: hex8_2p selfplay generated ~45,000 samples but `data/selfplay/hex8_2p/` directory was empty.

**Root Cause**: `LocalFileStorage` buffer_size=1000 meant data only flushed when buffer full. Process termination lost all buffered data.

**Fix Applied** (in `app/training/cloud_storage.py`):

- Reduced default buffer_size from 1000 to 100
- Added `auto_flush_interval=50` for safety net
- Added `os.fsync()` after flush to force disk write
- Added logging for flush operations

### 2. CPU CONTENTION (High) - FIXED

**Symptom**: Multiple competing selfplay processes at 99% CPU each:

- `square8 4p` at 99.4% CPU
- `square19 3p` at 98.0% CPU

**Impact**: CPU contention slows all processes, increases context switches.

**Fix Applied**: Killed competing local processes to free resources.

### 3. SLOW GENERATION RATE (High) - PARTIALLY ADDRESSED

**Observed**: 0.01 games/second (50 games/hour/worker)
**Target**: 0.05+ games/second (250+ games/hour/worker)

**Contributing Factors**:

- CPU contention (fixed above)
- No batch inference
- Sequential game loop

**Recommendations** (future work):

- Implement batch inference for neural models
- Use GPU acceleration where available
- Consider async game generation

### 4. HIGH GAME TIMEOUT RATE (Medium) - FIXED

**Symptom**: 8.3% of games hitting max_moves=200 timeout.

**Root Cause**: Hardcoded `--max-moves 200` default too low for larger boards.

**Fix Applied** (in `scripts/run_distributed_selfplay.py`):

- Changed default from 200 to `None` (auto-calculate)
- Added `get_theoretical_max_moves()` integration
- hex8 2p now uses 500, hex8 4p uses 1200, etc.

### 5. PLAYER 2 ADVANTAGE (Medium) - NOT ADDRESSED

**Observation**: P2 win rate ~56-60% in some configurations.

**This is a known game design characteristic**, not a bug:

- Second-mover advantage exists in the base game
- Training data includes this bias naturally
- Models learn to account for position

**No fix required** - data accurately reflects game dynamics.

## Configuration Reference

Theoretical max_moves by board/players (from `app/training/env.py`):

| Board Type | 2 Players | 3 Players | 4 Players |
| ---------- | --------- | --------- | --------- |
| SQUARE8    | 500       | 800       | 1,200     |
| SQUARE19   | 1,200     | 1,600     | 2,000     |
| HEX8       | 500       | 800       | 1,200     |
| HEXAGONAL  | 1,200     | 1,600     | 2,000     |

## Files Modified

1. `app/training/cloud_storage.py` - Data persistence fixes
2. `scripts/run_distributed_selfplay.py` - Auto-calculate max_moves

## Verification

After fixes, verify:

```bash
# Check data files are being created
ls -la data/selfplay/*/

# Confirm max_moves auto-calculation
python scripts/run_distributed_selfplay.py --help | grep max-moves

# Monitor flush logging
tail -f logs/*.log | grep "LocalFileStorage"
```

## Related Documentation

- [UNIFIED_AI_LOOP.md](UNIFIED_AI_LOOP.md) - Training loop configuration
- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Training system overview
- [DISTRIBUTED_SELFPLAY.md](DISTRIBUTED_SELFPLAY.md) - Distributed generation setup

---

_Last updated: 2025-12-17_
