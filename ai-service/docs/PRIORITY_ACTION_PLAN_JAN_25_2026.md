# RingRift Priority Action Plan - January 25, 2026

## Executive Summary

**Project Grade: B+ (Infrastructure A, AI Strength C+)**

The training infrastructure is mature and production-ready (95/100), but AI model strength is the critical blocker (65/100). Only 2 of 12 configs are actively improving, and none have reached the 2000 Elo target.

## Current State

### Cluster Status

- **Alive Nodes**: 19 (target: 20+)
- **Leader**: local-mac
- **Total Games**: 41,214 across 12 configs

### Model Elo Progress (Jan 25, 2026)

| Config           | Current Elo | Delta  | Status      | Games   |
| ---------------- | ----------- | ------ | ----------- | ------- |
| hexagonal_2p     | 1775        | +0     | STALLED     | 3,181   |
| square8_2p       | 1659        | +250   | STRONG      | 2,039   |
| hex8_3p          | 1590        | +0     | STALLED     | 5,776   |
| square8_3p       | 1590        | +0     | STALLED     | 3,751   |
| square19_3p      | 1581        | +369   | STRONG      | 826     |
| square8_4p       | 1507        | -113   | REGRESSED   | 6,354   |
| hex8_4p          | 1502        | -118   | REGRESSED   | 5,553   |
| hexagonal_3p     | 1400        | +0     | STALLED     | 2,966   |
| square19_2p      | 1362        | +0     | STALLED     | 755     |
| hex8_2p          | 1219        | -136   | REGRESSED   | 9,289   |
| square19_4p      | 1171        | +0     | STALLED     | 592     |
| **hexagonal_4p** | **1119**    | **+0** | **STALLED** | **132** |

### Key Issues

1. **3 configs regressed** - hex8_2p, hex8_4p, square8_4p lost significant Elo
2. **7 configs stalled** - No improvement despite training iterations
3. **hexagonal_4p critically undertrained** - Only 132 games vs 5,000+ for other configs
4. **No config at target** - All configs below 2000 Elo target

---

## Priority 0: Critical (This Week)

### P0.1: Fix hexagonal_4p Data Gap

**Impact**: Unblock 4-player large board capability
**Action**: Dispatch 2000+ selfplay games to hexagonal_4p

```bash
# On P2P leader node
curl -X POST http://localhost:8770/dispatch_selfplay \
  -d '{"config_key": "hexagonal_4p", "num_games": 500, "priority": "critical"}'
```

**Why**: hexagonal_4p has only 132 games - far below the ~500 minimum needed for bootstrap training. This config cannot improve until it has more data.

### P0.2: Investigate Regressions

**Impact**: Stop Elo decay
**Action**: Analyze why hex8_2p, hex8_4p, square8_4p regressed

Check:

1. Data quality scores for recent games
2. Training hyperparameters (learning rate too high?)
3. Opponent diversity (weak-vs-weak cycle?)

```bash
# Check quality scores
python3 -c "
from app.coordination.data_quality_scorer import DataQualityScorer
scorer = DataQualityScorer()
for config in ['hex8_2p', 'hex8_4p', 'square8_4p']:
    score = scorer.get_quality_score(config)
    print(f'{config}: {score}')
"
```

### P0.3: Verify Selfplay Allocation

**Impact**: Ensure training resources are used efficiently
**Action**: Check that selfplay scheduler is allocating to undertrained configs

```bash
# Check allocation weights
tail -f /tmp/master_loop.log | grep -i "allocation\|priority\|dispatch"

# Or via P2P
curl http://localhost:8770/allocation_weights
```

---

## Priority 1: High (Weeks 1-2)

### P1.1: Increase Gumbel Budget for Quality

**Impact**: Better training data quality
**Current**: 200 simulations (STANDARD tier)
**Target**: 400+ simulations for 2p configs

Edit `app/ai/gumbel_common.py`:

```python
GUMBEL_BUDGET_STANDARD = 400  # Was 200
GUMBEL_BUDGET_QUALITY = 1200  # Was 800
```

### P1.2: Focus Cluster on Undertrained Configs

**Impact**: Accelerate stalled configs
**Action**: Prioritize square19 and hexagonal configs

| Config       | Current Games | Target Games | Priority |
| ------------ | ------------- | ------------ | -------- |
| hexagonal_4p | 132           | 2,000        | CRITICAL |
| square19_4p  | 592           | 2,000        | HIGH     |
| square19_2p  | 755           | 2,000        | HIGH     |
| square19_3p  | 826           | 2,000        | HIGH     |

### P1.3: Add Opponent Diversity

**Impact**: Break weak-vs-weak training cycles
**Action**: Ensure selfplay uses diverse opponents

Current mix should be:

- 50% best model (quality games)
- 30% previous generations (diversity)
- 20% heuristic/random (exploration)

Check in `app/training/selfplay_runner.py`:

```python
OPPONENT_MIX = {
    "best": 0.5,
    "previous": 0.3,
    "heuristic": 0.2,
}
```

### P1.4: Monitor Daily Elo Progress

**Impact**: Visibility on improvements
**Action**: Run daily progress report

```bash
# Add to cron or master_loop
python scripts/elo_progress_report.py --format json > /tmp/daily_elo.json
```

---

## Priority 2: Medium (Weeks 2-4)

### P2.1: Evaluate v5-heavy Architecture

**Impact**: Potential +50-100 Elo for large boards
**Action**: Train hex8_2p with v5-heavy and compare

```bash
# Export with full heuristics
python scripts/export_replay_dataset.py \
  --use-discovery --board-type hex8 --num-players 2 \
  --full-heuristics --output data/training/hex8_2p_v5heavy.npz

# Train v5-heavy
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --model-version v5-heavy \
  --data-path data/training/hex8_2p_v5heavy.npz
```

### P2.2: Increase Training Epochs for Large Boards

**Impact**: Better convergence on complex boards
**Current**: 20 epochs
**Target**: 50 epochs for square19, hexagonal

### P2.3: Tune Early Stopping Patience

**Impact**: Avoid premature training termination
**Current**: 5 epochs patience
**Target**: 10 epochs for undertrained configs

---

## Priority 3: Low (Weeks 4+)

### P3.1: Large File Decomposition

- `selfplay_scheduler.py` (4,743 LOC) → 5 modules
- `feedback_loop_controller.py` (3,800 LOC) → 4 modules

### P3.2: Custom health_check() for P2P Loops

- 6 P2P loops need custom health metrics
- Base class provides sufficient coverage for now

### P3.3: Circuit Breaker TTL Decay

- Add time-based recovery for circuit breakers
- Prevents permanent node exclusion

---

## Success Metrics

### Week 1 Targets

| Metric              | Current | Target |
| ------------------- | ------- | ------ |
| hexagonal_4p games  | 132     | 500+   |
| Configs improving   | 2/12    | 4/12   |
| Cluster nodes alive | 19      | 22+    |

### Week 2 Targets

| Metric             | Current | Target |
| ------------------ | ------- | ------ |
| hexagonal_4p games | 500     | 1,500+ |
| Configs >1600 Elo  | 3/12    | 5/12   |
| Average 2p Elo     | ~1400   | 1550+  |

### Week 4 Targets

| Metric            | Current | Target |
| ----------------- | ------- | ------ |
| Configs >1600 Elo | 3/12    | 8/12   |
| Configs at target | 0/12    | 2/12   |
| Total games       | 41K     | 80K+   |

---

## Immediate Next Steps

1. **Now**: Dispatch hexagonal_4p selfplay jobs (P0.1)
2. **Now**: Check allocation weights to verify scheduler (P0.3)
3. **Today**: Investigate regression root causes (P0.2)
4. **This week**: Increase Gumbel budgets (P1.1)
5. **This week**: Focus cluster on undertrained configs (P1.2)

---

## Files to Monitor

| File                                     | Purpose             |
| ---------------------------------------- | ------------------- |
| `scripts/master_loop.py`                 | Cluster automation  |
| `app/coordination/selfplay_scheduler.py` | Allocation logic    |
| `data/elo_progress.db`                   | Elo tracking        |
| `scripts/elo_progress_report.py`         | Progress visibility |
| `logs/p2p_local.log`                     | P2P cluster health  |

---

## Infrastructure Health (Reference)

The infrastructure is mature and working:

| Component         | Grade       | Status                         |
| ----------------- | ----------- | ------------------------------ |
| P2P Network       | A- (94/100) | 19 nodes, quorum OK            |
| Training Pipeline | A (95/100)  | All feedback loops wired       |
| Code Quality      | A (95/100)  | 99.5% test coverage            |
| Data Sync         | A- (92/100) | Multi-transport, auto-recovery |
| Async Safety      | A (95/100)  | All SQLite wrapped             |

**The blocker is NOT infrastructure - it's training data quantity and quality for AI models.**
