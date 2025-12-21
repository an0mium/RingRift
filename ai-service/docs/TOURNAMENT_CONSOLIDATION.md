# Tournament Implementation Analysis

This document provides a comprehensive analysis of all tournament implementations in the RingRift codebase, with recommendations for consolidation that preserve all functionality.

## Implementation Matrix

### Core Implementations (Keep Exactly As-Is)

| Script                          | Purpose                         | Unique Features                                                                    | Dependencies                 |
| ------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------- |
| `run_distributed_tournament.py` | Tier difficulty ladder (D1-D10) | Deterministic Elo replay, Wilson intervals, training data export, device detection | Core - no dependencies       |
| `run_model_elo_tournament.py`   | Model vs model evaluation       | Model discovery, persistent Elo database, composite participants, event bus        | Core - critical for training |
| `run_eval_tournaments.py`       | Eval pool snapshots             | Start from mid/late-game states, per-scenario statistics                           | Unique niche                 |
| `run_profile_tournament.py`     | Heuristic weight testing        | Weight profile comparison, position-based analysis                                 | Specialized tool             |
| `unified_loop/tournament.py`    | Unified loop integration        | ShadowTournamentService, host load balancing, event integration                    | Modern replacement           |

### Library Modules (Keep As Stable APIs)

| Module                            | Purpose                    | Usage                           |
| --------------------------------- | -------------------------- | ------------------------------- |
| `app/tournament/runner.py`        | High-level match execution | Reusable API for Python imports |
| `app/tournament/orchestrator.py`  | Orchestration API          | Evaluation workflows            |
| `app/training/tournament.py`      | Low-level match execution  | Core match logic                |
| `app/training/auto_tournament.py` | Model version pipeline     | Champion promotion              |

### Specialized Variants (Keep for Scaling)

| Script                              | Purpose                  | Relationship                                            |
| ----------------------------------- | ------------------------ | ------------------------------------------------------- |
| `run_ssh_distributed_tournament.py` | Multi-host orchestration | Wraps `run_distributed_tournament.py` for cluster scale |
| `run_diverse_tournaments.py`        | All configurations       | Orchestrates runs across board/player combinations      |

### Redundant/Archive Candidates

| Script                         | Status                  | Replacement                                    |
| ------------------------------ | ----------------------- | ---------------------------------------------- |
| `shadow_tournament_service.py` | Deprecated in docstring | `unified_loop/tournament.py`                   |
| `auto_elo_tournament.py`       | Simple daemon wrapper   | Can fold into monitoring layer                 |
| `run_tournament.py`            | Router/wrapper          | Direct calls to implementations may be simpler |

## Dependency Graph

```
run_distributed_tournament.py (CORE)
├── LadderTierConfig
├── AIConfig, AIType
├── GameEngine
└── Training data export

run_ssh_distributed_tournament.py
└── Wraps: run_distributed_tournament.py

run_model_elo_tournament.py (CORE)
├── Model discovery (.pth files)
├── Elo database (SQLite)
├── Event bus (elo_updated)
└── Composite participants

run_eval_tournaments.py
├── EvalPoolConfig
└── Snapshot-based evaluation

unified_loop/tournament.py
├── ShadowTournamentService class
├── Unified loop events
└── Improvement optimizer
```

## Preserved Functionality Checklist

All features below MUST be preserved in any consolidation:

- [ ] Deterministic Elo replay (run_distributed_tournament.py)
- [ ] Wilson confidence intervals (run_distributed_tournament.py)
- [ ] Training data export (run_distributed_tournament.py)
- [ ] MPS/CUDA/CPU device detection (run_distributed_tournament.py)
- [ ] Model discovery + persistent leaderboard (run_model_elo_tournament.py)
- [ ] Composite participant tracking (run_model_elo_tournament.py)
- [ ] Event bus integration (run_model_elo_tournament.py, unified_loop/tournament.py)
- [ ] Eval pool snapshots (run_eval_tournaments.py)
- [ ] Heuristic weight comparison (run_profile_tournament.py)
- [ ] Multi-host SSH orchestration (run_ssh_distributed_tournament.py)
- [ ] Multiplayer support with filler AIs (run_distributed_tournament.py)
- [ ] ShadowTournamentService (unified_loop/tournament.py)

## Current Usage

### Active on Cluster

- `run_distributed_tournament.py` - Tier tournaments on lambda-h100, lambda-gh200-\*
- `auto_elo_tournament.py` - Continuous daemon on lambda-gh200-e
- `unified_loop/tournament.py` - Part of unified AI loop

### Active in Training Pipeline

- `run_model_elo_tournament.py` - Called by p2p_orchestrator.py
- `app/tournament/orchestrator.py` - Used by training loop

## Recommendations

### Phase 1: Document (Current)

- ✓ Create this analysis document
- ✓ Map all dependencies and unique features
- No deprecations until thorough testing

### Phase 2: Test Coverage

- Add integration tests for each tournament type
- Verify all features work correctly
- Create feature parity checklist

### Phase 3: Gradual Consolidation (Future)

- Migrate `shadow_tournament_service.py` users to `unified_loop/tournament.py`
- Consider if `run_tournament.py` router adds value vs. complexity
- Archive `auto_elo_tournament.py` if monitoring layer covers functionality

## File Locations

**CLI Scripts:**

- `/scripts/run_distributed_tournament.py` (1710 lines)
- `/scripts/run_model_elo_tournament.py` (2546 lines)
- `/scripts/run_ssh_distributed_tournament.py` (652 lines)
- `/scripts/run_eval_tournaments.py` (527 lines)
- `/scripts/run_diverse_tournaments.py` (697 lines)
- `/scripts/run_profile_tournament.py` (406 lines)
- `/scripts/run_tournament.py` (528 lines) - Router
- `/scripts/auto_elo_tournament.py` - Daemon
- `/scripts/shadow_tournament_service.py` - Deprecated

**Libraries:**

- `/app/tournament/runner.py`
- `/app/tournament/orchestrator.py`
- `/app/training/tournament.py`
- `/app/training/auto_tournament.py`

**Unified Loop:**

- `/scripts/unified_loop/tournament.py`

## Conclusion

The tournament system has **excellent separation of concerns** with clear unique functionality for each implementation. The main consolidation opportunity is reducing router/wrapper layers rather than merging core functionality. No functionality should be lost by maintaining the current core implementations.
