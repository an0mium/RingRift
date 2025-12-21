# Hexagonal Board Parity Bug

**Status:** OPEN - Blocking canonical hex training
**Priority:** CRITICAL
**Discovered:** December 2025

---

## Problem Statement

Hexagonal board games exhibit phase divergence between Python and TypeScript implementations after `no_territory_action` moves. This causes parity validation failures and blocks training on hexagonal board variants.

## Symptoms

- Phase divergence at move ~k=989 in hexagonal games
- Python reports `territory_processing` phase
- TypeScript reports `forced_elimination` phase
- Occurs after `NO_TERRITORY_ACTION` bookkeeping move
- Square board (8x8) games work correctly

## Root Cause Analysis

### Phase Transition Logic

The divergence occurs in the phase state machine transition after territory processing:

1. Player completes territory phase with `NO_TERRITORY_ACTION`
2. Python transitions to `territory_processing` for next player
3. TypeScript transitions to `forced_elimination` check first
4. The `_should_enter_forced_elimination()` gate differs between implementations

### Suspected Locations

| File                             | Lines   | Concern                                              |
| -------------------------------- | ------- | ---------------------------------------------------- |
| `app/rules/phase_machine.py`     | 100-150 | `_did_process_territory_region()` may have edge case |
| `app/rules/fsm.py`               | 300-400 | FSM orchestration for hex boards                     |
| `app/ai/gpu_canonical_export.py` | 400-500 | GPU export may skip bookkeeping moves                |

### Hex-Specific Factors

1. **Larger board**: 469 spaces (hex) vs 64 spaces (square8) = longer games
2. **Territory geometry**: Hexagonal adjacency affects FE eligibility checks
3. **Ring distribution**: Different ring counts per player (96 vs 44)

## Reproduction Steps

```bash
# 1. Generate hex selfplay with parity validation
python scripts/generate_gumbel_selfplay.py \
  --board hexagonal \
  --num-players 2 \
  --games 10 \
  --validate-parity

# 2. Check for parity failures
ls -la parity_failures/canonical_hexagonal_*.json

# 3. Analyze specific failure
python scripts/analyze_parity_failures.py \
  parity_failures/canonical_hexagonal__*.json \
  --verbose
```

## Diagnostic Data

Parity failure bundles are stored in:

```
parity_failures/canonical_hexagonal__<uuid>__k<move>.parity_failure.json
```

Bundle contents:

- `game_id`: UUID of failed game
- `move_k`: Move index where divergence occurred
- `python_state`: Python game state snapshot
- `typescript_state`: TypeScript game state snapshot
- `last_move`: Move that triggered divergence
- `move_history`: Full move history up to failure point

## Fix Strategy

### Step 1: Reproduce Locally

Use the state bundle diff tool first, then optional ad-hoc diffing via debug_utils:

```bash
# Emit state bundles during parity check
python scripts/check_ts_python_replay_parity.py \
  --db ai-service/data/games/canonical_hexagonal.db \
  --emit-state-bundles-dir parity_bundles

# Diff the first divergent bundle
python scripts/diff_state_bundle.py \
  --bundle parity_bundles/<bundle>.state_bundle.json \
  --k <diverged_at>
```

If you need custom comparisons beyond the bundle diff:

```python
import json
from pathlib import Path

from app.db.game_replay import GameReplayDB
from app.utils.debug_utils import StateDiffer, load_ts_state_dump

bundle = json.load(open("parity_failures/<bundle>.parity_failure.json"))
db = GameReplayDB(bundle["db_path"])
py_state = db.get_state_at_move(bundle["game_id"], bundle["diverged_at"])
ts_state = load_ts_state_dump(Path("parity_failures/<bundle>.ts_state.json"))
diff = StateDiffer().diff_py_ts_state(py_state, ts_state)
print(diff)
```

### Step 2: Trace Phase Transitions

In `app/rules/phase_machine.py`:

```python
def advance_phases(input: PhaseTransitionInput) -> None:
    if input.trace_mode:
        print(f"[PHASE] Before: {input.game_state.current_phase}")
        print(f"[PHASE] Last move: {input.last_move.type}")
```

### Step 3: Fix Transition Logic

Based on root cause, fix will likely be in one of:

- `_should_enter_forced_elimination()` - add hex-specific gating
- `_did_process_territory_region()` - fix hex territory detection
- FSM orchestration - ensure bookkeeping moves are consistent

### Step 4: Validate Fix

```bash
# Run hex parity validation
pytest tests/parity/test_fsm_parity.py -k hex -v

# Generate fresh hex games
python scripts/generate_gumbel_selfplay.py \
  --board hexagonal \
  --games 100 \
  --validate-parity \
  --fail-fast
```

## Related Files

- `app/rules/phase_machine.py` - Phase state machine
- `app/rules/fsm.py` - Canonical FSM orchestrator
- `app/ai/gpu_canonical_export.py` - GPU parity validation
- `app/utils/debug_utils.py` - Debug utilities
- `tests/parity/test_fsm_parity.py` - FSM parity tests
- `docs/infrastructure/GPU_RULES_PARITY_AUDIT.md` - Parity audit notes

## References

- 7-phase state machine spec: `docs/specs/GAME_NOTATION_SPEC.md`
- RR-CANON compliance: FSM module is canonical
- TS implementation: `phaseStateMachine.ts`, `turnOrchestrator.ts`

## Success Criteria

- [ ] Hex games complete without phase divergence
- [ ] Parity validation passes for 1000+ hex games
- [ ] Training data can be generated for hex boards
- [ ] Model evaluation works on hex boards

---

**Last Updated:** December 21, 2025
