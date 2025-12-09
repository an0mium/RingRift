# Parity & Canonical Replay Status Snapshot

_Generated locally on 2025-12-09; read-only diagnostic run. No code, configs, or DBs were modified._

## Scope

This snapshot covers replay databases referenced (directly or indirectly) in [`TRAINING_DATA_REGISTRY.md`](ai-service/TRAINING_DATA_REGISTRY.md:1) and closely related self-play/victory-study DBs discovered under [`ai-service/data/games/`](ai-service/data/games:1).

Databases included:

- [`canonical_square8.db`](ai-service/data/games/canonical_square8.db:1)
- [`selfplay.db`](ai-service/data/games/selfplay.db:1)
- [`selfplay_aws.db`](ai-service/data/games/selfplay_aws.db:1)
- [`victory_study/square8_2p.db`](ai-service/data/games/victory_study/square8_2p.db:1)
- [`victory_study/square8_3p.db`](ai-service/data/games/victory_study/square8_3p.db:1)
- [`victory_study/square8_4p.db`](ai-service/data/games/victory_study/square8_4p.db:1)
- Virtual/absent registry entries:
  - `canonical_square19.db` (referenced in registry notes, not present on disk)
  - `canonical_hex.db` (explicitly removed per registry/model docs)

All checks were run via the existing parity/canonical-history tooling:

- [`run_parity_and_history_gate.py`](ai-service/scripts/run_parity_and_history_gate.py:1)  
  (wrapper over [`check_ts_python_replay_parity.py`](ai-service/scripts/check_ts_python_replay_parity.py:1) and [`check_canonical_phase_history.py`](ai-service/scripts/check_canonical_phase_history.py:1))

Commands were executed from the repo root with `PYTHONPATH=ai-service`.

## Database Status Table

Legend:

- **Parity status**:
  - `pass` = parity script completed with **no structural issues and no semantic divergences** (end-of-game-only divergences allowed).
  - `fail` = any structural issues or semantic divergences.
  - `not_checked` = DB not present locally.
- **Canonical-history status**: `pass` = [`check_canonical_phase_history.py`](ai-service/scripts/check_canonical_phase_history.py:1) exited 0; `fail` = non-zero (phase/move invariant violations).

| DB                              | Path                                                                                                       | Board / Players         | Declared Status (Registry)                      | Parity Status (this run)                                                                                                                                                                                                                                                              | Canonical-History Status                                                                                                             | Notes / Gate Artifacts                                                                                                                                                                                                                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `canonical_square8.db`          | [`ai-service/data/games/canonical_square8.db`](ai-service/data/games/canonical_square8.db:1)               | square8 / 2p            | **pending_gate** (registry row)                 | **fail** – 1/1 games has TS structural error; `games_with_structural_issues = 1`, `games_with_semantic_divergence = 0`                                                                                                                                                                | **fail** – canonical phase-history violation (`process_territory_region` applied while `currentPhase = ring_placement`)              | Existing gate artifacts: [`canonical_square8.db.parity_gate.json`](ai-service/data/games/canonical_square8.db.parity_gate.json:1), [`db_health.canonical_square8.json`](ai-service/data/games/db_health.canonical_square8.json:1). Local run confirms DB is non-canonical and still requires regeneration. |
| `selfplay.db`                   | [`ai-service/data/games/selfplay.db`](ai-service/data/games/selfplay.db:1)                                 | square8 / mixed (2–4p)  | not_listed (implicit legacy/self-play)          | **fail** – `games_with_semantic_divergence = 2`, `games_with_structural_issues = 0`; both divergences are mid-game (not end-of-game-only)                                                                                                                                             | **pass** – all games replay via [`GameEngine.apply_move()`](ai-service/app/game_engine.py:1) with no phase/move invariant violations | Historical parity gate summary: [`selfplay.db.parity_gate.json`](ai-service/data/games/selfplay.db.parity_gate.json:1). Suitable for debugging, but not canonical.                                                                                                                                         |
| `selfplay_aws.db`               | [`ai-service/data/games/selfplay_aws.db`](ai-service/data/games/selfplay_aws.db:1)                         | square8 / mixed (2–4p)  | not_listed (legacy AWS self-play)               | **fail** – `games_with_semantic_divergence = 104`, `games_with_structural_issues = 117`, plus 4 end-of-game-only divergences; many errors are TS harness failures on non-canonical territory regions                                                                                  | **fail** – canonical phase-history violations, e.g. `place_ring` or `process_territory_region` applied in the wrong phase            | Strongly non-canonical legacy DB; keep only for historical analysis or targeted parity debugging.                                                                                                                                                                                                          |
| `square8_2p.db` (victory study) | [`ai-service/data/games/victory_study/square8_2p.db`](ai-service/data/games/victory_study/square8_2p.db:1) | square8 / 2p            | not_listed (victory-study only)                 | **fail** – `games_with_semantic_divergence = 4`, `games_with_structural_issues = 13`; divergences are line/territory → `game_over` transitions; structural issues come from TS replay PHASE_MOVE_INVARIANT errors on `process_territory_region` while `currentPhase = ring_placement` | **pass** – no canonical phase-history violations detected                                                                            | Useful for UX/victory analysis but not canonical for training.                                                                                                                                                                                                                                             |
| `square8_3p.db` (victory study) | [`ai-service/data/games/victory_study/square8_3p.db`](ai-service/data/games/victory_study/square8_3p.db:1) | square8 / 3p            | not_listed (victory-study only)                 | **fail** – `games_with_semantic_divergence = 0` but `games_with_structural_issues = 8`; additionally 2 end-of-game-only divergences (terminal currentPlayer mismatch with identical state hash)                                                                                       | **pass** – canonical phase-history checker completes with no violations                                                              | Structurally non-canonical for TS↔Python replay despite canonical history; safe for manual analysis, not for canonical training.                                                                                                                                                                           |
| `square8_4p.db` (victory study) | [`ai-service/data/games/victory_study/square8_4p.db`](ai-service/data/games/victory_study/square8_4p.db:1) | square8 / 4p            | not_listed (victory-study only)                 | **fail** – `games_with_semantic_divergence = 0`, `games_with_structural_issues = 4`; TS replay fails on several `process_territory_region` moves in `ring_placement`                                                                                                                  | **pass** – canonical phase-history checker completes with no violations                                                              | Same pattern as 3p DB: canonical history, but non-canonical TS replay; use for case studies only.                                                                                                                                                                                                          |
| `canonical_square19.db`         | _(not present on disk)_                                                                                    | square19 / 2p (implied) | **pending_gate** (registry gate notes)          | **not_checked** – DB file not found under [`ai-service/data/games/`](ai-service/data/games:1); gate notes say it currently contains zero games                                                                                                                                        | **not_checked** – no DB to replay                                                                                                    | Registry notes already flag this as needing regeneration; no local parity/canonical checks are currently possible.                                                                                                                                                                                         |
| `canonical_hex.db`              | _(removed)_                                                                                                | hex / radius-12 target  | removed / deprecated (see hex deprecation docs) | **not_checked** – DB intentionally removed                                                                                                                                                                                                                                            | **not_checked** – DB intentionally removed                                                                                           | Hex canonical data is pending a fresh radius-12 DB generation; see [`HEX_DATA_DEPRECATION_NOTICE.md`](ai-service/data/HEX_DATA_DEPRECATION_NOTICE.md:1) and [`HEX_ARTIFACTS_DEPRECATED.md`](ai-service/docs/HEX_ARTIFACTS_DEPRECATED.md:1).                                                                |

## Narrative Summary

- **No DB currently satisfies full canonical gate criteria** (zero structural issues, zero semantic divergences, and zero canonical phase-history violations).
- [`canonical_square8.db`](ai-service/data/games/canonical_square8.db:1), the only registry-row DB, is **still non-canonical**:
  - TS replay fails with `[PHASE_MOVE_INVARIANT] Cannot apply move type 'process_territory_region' in phase 'ring_placement'`.
  - Python canonical phase-history replay also fails on the same game, confirming the recording itself is non-canonical, not just the TS harness.
  - Existing gate summary [`db_health.canonical_square8.json`](ai-service/data/games/db_health.canonical_square8.json:1) already reports `canonical_ok = false`; this local run reinforces that status.
- [`selfplay.db`](ai-service/data/games/selfplay.db:1) passes canonical phase-history checks but has **2 semantic TS↔Python divergences**, so it is suitable for **debugging** but **not** as canonical training data.
- [`selfplay_aws.db`](ai-service/data/games/selfplay_aws.db:1) is **strongly non-canonical**, with >100 semantic divergences and >100 structural issues, plus canonical phase-history violations. It should remain legacy-only.
- Victory-study DBs (`square8_2p/3p/4p`) all:
  - Pass canonical phase-history replay,
  - But fail parity due to structural TS harness errors around `process_territory_region` immediately after territory phases, and (for the 2p DB) several line/territory → `game_over` mismatches.
  - These make them useful for UX/victory case analysis but unsafe as canonical training sources.
- `canonical_square19.db` and `canonical_hex.db` are effectively **absent** from the local environment; regeneration and re-gating remain outstanding preconditions for any square19/hex canonical training.

## Commands Used

All commands were run from the monorepo root (`/Users/armand/Development/RingRift`):

```bash
# Discovery
find ai-service -name "*.db" -print

# Parity + canonical-history gates
PYTHONPATH=ai-service python ai-service/scripts/run_parity_and_history_gate.py --db ai-service/data/games/canonical_square8.db
PYTHONPATH=ai-service python ai-service/scripts/run_parity_and_history_gate.py --db ai-service/data/games/selfplay.db
PYTHONPATH=ai-service python ai-service/scripts/run_parity_and_history_gate.py --db ai-service/data/games/selfplay_aws.db
PYTHONPATH=ai-service python ai-service/scripts/run_parity_and_history_gate.py --db ai-service/data/games/victory_study/square8_2p.db
PYTHONPATH=ai-service python ai-service/scripts/run_parity_and_history_gate.py --db ai-service/data/games/victory_study/square8_3p.db
PYTHONPATH=ai-service python ai-service/scripts/run_parity_and_history_gate.py --db ai-service/data/games/victory_study/square8_4p.db
```

## Next Steps (for Future Parity Tasks)

1. **Regenerate `canonical_square8.db` off-sandbox** using [`generate_canonical_selfplay.py`](ai-service/scripts/generate_canonical_selfplay.py:1) on a host without SHM/OMP restrictions, then:
   - Re-run TS↔Python parity with `--fail-on-divergence` and `--emit-state-bundles-dir`,
   - Re-run canonical phase-history checks,
   - Update [`TRAINING_DATA_REGISTRY.md`](ai-service/TRAINING_DATA_REGISTRY.md:1) only once `canonical_ok` and `passed_canonical_parity_gate` are both true.
2. **Create fresh canonical DBs** for square19 and hex:
   - `canonical_square19.db` (currently zero games / missing locally).
   - `canonical_hex.db` replacement using the new radius-12 geometry, consistent with [`HEX_ARTIFACTS_DEPRECATED.md`](ai-service/docs/HEX_ARTIFACTS_DEPRECATED.md:1).
3. **Keep legacy/self-play DBs out of the canonical allowlist**:
   - Treat [`selfplay.db`](ai-service/data/games/selfplay.db:1), [`selfplay_aws.db`](ai-service/data/games/selfplay_aws.db:1), and the victory-study DBs as **legacy_noncanonical** for training, while retaining them for debugging and UX analysis.
4. **Use parity fixtures/state bundles selectively** (outside this read-only subtask) to debug specific divergences, especially around `process_territory_region` and terminal game-over handling on square8.
