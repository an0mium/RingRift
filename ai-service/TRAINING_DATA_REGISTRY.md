# Training Data Registry

This document tracks the provenance and canonical status of all self-play databases and neural network models in the RingRift AI training pipeline.

## Data Classification

| Status                  | Meaning                                                                   |
| ----------------------- | ------------------------------------------------------------------------- |
| **canonical**           | Passes `run_canonical_selfplay_parity_gate.py`; safe for training         |
| **legacy_noncanonical** | Pre-dates line-length/territory/parity fixes; DO NOT use for new training |
| **pending_gate**        | Not yet validated; requires parity gate before training use               |

---

## Game Replay Databases

### Canonical (Parity-Gated)

| Database                | Board Type | Players | Status           | Gate Summary                          | Notes                                                                                                                                                                                                                                                                         |
| ----------------------- | ---------- | ------- | ---------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `canonical_square8.db`  | square8    | 2       | **canonical**    | parity_summary.canonical_square8.json | Fresh canonical self-play; **parity currently shows 0 semantic divergences and 1 end-of-game-only divergence (phase/player/status only with identical final state hash). Safe for training and evaluation, subject to re-running the gate script when regenerating this DB.** |
| `canonical_square19.db` | square19   | 2       | **pending_gate** | N/A                                   | Placeholder DB (no games recorded yet)                                                                                                                                                                                                                                        |
| `canonical_hex.db`      | hexagonal  | 2       | **pending_gate** | N/A                                   | Placeholder DB (no games recorded yet)                                                                                                                                                                                                                                        |
| `golden.db`             | mixed      | mixed   | **canonical**    | N/A                                   | Hand-curated golden games                                                                                                                                                                                                                                                     |

### Legacy / Non-Canonical

These databases were generated **before** the following fixes were applied:

- Line-length validation fix (RR-CANON-R120)
- Explicit line decision flow (RR-CANON-R121-R122)
- All turn actions/skips must be explicit (RR-CANON-R074)
- Forced elimination checks for correct player rotation

**DO NOT use these for new training runs.** They are retained for historical comparison only.

| Database                    | Board Type | Players | Status                  | Notes                    |
| --------------------------- | ---------- | ------- | ----------------------- | ------------------------ |
| `selfplay_square8_2p.db`    | square8    | 2       | **legacy_noncanonical** | Pre-parity-fix self-play |
| `selfplay_square19_2p.db`   | square19   | 2       | **legacy_noncanonical** | Pre-parity-fix self-play |
| `selfplay_square19_3p.db`   | square19   | 3       | **legacy_noncanonical** | Pre-parity-fix self-play |
| `selfplay_square19_4p.db`   | square19   | 4       | **legacy_noncanonical** | Pre-parity-fix self-play |
| `selfplay_hexagonal_2p.db`  | hexagonal  | 2       | **legacy_noncanonical** | Pre-parity-fix self-play |
| `selfplay_hexagonal_3p.db`  | hexagonal  | 3       | **legacy_noncanonical** | Pre-parity-fix self-play |
| `selfplay_hexagonal_4p.db`  | hexagonal  | 4       | **legacy_noncanonical** | Pre-parity-fix self-play |
| `selfplay.db`               | mixed      | mixed   | **legacy_noncanonical** | Ad-hoc testing DB        |
| `square8_2p.db`             | square8    | 2       | **legacy_noncanonical** | Early development DB     |
| `minimal_test.db`           | mixed      | mixed   | **legacy_noncanonical** | Test fixture DB          |
| `selfplay_hex_mps_smoke.db` | hexagonal  | 2       | **legacy_noncanonical** | MPS smoke test           |

---

## Neural Network Models

### Model Provenance Table

| Model File                                | Training Data               | Status                  | Notes                                          |
| ----------------------------------------- | --------------------------- | ----------------------- | ---------------------------------------------- |
| `ringrift_v1.pth`                         | Legacy selfplay DBs         | **legacy_noncanonical** | Main model, needs retraining on canonical data |
| `ringrift_v1.pth.legacy`                  | Early selfplay DBs          | **legacy_noncanonical** | Original v1, historical only                   |
| `ringrift_v1_legacy_nested.pth`           | Legacy nested replay export | **legacy_noncanonical** | Do not use                                     |
| `ringrift_v1_mps.pth`                     | Legacy selfplay DBs         | **legacy_noncanonical** | MPS variant, needs retraining                  |
| `ringrift_v1_mps_2025*.pth`               | Legacy selfplay DBs         | **legacy_noncanonical** | MPS checkpoints                                |
| `ringrift_v1_2025*.pth`                   | Legacy selfplay DBs         | **legacy_noncanonical** | v1 checkpoints from Nov 2025                   |
| `ringrift_from_replays_square8.pth`       | Mixed replay DBs            | **legacy_noncanonical** | Trained from legacy replays                    |
| `ringrift_from_replays_square8_2025*.pth` | Mixed replay DBs            | **legacy_noncanonical** | Checkpoints from legacy replays                |

### Target Canonical Models

Once canonical self-play DBs are generated and exported, retrain these models:

| Target Model               | Training Data Source  | Status  |
| -------------------------- | --------------------- | ------- |
| `ringrift_v2_square8.pth`  | canonical_square8.db  | Pending |
| `ringrift_v2_square19.pth` | canonical_square19.db | Pending |
| `ringrift_v2_hex.pth`      | canonical_hex.db      | Pending |

---

## Training Data Allowlist Policy

1. **All new training runs** must ONLY use databases listed as `canonical` in this registry.

2. **To add a new DB to the canonical allowlist:**

   ```bash
   # Run the parity gate
   PYTHONPATH=. python scripts/run_canonical_selfplay_parity_gate.py \
     --board-type <board> \
     --num-games 20 \
     --db data/games/<new_db>.db \
     --summary parity_gate.<board>.json

   # If passed_canonical_parity_gate: true, add to this registry
   ```

3. **Legacy DBs** may be used for:
   - Historical comparison experiments
   - Ablation studies (comparing legacy vs canonical)
   - Debugging parity issues

   They must NOT be used for:
   - Training official v2+ models
   - Evaluation baselines
   - Curriculum learning seeds

4. **Model version tracking:**
   - All new models must use the `ModelVersionManager` from `app/training/model_versioning.py`
   - Checkpoints include architecture version, config, and SHA256 checksum
   - Version mismatch errors are thrown explicitly (no silent fallback)

---

## Cleanup Recommendations

### Move to `data/games/legacy/`:

- `selfplay_square8_2p.db`
- `selfplay_square19_*.db`
- `selfplay_hexagonal_*.db`
- `selfplay.db`
- `square8_2p.db`
- `minimal_test.db`
- `selfplay_hex_mps_smoke.db`

### Move to `models/legacy/`:

- `ringrift_v1.pth.legacy`
- `ringrift_v1_legacy_nested.pth`
- `ringrift_v1_2025*.pth`
- `ringrift_from_replays_square8*.pth`

### Keep in place (but retrain when canonical data ready):

- `ringrift_v1.pth` (current production model)
- `ringrift_v1_mps.pth` (MPS production model)

---

## Parity Gate Results

When `run_canonical_selfplay_parity_gate.py` is used to generate and gate a new DB,
its summary should be stored alongside this document (for example as
`parity_gate.<board>.json`). The lower-level parity sweeps invoked directly by
`check_ts_python_replay_parity.py` can also be captured as
`parity_summary.<label>.json` for ad-hoc debugging (for example,
`parity_summary.canonical_square8.json`).

Each gate summary contains:

```json
{
  "board_type": "...",
  "db_path": "...",
  "num_games": N,
  "parity_summary": {
    "games_checked": N,
    "games_with_structural_issues": 0,
    "games_with_semantic_divergence": 0
  },
  "passed_canonical_parity_gate": true
}
```

---

_Last updated: 2025-12-05_
