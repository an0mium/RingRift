# Tier Promotion System

The tier promotion system manages model progression through difficulty tiers D1–D10.
D11 is reserved for internal benchmarks and is not exposed via the public API.

> Source of truth: `docs/ai/AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md` for the end-to-end workflow.  
> This document is a concise module/API reference aligned with the current scripts.

## Architecture

```
Training -> Tier Gate + Perf -> Promotion Plan -> Registry -> Ladder Update
```

## Core Modules & Files

### `app/training/tier_promotion_registry.py`

Tracks tier promotion candidates and their status.

Key helpers:

- `load_square8_two_player_registry()`
- `save_square8_two_player_registry()`
- `record_promotion_plan(...)`
- `update_square8_two_player_registry_for_run(...)`
- `get_current_ladder_model_for_tier(...)`

Example:

```python
from app.training.tier_promotion_registry import (
    load_square8_two_player_registry,
    get_current_ladder_model_for_tier,
)

registry = load_square8_two_player_registry()
current = get_current_ladder_model_for_tier("D4")
print(f"D4 model: {current['model_id']}")
```

Default registry path:

- `ai-service/config/tier_candidate_registry.square8_2p.json`

Registry shape (example):

```json
{
  "board": "square8",
  "num_players": 2,
  "tiers": {
    "D4": {
      "current": {
        "tier": "D4",
        "difficulty": 4,
        "board": "square8",
        "board_type": "square8",
        "num_players": 2,
        "model_id": "nnue_square8_2p",
        "heuristic_profile_id": "heuristic_v1_sq8_2p",
        "ai_type": "minimax",
        "ladder_source": "app.config.ladder_config._build_default_square8_two_player_configs"
      },
      "candidates": [
        {
          "candidate_id": "d4_square8_2p_20251225_abcd1234",
          "candidate_model_id": "d4_square8_2p_20251225_abcd1234",
          "tier": "D4",
          "board": "square8",
          "num_players": 2,
          "source_run_dir": "runs/tier_training/D4_square8_2p_20251225_123000",
          "training_report": "training_report.json",
          "gate_report": "gate_report.json",
          "promotion_plan": "promotion_plan.json",
          "model_id": "d4_square8_2p_20251225_abcd1234",
          "status": "gated_promote"
        }
      ]
    }
  }
}
```

### `app/training/tier_eval_config.py`

Defines per-tier evaluation thresholds and opponent mixes.

Key helpers:

- `TierEvaluationConfig`
- `TIER_EVAL_CONFIGS`
- `get_tier_config("D4")`

### `app/training/tier_eval_runner.py`

Runs tier evaluation games and returns a `TierEvaluationResult`.

Entry point:

- `run_tier_evaluation(...)`

### `app/config/ladder_config.py`

Production ladder assignments plus runtime overrides.

Key helpers:

- `get_ladder_tier_config(...)` (base ladder)
- `get_effective_ladder_config(...)` (applies runtime overrides)
- Runtime overrides file: `ai-service/data/ladder_runtime_overrides.json`

### `app/config/perf_budgets.py` + `app/training/tier_perf_benchmark.py`

Perf budgets for D3–D8 and the benchmark runner.

## Promotion Workflow (Scripted)

### 1) Train a candidate

Use the tier training pipeline (D2–D10):

```bash
cd ai-service
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D4 \
  --board square8 \
  --num-players 2 \
  --output-dir runs/tier_training
```

This creates a timestamped run directory containing `training_report.json`.

### 2) Gate the candidate

Use the combined gate + perf wrapper:

```bash
RUN_DIR="runs/tier_training/D4_square8_2p_20251225_123000"
CANDIDATE_ID=$(jq -r '.candidate_id' "$RUN_DIR/training_report.json")

PYTHONPATH=. python scripts/run_full_tier_gating.py \
  --tier D4 \
  --candidate-id "$CANDIDATE_ID" \
  --run-dir "$RUN_DIR"
```

Outputs in `RUN_DIR`:

- `tier_eval_result.json`
- `promotion_plan.json`
- `tier_perf_report.json` (when a budget exists)
- `gate_report.json`

### 3) Update the candidate registry

```bash
python scripts/apply_tier_promotion_plan.py \
  --plan-path "$RUN_DIR/promotion_plan.json"
```

This updates `config/tier_candidate_registry.square8_2p.json` and emits:

- `promotion_summary.json`
- `promotion_patch_guide.txt`

### 4) Promote the ladder

Choose one of:

- **Runtime override:** update `data/ladder_runtime_overrides.json` (fast).
- **Permanent change:** update `app/config/ladder_config.py` and commit.

## Monitoring & Debugging

### Registry inspection

```bash
python -c "
from app.training.tier_promotion_registry import load_square8_two_player_registry
import json
print(json.dumps(load_square8_two_player_registry(), indent=2))
"
```

### Runtime override inspection

```bash
cat data/ladder_runtime_overrides.json
```

## Notes

- D1 is a random baseline and is not trained.
- D11 is internal-only; treat it as a stress tier, not a public ladder tier.
