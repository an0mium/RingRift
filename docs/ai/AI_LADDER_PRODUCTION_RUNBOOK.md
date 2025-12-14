# AI Ladder Production Runbook – Square-8 2-Player

> **Status:** Active
> Quick-start guide for running AI tier training and promotion in production.
> For full details, see [AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md).

---

## Prerequisites

### Environment

```bash
# From project root
cd ai-service
source venv/bin/activate  # or your Python env
pip install -r requirements.txt
```

### Compute Requirements

| Tier | Training Time (Demo) | Training Time (Full) | Notes                         |
| ---- | -------------------- | -------------------- | ----------------------------- |
| D2   | ~30s                 | 5-15 min             | Heuristic CMA-ES optimization |
| D4   | ~30s                 | 10-30 min            | Minimax persona tuning        |
| D6   | ~30s                 | 1-4 hours            | Neural network training       |
| D8   | ~30s                 | 2-8 hours            | MCTS + NN training            |

Full training runs require GPU for D6/D8. Demo mode runs on CPU.

### Required Files

- Canonical selfplay DBs in `TRAINING_DATA_REGISTRY.md` with status=`canonical`
- Tier candidate registry: `ai-service/config/tier_candidate_registry.square8_2p.json`

---

## Quick Start: Demo Mode (CI/Smoke)

Run both training and gating in demo mode (fast, no heavy compute):

```bash
cd ai-service

# Step 1: Train a candidate (demo mode)
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D4 \
  --board square8 \
  --num-players 2 \
  --run-dir /tmp/tier_training_demo \
  --demo \
  --seed 42

# Step 2: Gate the candidate (demo mode)
CANDIDATE_ID=$(jq -r '.candidate_id' /tmp/tier_training_demo/training_report.json)
PYTHONPATH=. python scripts/run_full_tier_gating.py \
  --tier D4 \
  --candidate-id "$CANDIDATE_ID" \
  --run-dir /tmp/tier_training_demo \
  --demo
```

Outputs in `--run-dir`:

- `training_report.json` – Training configuration and candidate ID
- `tier_eval_result.json` – Evaluation results vs baselines
- `promotion_plan.json` – Promotion decision (promote/reject)
- `tier_perf_report.json` – Performance benchmark results
- `gate_report.json` – Combined summary with `final_decision`
- `status.json` – Pipeline status tracker

---

## Production Run: Full Training Cycle

### Step 1: Create Run Directory

```bash
RUN_DIR="ai-service/logs/tier_gate/D4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
```

### Step 2: Train Candidate

```bash
cd ai-service
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D4 \
  --board square8 \
  --num-players 2 \
  --run-dir "$RUN_DIR" \
  --seed 1
```

Monitor progress in `$RUN_DIR/status.json`.

### Step 3: Gate Candidate

```bash
CANDIDATE_ID=$(jq -r '.candidate_id' "$RUN_DIR/training_report.json")
PYTHONPATH=. python scripts/run_full_tier_gating.py \
  --tier D4 \
  --candidate-id "$CANDIDATE_ID" \
  --run-dir "$RUN_DIR"
```

### Step 4: Verify Results

```bash
# Check final decision
jq '.final_decision' "$RUN_DIR/gate_report.json"

# Check evaluation pass
jq '.tier_eval.overall_pass' "$RUN_DIR/gate_report.json"

# Check perf pass (D4/D6/D8 only)
jq '.tier_perf.overall_pass' "$RUN_DIR/gate_report.json"
```

### Step 5: Promotion (if passed)

If `final_decision: "promote"`, update the ladder config:

1. Update `ai-service/app/config/ladder_config.py`:
   - Set `model_id` to the new candidate ID
   - Update `heuristic_profile_id` if applicable

2. Update tier candidate registry:

   ```bash
   # Add promoted candidate to registry
   # File: ai-service/config/tier_candidate_registry.square8_2p.json
   ```

3. Archive artifacts:
   ```bash
   mv "$RUN_DIR" ai-service/data/promotions/square8_2p/D4/
   ```

---

## Tier-Specific Commands

### D2 (Heuristic)

```bash
# Training (heuristic CMA-ES optimization)
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D2 --board square8 --num-players 2 --run-dir "$RUN_DIR"

# Gating (no perf budget for D2)
PYTHONPATH=. python scripts/run_full_tier_gating.py \
  --tier D2 --candidate-id "$CANDIDATE_ID" --run-dir "$RUN_DIR" --no-perf
```

### D6 (Neural Network)

```bash
# Requires GPU for full training
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D6 --board square8 --num-players 2 --run-dir "$RUN_DIR"
```

### D8 (MCTS + Neural)

```bash
# Longest training time, requires GPU
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D8 --board square8 --num-players 2 --run-dir "$RUN_DIR"
```

---

## Gating Criteria Summary

| Tier | Min Win Rate vs Baseline | Max Regression vs Previous | Perf Budget |
| ---- | ------------------------ | -------------------------- | ----------- |
| D2   | ≥60% vs D1 random        | N/A                        | None        |
| D4   | ≥70% vs baselines        | ≤5% vs D2                  | Required    |
| D6   | ≥75% vs baselines        | ≤5% vs D4                  | Required    |
| D8   | ≥80% vs baselines        | ≤5% vs D6                  | Required    |

From `ai-service/app/training/tier_eval_config.py`.

---

## Troubleshooting

### Training Report Missing

```bash
# Verify training completed
cat "$RUN_DIR/status.json"
# Should show: "training": {"status": "completed"}
```

### Gate Failed

```bash
# Check specific failure reason
jq '.tier_eval' "$RUN_DIR/gate_report.json"
jq '.reason' "$RUN_DIR/promotion_plan.json"
```

### Perf Budget Exceeded

Options:

1. Reduce search depth/iterations and retrain
2. Update perf budgets in `ai-service/app/config/perf_budgets.py` (requires justification)
3. Reject candidate and keep current production tier

### Missing Canonical Data

```bash
# Check registry for canonical DBs
cat ai-service/TRAINING_DATA_REGISTRY.md | grep -A5 "canonical"
```

Only DBs with status=`canonical` should be used for production training.

---

## Related Documentation

- [AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md](AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md) – Full pipeline design
- [AI_CALIBRATION_RUNBOOK.md](AI_CALIBRATION_RUNBOOK.md) – Human calibration procedures
- [AI_TIER_PERF_BUDGETS.md](AI_TIER_PERF_BUDGETS.md) – Performance budget specifications
- [AI_HUMAN_CALIBRATION_GUIDE.md](AI_HUMAN_CALIBRATION_GUIDE.md) – Human testing templates
