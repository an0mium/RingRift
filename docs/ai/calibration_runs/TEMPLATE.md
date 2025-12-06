# Square‑8 2‑Player Calibration Run – Template

> Copy this file into a new directory under `docs/ai/calibration_runs/` and rename it to `notes.md` before filling it in.

## 1. Run metadata

- **Window label:** `YYYY-MM-square8-2p-windowNN`
- **Calendar window:** `YYYY-MM-DD` to `YYYY-MM-DD`
- **Board / players:** `square8`, `2`
- **Tiers covered:** `D2`, `D4`, `D6`, `D8` (adjust if needed)
- **Operator(s):**
- **Date run:**

## 2. Inputs

- **Calibration aggregates JSON:**
  - Path:
  - Notes (source query, version, filters):
- **Tier candidate registry used:**
  - Path:
  - Snapshot path (if any):
- **Eval root (`--eval-root`):**
  - Path:
  - Example tier run dirs inspected:

## 3. Command(s) run

Paste the exact CLI invocation(s) used for this run (including paths and flags).

```bash
# Example
python -m ai-service.scripts.analyze_difficulty_calibration \
  --calibration-aggregates docs/ai/calibration_runs/2025_12_square8_2p_window01/aggregates.square8_2p.window01.json \
  --registry-path ai-service/config/tier_candidate_registry.square8_2p.json \
  --eval-root ai-service/logs \
  --output-json docs/ai/calibration_runs/2025_12_square8_2p_window01/calibration_summary.json \
  --output-md docs/ai/calibration_runs/2025_12_square8_2p_window01/calibration_summary.md \
  --window-label 2025-12-square8-2p-window01
```

## 4. Summary of results

_Briefly summarise the calibration outcome per tier. Refer to `calibration_summary.md` for details._

### 4.1 Tier D2

- **Overall status:**
- **Key metrics / notes:**

### 4.2 Tier D4

- **Overall status:**
- **Key metrics / notes:**

### 4.3 Tier D6

- **Overall status:**
- **Key metrics / notes:**

### 4.4 Tier D8

- **Overall status:**
- **Key metrics / notes:**

## 5. Decisions & follow-ups

_Record concrete actions that follow from this calibration run. These should be referenceable from PRs or tickets._

- **Ladder / model decisions:**
  -
- **Training / gating follow-ups:**
  -
- **UX / difficulty descriptor changes:**
  -
- **Additional calibration or playtests to schedule:**
  -

## 6. Links

- Related training or gating runs (paths or external links):
- Related PRs:
- Related tickets or tasks:
