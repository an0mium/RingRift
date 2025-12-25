# Tier performance budgets and benchmarks

This document summarises the tier performance budgets added in the H‑AI‑8
work and how to run the small benchmark harness used to enforce them.

## Scope

- Currently budgeted tiers: Square‑8 2‑player D3 / D4 / D5 / D6 / D7 / D8.
- Budgets are **guard rails**, not UX promises; they are tuned to detect
  large regressions relative to the existing ladder configuration.
- Budgets are expressed in **per‑move wall‑clock milliseconds** and derived
  from the canonical `think_time_ms` values already used by the ladder.

## Configuration: perf budgets

Single source of truth:

- `TierPerfBudget` dataclass and `TIER_PERF_BUDGETS` live in
  `ai-service/app/config/perf_budgets.py`.

Key fields on `TierPerfBudget`:

- `tier_name`: canonical tier identifier, e.g. `"D6_SQ8_2P"`.
- `difficulty`: ladder difficulty (3–8).
- `board_type`, `num_players`: board geometry and player count.
- `max_avg_move_ms`: soft upper bound for **average** per‑move latency
  over a small benchmark sample.
- `max_p95_move_ms`: soft upper bound for the **p95** per‑move latency.

Budgets are derived directly from the ladder:

- For each `(difficulty, BoardType.SQUARE8, num_players=2)` ladder entry,
  we read `LadderTierConfig.think_time_ms` and compute:
  - `max_avg_move_ms = think_time_ms * 1.10`
  - `max_p95_move_ms = think_time_ms * 1.25`

This keeps the perf budgets aligned with the existing search budgets in
the ladder (roughly sub‑second at low difficulties and up to ~20s at the
highest difficulties), while allowing modest headroom for environment and
host variance.

Helper:

- `get_tier_perf_budget(tier_name: str) -> TierPerfBudget` looks up either
  full names (`"D6_SQ8_2P"`) or short aliases (`"D6"`) in
  `TIER_PERF_BUDGETS` (case‑insensitive).

## Benchmark helper

The benchmark harness lives in:

- `ai-service/app/training/tier_perf_benchmark.py`

Main entry point:

- `run_tier_perf_benchmark(tier_name: str, num_games: int = 4,
moves_per_game: int = 16, seed: int = 1) -> TierPerfResult`

Behaviour overview:

- Resolves a `TierPerfBudget` for the requested tier.
- Resolves the matching `TierEvaluationConfig` (D3–D8, Square‑8 2p).
- Creates a `RingRiftEnv` with the correct board type and player count.
- Instantiates ladder AIs for both seats at the tier’s candidate difficulty.
- For each move sampled it measures wall‑clock time around:
  - `select_move(state)`
  - `evaluate_position(state)`
- Aggregates all measured latencies and returns:
  - `average_ms`
  - `p95_ms`
  - the associated `TierPerfBudget`

The helper does **not** assert on budgets; it only reports measurements.
Assertions live in the unit tests (see below).

## CLI: manual tier benchmarks

The CLI wrapper is:

- `ai-service/scripts/run_tier_perf_benchmark.py`

Example usage (from the `ai-service` directory):

```bash
python scripts/run_tier_perf_benchmark.py --tier D6 --num-games 4 \
  --moves-per-game 16 --seed 1
```

Flags:

- `--tier`: required; accepts `D3`–`D8` or full names like
  `D6_SQ8_2P`.
- `--num-games`: number of self‑play games to run (default 4).
- `--moves-per-game`: max moves sampled per game (default 16).
- `--seed`: base RNG seed (default 1).
- `--output-json PATH`: optional JSON summary output.

The CLI prints a human‑readable summary including average / p95 latencies,
configured budgets, and PASS/FAIL flags for each threshold.

## CI perf tests

CI‑facing tests live in:

- `ai-service/tests/test_tier_perf_budgets.py`
- `ai-service/tests/test_run_tier_perf_benchmark.py`

They cover:

- Config sanity: each budgeted tier has matching `LadderTierConfig` and
  `TierEvaluationConfig`, and budgets are positive and bounded relative to
  `think_time_ms`.
- Smoke perf tests: for D4/D6/D8 Square‑8 2p, a tiny benchmark run (1 game,
  4 moves) must have `average_ms` and `p95_ms` below the configured
  `max_avg_move_ms` / `max_p95_move_ms` for that tier. D3/D5/D7 budgets are
  configured but not currently exercised in CI smoke.
- Budget evaluation semantics: `_eval_budget` (used by the CLI and any
  future automation) sets `within_avg`, `within_p95`, and `overall_pass`
  consistently based on the observed metrics and per‑tier budgets.
