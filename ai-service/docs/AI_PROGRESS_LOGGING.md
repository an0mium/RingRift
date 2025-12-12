# AI Progress Logging Standards

> **SSOT**: This document defines the canonical progress logging patterns  
> for long-running AI jobs in the `ai-service` Python microservice.

## Overview

Long-running AI jobs (training, optimization, self-play soaks, evaluation)  
must emit regular progress heartbeats to avoid appearing stalled in CI/CD  
pipelines and monitoring systems. The target cadence is approximately  
**~10 seconds** between progress lines, with a hard upper bound of  
**60 seconds** of silence allowed.

## Shared Reporter Classes

All progress logging uses helpers from `app/utils/progress_reporter.py`:

| Class                          | Use Case                      | Key Metrics                            |
| ------------------------------ | ----------------------------- | -------------------------------------- |
| `ProgressReporter`             | General-purpose unit tracking | completed/total, units/sec, ETA        |
| `SoakProgressReporter`         | Game soaks (self-play)        | games/sec, moves/sec, avg game length  |
| `OptimizationProgressReporter` | CMA-ES/GA optimization        | generations, candidates, fitness stats |

## CLI Flag Conventions

Scripts that support tunable progress intervals should use consistent flags:

```bash
--progress-interval-sec FLOAT   # Minimum seconds between progress lines (default: 10.0)
--disable-progress              # Suppress progress output entirely
--disable-eval-progress         # Suppress per-evaluation progress (optimization scripts)
```

## Usage Examples

### General Progress (units processed)

```python
from app.utils.progress_reporter import ProgressReporter

reporter = ProgressReporter(
    total_units=1000,
    unit_name="games",
    report_interval_sec=10.0,
    context_label="heuristic_eval",
)

for i, item in enumerate(items, start=1):
    process(item)
    reporter.update(completed=i)

reporter.finish()
```

### Game Soak Progress

```python
from app.utils.progress_reporter import SoakProgressReporter

reporter = SoakProgressReporter(
    total_games=100,
    report_interval_sec=10.0,
    context_label="square8_mixed_2p",
)

for game_idx in range(num_games):
    moves, duration_sec = play_game()
    reporter.record_game(moves=moves, duration_sec=duration_sec)

reporter.finish()
```

### Optimization Progress (CMA-ES / GA)

```python
from app.utils.progress_reporter import OptimizationProgressReporter

reporter = OptimizationProgressReporter(
    total_generations=50,
    candidates_per_generation=16,
    report_interval_sec=10.0,
)

for gen in range(1, generations + 1):
    reporter.start_generation(gen)

    for idx, candidate in enumerate(population, start=1):
        fitness = evaluate(candidate)
        reporter.record_candidate(
            candidate_idx=idx,
            fitness=fitness,
            games_played=games_per_eval,
        )

    reporter.finish_generation(
        mean_fitness=mean_f,
        best_fitness=best_f,
        std_fitness=std_f,
    )

reporter.finish()
```

## Instrumented Scripts

The following scripts have been instrumented with progress reporters:

### Optimization Drivers

| Script                            | Reporter Type                     | CLI Flags                                                                  |
| --------------------------------- | --------------------------------- | -------------------------------------------------------------------------- |
| `run_cmaes_optimization.py`       | `OptimizationProgressReporter`    | `--progress-interval-sec`, `--disable-eval-progress`                       |
| `run_genetic_heuristic_search.py` | `OptimizationProgressReporter`    | `--progress-interval-sec`, `--disable-progress`, `--disable-eval-progress` |
| `run_heuristic_experiment.py`     | `ProgressReporter` + CMA-ES flags | `--cmaes-progress-interval-sec`, `--cmaes-disable-eval-progress`           |

### Self-Play & Evaluation

| Script                                  | Reporter Type               | Default Interval                               |
| --------------------------------------- | --------------------------- | ---------------------------------------------- |
| `run_self_play_soak.py`                 | `SoakProgressReporter`      | 10s (hardcoded)                                |
| `run_parallel_self_play.py`             | `ProgressReporter`          | 10s (hardcoded)                                |
| `run_canonical_selfplay_parity_gate.py` | Heartbeat + parity progress | 60s heartbeat; parity progress every 200 steps |

`run_canonical_selfplay_parity_gate.py` is expected to be “chatty” even for long runs:

- Emits a heartbeat to stderr every `--heartbeat-seconds` (default: 60).
- Refreshes the `--summary` JSON on each heartbeat so you can `tail -f` it.
- Emits TS↔Python replay progress every `--parity-progress-every` steps (default: 200).
- Supports `--soak-timeout-seconds` and `--parity-timeout-seconds` to prevent silent hangs.

## Configuration Dataclasses

For optimization scripts, progress configuration is encapsulated in dataclasses:

```python
@dataclass
class CMAESConfig:
    # ... other fields ...
    progress_interval_sec: float = 10.0
    enable_eval_progress: bool = True
```

The `CMAESConfig` is constructed from CLI args and passed to evaluation  
functions, ensuring consistent progress behavior across the pipeline.

## Evaluation Function Progress

Multi-board evaluation functions accept optional progress parameters:

```python
def evaluate_fitness_over_boards(
    # ... required params ...
    progress_label: str | None = None,        # None = suppress per-board progress
    progress_interval_sec: float = 10.0,
    enable_eval_progress: bool = True,        # False = suppress even if label provided
) -> Tuple[float, Dict[BoardType, float]]:
```

Per-board reporters are only created when **both**:

1. `progress_label is not None`
2. `enable_eval_progress=True`

This allows suppressing nested progress while keeping outer-level reporting.

## Log Format

Progress lines follow a consistent format for parsing:

```
[PROGRESS] context=<label> completed=<N>/<total> rate=<X.XX>/sec ETA=<MM:SS>
[SOAK PROGRESS] context=<label> games=<N>/<total> moves/sec=<X.XX> avg_len=<N>
[OPT PROGRESS] gen=<G>/<total> candidates=<N> best=<fitness> mean=<fitness>
```

## Testing

Progress reporters are tested via:

- `ai-service/tests/test_cmaes_optimization.py` - CMA-ES CLI integration
- Smoke tests in nightly CI pipelines
- Manual observation during long-running jobs

## Future Work

- Add `--progress-interval-sec` flag to remaining evaluation scripts
- Integrate with Prometheus metrics for real-time dashboards
- Consider structured JSON logging for machine parsing
