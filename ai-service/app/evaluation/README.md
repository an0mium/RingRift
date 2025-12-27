# Evaluation Module

Comprehensive evaluation framework for RingRift AI models.

## Overview

This module provides:

- Benchmark suites for reproducible model evaluation
- Human evaluation interfaces
- Performance, quality, and tactical metrics

## Key Components

### `benchmark_suite.py` - Benchmark Framework

Structured benchmarks with categories:

```python
from app.evaluation.benchmark_suite import (
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuiteResult,
)

# Create a benchmark result
result = BenchmarkResult(
    benchmark_name="win_rate_vs_random",
    category=BenchmarkCategory.QUALITY,
    score=0.95,
    unit="win_rate",
    higher_is_better=True,
    details={"games": 100, "wins": 95},
)

# Create a suite with multiple benchmarks
suite = BenchmarkSuiteResult(
    suite_name="full_evaluation",
    model_id="hex8_2p_v3",
    results=[result],
)

# Query results
win_rate = suite.get_score("win_rate_vs_random")
quality_scores = suite.get_category_scores(BenchmarkCategory.QUALITY)
aggregate = suite.compute_aggregate_score()
```

### Benchmark Categories

| Category      | Description                  |
| ------------- | ---------------------------- |
| `PERFORMANCE` | Speed, throughput metrics    |
| `QUALITY`     | Win rate, Elo ratings        |
| `TACTICAL`    | Pattern recognition accuracy |
| `STRATEGIC`   | Long-term planning ability   |
| `ROBUSTNESS`  | Edge case handling           |
| `EFFICIENCY`  | Memory, compute usage        |

### `human_eval.py` - Human Evaluation

Framework for collecting human feedback:

```python
from app.evaluation.human_eval import (
    EvaluationType,
    MoveQuality,
    EvaluationTask,
    EvaluationResponse,
    EvaluatorProfile,
    HumanEvalServer,
)

# Create an evaluation task
task = EvaluationTask(
    task_id="task_001",
    game_id="game_123",
    eval_type=EvaluationType.MOVE_QUALITY,
    position_fen="...",  # Board state
)

# Create evaluator response
response = EvaluationResponse(
    task_id="task_001",
    evaluator_id="expert_1",
    move_quality=MoveQuality.EXCELLENT,
    confidence=0.9,
    reasoning="Creates line threat",
)
```

## Usage Examples

### Running a Benchmark Suite

```python
from app.evaluation.benchmark_suite import (
    BenchmarkSuite,
    BenchmarkResult,
    BenchmarkSuiteResult,
    InferenceBenchmark,
    PolicyAccuracyBenchmark,
)

# Create benchmarks
inference = InferenceBenchmark(
    name="inference_time",
    model_path="models/hex8_2p.pth",
)
policy = PolicyAccuracyBenchmark(
    name="policy_accuracy",
    model_path="models/hex8_2p.pth",
)

# Run benchmarks
results = [
    inference.run(),
    policy.run(),
]

# Create suite result
suite = BenchmarkSuiteResult(
    suite_name="standard_eval",
    model_id="hex8_2p_v3",
    results=results,
)

print(f"Aggregate score: {suite.compute_aggregate_score():.3f}")
for result in suite.results:
    print(f"  {result.benchmark_name}: {result.score:.3f} {result.unit}")
```

### Exporting Results

```python
import json

# Export to JSON
results_dict = suite.to_dict()
with open("results.json", "w") as f:
    json.dump(results_dict, f, indent=2)
```

## Metrics Computed

### Quality Metrics

- Win rate vs random opponent
- Win rate vs heuristic baseline
- Elo rating in ladder

### Performance Metrics

- Inference time (ms)
- Throughput (games/sec)
- GPU utilization

### Tactical Metrics

- Line detection accuracy
- Territory recognition
- Threat assessment

## Integration with Training

Benchmark results feed into:

- Model promotion decisions
- Training curriculum adjustments
- Hyperparameter optimization

```python
from app.config.thresholds import (
    WIN_RATE_RANDOM_THRESHOLD,
    WIN_RATE_HEURISTIC_THRESHOLD,
)

# Check promotion criteria
if suite.get_score("win_rate_vs_random") >= WIN_RATE_RANDOM_THRESHOLD:
    print("Meets random baseline")
```
