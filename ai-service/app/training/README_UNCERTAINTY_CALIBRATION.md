# Uncertainty Calibration Study

This module evaluates whether the GMO AI's uncertainty estimates are well-calibrated.

## Overview

When a neural network predicts a game outcome, it also provides a variance estimate
indicating its confidence. Well-calibrated uncertainty means:

- **High predicted uncertainty** should correlate with **high prediction errors**
- **Confidence intervals** should have correct coverage (68% within 1σ, 95% within 2σ)

## Metrics Computed

1. **Error-Uncertainty Correlation**: Spearman correlation between |error| and variance
   - Ideal: positive correlation (higher variance predicts larger errors)

2. **Coverage Metrics**:
   - 1σ coverage: % of outcomes within 1 standard deviation (ideal: 68%)
   - 2σ coverage: % of outcomes within 2 standard deviations (ideal: 95%)

3. **Calibration Error**: ECE-style metric measuring deviation from ideal calibration

4. **Brier Score**: For win probability calibration (lower is better)

## Usage

```bash
# Run calibration study with 50 games against random opponent
python -m app.training.uncertainty_calibration --games 50 --opponent random

# Use a specific checkpoint
python -m app.training.uncertainty_calibration --checkpoint models/gmo/gmo_best.pt

# Against heuristic opponent
python -m app.training.uncertainty_calibration --games 100 --opponent heuristic
```

## Interpreting Results

### Error-Uncertainty Correlation

- `> 0.3`: Good - uncertainty correlates well with error
- `0.1-0.3`: Moderate - some correlation
- `< 0.1`: Poor - model is overconfident

### Coverage

- 1σ coverage near 68%: Well-calibrated
- 1σ coverage < 60%: Overconfident (underestimates uncertainty)
- 1σ coverage > 76%: Underconfident (overestimates uncertainty)

### Brier Score

- `< 0.2`: Good calibration
- `0.2-0.3`: Moderate
- `> 0.3`: Poor calibration

## Output

Results are saved to `data/calibration/calibration_<opponent>_<timestamp>.json` including:

- Game results (wins/losses/draws)
- All calibration metrics
- Uncertainty bins for detailed analysis

## Implementation Details

The study:

1. Creates a GMO AI with uncertainty estimation enabled
2. Plays games against a specified opponent
3. Records predicted values and variances for each move
4. Computes calibration metrics after all games complete
