"""Lightweight statistical helpers for promotion gates.

These functions are used by curriculum and tier-gating code to avoid
promoting candidates on noisy, small-sample win rates.
"""

from __future__ import annotations

import math
from typing import Tuple


def wilson_score_interval(
    wins: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Return Wilson score confidence interval for a binomial win rate.

    Args:
        wins: Number of wins.
        total: Total trials.
        confidence: Confidence level (e.g. 0.95).

    Returns:
        (lower_bound, upper_bound) as floats in [0, 1].
    """
    if total <= 0:
        return (0.0, 0.0)

    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.90:
        z = 1.645
    elif confidence == 0.99:
        z = 2.576
    else:
        # Fall back to a conservative z-score for uncommon confidences.
        z = 2.576

    p_hat = wins / float(total)
    n = float(total)

    denominator = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denominator
    spread = (z / denominator) * math.sqrt(
        (p_hat * (1.0 - p_hat) / n) + (z ** 2) / (4.0 * n ** 2)
    )

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (round(lower, 4), round(upper, 4))


def wilson_lower_bound(
    wins: int,
    total: int,
    confidence: float = 0.95,
) -> float:
    """Convenience wrapper returning only the Wilson lower bound."""
    lower, _ = wilson_score_interval(wins, total, confidence=confidence)
    return lower

