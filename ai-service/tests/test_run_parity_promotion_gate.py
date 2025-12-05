"""Tests for the cross-board parity promotion gate helper."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pytest

# Ensure app/ is importable when running tests directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.run_parity_promotion_gate import (  # type: ignore[import]  # noqa: E402
    _evaluate_promotion,
)


def _make_formatted(
    *,
    board: str,
    games: int,
    win_rate: float,
    ci: tuple[float, float],
    piece_advantage: float,
) -> Dict[str, Any]:
    """Build a minimal ``format_results_json``-like payload."""
    return {
        "config": {
            "player1": "candidate",
            "player2": "baseline",
            "games": games,
            "board": board,
        },
        "results": {
            "player1_wins": int(round(win_rate * games)),
            "player2_wins": games - int(round(win_rate * games)),
            "draws": 0,
            "player1_win_rate": win_rate,
            "player1_win_rate_ci95": [ci[0], ci[1]],
            "avg_game_length": 40.0,
            "avg_game_length_std": 5.0,
            "avg_decision_time_p1": 0.01,
            "avg_decision_time_p2": 0.01,
            "total_runtime_seconds": 1.0,
            "victory_types": {},
            "avg_p1_final_pieces": 10.0,
            "avg_p2_final_pieces": 9.0,
            "piece_advantage_p1": piece_advantage,
        },
        "games": [],
    }


def test_evaluate_promotion_all_matrices_pass() -> None:
    """Gate should pass when every matrix has CI lower bound above threshold."""
    matrices = {
        "square8_2p": _make_formatted(
            board="square8",
            games=200,
            win_rate=0.62,
            ci=(0.55, 0.68),
            piece_advantage=1.5,
        ),
        "square19_2p": _make_formatted(
            board="square19",
            games=200,
            win_rate=0.58,
            ci=(0.51, 0.65),
            piece_advantage=0.7,
        ),
    }

    summary = _evaluate_promotion(matrices, min_ci_lower_bound=0.5)

    assert summary["overall_pass"] is True
    assert summary["thresholds"]["min_ci_lower_bound"] == 0.5
    assert summary["worst_case_ci_lower_bound"] == pytest.approx(0.51)

    sq8 = summary["matrices"]["square8_2p"]
    assert sq8["board"] == "square8"
    assert sq8["games"] == 200
    assert sq8["player1_win_rate"] == pytest.approx(0.62)
    assert sq8["player1_win_rate_ci95"] == [pytest.approx(0.55), pytest.approx(0.68)]
    assert sq8["piece_advantage_p1"] == pytest.approx(1.5)
    assert sq8["passes"] is True


def test_evaluate_promotion_any_failure_fails_gate() -> None:
    """Gate should fail when any matrix has CI lower bound below threshold."""
    matrices = {
        "square8_2p": _make_formatted(
            board="square8",
            games=200,
            win_rate=0.55,
            ci=(0.49, 0.61),
            piece_advantage=0.2,
        ),
        "square19_2p": _make_formatted(
            board="square19",
            games=200,
            win_rate=0.6,
            ci=(0.53, 0.67),
            piece_advantage=0.9,
        ),
    }

    summary = _evaluate_promotion(matrices, min_ci_lower_bound=0.5)

    assert summary["overall_pass"] is False
    assert summary["matrices"]["square8_2p"]["passes"] is False
    assert summary["matrices"]["square19_2p"]["passes"] is True
    assert summary["worst_case_ci_lower_bound"] == pytest.approx(0.49)


def test_evaluate_promotion_empty_matrix_is_conservative_failure() -> None:
    """With no matrices provided, gate should fail rather than pass vacuously."""
    summary = _evaluate_promotion({}, min_ci_lower_bound=0.5)

    assert summary["overall_pass"] is False
    assert summary["worst_case_ci_lower_bound"] == pytest.approx(0.0)
    assert summary["matrices"] == {}

