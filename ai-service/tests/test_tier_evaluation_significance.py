from __future__ import annotations

import pytest

from app.models import AIType, BoardType
from app.training.tier_eval_config import TierEvaluationConfig, TierOpponentConfig
from app.training import tier_eval_runner as runner


def test_min_win_rate_uses_wilson_lower_bound(monkeypatch) -> None:
    """Candidate should not pass baseline gate without significance."""

    def fake_play_matchup(*, tier_config, opponent, **_kwargs):
        stats = runner.MatchupStats(
            opponent_id=opponent.id,
            opponent_difficulty=opponent.difficulty,
            opponent_ai_type="random",
        )
        stats.games = 10
        stats.total_moves = 10
        stats.victory_reasons = {"unknown": 10}
        if opponent.role == "baseline":
            stats.wins = 8
            stats.losses = 2
            stats.draws = 0
        elif opponent.role == "previous_tier":
            stats.wins = 5
            stats.losses = 5
            stats.draws = 0
        return stats

    monkeypatch.setattr(runner, "_play_matchup", fake_play_matchup)

    tier_config = TierEvaluationConfig(
        tier_name="TEST_SIG",
        display_name="Test significance gate",
        board_type=BoardType.SQUARE8,
        num_players=2,
        num_games=10,
        candidate_difficulty=1,
        time_budget_ms=0,
        opponents=[
            TierOpponentConfig(
                id="baseline",
                description="baseline",
                difficulty=1,
                ai_type=AIType.RANDOM,
                role="baseline",
            ),
            TierOpponentConfig(
                id="prev",
                description="prev tier",
                difficulty=1,
                ai_type=AIType.RANDOM,
                role="previous_tier",
            ),
        ],
        min_win_rate_vs_baseline=0.7,
        max_regression_vs_previous_tier=None,
        promotion_confidence=0.95,
    )

    result = runner.run_tier_evaluation(
        tier_config=tier_config,
        candidate_id="cand",
    )

    assert result.metrics["win_rate_vs_baseline"] == pytest.approx(0.8)
    assert result.metrics["win_rate_vs_baseline_ci_low"] < 0.7
    assert result.criteria["min_win_rate_vs_baseline"] is False

