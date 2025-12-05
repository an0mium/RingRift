#!/usr/bin/env python
"""Tests for tier evaluation config, runner, and CLI harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# Ensure app/ and scripts/ are importable when running tests directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from app.models import AIType, BoardType  # noqa: E402
from app.training.tier_eval_config import (  # noqa: E402
    TierEvaluationConfig,
    TierOpponentConfig,
)
from app.training.tier_eval_runner import run_tier_evaluation  # noqa: E402
from app.training.eval_pools import (  # noqa: E402
    HEURISTIC_TIER_SPECS,
    run_all_heuristic_tiers,
    run_heuristic_tier_eval,
)


class TestTierEvaluationRunner:
    """Tests for the tier evaluation runner and config wiring."""

    def test_random_vs_random_basic_stats(self) -> None:
        """Random-vs-random smoke test for wiring and statistics."""
        tier_config = TierEvaluationConfig(
            tier_name="TEST",
            display_name="Test random vs random (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=4,
            candidate_difficulty=1,
            time_budget_ms=0,
            opponents=[
                TierOpponentConfig(
                    id="random_baseline",
                    description="Random baseline",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
            ],
            min_win_rate_vs_baseline=0.0,
            max_regression_vs_previous_tier=None,
            description="Unit-test tier profile for random-vs-random.",
        )

        result = run_tier_evaluation(
            tier_config=tier_config,
            candidate_id="random_candidate",
            seed=123,
            num_games_override=4,
        )

        assert result.tier_name == "TEST"
        assert result.total_games == 4
        assert len(result.matchups) == 1

        matchup = result.matchups[0]
        assert matchup.opponent_id == "random_baseline"
        assert matchup.games == 4
        assert (
            matchup.wins + matchup.losses + matchup.draws == matchup.games
        )
        assert matchup.average_game_length > 0.0

        # Victory reasons should be populated from RingRiftEnv.info
        assert matchup.victory_reasons
        assert sum(matchup.victory_reasons.values()) == matchup.games

        # Gating metrics and criteria should be present and JSON-serialisable
        data = result.to_dict()
        assert data["tier"] == "TEST"
        assert data["candidate"]["id"] == "random_candidate"
        assert "stats" in data
        assert "by_opponent" in data["stats"]
        assert "random_baseline" in data["stats"]["by_opponent"]
        assert (
            data["stats"]["by_opponent"]["random_baseline"]["games"] == 4
        )

        # With min_win_rate_vs_baseline == 0.0 this criterion should pass
        assert result.criteria["min_win_rate_vs_baseline"] is True
        # No previous-tier constraint configured for this synthetic tier
        assert (
            result.criteria["no_major_regression_vs_previous_tier"]
            is None
        )


class TestTierEvaluationCli:
    """CLI smoke test for run_tier_evaluation.py."""

    @pytest.mark.slow
    def test_run_tier_evaluation_cli_smoke(self, tmp_path) -> None:
        """Run a tiny D2 evaluation via the CLI and check JSON output."""
        output_path = tmp_path / "tier_eval_d2.json"
        cmd = [
            sys.executable,
            "scripts/run_tier_evaluation.py",
            "--tier",
            "D2",
            "--candidate-config",
            "baseline_smoke",
            "--num-games",
            "4",
            "--seed",
            "123",
            "--output-json",
            str(output_path),
        ]

        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        assert proc.returncode == 0, proc.stderr
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["tier"] == "D2"
        assert "overall_pass" in data
        assert isinstance(data["overall_pass"], bool)


def test_heuristic_tier_eval_smoke(monkeypatch) -> None:
    """Smoke test for heuristic tier eval harness on eval pools.

    Uses a stubbed load_state_pool so the test does not depend on on-disk
    JSONL eval pools and runs quickly.
    """
    # Ensure we have at least one heuristic tier defined.
    assert HEURISTIC_TIER_SPECS, "Expected at least one heuristic heuristic tier spec"
    tier = HEURISTIC_TIER_SPECS[0]

    # Stub load_state_pool to return a single valid GameState snapshot.
    from app.training import eval_pools as eval_pools_mod  # noqa: E402
    from app.training.generate_data import create_initial_state  # noqa: E402

    def _fake_load_state_pool(board_type, pool_id="v1", max_states=None, num_players=None):
        state = create_initial_state(
            board_type=board_type,
            num_players=num_players or tier.num_players,
        )
        return [state]

    monkeypatch.setattr(
        eval_pools_mod,
        "load_state_pool",
        _fake_load_state_pool,
    )

    # Run a tiny single-game eval for the first tier.
    result = run_heuristic_tier_eval(
        tier_spec=tier,
        rng_seed=123,
        max_games=1,
    )

    assert result["tier_id"] == tier.id
    assert result["games_played"] == 1
    assert set(result["results"].keys()) == {"wins", "losses", "draws"}
    assert "ring_margin_mean" in result["margins"]
    assert "territory_margin_mean" in result["margins"]
    assert "mean" in result["latency_ms"]
    assert "p95" in result["latency_ms"]

    # Also exercise the multi-tier wrapper with a single-tier filter so the
    # top-level report structure is validated.
    report = run_all_heuristic_tiers(
        tiers=HEURISTIC_TIER_SPECS,
        rng_seed=321,
        max_games=1,
        tier_ids=[tier.id],
    )
    assert "run_id" in report
    assert "timestamp" in report
    assert "tiers" in report
    assert report["tiers"]
    assert len(report["tiers"]) == 1
    tier_entry = report["tiers"][0]
    assert tier_entry["tier_id"] == tier.id