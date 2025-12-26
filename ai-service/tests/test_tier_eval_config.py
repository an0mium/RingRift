"""Tests for tier evaluation configuration."""

from __future__ import annotations

import unittest

from app.training.tier_eval_config import (
    TIER_EVAL_CONFIGS,
    TierEvaluationConfig,
    TierOpponentConfig,
    get_tier_config,
)


class TestTierEvalConfig(unittest.TestCase):
    """Tests for tier evaluation configuration."""

    def test_all_tiers_d1_to_d11_exist(self) -> None:
        """Verify all difficulty tiers D1-D11 have configs."""
        for i in range(1, 12):
            tier_name = f"D{i}"
            config = get_tier_config(tier_name)
            self.assertEqual(config.tier_name, tier_name)
            self.assertEqual(config.candidate_difficulty, i)

    def test_baseline_thresholds_monotonic(self) -> None:
        """Verify baseline thresholds never decrease for higher tiers."""
        prev_threshold = 0.0
        for i in range(2, 12):  # D2-D11 (D1 has no baseline)
            config = get_tier_config(f"D{i}")
            threshold = config.min_win_rate_vs_baseline or 0.0
            self.assertGreaterEqual(
                threshold,
                prev_threshold,
                f"D{i} threshold {threshold} < D{i-1} threshold {prev_threshold}",
            )
            prev_threshold = threshold

    def test_all_tiers_have_previous_tier_opponent(self) -> None:
        """Verify D2+ tiers have a previous_tier opponent."""
        for i in range(2, 12):  # D2-D11
            config = get_tier_config(f"D{i}")
            prev_tier_opponents = [
                o for o in config.opponents if o.role == "previous_tier"
            ]
            self.assertEqual(
                len(prev_tier_opponents),
                1,
                f"D{i} should have exactly 1 previous_tier opponent",
            )
            # Verify it references the correct previous difficulty
            expected_prev_diff = i - 1
            self.assertEqual(
                prev_tier_opponents[0].difficulty,
                expected_prev_diff,
                f"D{i} previous_tier should reference D{i-1}",
            )

    def test_min_win_rate_vs_previous_tier_default(self) -> None:
        """Verify min_win_rate_vs_previous_tier defaults to 0.50."""
        for i in range(1, 12):
            config = get_tier_config(f"D{i}")
            self.assertEqual(
                config.min_win_rate_vs_previous_tier,
                0.50,
                f"D{i} should have min_win_rate_vs_previous_tier=0.50",
            )

    def test_d1_has_no_gating(self) -> None:
        """Verify D1 (entry tier) has no baseline gating."""
        config = get_tier_config("D1")
        self.assertIsNone(config.min_win_rate_vs_baseline)
        self.assertEqual(len(config.opponents), 0)

    def test_d7_plus_capped_at_75_percent(self) -> None:
        """Verify D7-D11 are capped at 75% baseline threshold."""
        for i in range(7, 12):
            config = get_tier_config(f"D{i}")
            self.assertLessEqual(
                config.min_win_rate_vs_baseline,
                0.75,
                f"D{i} should be capped at 75%",
            )

    def test_heuristic_baselines_at_d5_plus(self) -> None:
        """Verify D5+ tiers have heuristic baseline opponents."""
        for i in range(5, 12):
            config = get_tier_config(f"D{i}")
            heuristic_opponents = [
                o for o in config.opponents
                if o.role == "baseline" and o.ai_type is not None
                and o.ai_type.value == "heuristic"
            ]
            self.assertGreaterEqual(
                len(heuristic_opponents),
                1,
                f"D{i} should have at least 1 heuristic baseline",
            )

    def test_get_tier_config_case_insensitive(self) -> None:
        """Verify get_tier_config is case-insensitive."""
        config_upper = get_tier_config("D5")
        config_lower = get_tier_config("d5")
        self.assertEqual(config_upper.tier_name, config_lower.tier_name)

    def test_get_tier_config_invalid_raises(self) -> None:
        """Verify get_tier_config raises for invalid tier."""
        with self.assertRaises(KeyError):
            get_tier_config("D99")

    def test_threshold_progression(self) -> None:
        """Verify expected threshold progression."""
        expected = {
            "D1": None,
            "D2": 0.60,
            "D3": 0.55,
            "D4": 0.68,
            "D5": 0.60,
            "D6": 0.72,
            "D7": 0.65,
            "D8": 0.75,
            "D9": 0.75,
            "D10": 0.75,
            "D11": 0.75,
        }
        for tier, expected_threshold in expected.items():
            config = get_tier_config(tier)
            self.assertEqual(
                config.min_win_rate_vs_baseline,
                expected_threshold,
                f"{tier} threshold mismatch",
            )


if __name__ == "__main__":
    unittest.main()
