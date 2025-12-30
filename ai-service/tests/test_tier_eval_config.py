"""Tests for tier evaluation configuration."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from app.models import AIType, BoardType
from app.training.tier_eval_config import (
    HEURISTIC_TIER_SPECS,
    TIER_EVAL_CONFIGS,
    HeuristicTierSpec,
    TierEvaluationConfig,
    TierOpponentConfig,
    TierRole,
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

    def test_baseline_thresholds_have_expected_pattern(self) -> None:
        """Verify baseline thresholds follow expected pattern.

        Note: Thresholds are NOT monotonic by design - they are tuned per tier.
        D2 and D3 have different calibration goals.
        """
        expected_thresholds = {
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
        for tier, expected in expected_thresholds.items():
            config = get_tier_config(tier)
            self.assertEqual(
                config.min_win_rate_vs_baseline,
                expected,
                f"{tier} threshold should be {expected}",
            )

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
        """Verify min_win_rate_vs_previous_tier defaults to 0.55.

        The canonical threshold requires D(n) to beat D(n-1) at 55%+ win rate.
        """
        for i in range(1, 12):
            config = get_tier_config(f"D{i}")
            self.assertEqual(
                config.min_win_rate_vs_previous_tier,
                0.55,
                f"D{i} should have min_win_rate_vs_previous_tier=0.55",
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


class TestTierOpponentConfig(unittest.TestCase):
    """Tests for TierOpponentConfig dataclass."""

    def test_required_fields(self) -> None:
        """Verify required fields are enforced."""
        config = TierOpponentConfig(
            id="test_opponent",
            description="Test opponent",
            difficulty=3,
        )
        self.assertEqual(config.id, "test_opponent")
        self.assertEqual(config.description, "Test opponent")
        self.assertEqual(config.difficulty, 3)

    def test_default_values(self) -> None:
        """Verify default values for optional fields."""
        config = TierOpponentConfig(
            id="test",
            description="test",
            difficulty=1,
        )
        self.assertIsNone(config.ai_type)
        self.assertEqual(config.role, "baseline")
        self.assertEqual(config.weight, 1.0)
        self.assertIsNone(config.games)

    def test_all_fields(self) -> None:
        """Verify all fields can be set."""
        config = TierOpponentConfig(
            id="full_config",
            description="Full configuration",
            difficulty=5,
            ai_type=AIType.HEURISTIC,
            role="previous_tier",
            weight=2.5,
            games=100,
        )
        self.assertEqual(config.id, "full_config")
        self.assertEqual(config.ai_type, AIType.HEURISTIC)
        self.assertEqual(config.role, "previous_tier")
        self.assertEqual(config.weight, 2.5)
        self.assertEqual(config.games, 100)

    def test_frozen_immutability(self) -> None:
        """Verify dataclass is frozen (immutable)."""
        config = TierOpponentConfig(
            id="test",
            description="test",
            difficulty=1,
        )
        with self.assertRaises(FrozenInstanceError):
            config.difficulty = 5  # type: ignore[misc]

    def test_valid_roles(self) -> None:
        """Verify all TierRole values can be used."""
        for role in ["baseline", "previous_tier", "peer", "other"]:
            config = TierOpponentConfig(
                id="test",
                description="test",
                difficulty=1,
                role=role,  # type: ignore[arg-type]
            )
            self.assertEqual(config.role, role)


class TestTierEvaluationConfig(unittest.TestCase):
    """Tests for TierEvaluationConfig dataclass."""

    def test_required_fields(self) -> None:
        """Verify required fields can be constructed."""
        config = TierEvaluationConfig(
            tier_name="TEST",
            display_name="Test Tier",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=100,
            candidate_difficulty=5,
            time_budget_ms=1000,
        )
        self.assertEqual(config.tier_name, "TEST")
        self.assertEqual(config.board_type, BoardType.SQUARE8)
        self.assertEqual(config.num_players, 2)

    def test_default_values(self) -> None:
        """Verify default values for optional fields."""
        config = TierEvaluationConfig(
            tier_name="TEST",
            display_name="Test",
            board_type=BoardType.HEX8,
            num_players=2,
            num_games=50,
            candidate_difficulty=3,
            time_budget_ms=None,
        )
        self.assertEqual(config.opponents, [])
        self.assertIsNone(config.min_win_rate_vs_baseline)
        self.assertEqual(config.min_win_rate_vs_previous_tier, 0.55)
        self.assertIsNone(config.max_regression_vs_previous_tier)
        self.assertEqual(config.promotion_confidence, 0.95)
        self.assertEqual(config.description, "")

    def test_with_opponents(self) -> None:
        """Verify config can include opponents."""
        opponent = TierOpponentConfig(
            id="random",
            description="Random baseline",
            difficulty=1,
            ai_type=AIType.RANDOM,
        )
        config = TierEvaluationConfig(
            tier_name="D2",
            display_name="D2 Test",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=100,
            candidate_difficulty=2,
            time_budget_ms=None,
            opponents=[opponent],
        )
        self.assertEqual(len(config.opponents), 1)
        self.assertEqual(config.opponents[0].ai_type, AIType.RANDOM)

    def test_frozen_immutability(self) -> None:
        """Verify dataclass is frozen (immutable)."""
        config = TierEvaluationConfig(
            tier_name="TEST",
            display_name="Test",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=100,
            candidate_difficulty=1,
            time_budget_ms=None,
        )
        with self.assertRaises(FrozenInstanceError):
            config.num_games = 200  # type: ignore[misc]


class TestMultiBoardTierConfigs(unittest.TestCase):
    """Tests for multi-board and multi-player tier configurations."""

    def test_square19_tiers_exist(self) -> None:
        """Verify square19 tier configs exist."""
        d2_sq19 = get_tier_config("D2_SQ19_2P")
        self.assertEqual(d2_sq19.board_type, BoardType.SQUARE19)
        self.assertEqual(d2_sq19.num_players, 2)
        self.assertEqual(d2_sq19.candidate_difficulty, 2)

        d4_sq19 = get_tier_config("D4_SQ19_2P")
        self.assertEqual(d4_sq19.board_type, BoardType.SQUARE19)
        self.assertEqual(d4_sq19.candidate_difficulty, 4)

    def test_hex8_tiers_d2_to_d10_exist(self) -> None:
        """Verify hex8 2-player tiers D2-D10 exist."""
        for i in range(2, 11):
            tier_name = f"D{i}_HEX8_2P"
            config = get_tier_config(tier_name)
            self.assertEqual(config.board_type, BoardType.HEX8)
            self.assertEqual(config.num_players, 2)
            self.assertEqual(config.candidate_difficulty, i)

    def test_hex8_tiers_have_previous_tier(self) -> None:
        """Verify hex8 tiers D3+ reference previous tier."""
        for i in range(3, 11):
            tier_name = f"D{i}_HEX8_2P"
            config = get_tier_config(tier_name)
            prev_tier_opponents = [
                o for o in config.opponents if o.role == "previous_tier"
            ]
            self.assertEqual(
                len(prev_tier_opponents),
                1,
                f"{tier_name} should have exactly 1 previous_tier opponent",
            )
            self.assertEqual(
                prev_tier_opponents[0].difficulty,
                i - 1,
                f"{tier_name} should reference D{i-1}",
            )

    def test_hexagonal_legacy_tiers_exist(self) -> None:
        """Verify legacy hexagonal (469 cells) tiers exist."""
        d2_hex = get_tier_config("D2_HEX_2P")
        self.assertEqual(d2_hex.board_type, BoardType.HEXAGONAL)
        self.assertEqual(d2_hex.num_players, 2)

        d4_hex = get_tier_config("D4_HEX_2P")
        self.assertEqual(d4_hex.board_type, BoardType.HEXAGONAL)

    def test_multiplayer_tiers_exist(self) -> None:
        """Verify 3-player and 4-player tier configs exist."""
        d2_sq8_3p = get_tier_config("D2_SQ8_3P")
        self.assertEqual(d2_sq8_3p.board_type, BoardType.SQUARE8)
        self.assertEqual(d2_sq8_3p.num_players, 3)

        d2_sq8_4p = get_tier_config("D2_SQ8_4P")
        self.assertEqual(d2_sq8_4p.board_type, BoardType.SQUARE8)
        self.assertEqual(d2_sq8_4p.num_players, 4)

    def test_multiplayer_win_rate_thresholds(self) -> None:
        """Verify multiplayer tiers have appropriate win rate thresholds."""
        d2_sq8_3p = get_tier_config("D2_SQ8_3P")
        self.assertEqual(d2_sq8_3p.min_win_rate_vs_baseline, 0.55)

        d2_sq8_4p = get_tier_config("D2_SQ8_4P")
        self.assertEqual(d2_sq8_4p.min_win_rate_vs_baseline, 0.55)


class TestHeuristicTierSpec(unittest.TestCase):
    """Tests for HeuristicTierSpec dataclass."""

    def test_heuristic_tier_specs_not_empty(self) -> None:
        """Verify HEURISTIC_TIER_SPECS is not empty."""
        self.assertGreater(len(HEURISTIC_TIER_SPECS), 0)

    def test_heuristic_tier_spec_fields(self) -> None:
        """Verify HeuristicTierSpec has expected fields."""
        spec = HEURISTIC_TIER_SPECS[0]
        self.assertIsInstance(spec.id, str)
        self.assertIsInstance(spec.name, str)
        self.assertIsInstance(spec.board_type, BoardType)
        self.assertIsInstance(spec.num_players, int)
        self.assertIsInstance(spec.eval_pool_id, str)
        self.assertIsInstance(spec.num_games, int)
        self.assertIsInstance(spec.candidate_profile_id, str)
        self.assertIsInstance(spec.baseline_profile_id, str)

    def test_heuristic_tier_spec_creation(self) -> None:
        """Verify HeuristicTierSpec can be created."""
        spec = HeuristicTierSpec(
            id="test_spec",
            name="Test Spec",
            board_type=BoardType.HEX8,
            num_players=2,
            eval_pool_id="v2",
            num_games=32,
            candidate_profile_id="cma_optimized",
            baseline_profile_id="balanced_v1",
            description="Test description",
        )
        self.assertEqual(spec.id, "test_spec")
        self.assertEqual(spec.board_type, BoardType.HEX8)
        self.assertEqual(spec.description, "Test description")

    def test_heuristic_tier_spec_frozen(self) -> None:
        """Verify HeuristicTierSpec is frozen (immutable)."""
        spec = HEURISTIC_TIER_SPECS[0]
        with self.assertRaises(FrozenInstanceError):
            spec.num_games = 999  # type: ignore[misc]


class TestTierConfigGlobalDict(unittest.TestCase):
    """Tests for the global TIER_EVAL_CONFIGS dictionary."""

    def test_config_count(self) -> None:
        """Verify expected number of tier configs exist."""
        # D1-D11 (11) + D2/D4_SQ19_2P (2) + D2-D10_HEX8_2P (9) + D2/D4_HEX_2P (2)
        # + D2_SQ8_3P + D2_SQ8_4P (2) = 26
        self.assertGreaterEqual(len(TIER_EVAL_CONFIGS), 26)

    def test_all_configs_have_required_fields(self) -> None:
        """Verify all configs have non-empty required fields."""
        for tier_name, config in TIER_EVAL_CONFIGS.items():
            self.assertTrue(
                len(config.tier_name) > 0,
                f"{tier_name} has empty tier_name",
            )
            self.assertTrue(
                len(config.display_name) > 0,
                f"{tier_name} has empty display_name",
            )
            self.assertIsNotNone(
                config.board_type,
                f"{tier_name} has None board_type",
            )
            self.assertGreater(
                config.num_players,
                0,
                f"{tier_name} has invalid num_players",
            )
            self.assertGreater(
                config.num_games,
                0,
                f"{tier_name} has invalid num_games",
            )

    def test_all_configs_have_valid_board_types(self) -> None:
        """Verify all configs use valid BoardType enum values."""
        valid_board_types = {BoardType.SQUARE8, BoardType.SQUARE19,
                           BoardType.HEX8, BoardType.HEXAGONAL}
        for tier_name, config in TIER_EVAL_CONFIGS.items():
            self.assertIn(
                config.board_type,
                valid_board_types,
                f"{tier_name} has invalid board_type",
            )

    def test_opponent_configs_are_valid(self) -> None:
        """Verify all opponent configs within tiers are valid."""
        for tier_name, config in TIER_EVAL_CONFIGS.items():
            for opp in config.opponents:
                self.assertTrue(
                    len(opp.id) > 0,
                    f"{tier_name} opponent has empty id",
                )
                self.assertGreaterEqual(
                    opp.difficulty,
                    1,
                    f"{tier_name} opponent has invalid difficulty",
                )
                self.assertIn(
                    opp.role,
                    ["baseline", "previous_tier", "peer", "other"],
                    f"{tier_name} opponent has invalid role",
                )


class TestPromotionConfidence(unittest.TestCase):
    """Tests for promotion_confidence field used in Wilson interval gating."""

    def test_default_promotion_confidence(self) -> None:
        """Verify default promotion_confidence is 0.95."""
        for config in TIER_EVAL_CONFIGS.values():
            self.assertEqual(
                config.promotion_confidence,
                0.95,
                f"{config.tier_name} has non-default promotion_confidence",
            )

    def test_custom_promotion_confidence(self) -> None:
        """Verify custom promotion_confidence can be set."""
        config = TierEvaluationConfig(
            tier_name="TEST",
            display_name="Test",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=100,
            candidate_difficulty=1,
            time_budget_ms=None,
            promotion_confidence=0.90,
        )
        self.assertEqual(config.promotion_confidence, 0.90)


if __name__ == "__main__":
    unittest.main()
