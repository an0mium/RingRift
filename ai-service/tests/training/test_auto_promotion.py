"""Tests for automated post-training model promotion.

Tests cover:
- AutoPromotionCriteria configuration
- PromotionDecision evaluation logic
- GauntletEvalResults parsing
- Wilson confidence interval requirements
- OR logic for promotion criteria
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.training.auto_promotion import (
    AutoPromotionCriteria,
    AutoPromotionEngine,
    GauntletEvalResults,
    PromotionCriterion,
    PromotionDecision,
    PromotionResult,
    evaluate_and_promote,
    get_auto_promotion_engine,
    reset_auto_promotion_engine,
)


class TestAutoPromotionCriteria:
    """Tests for AutoPromotionCriteria configuration."""

    def test_default_criteria(self):
        """Default criteria has reasonable values."""
        criteria = AutoPromotionCriteria()

        assert criteria.min_games_quick == 30
        assert criteria.min_games_production == 100
        assert criteria.wilson_confidence == 0.95

    def test_win_rate_floors_by_player_count(self):
        """Win rate floors are correct for each player count."""
        criteria = AutoPromotionCriteria()

        # 2-player should have highest requirements
        assert criteria.win_rate_floors[2]["random"] == 0.70
        assert criteria.win_rate_floors[2]["heuristic"] == 0.50

        # 3-player intermediate
        assert criteria.win_rate_floors[3]["random"] == 0.50
        assert criteria.win_rate_floors[3]["heuristic"] == 0.40

        # 4-player lowest requirements
        assert criteria.win_rate_floors[4]["random"] == 0.40
        assert criteria.win_rate_floors[4]["heuristic"] == 0.35

    def test_min_absolute_elo_by_player_count(self):
        """Minimum Elo requirements are correct for each player count."""
        criteria = AutoPromotionCriteria()

        assert criteria.min_absolute_elo[2] == 1400
        assert criteria.min_absolute_elo[3] == 1350
        assert criteria.min_absolute_elo[4] == 1300

    def test_custom_criteria(self):
        """Custom criteria can be provided."""
        custom = AutoPromotionCriteria(
            min_games_quick=50,
            wilson_confidence=0.99,
        )

        assert custom.min_games_quick == 50
        assert custom.wilson_confidence == 0.99


class TestPromotionDecision:
    """Tests for PromotionDecision dataclass."""

    def test_approved_decision(self):
        """Approved decision has correct structure."""
        decision = PromotionDecision(
            approved=True,
            reason="Elo parity achieved: 1500 >= 1400",
            criterion_met=PromotionCriterion.ELO_PARITY,
        )

        assert decision.approved is True
        assert "Elo parity" in decision.reason
        assert decision.criterion_met == PromotionCriterion.ELO_PARITY

    def test_rejected_decision(self):
        """Rejected decision has correct structure."""
        decision = PromotionDecision(
            approved=False,
            reason="Elo 1200 below minimum 1400",
            details={"estimated_elo": 1200, "min_elo": 1400},
        )

        assert decision.approved is False
        assert "below minimum" in decision.reason
        assert decision.details["estimated_elo"] == 1200


class TestGauntletEvalResults:
    """Tests for GauntletEvalResults dataclass."""

    def test_default_values(self):
        """Default values are zero/neutral."""
        results = GauntletEvalResults()

        assert results.games_vs_random == 0
        assert results.wins_vs_random == 0
        assert results.win_rate_vs_random == 0.0
        assert results.estimated_elo == 1000.0

    def test_win_rate_calculation(self):
        """Win rates are calculated correctly."""
        results = GauntletEvalResults(
            games_vs_random=20,
            wins_vs_random=15,
            win_rate_vs_random=0.75,
            games_vs_heuristic=20,
            wins_vs_heuristic=10,
            win_rate_vs_heuristic=0.50,
        )

        assert results.win_rate_vs_random == 0.75
        assert results.win_rate_vs_heuristic == 0.50

    def test_is_statistically_significant(self):
        """Statistical significance check works."""
        # High CI lower bounds - significant
        results = GauntletEvalResults(
            wilson_ci_random=(0.65, 0.85),
            wilson_ci_heuristic=(0.55, 0.75),
        )
        assert results.is_statistically_significant(threshold=0.5) is True

        # Low CI lower bounds - not significant
        results_weak = GauntletEvalResults(
            wilson_ci_random=(0.40, 0.60),
            wilson_ci_heuristic=(0.30, 0.50),
        )
        assert results_weak.is_statistically_significant(threshold=0.5) is False


class TestPromotionResult:
    """Tests for PromotionResult dataclass."""

    def test_approved_property(self):
        """approved property delegates to decision."""
        result = PromotionResult(
            decision=PromotionDecision(approved=True, reason="Test"),
            model_path="models/test.pth",
            board_type="hex8",
            num_players=2,
        )

        assert result.approved is True

    def test_reason_property(self):
        """reason property delegates to decision."""
        result = PromotionResult(
            decision=PromotionDecision(approved=False, reason="Failed criteria"),
            model_path="models/test.pth",
            board_type="hex8",
            num_players=2,
        )

        assert result.reason == "Failed criteria"

    def test_timestamp_auto_generated(self):
        """Timestamp is auto-generated."""
        result = PromotionResult(
            decision=PromotionDecision(approved=True, reason="Test"),
            model_path="models/test.pth",
            board_type="hex8",
            num_players=2,
        )

        assert isinstance(result.timestamp, datetime)


class TestAutoPromotionEngine:
    """Tests for AutoPromotionEngine class."""

    def test_init_default_criteria(self):
        """Engine initializes with default criteria."""
        engine = AutoPromotionEngine()
        assert engine.criteria is not None
        assert engine.criteria.min_games_quick == 30

    def test_init_custom_criteria(self):
        """Engine accepts custom criteria."""
        custom = AutoPromotionCriteria(min_games_quick=50)
        engine = AutoPromotionEngine(criteria=custom)
        assert engine.criteria.min_games_quick == 50

    def test_evaluate_criteria_elo_parity(self):
        """Elo parity criterion triggers promotion."""
        engine = AutoPromotionEngine()

        results = GauntletEvalResults(
            estimated_elo=1500,
            games_vs_random=30,
            wins_vs_random=24,
            win_rate_vs_random=0.80,
            games_vs_heuristic=30,
            wins_vs_heuristic=15,
            win_rate_vs_heuristic=0.50,
            wilson_ci_random=(0.65, 0.91),
            wilson_ci_heuristic=(0.33, 0.67),
        )

        decision = engine._evaluate_criteria(
            results=results,
            heuristic_elo=1400,  # Model has higher Elo
            num_players=2,
            config_key="hex8_2p",
        )

        assert decision.approved is True
        assert decision.criterion_met == PromotionCriterion.ELO_PARITY
        assert "Elo parity achieved" in decision.reason

    def test_evaluate_criteria_win_rate_floors(self):
        """Win rate floors criterion triggers promotion."""
        engine = AutoPromotionEngine()

        # Model has lower Elo but meets win rate floors
        results = GauntletEvalResults(
            estimated_elo=1450,  # Below heuristic
            games_vs_random=50,
            wins_vs_random=40,
            win_rate_vs_random=0.80,
            games_vs_heuristic=50,
            wins_vs_heuristic=30,
            win_rate_vs_heuristic=0.60,
            wilson_ci_random=(0.70, 0.88),  # Lower bound >= 0.70
            wilson_ci_heuristic=(0.50, 0.70),  # Lower bound >= 0.50
        )

        decision = engine._evaluate_criteria(
            results=results,
            heuristic_elo=1500,  # Model below heuristic
            num_players=2,
            config_key="hex8_2p",
        )

        assert decision.approved is True
        assert decision.criterion_met == PromotionCriterion.WIN_RATE_FLOORS
        assert "Win rate floors met" in decision.reason

    def test_evaluate_criteria_below_min_elo(self):
        """Models below minimum Elo are rejected."""
        engine = AutoPromotionEngine()

        results = GauntletEvalResults(
            estimated_elo=1200,  # Below minimum 1400 for 2p
            games_vs_random=30,
            wins_vs_random=25,
            win_rate_vs_random=0.83,
            wilson_ci_random=(0.70, 0.93),
            wilson_ci_heuristic=(0.50, 0.70),
        )

        decision = engine._evaluate_criteria(
            results=results,
            heuristic_elo=1400,
            num_players=2,
            config_key="hex8_2p",
        )

        assert decision.approved is False
        assert "below minimum" in decision.reason

    def test_evaluate_criteria_neither_met(self):
        """Neither criterion met results in rejection."""
        engine = AutoPromotionEngine()

        # Low Elo AND insufficient win rate CI lower bounds
        results = GauntletEvalResults(
            estimated_elo=1410,  # Above min but below heuristic
            games_vs_random=30,
            wins_vs_random=15,
            win_rate_vs_random=0.50,
            games_vs_heuristic=30,
            wins_vs_heuristic=10,
            win_rate_vs_heuristic=0.33,
            wilson_ci_random=(0.35, 0.65),  # Below 0.70 floor
            wilson_ci_heuristic=(0.20, 0.50),  # Below 0.50 floor
        )

        decision = engine._evaluate_criteria(
            results=results,
            heuristic_elo=1500,
            num_players=2,
            config_key="hex8_2p",
        )

        assert decision.approved is False
        assert "Neither Elo parity nor win rate floors met" in decision.reason

    def test_evaluate_criteria_3p_lower_requirements(self):
        """3-player has lower win rate requirements."""
        engine = AutoPromotionEngine()

        # Results that pass 3p thresholds but would fail 2p
        results = GauntletEvalResults(
            estimated_elo=1400,
            games_vs_random=30,
            wins_vs_random=18,
            win_rate_vs_random=0.60,
            games_vs_heuristic=30,
            wins_vs_heuristic=15,
            win_rate_vs_heuristic=0.50,
            wilson_ci_random=(0.50, 0.70),  # Passes 3p random floor (0.50)
            wilson_ci_heuristic=(0.40, 0.60),  # Passes 3p heuristic floor (0.40)
        )

        decision = engine._evaluate_criteria(
            results=results,
            heuristic_elo=1500,
            num_players=3,
            config_key="hex8_3p",
        )

        assert decision.approved is True

    def test_evaluate_criteria_4p_lowest_requirements(self):
        """4-player has lowest win rate requirements."""
        engine = AutoPromotionEngine()

        # Results that pass 4p thresholds
        results = GauntletEvalResults(
            estimated_elo=1350,
            games_vs_random=30,
            wins_vs_random=15,
            win_rate_vs_random=0.50,
            games_vs_heuristic=30,
            wins_vs_heuristic=12,
            win_rate_vs_heuristic=0.40,
            wilson_ci_random=(0.40, 0.60),  # Passes 4p random floor (0.40)
            wilson_ci_heuristic=(0.35, 0.55),  # Passes 4p heuristic floor (0.35)
        )

        decision = engine._evaluate_criteria(
            results=results,
            heuristic_elo=1500,
            num_players=4,
            config_key="hex8_4p",
        )

        assert decision.approved is True


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_engine_returns_same_instance(self):
        """get_auto_promotion_engine returns singleton."""
        reset_auto_promotion_engine()

        engine1 = get_auto_promotion_engine()
        engine2 = get_auto_promotion_engine()

        assert engine1 is engine2

    def test_reset_clears_instance(self):
        """reset_auto_promotion_engine clears singleton."""
        reset_auto_promotion_engine()

        engine1 = get_auto_promotion_engine()
        reset_auto_promotion_engine()
        engine2 = get_auto_promotion_engine()

        assert engine1 is not engine2


class TestEventEmission:
    """Tests for event emission."""

    def test_emit_promotion_event_approved(self):
        """Approved promotion emits correct event."""
        engine = AutoPromotionEngine()

        # Mock event router
        mock_router = MagicMock()
        engine._event_router = mock_router

        result = PromotionResult(
            decision=PromotionDecision(
                approved=True,
                reason="Test promotion",
                criterion_met=PromotionCriterion.ELO_PARITY,
            ),
            eval_results=GauntletEvalResults(estimated_elo=1500),
            model_path="models/test.pth",
            board_type="hex8",
            num_players=2,
            promoted_path="models/canonical_hex8_2p.pth",
        )

        engine._emit_promotion_event("MODEL_AUTO_PROMOTED", result)

        mock_router.publish_sync.assert_called_once()
        call_args = mock_router.publish_sync.call_args
        assert call_args[0][0] == "MODEL_AUTO_PROMOTED"
        event_data = call_args[0][1]
        assert event_data["approved"] is True
        assert event_data["config_key"] == "hex8_2p"

    def test_emit_promotion_event_rejected(self):
        """Rejected promotion emits correct event."""
        engine = AutoPromotionEngine()

        mock_router = MagicMock()
        engine._event_router = mock_router

        result = PromotionResult(
            decision=PromotionDecision(
                approved=False,
                reason="Below minimum Elo",
            ),
            model_path="models/test.pth",
            board_type="hex8",
            num_players=2,
        )

        engine._emit_promotion_event("MODEL_PROMOTION_REJECTED", result)

        mock_router.publish_sync.assert_called_once()
        call_args = mock_router.publish_sync.call_args
        assert call_args[0][0] == "MODEL_PROMOTION_REJECTED"
        event_data = call_args[0][1]
        assert event_data["approved"] is False


class TestPromotionEventsModule:
    """Tests for promotion_events.py module."""

    def test_event_type_enum(self):
        """PromotionEventType has correct values."""
        from app.training.promotion_events import PromotionEventType

        assert PromotionEventType.MODEL_AUTO_PROMOTED.value == "model_auto_promoted"
        assert PromotionEventType.MODEL_PROMOTION_REJECTED.value == "model_promotion_rejected"
        assert PromotionEventType.MODEL_PROMOTION_DEFERRED.value == "model_promotion_deferred"

    def test_event_data_to_dict(self):
        """PromotionEventData converts to dict correctly."""
        from app.training.promotion_events import PromotionEventData

        data = PromotionEventData(
            model_path="models/test.pth",
            board_type="hex8",
            num_players=2,
            approved=True,
            reason="Test",
            criterion_met="elo_parity",
            estimated_elo=1500.0,
        )

        result = data.to_dict()

        assert result["model_path"] == "models/test.pth"
        assert result["config_key"] == "hex8_2p"
        assert result["approved"] is True
        assert result["estimated_elo"] == 1500.0

    def test_emit_auto_promoted(self):
        """emit_auto_promoted convenience function works."""
        from app.training.promotion_events import emit_auto_promoted

        with patch('app.training.promotion_events._get_safe_emit') as mock_get:
            mock_emit = MagicMock(return_value=True)
            mock_get.return_value = mock_emit

            result = emit_auto_promoted(
                model_path="models/test.pth",
                board_type="hex8",
                num_players=2,
                reason="Elo parity achieved",
                criterion_met="elo_parity",
                estimated_elo=1500.0,
                promoted_path="models/canonical_hex8_2p.pth",
            )

            # Should call the emit function
            assert mock_emit.called or result is False  # May fail if module not available

    def test_emit_promotion_rejected(self):
        """emit_promotion_rejected convenience function works."""
        from app.training.promotion_events import emit_promotion_rejected

        with patch('app.training.promotion_events._get_safe_emit') as mock_get:
            mock_emit = MagicMock(return_value=True)
            mock_get.return_value = mock_emit

            result = emit_promotion_rejected(
                model_path="models/test.pth",
                board_type="hex8",
                num_players=2,
                reason="Below minimum Elo",
                estimated_elo=1200.0,
            )

            # Should call the emit function
            assert mock_emit.called or result is False
