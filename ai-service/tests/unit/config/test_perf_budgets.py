"""Tests for app.config.perf_budgets module.

Tests performance budget configuration for tiered AI move latency.
"""

import pytest

from app.config.perf_budgets import (
    TierPerfBudget,
    get_tier_perf_budget,
    TIER_PERF_BUDGETS,
)
from app.models import BoardType


class TestTierPerfBudget:
    """Tests for TierPerfBudget dataclass."""

    def test_creation(self):
        """Test creating a TierPerfBudget."""
        budget = TierPerfBudget(
            tier_name="TEST_TIER",
            difficulty=5,
            board_type=BoardType.SQUARE8,
            num_players=2,
            max_avg_move_ms=500.0,
            max_p95_move_ms=750.0,
            notes="Test budget",
        )
        assert budget.tier_name == "TEST_TIER"
        assert budget.difficulty == 5
        assert budget.board_type == BoardType.SQUARE8
        assert budget.num_players == 2
        assert budget.max_avg_move_ms == 500.0
        assert budget.max_p95_move_ms == 750.0

    def test_frozen(self):
        """Test that TierPerfBudget is frozen (immutable)."""
        budget = TierPerfBudget(
            tier_name="TEST",
            difficulty=5,
            board_type=BoardType.SQUARE8,
            num_players=2,
            max_avg_move_ms=500.0,
            max_p95_move_ms=750.0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            budget.difficulty = 6  # type: ignore

    def test_default_notes(self):
        """Test default notes is empty string."""
        budget = TierPerfBudget(
            tier_name="TEST",
            difficulty=5,
            board_type=BoardType.SQUARE8,
            num_players=2,
            max_avg_move_ms=500.0,
            max_p95_move_ms=750.0,
        )
        assert budget.notes == ""


class TestTierPerfBudgets:
    """Tests for TIER_PERF_BUDGETS dictionary."""

    def test_budgets_exist(self):
        """Test that tier budgets are defined."""
        assert TIER_PERF_BUDGETS is not None
        assert len(TIER_PERF_BUDGETS) > 0

    def test_d4_budget_exists(self):
        """Test D4 tier budget exists."""
        assert "D4_SQ8_2P" in TIER_PERF_BUDGETS
        assert "D4" in TIER_PERF_BUDGETS  # Short alias

    def test_d6_budget_exists(self):
        """Test D6 tier budget exists."""
        assert "D6_SQ8_2P" in TIER_PERF_BUDGETS
        assert "D6" in TIER_PERF_BUDGETS  # Short alias

    def test_d8_budget_exists(self):
        """Test D8 tier budget exists."""
        assert "D8_SQ8_2P" in TIER_PERF_BUDGETS
        assert "D8" in TIER_PERF_BUDGETS  # Short alias

    def test_budgets_have_valid_times(self):
        """Test all budgets have positive time limits."""
        for name, budget in TIER_PERF_BUDGETS.items():
            assert budget.max_avg_move_ms > 0, f"{name} has invalid avg time"
            assert budget.max_p95_move_ms > 0, f"{name} has invalid p95 time"

    def test_p95_greater_than_avg(self):
        """Test p95 is greater than or equal to avg for all budgets."""
        for name, budget in TIER_PERF_BUDGETS.items():
            assert budget.max_p95_move_ms >= budget.max_avg_move_ms, \
                f"{name}: p95 ({budget.max_p95_move_ms}) < avg ({budget.max_avg_move_ms})"

    def test_higher_difficulty_may_have_higher_budget(self):
        """Test higher difficulty tiers may have higher time budgets."""
        d4 = TIER_PERF_BUDGETS.get("D4_SQ8_2P")
        d8 = TIER_PERF_BUDGETS.get("D8_SQ8_2P")
        if d4 and d8:
            # Higher difficulty may have longer think time
            assert d8.max_avg_move_ms >= d4.max_avg_move_ms * 0.5, \
                "D8 budget unexpectedly much lower than D4"


class TestGetTierPerfBudget:
    """Tests for get_tier_perf_budget function."""

    def test_get_existing_budget(self):
        """Test getting an existing budget."""
        budget = get_tier_perf_budget("D6_SQ8_2P")
        assert budget is not None
        assert budget.tier_name == "D6_SQ8_2P"

    def test_get_by_short_name(self):
        """Test getting budget by short difficulty name."""
        budget = get_tier_perf_budget("D6")
        assert budget is not None
        assert budget.difficulty == 6

    def test_case_insensitive_lookup(self):
        """Test case-insensitive budget lookup."""
        budget_upper = get_tier_perf_budget("D6_SQ8_2P")
        budget_lower = get_tier_perf_budget("d6_sq8_2p")
        assert budget_upper == budget_lower

    def test_get_nonexistent_raises_keyerror(self):
        """Test getting non-existent budget raises KeyError."""
        with pytest.raises(KeyError):
            get_tier_perf_budget("NONEXISTENT_TIER")

    def test_budget_has_correct_board_type(self):
        """Test returned budget has correct board type."""
        budget = get_tier_perf_budget("D6_SQ8_2P")
        assert budget.board_type == BoardType.SQUARE8

    def test_budget_has_correct_player_count(self):
        """Test returned budget has correct player count."""
        budget = get_tier_perf_budget("D6_SQ8_2P")
        assert budget.num_players == 2


class TestBudgetConsistency:
    """Tests for budget consistency with ladder config."""

    def test_budgets_align_with_ladder_difficulties(self):
        """Test budget difficulties match tier names."""
        for name, budget in TIER_PERF_BUDGETS.items():
            # Extract difficulty from name (e.g., "D6_SQ8_2P" -> 6 or "D6" -> 6)
            if name.startswith("D"):
                if "_" in name:
                    expected_diff = int(name[1:name.index("_")])
                else:
                    expected_diff = int(name[1:])
                assert budget.difficulty == expected_diff, \
                    f"{name}: difficulty {budget.difficulty} != expected {expected_diff}"

    def test_all_budgets_are_square8(self):
        """Test all current budgets are for Square8 board."""
        for name, budget in TIER_PERF_BUDGETS.items():
            assert budget.board_type == BoardType.SQUARE8, \
                f"{name} has wrong board type: {budget.board_type}"

    def test_all_budgets_are_2p(self):
        """Test all current budgets are for 2 players."""
        for name, budget in TIER_PERF_BUDGETS.items():
            assert budget.num_players == 2, \
                f"{name} has wrong player count: {budget.num_players}"
