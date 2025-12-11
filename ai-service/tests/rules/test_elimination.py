"""
Tests for EliminationAggregate - Python parity with TypeScript implementation.

Mirrors: tests/unit/engine/EliminationAggregate.test.ts
"""

import pytest
from app.rules.elimination import (
    EliminationContext,
    EliminationReason,
    calculate_cap_height,
    is_stack_eligible_for_elimination,
    get_rings_to_eliminate,
    eliminate_from_stack,
    enumerate_eligible_stacks,
    has_eligible_elimination_target,
)


# =============================================================================
# CALCULATE CAP HEIGHT
# =============================================================================


class TestCalculateCapHeight:
    """Test cap height calculation."""

    def test_returns_0_for_empty_array(self):
        assert calculate_cap_height([]) == 0

    def test_returns_1_for_single_ring(self):
        assert calculate_cap_height([1]) == 1

    def test_returns_full_height_for_uniform_stack(self):
        assert calculate_cap_height([1, 1, 1]) == 3

    def test_returns_cap_height_for_multicolor_stack(self):
        # Note: Python uses bottom-to-top (index 0 = bottom, -1 = top)
        # TypeScript uses top-to-bottom (index 0 = top)
        assert calculate_cap_height([2, 1, 1]) == 2  # [bottom, ..., top]
        assert calculate_cap_height([2, 2, 1]) == 1
        assert calculate_cap_height([1, 1, 2, 2]) == 2


# =============================================================================
# GET RINGS TO ELIMINATE
# =============================================================================


class TestGetRingsToEliminate:
    """Test ring elimination count calculation."""

    def test_returns_1_for_line_context(self):
        rings_1 = [1]
        rings_3 = [1, 1, 1]
        multicolor = [2, 1, 1]

        assert get_rings_to_eliminate(rings_1, EliminationContext.LINE) == 1
        assert get_rings_to_eliminate(rings_3, EliminationContext.LINE) == 1
        assert get_rings_to_eliminate(multicolor, EliminationContext.LINE) == 1

    def test_returns_cap_height_for_territory_context(self):
        rings_1 = [1]
        rings_3 = [1, 1, 1]
        multicolor = [2, 1, 1]  # cap = 2

        assert get_rings_to_eliminate(rings_1, EliminationContext.TERRITORY) == 1
        assert get_rings_to_eliminate(rings_3, EliminationContext.TERRITORY) == 3
        assert get_rings_to_eliminate(multicolor, EliminationContext.TERRITORY) == 2

    def test_returns_cap_height_for_forced_context(self):
        rings_1 = [1]
        rings_3 = [1, 1, 1]

        assert get_rings_to_eliminate(rings_1, EliminationContext.FORCED) == 1
        assert get_rings_to_eliminate(rings_3, EliminationContext.FORCED) == 3

    def test_returns_1_for_recovery_context(self):
        rings_3 = [2, 1, 1]
        assert get_rings_to_eliminate(rings_3, EliminationContext.RECOVERY) == 1


# =============================================================================
# STACK ELIGIBILITY
# =============================================================================


class TestIsStackEligibleForElimination:
    """Test stack eligibility checking."""

    class TestLineContext:
        """Tests for line context (RR-CANON-R122)."""

        def test_allows_any_controlled_stack_including_height_1(self):
            height_1 = [1]
            height_3 = [1, 1, 1]
            multicolor = [2, 1, 1]

            assert is_stack_eligible_for_elimination(height_1, 1, EliminationContext.LINE, 1).eligible
            assert is_stack_eligible_for_elimination(height_3, 1, EliminationContext.LINE, 1).eligible
            assert is_stack_eligible_for_elimination(multicolor, 1, EliminationContext.LINE, 1).eligible

        def test_rejects_stacks_not_controlled_by_player(self):
            opponent_stack = [1, 2, 2]  # P2 controls (top = 2)
            result = is_stack_eligible_for_elimination(opponent_stack, 2, EliminationContext.LINE, 1)
            assert not result.eligible

    class TestTerritoryContext:
        """Tests for territory context (RR-CANON-R145)."""

        def test_rejects_height_1_standalone_stacks(self):
            height_1 = [1]
            result = is_stack_eligible_for_elimination(
                height_1, 1, EliminationContext.TERRITORY, 1
            )
            assert not result.eligible
            assert "height-1" in result.reason.lower()

        def test_allows_multicolor_stacks(self):
            multicolor = [2, 1, 1]  # P1 controls with P2 buried
            result = is_stack_eligible_for_elimination(
                multicolor, 1, EliminationContext.TERRITORY, 1
            )
            assert result.eligible

        def test_allows_single_color_stacks_with_height_greater_than_1(self):
            height_2 = [1, 1]
            height_3 = [1, 1, 1]
            assert is_stack_eligible_for_elimination(
                height_2, 1, EliminationContext.TERRITORY, 1
            ).eligible
            assert is_stack_eligible_for_elimination(
                height_3, 1, EliminationContext.TERRITORY, 1
            ).eligible

        def test_rejects_stacks_not_controlled_by_player(self):
            opponent_multicolor = [1, 2, 2]  # P2 controls
            result = is_stack_eligible_for_elimination(
                opponent_multicolor, 2, EliminationContext.TERRITORY, 1
            )
            assert not result.eligible

    class TestForcedContext:
        """Tests for forced context (RR-CANON-R100)."""

        def test_allows_any_controlled_stack_including_height_1(self):
            height_1 = [1]
            multicolor = [2, 1, 1]

            assert is_stack_eligible_for_elimination(
                height_1, 1, EliminationContext.FORCED, 1
            ).eligible
            assert is_stack_eligible_for_elimination(
                multicolor, 1, EliminationContext.FORCED, 1
            ).eligible

    class TestRecoveryContext:
        """Tests for recovery context (RR-CANON-R113)."""

        def test_allows_stacks_with_buried_rings_of_player(self):
            # P1 ring buried, P2 controls
            buried_p1 = [1, 2, 2]  # P1 at bottom, P2 controls
            result = is_stack_eligible_for_elimination(
                buried_p1, 2, EliminationContext.RECOVERY, 1
            )
            assert result.eligible

        def test_rejects_stacks_without_buried_rings_of_player(self):
            # P1 only on top
            no_buried = [2, 2, 1]  # P2 buried, P1 controls
            result = is_stack_eligible_for_elimination(
                no_buried, 1, EliminationContext.RECOVERY, 1
            )
            assert not result.eligible
            assert "buried" in result.reason.lower()

        def test_does_not_require_control_for_recovery(self):
            # Player 1 can extract buried ring even if P2 controls
            buried_p1 = [1, 2, 2]  # P1 buried at bottom
            result = is_stack_eligible_for_elimination(
                buried_p1, 2, EliminationContext.RECOVERY, 1
            )
            assert result.eligible


# =============================================================================
# ENUMERATE ELIGIBLE STACKS
# =============================================================================


class TestEnumerateEligibleStacks:
    """Test eligible stack enumeration."""

    def test_filters_by_context_correctly(self):
        stacks = {
            (0, 0): {"rings": [1], "controlling_player": 1},  # height-1 P1
            (1, 0): {"rings": [1, 1], "controlling_player": 1},  # height-2 P1
            (2, 0): {"rings": [2, 1, 1], "controlling_player": 1},  # multicolor P1
            (3, 0): {"rings": [2], "controlling_player": 2},  # height-1 P2
        }

        # Line: all P1 stacks (3)
        line_eligible = enumerate_eligible_stacks(stacks, 1, EliminationContext.LINE)
        assert len(line_eligible) == 3

        # Territory: P1 stacks excluding height-1 (2)
        territory_eligible = enumerate_eligible_stacks(stacks, 1, EliminationContext.TERRITORY)
        assert len(territory_eligible) == 2
        assert (0, 0) not in territory_eligible

    def test_respects_exclude_positions(self):
        stacks = {
            (0, 0): {"rings": [1, 1], "controlling_player": 1},
            (1, 0): {"rings": [1, 1], "controlling_player": 1},
        }

        excluded = {(0, 0)}
        eligible = enumerate_eligible_stacks(stacks, 1, EliminationContext.TERRITORY, excluded)
        assert len(eligible) == 1
        assert (1, 0) in eligible


# =============================================================================
# HAS ELIGIBLE ELIMINATION TARGET
# =============================================================================


class TestHasEligibleEliminationTarget:
    """Test eligible target checking."""

    def test_returns_true_when_eligible_stacks_exist(self):
        stacks = {(0, 0): {"rings": [1, 1], "controlling_player": 1}}
        assert has_eligible_elimination_target(stacks, 1, EliminationContext.TERRITORY)

    def test_returns_false_when_no_eligible_stacks_exist(self):
        stacks = {(0, 0): {"rings": [1], "controlling_player": 1}}  # height-1
        assert not has_eligible_elimination_target(stacks, 1, EliminationContext.TERRITORY)

    def test_respects_exclude_positions(self):
        stacks = {(0, 0): {"rings": [1, 1], "controlling_player": 1}}
        excluded = {(0, 0)}
        assert not has_eligible_elimination_target(
            stacks, 1, EliminationContext.TERRITORY, excluded
        )


# =============================================================================
# ELIMINATE FROM STACK
# =============================================================================


class TestEliminateFromStack:
    """Test elimination execution."""

    class TestLineContext:
        """Tests for line elimination."""

        def test_eliminates_exactly_1_ring_from_top(self):
            rings = [1, 1, 1]

            result = eliminate_from_stack(
                rings=rings,
                controlling_player=1,
                context=EliminationContext.LINE,
                player=1,
                reason=EliminationReason.LINE_REWARD_EXACT,
            )

            assert result.success
            assert result.rings_eliminated == 1
            assert result.updated_stack == [1, 1]

    class TestTerritoryContext:
        """Tests for territory elimination."""

        def test_eliminates_entire_cap(self):
            rings = [2, 1, 1]  # cap = 2

            result = eliminate_from_stack(
                rings=rings,
                controlling_player=1,
                context=EliminationContext.TERRITORY,
                player=1,
                reason=EliminationReason.TERRITORY_SELF_ELIMINATION,
            )

            assert result.success
            assert result.rings_eliminated == 2
            assert result.updated_stack == [2]

        def test_rejects_height_1_stacks(self):
            rings = [1]

            result = eliminate_from_stack(
                rings=rings,
                controlling_player=1,
                context=EliminationContext.TERRITORY,
                player=1,
            )

            assert not result.success
            assert "height-1" in result.error.lower()

    class TestForcedContext:
        """Tests for forced elimination."""

        def test_eliminates_entire_cap_including_height_1(self):
            rings = [1]

            result = eliminate_from_stack(
                rings=rings,
                controlling_player=1,
                context=EliminationContext.FORCED,
                player=1,
                reason=EliminationReason.FORCED_ELIMINATION_ANM,
            )

            assert result.success
            assert result.rings_eliminated == 1
            assert result.updated_stack is None  # Stack removed

    class TestRecoveryContext:
        """Tests for recovery elimination."""

        def test_extracts_bottommost_buried_ring(self):
            # Stack: [2, 1, 1, 2] - bottom to top
            # P1 rings at index 1 and 2, should extract bottommost (index 1)
            rings = [2, 1, 1, 2]

            result = eliminate_from_stack(
                rings=rings,
                controlling_player=2,
                context=EliminationContext.RECOVERY,
                player=1,
                reason=EliminationReason.RECOVERY_BURIED_EXTRACTION,
            )

            assert result.success
            assert result.rings_eliminated == 1
            # One P1 ring should remain
            assert result.updated_stack.count(1) == 1

        def test_fails_when_no_buried_rings_of_player_exist(self):
            rings = [2, 2, 1]  # P1 only on top

            result = eliminate_from_stack(
                rings=rings,
                controlling_player=1,
                context=EliminationContext.RECOVERY,
                player=1,
            )

            assert not result.success
            assert "buried" in result.error.lower()

    class TestErrorHandling:
        """Tests for error cases."""

        def test_fails_for_empty_stack(self):
            result = eliminate_from_stack(
                rings=[],
                controlling_player=1,
                context=EliminationContext.LINE,
                player=1,
            )

            assert not result.success
            assert "no rings" in result.error.lower()

        def test_fails_when_player_does_not_control_stack(self):
            rings = [1, 2, 2]  # P2 controls

            result = eliminate_from_stack(
                rings=rings,
                controlling_player=2,
                context=EliminationContext.LINE,
                player=1,
            )

            assert not result.success
            assert "does not control" in result.error.lower()

    class TestAuditEvents:
        """Tests for audit event generation."""

        def test_includes_audit_event_in_result(self):
            rings = [1, 1]

            result = eliminate_from_stack(
                rings=rings,
                controlling_player=1,
                context=EliminationContext.TERRITORY,
                player=1,
                reason=EliminationReason.TERRITORY_SELF_ELIMINATION,
                stack_position=(0, 0),
            )

            assert result.audit_event is not None
            assert result.audit_event.context == EliminationContext.TERRITORY
            assert result.audit_event.reason == EliminationReason.TERRITORY_SELF_ELIMINATION
            assert result.audit_event.rings_eliminated == 2
            assert result.audit_event.stack_height_before == 2
            assert result.audit_event.stack_height_after == 0
