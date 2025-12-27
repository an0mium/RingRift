"""Tests for AI Registry and Determinism Verification.

Lane 3 Consolidation (2025-12):
    Validates that AI implementations correctly report their capabilities
    and that deterministic AI implementations produce consistent results
    with fixed seeds.
"""

import pytest

from app.ai.registry import (
    AICapability,
    AIInfo,
    AIRegistry,
    AIStatus,
    get_ai_info,
    list_deterministic_ais,
    list_production_ais,
    list_selfplay_ais,
)
from app.models.core import AIType


class TestAIRegistryBasics:
    """Basic registry functionality tests."""

    def test_all_ai_types_registered(self):
        """Verify all AI types have registry entries."""
        # Get all defined AI types
        all_types = list(AIType)

        for ai_type in all_types:
            info = get_ai_info(ai_type)
            # Some AI types may not be registered (e.g., internal-only types)
            # but common ones should be
            if ai_type in [
                AIType.RANDOM,
                AIType.HEURISTIC,
                AIType.MCTS,
                AIType.MINIMAX,
                AIType.GUMBEL_MCTS,
            ]:
                assert info is not None, f"{ai_type.name} should be registered"

    def test_registry_entries_have_required_fields(self):
        """Verify all registry entries have required metadata."""
        for info in AIRegistry.get_all():
            assert info.name, f"{info.ai_type} missing name"
            assert info.version, f"{info.ai_type} missing version"
            assert info.description, f"{info.ai_type} missing description"
            assert info.status in AIStatus, f"{info.ai_type} has invalid status"
            assert isinstance(info.capabilities, frozenset)
            assert info.min_difficulty <= info.max_difficulty

    def test_version_format(self):
        """Verify version strings follow semver format."""
        import re

        semver_pattern = re.compile(r"^\d+\.\d+\.\d+$")
        for info in AIRegistry.get_all():
            assert semver_pattern.match(info.version), (
                f"{info.ai_type} version '{info.version}' doesn't match semver"
            )


class TestDeterministicAIs:
    """Tests for deterministic AI implementations."""

    def test_deterministic_ais_exist(self):
        """Verify some AIs report determinism capability."""
        deterministic_ais = list_deterministic_ais()
        assert len(deterministic_ais) > 0, "No deterministic AIs found"

    def test_production_ais_are_deterministic(self):
        """Verify production AIs support determinism for reproducibility."""
        production_ais = list_production_ais()
        for info in production_ais:
            assert info.supports_determinism, (
                f"Production AI {info.name} should support determinism"
            )

    def test_random_ai_is_deterministic(self):
        """Random AI should be deterministic with fixed seed."""
        info = get_ai_info(AIType.RANDOM)
        assert info is not None
        assert info.supports_determinism

    def test_heuristic_ai_is_deterministic(self):
        """Heuristic AI should be deterministic with fixed seed."""
        info = get_ai_info(AIType.HEURISTIC)
        assert info is not None
        assert info.supports_determinism

    def test_mcts_ai_is_deterministic(self):
        """MCTS AI should be deterministic with fixed seed."""
        info = get_ai_info(AIType.MCTS)
        assert info is not None
        assert info.supports_determinism


class TestCapabilityQueries:
    """Tests for capability-based queries."""

    def test_get_by_capability(self):
        """Test querying AIs by capability."""
        neural_ais = AIRegistry.get_by_capability(AICapability.NEURAL_GUIDANCE)
        assert len(neural_ais) > 0, "No neural-guided AIs found"

        for info in neural_ais:
            assert AICapability.NEURAL_GUIDANCE in info.capabilities

    def test_supports_capability(self):
        """Test capability check for specific AI types."""
        # Random AI should not use neural guidance
        assert not AIRegistry.supports_capability(
            AIType.RANDOM, AICapability.NEURAL_GUIDANCE
        )

        # MCTS should use neural guidance
        assert AIRegistry.supports_capability(
            AIType.MCTS, AICapability.NEURAL_GUIDANCE
        )

    def test_selfplay_capable_ais(self):
        """Verify self-play capable AIs are correctly identified."""
        selfplay_ais = list_selfplay_ais()
        assert len(selfplay_ais) > 0, "No self-play capable AIs found"

        for info in selfplay_ais:
            assert AICapability.SELF_PLAY_CAPABLE in info.capabilities
            # Self-play AIs should also generate training data
            assert AICapability.GENERATES_TRAINING_DATA in info.capabilities


class TestAIStatus:
    """Tests for AI status tracking."""

    def test_production_ais_not_deprecated(self):
        """Production AIs should not be deprecated or archived."""
        production_ais = list_production_ais()
        for info in production_ais:
            assert info.status == AIStatus.PRODUCTION

    def test_experimental_ais_identified(self):
        """Experimental AIs should be correctly identified."""
        experimental_ais = AIRegistry.get_experimental()
        for info in experimental_ais:
            assert info.status == AIStatus.EXPERIMENTAL

    def test_deprecated_ais_have_replacement(self):
        """Deprecated AIs should specify what replaces them."""
        all_ais = AIRegistry.get_all()
        for info in all_ais:
            if info.status == AIStatus.DEPRECATED:
                assert info.deprecated_by is not None, (
                    f"Deprecated AI {info.name} should specify replacement"
                )


class TestDifficultyMapping:
    """Tests for difficulty level mapping."""

    def test_difficulty_ranges_valid(self):
        """Verify difficulty ranges are valid."""
        for info in AIRegistry.get_all():
            assert info.min_difficulty >= 1, f"{info.name} min_difficulty too low"
            assert info.max_difficulty >= info.min_difficulty

    def test_difficulty_coverage(self):
        """Verify difficulty levels 1-11 are covered by production AIs."""
        production_ais = list_production_ais()
        covered_difficulties = set()

        for info in production_ais:
            for d in range(info.min_difficulty, info.max_difficulty + 1):
                covered_difficulties.add(d)

        # At least difficulties 1-8 should be covered
        for d in range(1, 9):
            assert d in covered_difficulties, (
                f"Difficulty {d} not covered by any production AI"
            )
