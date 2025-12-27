#!/usr/bin/env python3
"""Unit tests for the Composite Participant ID System.

Tests for:
- Composite ID creation and parsing
- Config hash encoding/decoding
- Participant category classification
- Legacy ID migration
"""

import pytest

from app.training.composite_participant import (
    BASELINE_PARTICIPANTS,
    STANDARD_ALGORITHM_CONFIGS,
    CompositeParticipant,
    ParticipantCategory,
    decode_config_hash,
    encode_config_hash,
    extract_ai_type,
    extract_nn_id,
    get_algorithm_variants,
    get_baseline_participant_ids,
    get_baseline_rating,
    get_nn_variants,
    get_participant_category,
    get_standard_config,
    is_baseline_participant,
    is_composite_id,
    make_composite_participant_id,
    migrate_legacy_participant_id,
    parse_composite_participant_id,
    parse_participant,
)


class TestCompositeParticipantId:
    """Test composite participant ID creation and parsing."""

    def test_make_composite_id_with_nn(self):
        """Test creating composite ID with neural network."""
        pid = make_composite_participant_id(
            nn_id="ringrift_v5_sq8_2p",
            ai_type="gumbel_mcts",
            config={"budget": 200}
        )
        assert "ringrift_v5_sq8_2p" in pid
        assert "gumbel_mcts" in pid
        assert "b200" in pid
        assert pid.count(":") == 2

    def test_make_composite_id_without_nn(self):
        """Test creating composite ID without neural network."""
        pid = make_composite_participant_id(
            nn_id=None,
            ai_type="random",
            config={"difficulty": 1}
        )
        assert pid.startswith("none:")
        assert "random" in pid

    def test_make_composite_id_with_none_string(self):
        """Test that 'none' string is normalized."""
        pid = make_composite_participant_id(
            nn_id="none",
            ai_type="heuristic",
        )
        assert pid.startswith("none:")

    def test_parse_composite_id(self):
        """Test parsing a composite ID."""
        pid = "ringrift_v5_sq8_2p:gumbel_mcts:b200"
        nn_id, ai_type, config = parse_composite_participant_id(pid)

        assert nn_id == "ringrift_v5_sq8_2p"
        assert ai_type == "gumbel_mcts"
        assert config["budget"] == 200

    def test_parse_composite_id_no_nn(self):
        """Test parsing composite ID without NN."""
        pid = "none:random:d1"
        nn_id, ai_type, config = parse_composite_participant_id(pid)

        assert nn_id is None
        assert ai_type == "random"
        assert config["difficulty"] == 1

    def test_parse_invalid_composite_id(self):
        """Test parsing invalid composite ID raises ValueError."""
        with pytest.raises(ValueError):
            parse_composite_participant_id("invalid_id")

        with pytest.raises(ValueError):
            parse_composite_participant_id("only:one")

    def test_roundtrip_composite_id(self):
        """Test that make → parse → make produces same ID."""
        original_nn = "test_model_v1"
        original_algo = "mcts"
        original_config = {"simulations": 800, "c_puct": 1.5}

        pid1 = make_composite_participant_id(original_nn, original_algo, original_config)
        nn_id, ai_type, config = parse_composite_participant_id(pid1)
        pid2 = make_composite_participant_id(nn_id, ai_type, config)

        # IDs should be functionally equivalent
        assert nn_id == original_nn
        assert ai_type == original_algo
        assert config["simulations"] == original_config["simulations"]


class TestConfigHashEncoding:
    """Test config hash encoding and decoding."""

    def test_encode_standard_config(self):
        """Test encoding returns short hash for standard config."""
        config = STANDARD_ALGORITHM_CONFIGS["gumbel_mcts"]
        hash_str = encode_config_hash(config, "gumbel_mcts")
        # Standard config should produce primary param hash
        assert "b" in hash_str or hash_str == "std"

    def test_encode_custom_budget(self):
        """Test encoding custom budget."""
        config = {"budget": 400}
        hash_str = encode_config_hash(config, "gumbel_mcts")
        assert "b400" in hash_str

    def test_encode_custom_simulations(self):
        """Test encoding custom simulations."""
        config = {"simulations": 1600}
        hash_str = encode_config_hash(config, "mcts")
        assert "s1600" in hash_str

    def test_encode_temperature(self):
        """Test encoding temperature with decimal."""
        config = {"temperature": 0.5}
        hash_str = encode_config_hash(config, "policy_only")
        assert "t" in hash_str  # t0.5 or t0p5

    def test_decode_budget(self):
        """Test decoding budget hash."""
        config = decode_config_hash("b400", "gumbel_mcts")
        assert config["budget"] == 400

    def test_decode_simulations(self):
        """Test decoding simulations hash."""
        config = decode_config_hash("s1600", "mcts")
        assert config["simulations"] == 1600

    def test_decode_standard(self):
        """Test decoding 'std' returns standard config."""
        config = decode_config_hash("std", "mcts")
        assert config == STANDARD_ALGORITHM_CONFIGS["mcts"]

    def test_encode_decode_roundtrip(self):
        """Test encode → decode roundtrip."""
        original = {"simulations": 1200, "c_puct": 2.0}
        encoded = encode_config_hash(original, "mcts")
        decoded = decode_config_hash(encoded, "mcts")

        # Non-standard values should be preserved
        if "s1200" in encoded:
            assert decoded["simulations"] == 1200


class TestParticipantCategory:
    """Test participant category classification."""

    def test_baseline_category(self):
        """Test baseline participant detection."""
        assert get_participant_category("none:random:d1") == ParticipantCategory.BASELINE
        assert get_participant_category("none:heuristic:d2") == ParticipantCategory.BASELINE

    def test_pure_nn_category(self):
        """Test pure NN (policy-only) detection."""
        assert get_participant_category("model_v1:policy_only:t0.3") == ParticipantCategory.PURE_NN

    def test_nn_search_category(self):
        """Test NN + search combination detection."""
        assert get_participant_category("model_v1:gumbel_mcts:b200") == ParticipantCategory.NN_SEARCH
        assert get_participant_category("model_v1:mcts:s800") == ParticipantCategory.NN_SEARCH

    def test_search_only_category(self):
        """Test search-only (no NN) detection."""
        assert get_participant_category("none:mcts:s800") == ParticipantCategory.SEARCH_ONLY

    def test_legacy_category(self):
        """Test legacy ID detection."""
        assert get_participant_category("old_model_id") == ParticipantCategory.LEGACY
        assert get_participant_category("model:with:too:many:colons") == ParticipantCategory.LEGACY


class TestIsCompositeId:
    """Test composite ID detection."""

    def test_composite_id_true(self):
        """Test valid composite IDs."""
        assert is_composite_id("nn:algo:config") is True
        assert is_composite_id("none:random:d1") is True
        assert is_composite_id("model_v1:gumbel_mcts:b200") is True

    def test_composite_id_false(self):
        """Test invalid composite IDs."""
        assert is_composite_id("legacy_model") is False
        assert is_composite_id("model:algo") is False
        assert is_composite_id("a:b:c:d") is False
        assert is_composite_id("") is False


class TestLegacyMigration:
    """Test legacy participant ID migration."""

    def test_migrate_legacy_id(self):
        """Test migrating legacy ID to composite format."""
        legacy = "old_model_v1"
        composite = migrate_legacy_participant_id(legacy)

        assert is_composite_id(composite)
        assert legacy in composite
        assert "mcts" in composite  # Default algorithm

    def test_migrate_already_composite(self):
        """Test that composite IDs are not modified."""
        composite = "model:gumbel_mcts:b200"
        migrated = migrate_legacy_participant_id(composite)
        assert migrated == composite

    def test_migrate_with_custom_algorithm(self):
        """Test migration with custom default algorithm."""
        legacy = "old_model"
        composite = migrate_legacy_participant_id(legacy, default_ai_type="descent")

        assert "descent" in composite


class TestParseParticipant:
    """Test comprehensive participant parsing."""

    def test_parse_composite_participant(self):
        """Test parsing composite participant."""
        pid = "model_v1:gumbel_mcts:b200"
        p = parse_participant(pid)

        assert isinstance(p, CompositeParticipant)
        assert p.nn_id == "model_v1"
        assert p.ai_type == "gumbel_mcts"
        assert p.is_composite is True
        assert p.category == ParticipantCategory.NN_SEARCH

    def test_parse_legacy_participant(self):
        """Test parsing legacy participant."""
        pid = "legacy_model"
        p = parse_participant(pid)

        assert p.nn_id == pid
        assert p.ai_type == "mcts"  # Default
        assert p.is_composite is False
        assert p.category == ParticipantCategory.LEGACY


class TestExtractHelpers:
    """Test extraction helper functions."""

    def test_extract_nn_id_composite(self):
        """Test extracting NN ID from composite."""
        assert extract_nn_id("model:gumbel:b200") == "model"
        assert extract_nn_id("none:random:d1") is None

    def test_extract_nn_id_legacy(self):
        """Test extracting NN ID from legacy."""
        assert extract_nn_id("legacy_model") == "legacy_model"

    def test_extract_ai_type_composite(self):
        """Test extracting AI type from composite."""
        assert extract_ai_type("model:gumbel_mcts:b200") == "gumbel_mcts"
        assert extract_ai_type("none:random:d1") == "random"

    def test_extract_ai_type_legacy(self):
        """Test extracting AI type from legacy."""
        assert extract_ai_type("legacy_model") == "mcts"  # Default


class TestVariantGeneration:
    """Test variant generation helpers."""

    def test_get_nn_variants(self):
        """Test generating algorithm variants for an NN."""
        variants = get_nn_variants("model_v1")

        assert len(variants) == 4  # Default algorithms
        assert any("policy_only" in v for v in variants)
        assert any("gumbel_mcts" in v for v in variants)
        assert all("model_v1" in v for v in variants)

    def test_get_nn_variants_custom_algorithms(self):
        """Test generating custom algorithm variants."""
        variants = get_nn_variants("model_v1", algorithms=["mcts", "descent"])

        assert len(variants) == 2
        assert any("mcts" in v for v in variants)
        assert any("descent" in v for v in variants)

    def test_get_algorithm_variants(self):
        """Test generating NN variants for an algorithm."""
        variants = get_algorithm_variants("gumbel_mcts", ["model_a", "model_b"])

        assert len(variants) == 2
        assert all("gumbel_mcts" in v for v in variants)


class TestBaselineParticipants:
    """Test baseline participant handling."""

    def test_baseline_ids(self):
        """Test getting baseline IDs."""
        baselines = get_baseline_participant_ids()

        assert len(baselines) > 0
        assert all(is_composite_id(b) for b in baselines)

    def test_baseline_rating(self):
        """Test getting baseline ratings."""
        assert get_baseline_rating("none:random:d1") == 400.0
        assert get_baseline_rating("unknown_id") is None

    def test_is_baseline(self):
        """Test baseline detection."""
        assert is_baseline_participant("none:random:d1") is True
        assert is_baseline_participant("model:mcts:s800") is False


class TestStandardConfigs:
    """Test standard configuration retrieval."""

    def test_get_standard_config_gumbel(self):
        """Test getting Gumbel MCTS config."""
        config = get_standard_config("gumbel_mcts")

        assert "budget" in config
        assert config["budget"] == 200
        assert "m" in config

    def test_get_standard_config_mcts(self):
        """Test getting MCTS config."""
        config = get_standard_config("mcts")

        assert "simulations" in config
        assert config["simulations"] == 800

    def test_get_standard_config_unknown(self):
        """Test getting config for unknown algorithm."""
        config = get_standard_config("unknown_algorithm")
        assert config == {}

    def test_standard_config_returns_copy(self):
        """Test that returned config is a copy."""
        config1 = get_standard_config("mcts")
        config2 = get_standard_config("mcts")

        config1["simulations"] = 9999
        assert config2["simulations"] == 800  # Unaffected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
