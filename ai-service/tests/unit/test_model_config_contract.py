"""
Tests for Model Configuration Contract System

Tests cover:
- Contract creation for different board types and player counts
- Model validation (value head, policy size)
- Checkpoint metadata validation
- Pre-save validation
- Pre-promotion validation
- Error handling for invalid configurations

This module ensures that misconfigured models are caught before
they are saved or promoted to production.
"""

import pytest
import torch
import torch.nn as nn

from app.models import BoardType
from app.ai.neural_net.constants import BOARD_POLICY_SIZES
from app.training.model_config_contract import (
    ModelConfigContract,
    ModelConfigError,
    validate_model_for_save,
    validate_checkpoint_for_promotion,
    get_canonical_model_name,
)


class MockModel(nn.Module):
    """Mock model with configurable value and policy head dimensions."""

    def __init__(
        self,
        value_outputs: int = 2,
        policy_size: int = 7000,
    ):
        super().__init__()
        # Value head
        self.value_fc1 = nn.Linear(128, 128)
        self.value_fc2 = nn.Linear(128, value_outputs)  # Output dimension = num_players

        # Policy head with fc layer
        self.policy_head = nn.Module()
        self.policy_head.fc = nn.Linear(128, policy_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_head.fc(x), self.value_fc2(x)


class TestModelConfigContract:
    """Tests for ModelConfigContract class."""

    def test_for_config_square8_2p(self):
        """Test contract creation for square8 2-player."""
        contract = ModelConfigContract.for_config(BoardType.SQUARE8, 2)
        assert contract.board_type == BoardType.SQUARE8
        assert contract.num_players == 2
        assert contract.policy_size == 7000
        assert contract.value_head_outputs == 2

    def test_for_config_hex8_3p(self):
        """Test contract creation for hex8 3-player."""
        contract = ModelConfigContract.for_config(BoardType.HEX8, 3)
        assert contract.board_type == BoardType.HEX8
        assert contract.num_players == 3
        assert contract.policy_size == 4500
        assert contract.value_head_outputs == 3

    def test_for_config_square19_4p(self):
        """Test contract creation for square19 4-player."""
        contract = ModelConfigContract.for_config(BoardType.SQUARE19, 4)
        assert contract.board_type == BoardType.SQUARE19
        assert contract.num_players == 4
        assert contract.policy_size == 67000
        assert contract.value_head_outputs == 4

    def test_for_config_invalid_num_players(self):
        """Test that invalid num_players raises error."""
        with pytest.raises(ValueError, match="num_players must be 2, 3, or 4"):
            ModelConfigContract.for_config(BoardType.SQUARE8, 5)

        with pytest.raises(ValueError, match="num_players must be 2, 3, or 4"):
            ModelConfigContract.for_config(BoardType.SQUARE8, 1)

    def test_validate_model_correct_config(self):
        """Test validation passes for correctly configured model."""
        model = MockModel(value_outputs=3, policy_size=4500)
        contract = ModelConfigContract.for_config(BoardType.HEX8, 3)

        violations = contract.validate_model(model)
        assert len(violations) == 0

    def test_validate_model_wrong_value_head(self):
        """Test validation catches wrong value head dimension."""
        # Model has 4-player value head, but contract expects 3
        model = MockModel(value_outputs=4, policy_size=4500)
        contract = ModelConfigContract.for_config(BoardType.HEX8, 3)

        violations = contract.validate_model(model)
        assert len(violations) == 1
        assert "Value head outputs" in violations[0]
        assert "expected 3" in violations[0]
        assert "got 4" in violations[0]

    def test_validate_model_wrong_policy_size(self):
        """Test validation catches legacy encoding (wrong policy size)."""
        # Model has legacy policy size (~59K), but contract expects 7000
        model = MockModel(value_outputs=2, policy_size=59361)
        contract = ModelConfigContract.for_config(BoardType.SQUARE8, 2)

        violations = contract.validate_model(model)
        assert len(violations) == 1
        assert "Policy size" in violations[0]
        assert "legacy_max_n" in violations[0]

    def test_validate_model_multiple_violations(self):
        """Test validation catches multiple issues at once."""
        # Model has both wrong value head AND wrong policy size
        model = MockModel(value_outputs=4, policy_size=59361)
        contract = ModelConfigContract.for_config(BoardType.SQUARE8, 3)

        violations = contract.validate_model(model)
        assert len(violations) == 2

    def test_validate_checkpoint_metadata_correct(self):
        """Test metadata validation passes for correct config."""
        metadata = {
            'config': {
                'num_players': 2,
                'policy_size': 7000,
                'board_type': 'square8',
            }
        }
        contract = ModelConfigContract.for_config(BoardType.SQUARE8, 2)

        violations = contract.validate_checkpoint_metadata(metadata)
        assert len(violations) == 0

    def test_validate_checkpoint_metadata_wrong_num_players(self):
        """Test metadata validation catches wrong num_players."""
        metadata = {
            'config': {
                'num_players': 4,  # Wrong - checkpoint has 4p but target is 3p
                'policy_size': 4500,
            }
        }
        contract = ModelConfigContract.for_config(BoardType.HEX8, 3)

        violations = contract.validate_checkpoint_metadata(metadata)
        assert len(violations) == 1
        assert "num_players" in violations[0]
        assert "expected 3" in violations[0]
        assert "got 4" in violations[0]

    def test_validate_checkpoint_metadata_legacy_encoding(self):
        """Test metadata validation catches legacy policy encoding."""
        metadata = {
            'config': {
                'num_players': 2,
                'policy_size': 59361,  # Legacy encoding
            }
        }
        contract = ModelConfigContract.for_config(BoardType.SQUARE8, 2)

        violations = contract.validate_checkpoint_metadata(metadata)
        assert len(violations) == 1
        assert "policy_size" in violations[0]
        assert "legacy" in violations[0].lower()


class TestValidateModelForSave:
    """Tests for validate_model_for_save function."""

    def test_valid_model_passes(self):
        """Test that valid model passes without error."""
        model = MockModel(value_outputs=2, policy_size=7000)
        # Should not raise
        violations = validate_model_for_save(model, BoardType.SQUARE8, 2, strict=True)
        assert len(violations) == 0

    def test_invalid_model_raises_strict(self):
        """Test that invalid model raises in strict mode."""
        model = MockModel(value_outputs=4, policy_size=7000)  # Wrong value head

        with pytest.raises(ModelConfigError) as exc_info:
            validate_model_for_save(model, BoardType.SQUARE8, 3, strict=True)

        assert "Value head" in str(exc_info.value)
        assert len(exc_info.value.violations) == 1

    def test_invalid_model_warns_nonstrict(self):
        """Test that invalid model only warns in non-strict mode."""
        model = MockModel(value_outputs=4, policy_size=7000)

        # Should not raise, but return violations
        violations = validate_model_for_save(model, BoardType.SQUARE8, 3, strict=False)
        assert len(violations) == 1


class TestValidateCheckpointForPromotion:
    """Tests for validate_checkpoint_for_promotion function."""

    def test_valid_checkpoint_passes(self):
        """Test that valid checkpoint passes validation."""
        metadata = {
            'config': {
                'num_players': 2,
                'policy_size': 7000,
            }
        }

        is_valid, violations = validate_checkpoint_for_promotion(
            metadata, BoardType.SQUARE8, 2
        )

        assert is_valid is True
        assert len(violations) == 0

    def test_invalid_checkpoint_fails(self):
        """Test that invalid checkpoint fails validation."""
        metadata = {
            'config': {
                'num_players': 4,  # Wrong
                'policy_size': 59361,  # Legacy
            }
        }

        is_valid, violations = validate_checkpoint_for_promotion(
            metadata, BoardType.SQUARE8, 3
        )

        assert is_valid is False
        assert len(violations) == 2  # Wrong num_players AND legacy encoding


class TestGetCanonicalModelName:
    """Tests for get_canonical_model_name function."""

    def test_square8_2p(self):
        """Test canonical name for square8 2-player."""
        name = get_canonical_model_name(BoardType.SQUARE8, 2)
        assert name == "canonical_square8_2p.pth"

    def test_hex8_3p(self):
        """Test canonical name for hex8 3-player."""
        name = get_canonical_model_name(BoardType.HEX8, 3)
        assert name == "canonical_hex8_3p.pth"

    def test_hexagonal_4p(self):
        """Test canonical name for hexagonal 4-player."""
        name = get_canonical_model_name(BoardType.HEXAGONAL, 4)
        assert name == "canonical_hexagonal_4p.pth"


class TestContractImmutability:
    """Tests to verify contract immutability."""

    def test_contract_is_frozen(self):
        """Test that contract attributes cannot be modified."""
        contract = ModelConfigContract.for_config(BoardType.SQUARE8, 2)

        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            contract.num_players = 4

    def test_contract_str_representation(self):
        """Test human-readable string representation."""
        contract = ModelConfigContract.for_config(BoardType.HEX8, 3)
        str_repr = str(contract)

        assert "hex8" in str_repr
        assert "players=3" in str_repr
        assert "policy=4500" in str_repr


class TestAllBoardTypes:
    """Tests to verify all board types have correct policy sizes."""

    @pytest.mark.parametrize("board_type,expected_policy", [
        (BoardType.SQUARE8, 7000),
        (BoardType.SQUARE19, 67000),
        (BoardType.HEX8, 4500),
        (BoardType.HEXAGONAL, 91876),
    ])
    def test_policy_size_matches_constants(self, board_type, expected_policy):
        """Test that contract policy sizes match BOARD_POLICY_SIZES."""
        contract = ModelConfigContract.for_config(board_type, 2)
        assert contract.policy_size == expected_policy
        assert contract.policy_size == BOARD_POLICY_SIZES[board_type]

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_value_head_equals_num_players(self, num_players):
        """Test that value head output always equals num_players."""
        contract = ModelConfigContract.for_config(BoardType.SQUARE8, num_players)
        assert contract.value_head_outputs == num_players
