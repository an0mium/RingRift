from app.models.core import MoveType
from app.rules.history_contract import (
    derive_phase_from_move_type,
    phase_move_contract,
    validate_canonical_move,
)


def test_phase_move_contract_has_canonical_phases():
    expected = {
        "ring_placement",
        "movement",
        "capture",
        "chain_capture",
        "line_processing",
        "territory_processing",
        "forced_elimination",
    }

    contract = phase_move_contract()

    assert set(contract.keys()) == expected


def test_validate_canonical_move_accepts_contract_pairs():
    contract = phase_move_contract()

    for phase, move_types in contract.items():
        for move_type in move_types:
            result = validate_canonical_move(phase, move_type)
            assert result.ok is True
            assert result.effective_phase == phase
            assert result.reason is None


def test_validate_canonical_move_rejects_phase_mismatch():
    result = validate_canonical_move("territory_processing", MoveType.MOVE_STACK.value)

    assert result.ok is False
    assert result.reason == "phase_move_mismatch:territory_processing:move_stack"


def test_validate_canonical_move_infers_phase_from_move_type():
    result = validate_canonical_move("", MoveType.PLACE_RING.value)

    assert result.ok is True
    assert result.effective_phase == "ring_placement"
    assert result.reason is None


def test_validate_canonical_move_infers_no_action_and_forced_elimination_phases():
    cases = {
        MoveType.NO_PLACEMENT_ACTION: "ring_placement",
        MoveType.NO_MOVEMENT_ACTION: "movement",
        MoveType.NO_LINE_ACTION: "line_processing",
        MoveType.NO_TERRITORY_ACTION: "territory_processing",
        MoveType.FORCED_ELIMINATION: "forced_elimination",
    }

    for move_type, expected_phase in cases.items():
        result = validate_canonical_move("", move_type.value)

        assert result.ok is True
        assert result.effective_phase == expected_phase
        assert result.reason is None


def test_validate_canonical_move_rejects_non_canonical_move_type():
    result = validate_canonical_move("", MoveType.CHAIN_CAPTURE.value)

    assert result.ok is False
    assert result.reason == "non_canonical_move_type:chain_capture"


def test_derive_phase_from_move_type_prefers_movement_for_capture_moves():
    phase = derive_phase_from_move_type(MoveType.OVERTAKING_CAPTURE.value)

    assert phase == "movement"
