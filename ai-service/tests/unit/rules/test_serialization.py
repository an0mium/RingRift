"""Tests for app.rules.serialization module.

Tests round-trip serialization/deserialization for cross-language parity.
Each type should survive: object -> serialize -> deserialize -> equals original.
"""

import pytest

from app.models import (
    BoardType,
    GamePhase,
    MarkerInfo,
    Move,
    MoveType,
    Position,
    RingStack,
)
from app.rules.serialization import (
    deserialize_board_state,
    deserialize_game_state,
    deserialize_marker,
    deserialize_move,
    deserialize_position,
    deserialize_stack,
    serialize_board_state,
    serialize_game_state,
    serialize_marker,
    serialize_move,
    serialize_position,
    serialize_stack,
)
from app.testing.fixtures import (
    create_board_state,
    create_game_state,
)


class TestPositionSerialization:
    """Round-trip tests for Position serialization."""

    def test_position_2d_roundtrip(self):
        """2D position serializes and deserializes correctly."""
        pos = Position(x=3, y=5)
        serialized = serialize_position(pos)
        deserialized = deserialize_position(serialized)
        assert deserialized.x == pos.x
        assert deserialized.y == pos.y
        assert deserialized.z == pos.z

    def test_position_3d_roundtrip(self):
        """3D position (with z) serializes correctly."""
        pos = Position(x=1, y=2, z=3)
        serialized = serialize_position(pos)
        assert "z" in serialized
        assert serialized["z"] == 3
        deserialized = deserialize_position(serialized)
        assert deserialized.z == 3

    def test_position_zero_coords(self):
        """Position at origin serializes correctly."""
        pos = Position(x=0, y=0)
        serialized = serialize_position(pos)
        deserialized = deserialize_position(serialized)
        assert deserialized.x == 0
        assert deserialized.y == 0

    def test_position_negative_coords(self):
        """Negative coordinates (hex boards) serialize correctly."""
        pos = Position(x=-3, y=2)
        serialized = serialize_position(pos)
        deserialized = deserialize_position(serialized)
        assert deserialized.x == -3
        assert deserialized.y == 2


class TestRingStackSerialization:
    """Round-trip tests for RingStack serialization."""

    def test_single_ring_stack_roundtrip(self):
        """Single-ring stack serializes correctly."""
        stack = RingStack(
            position=Position(x=2, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        serialized = serialize_stack(stack)
        deserialized = deserialize_stack(serialized)
        assert deserialized.stack_height == 1
        assert deserialized.cap_height == 1
        assert deserialized.controlling_player == 1

    def test_multi_ring_stack_roundtrip(self):
        """Multi-ring stack with multiple players serializes correctly."""
        # Rings from bottom to top: player 1, player 2, player 1
        stack = RingStack(
            position=Position(x=4, y=4),
            rings=[1, 2, 1],
            stackHeight=3,
            capHeight=1,
            controllingPlayer=1,
        )
        serialized = serialize_stack(stack)
        deserialized = deserialize_stack(serialized)
        assert deserialized.stack_height == 3
        # Note: deserialization reverses rings for TS/Python parity
        assert len(deserialized.rings) == 3

    def test_tall_stack_roundtrip(self):
        """Tall stack (5 rings) serializes correctly."""
        stack = RingStack(
            position=Position(x=0, y=0),
            rings=[1, 2, 1, 2, 1],
            stackHeight=5,
            capHeight=1,
            controllingPlayer=1,
        )
        serialized = serialize_stack(stack)
        deserialized = deserialize_stack(serialized)
        assert deserialized.stack_height == 5
        assert len(deserialized.rings) == 5


class TestMarkerInfoSerialization:
    """Round-trip tests for MarkerInfo serialization."""

    def test_marker_roundtrip(self):
        """MarkerInfo serializes correctly."""
        marker = MarkerInfo(
            position=Position(x=1, y=2),
            player=2,
            type="regular",
        )
        serialized = serialize_marker(marker)
        deserialized = deserialize_marker(serialized)
        assert deserialized.position.x == 1
        assert deserialized.position.y == 2
        assert deserialized.player == 2


class TestMoveSerialization:
    """Round-trip tests for Move serialization."""

    def test_place_ring_move_roundtrip(self):
        """PLACE_RING move serializes correctly."""
        move = Move(
            id="test-move-1",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=3, y=4),
            timestamp=None,
            thinkTime=0,
            moveNumber=5,
        )
        serialized = serialize_move(move)
        deserialized = deserialize_move(serialized)
        assert deserialized is not None
        assert deserialized.type == MoveType.PLACE_RING
        assert deserialized.player == 1
        assert deserialized.to.x == 3
        assert deserialized.to.y == 4

    def test_move_stack_move_roundtrip(self):
        """MOVE_STACK move with from/to serializes correctly."""
        move = Move(
            id="test-move-2",
            type=MoveType.MOVE_STACK,
            player=2,
            **{"from": Position(x=1, y=1)},  # 'from' is reserved keyword
            to=Position(x=4, y=4),
            timestamp=None,
            thinkTime=100,
            moveNumber=10,
        )
        serialized = serialize_move(move)
        deserialized = deserialize_move(serialized)
        assert deserialized is not None
        assert deserialized.type == MoveType.MOVE_STACK
        assert deserialized.from_pos is not None
        assert deserialized.from_pos.x == 1
        assert deserialized.to.x == 4


class TestBoardStateSerialization:
    """Round-trip tests for BoardState serialization."""

    def test_empty_board_roundtrip(self):
        """Empty board serializes correctly."""
        board = create_board_state()
        serialized = serialize_board_state(board)
        deserialized = deserialize_board_state(serialized)
        assert deserialized.type == board.type

    def test_board_with_stacks_roundtrip(self):
        """Board with stacks serializes correctly."""
        stack = RingStack(
            position=Position(x=2, y=2),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        board = create_board_state(stacks={"2,2": stack})
        serialized = serialize_board_state(board)
        deserialized = deserialize_board_state(serialized)
        assert "2,2" in deserialized.stacks
        assert deserialized.stacks["2,2"].controlling_player == 1

    def test_board_with_markers_roundtrip(self):
        """Board with markers serializes correctly."""
        marker = MarkerInfo(
            position=Position(x=3, y=3),
            player=1,
            type="regular",
        )
        board = create_board_state(markers={"3,3": marker})
        serialized = serialize_board_state(board)
        deserialized = deserialize_board_state(serialized)
        assert "3,3" in deserialized.markers
        assert deserialized.markers["3,3"].player == 1

    def test_hex_board_roundtrip(self):
        """Hex board type is preserved."""
        board = create_board_state(board_type=BoardType.HEX8)
        serialized = serialize_board_state(board)
        deserialized = deserialize_board_state(serialized)
        assert deserialized.type == BoardType.HEX8


class TestGameStateSerialization:
    """Round-trip tests for GameState serialization."""

    def test_game_state_roundtrip(self):
        """Game state serializes correctly."""
        state = create_game_state()
        serialized = serialize_game_state(state)
        deserialized = deserialize_game_state(serialized)
        assert deserialized.id == state.id
        assert deserialized.current_phase == state.current_phase
        assert deserialized.current_player == state.current_player

    def test_game_state_preserves_phase(self):
        """Phase is preserved through serialization."""
        for phase in [
            GamePhase.RING_PLACEMENT,
            GamePhase.MOVEMENT,
            GamePhase.LINE_PROCESSING,
        ]:
            state = create_game_state(current_phase=phase.value)
            serialized = serialize_game_state(state)
            deserialized = deserialize_game_state(serialized)
            assert deserialized.current_phase == phase

    def test_game_state_with_stacks(self):
        """Game state with board stacks serializes correctly."""
        stack = RingStack(
            position=Position(x=1, y=1),
            rings=[1, 2],
            stackHeight=2,
            capHeight=1,
            controllingPlayer=2,
        )
        board = create_board_state(stacks={"1,1": stack})
        state = create_game_state(board=board)
        serialized = serialize_game_state(state)
        deserialized = deserialize_game_state(serialized)
        assert "1,1" in deserialized.board.stacks
        assert deserialized.board.stacks["1,1"].stack_height == 2

    def test_multiplayer_game_state(self):
        """3-player and 4-player games serialize correctly."""
        for num_players in [2, 3, 4]:
            state = create_game_state(num_players=num_players)
            serialized = serialize_game_state(state)
            deserialized = deserialize_game_state(serialized)
            assert len(deserialized.players) == num_players
