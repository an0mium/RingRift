"""Tests for app.models.core module.

Tests the core Pydantic models, enums, and data structures.
"""

import pytest
from pydantic import ValidationError

from app.models.core import (
    # Enums
    BoardType,
    GamePhase,
    GameStatus,
    MoveType,
    AIType,
    # Core models
    Position,
    LineInfo,
    Territory,
    RingStack,
    MarkerInfo,
    Player,
    TimeControl,
    Move,
    # Utility functions
    clear_position_key_cache,
)


class TestBoardType:
    """Tests for BoardType enum."""

    def test_standard_values(self):
        """Test that standard board types are available."""
        assert BoardType.SQUARE8.value == "square8"
        assert BoardType.SQUARE19.value == "square19"
        assert BoardType.HEX8.value == "hex8"
        assert BoardType.HEXAGONAL.value == "hexagonal"

    def test_aliases(self):
        """Test board type aliases."""
        assert BoardType.FULL_HEX.value == "hexagonal"
        assert BoardType.FULL_HEXAGONAL.value == "hexagonal"

    def test_from_string(self):
        """Test creating BoardType from string."""
        assert BoardType("square8") == BoardType.SQUARE8
        assert BoardType("hex8") == BoardType.HEX8


class TestGamePhase:
    """Tests for GamePhase enum."""

    def test_all_phases_defined(self):
        """Test that all required phases are defined."""
        phases = [
            "ring_placement",
            "movement",
            "capture",
            "chain_capture",
            "line_processing",
            "territory_processing",
            "forced_elimination",
            "game_over",
        ]
        for phase in phases:
            assert GamePhase(phase) is not None

    def test_phase_count(self):
        """Test that we have the expected number of phases."""
        assert len(GamePhase) == 8


class TestGameStatus:
    """Tests for GameStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all statuses are defined."""
        statuses = ["waiting", "active", "paused", "abandoned", "completed"]
        for status in statuses:
            assert GameStatus(status) is not None


class TestMoveType:
    """Tests for MoveType enum."""

    def test_placement_move_types(self):
        """Test ring placement move types."""
        assert MoveType.PLACE_RING.value == "place_ring"
        assert MoveType.SKIP_PLACEMENT.value == "skip_placement"
        assert MoveType.NO_PLACEMENT_ACTION.value == "no_placement_action"

    def test_movement_move_types(self):
        """Test movement move types."""
        assert MoveType.MOVE_STACK.value == "move_stack"
        assert MoveType.MOVE_RING.value == "move_ring"  # Legacy alias
        assert MoveType.NO_MOVEMENT_ACTION.value == "no_movement_action"

    def test_capture_move_types(self):
        """Test capture move types."""
        assert MoveType.OVERTAKING_CAPTURE.value == "overtaking_capture"
        assert MoveType.SKIP_CAPTURE.value == "skip_capture"
        assert MoveType.CONTINUE_CAPTURE_SEGMENT.value == "continue_capture_segment"

    def test_swap_move_type(self):
        """Test swap sides (pie rule) move type."""
        assert MoveType.SWAP_SIDES.value == "swap_sides"


class TestAIType:
    """Tests for AIType enum."""

    def test_basic_ai_types(self):
        """Test basic AI types are available."""
        assert AIType.HEURISTIC.value == "heuristic"
        assert AIType.MCTS.value == "mcts"
        assert AIType.POLICY_ONLY.value == "policy_only"

    def test_advanced_ai_types(self):
        """Test advanced AI types."""
        assert AIType.GUMBEL_MCTS.value == "gumbel_mcts"
        assert AIType.GMO.value == "gmo"
        assert AIType.CAGE.value == "cage"


class TestPosition:
    """Tests for Position model."""

    def test_2d_position(self):
        """Test creating a 2D position."""
        pos = Position(x=3, y=4)
        assert pos.x == 3
        assert pos.y == 4
        assert pos.z is None

    def test_3d_position(self):
        """Test creating a 3D position (hexagonal)."""
        pos = Position(x=1, y=2, z=3)
        assert pos.x == 1
        assert pos.y == 2
        assert pos.z == 3

    def test_position_immutable(self):
        """Test that Position is immutable (frozen)."""
        pos = Position(x=1, y=2)
        with pytest.raises(ValidationError):
            pos.x = 5

    def test_to_key_2d(self):
        """Test position to_key for 2D position."""
        clear_position_key_cache()
        pos = Position(x=3, y=4)
        assert pos.to_key() == "3,4"

    def test_to_key_3d(self):
        """Test position to_key for 3D position."""
        clear_position_key_cache()
        pos = Position(x=1, y=2, z=3)
        assert pos.to_key() == "1,2,3"

    def test_to_key_cached(self):
        """Test that to_key uses cache for repeated calls."""
        clear_position_key_cache()
        pos = Position(x=5, y=6)
        key1 = pos.to_key()
        key2 = pos.to_key()
        assert key1 == key2 == "5,6"

    def test_position_equality(self):
        """Test position equality comparison."""
        pos1 = Position(x=1, y=2)
        pos2 = Position(x=1, y=2)
        pos3 = Position(x=1, y=3)
        assert pos1 == pos2
        assert pos1 != pos3


class TestRingStack:
    """Tests for RingStack model."""

    def test_basic_stack(self):
        """Test creating a basic ring stack."""
        stack = RingStack(
            position=Position(x=0, y=0),
            rings=[1, 2, 1],
            stackHeight=3,
            capHeight=2,
            controllingPlayer=1,
        )
        assert stack.stack_height == 3
        assert stack.cap_height == 2
        assert stack.controlling_player == 1
        assert len(stack.rings) == 3

    def test_stack_alias_fields(self):
        """Test that camelCase aliases work."""
        stack = RingStack(
            position=Position(x=0, y=0),
            rings=[1],
            stack_height=1,
            cap_height=1,
            controlling_player=1,
        )
        assert stack.stack_height == 1


class TestPlayer:
    """Tests for Player model."""

    def test_human_player(self):
        """Test creating a human player."""
        player = Player(
            id="player-1",
            username="TestUser",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=300000,
            aiDifficulty=None,
            ringsInHand=10,
            eliminatedRings=0,
            territorySpaces=0,
        )
        assert player.player_number == 1
        assert player.is_ready is True
        assert player.ai_difficulty is None
        assert player.rings_in_hand == 10

    def test_ai_player(self):
        """Test creating an AI player."""
        player = Player(
            id="ai-player",
            username="AI",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=300000,
            aiDifficulty=5,
            ringsInHand=10,
            eliminatedRings=0,
            territorySpaces=0,
        )
        assert player.ai_difficulty == 5


class TestMove:
    """Tests for Move model."""

    def test_placement_move(self):
        """Test creating a placement move."""
        move = Move(
            id="move-1",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=3, y=3),
        )
        assert move.type == MoveType.PLACE_RING
        assert move.player == 1
        assert move.to is not None
        assert move.to.x == 3

    def test_movement_move(self):
        """Test creating a movement move."""
        move = Move(
            id="move-2",
            type=MoveType.MOVE_STACK,
            player=1,
            **{"from": Position(x=0, y=0)},  # 'from' is a reserved word
            to=Position(x=2, y=0),
        )
        assert move.type == MoveType.MOVE_STACK
        assert move.from_pos is not None
        assert move.from_pos.x == 0
        assert move.to.x == 2

    def test_skip_capture_move(self):
        """Test creating a skip capture move."""
        move = Move(
            id="move-3",
            type=MoveType.SKIP_CAPTURE,
            player=1,
        )
        assert move.type == MoveType.SKIP_CAPTURE
        assert move.to is None
        assert move.from_pos is None

    def test_move_immutable(self):
        """Test that Move is immutable (frozen)."""
        move = Move(
            id="move-4",
            type=MoveType.PLACE_RING,
            player=1,
        )
        with pytest.raises(ValidationError):
            move.player = 2


class TestLineInfo:
    """Tests for LineInfo model."""

    def test_line_info(self):
        """Test creating a line info."""
        positions = [Position(x=i, y=0) for i in range(4)]
        line = LineInfo(
            positions=positions,
            player=1,
            length=4,
            direction=Position(x=1, y=0),
        )
        assert line.length == 4
        assert line.player == 1
        assert len(line.positions) == 4


class TestTerritory:
    """Tests for Territory model."""

    def test_territory(self):
        """Test creating a territory."""
        spaces = [Position(x=0, y=0), Position(x=0, y=1), Position(x=1, y=0)]
        territory = Territory(
            spaces=spaces,
            controllingPlayer=1,
            isDisconnected=False,
        )
        assert territory.controlling_player == 1
        assert territory.is_disconnected is False
        assert len(territory.spaces) == 3


class TestTimeControl:
    """Tests for TimeControl model."""

    def test_time_control(self):
        """Test creating time control settings."""
        tc = TimeControl(
            initialTime=300000,
            increment=5000,
            type="fischer",
        )
        assert tc.initial_time == 300000
        assert tc.increment == 5000
        assert tc.type == "fischer"
