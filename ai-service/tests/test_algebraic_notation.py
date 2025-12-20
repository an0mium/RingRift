"""Tests for algebraic notation conversion."""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import BoardType, Move, MoveType, Position
from app.notation import (
    MOVE_TYPE_TO_CODE,
    algebraic_to_position,
    game_to_pgn,
    move_to_algebraic,
    moves_to_notation_list,
    parse_pgn,
    position_to_algebraic,
)


class TestPositionConversion:
    """Test position <-> algebraic conversion."""

    def test_square8_position_to_algebraic(self):
        """Test square 8x8 position conversion."""
        assert position_to_algebraic(Position(x=0, y=0), BoardType.SQUARE8) == "a1"
        assert position_to_algebraic(Position(x=7, y=7), BoardType.SQUARE8) == "h8"
        assert position_to_algebraic(Position(x=3, y=4), BoardType.SQUARE8) == "d5"

    def test_square19_position_to_algebraic(self):
        """Test square 19x19 position conversion."""
        assert position_to_algebraic(Position(x=0, y=0), BoardType.SQUARE19) == "a1"
        assert position_to_algebraic(Position(x=18, y=18), BoardType.SQUARE19) == "s19"
        assert position_to_algebraic(Position(x=10, y=9), BoardType.SQUARE19) == "k10"

    def test_hex_position_to_algebraic(self):
        """Test hexagonal position conversion."""
        assert position_to_algebraic(Position(x=0, y=0, z=0), BoardType.HEXAGONAL) == "0.0"
        assert position_to_algebraic(Position(x=1, y=0, z=-1), BoardType.HEXAGONAL) == "1.0"
        assert position_to_algebraic(Position(x=-3, y=5, z=-2), BoardType.HEXAGONAL) == "-3.5"
        assert position_to_algebraic(Position(x=11, y=-11, z=0), BoardType.HEXAGONAL) == "11.-11"

    def test_square_algebraic_to_position(self):
        """Test parsing square board notation."""
        p = algebraic_to_position("a1", BoardType.SQUARE8)
        assert p.x == 0 and p.y == 0

        p = algebraic_to_position("h8", BoardType.SQUARE8)
        assert p.x == 7 and p.y == 7

        p = algebraic_to_position("d5", BoardType.SQUARE8)
        assert p.x == 3 and p.y == 4

        p = algebraic_to_position("s19", BoardType.SQUARE19)
        assert p.x == 18 and p.y == 18

    def test_hex_algebraic_to_position(self):
        """Test parsing hex board notation."""
        p = algebraic_to_position("0.0", BoardType.HEXAGONAL)
        assert p.x == 0 and p.y == 0 and p.z == 0

        p = algebraic_to_position("3.-2", BoardType.HEXAGONAL)
        assert p.x == 3 and p.y == -2 and p.z == -1

        p = algebraic_to_position("-5.3", BoardType.HEXAGONAL)
        assert p.x == -5 and p.y == 3 and p.z == 2

    def test_round_trip_square(self):
        """Test round-trip conversion for square board."""
        for x in range(8):
            for y in range(8):
                pos = Position(x=x, y=y)
                notation = position_to_algebraic(pos, BoardType.SQUARE8)
                recovered = algebraic_to_position(notation, BoardType.SQUARE8)
                assert recovered.x == pos.x and recovered.y == pos.y

    def test_round_trip_hex(self):
        """Test round-trip conversion for hex board."""
        test_positions = [
            (0, 0, 0),
            (1, 0, -1),
            (-1, 1, 0),
            (5, -3, -2),
            (-11, 11, 0),
        ]
        for x, y, z in test_positions:
            pos = Position(x=x, y=y, z=z)
            notation = position_to_algebraic(pos, BoardType.HEXAGONAL)
            recovered = algebraic_to_position(notation, BoardType.HEXAGONAL)
            assert recovered.x == pos.x
            assert recovered.y == pos.y
            assert recovered.z == pos.z


class TestMoveConversion:
    """Test move <-> algebraic conversion."""

    def create_move(
        self,
        move_type: MoveType,
        to: Position,
        from_pos: Position = None,
        capture_target: Position = None,
        marker_left: Position = None,
    ) -> Move:
        """Helper to create a test move."""
        return Move(
            id="test-move",
            type=move_type,
            player=1,
            from_pos=from_pos,
            to=to,
            capture_target=capture_target,
            marker_left=marker_left,
            timestamp=datetime.now(),
            think_time=100,
            move_number=1,
        )

    def test_placement_move(self):
        """Test placement move notation."""
        move = self.create_move(MoveType.PLACE_RING, Position(x=3, y=4))
        notation = move_to_algebraic(move, BoardType.SQUARE8)
        assert notation == "P d5"

    def test_movement_move(self):
        """Test movement move notation."""
        move = self.create_move(
            MoveType.MOVE_STACK,
            to=Position(x=4, y=5),
            from_pos=Position(x=3, y=4),
        )
        notation = move_to_algebraic(move, BoardType.SQUARE8)
        assert notation == "M d5-e6"

    def test_capture_move(self):
        """Test capture move notation."""
        move = self.create_move(
            MoveType.OVERTAKING_CAPTURE,
            to=Position(x=2, y=0),
            from_pos=Position(x=0, y=0),
            capture_target=Position(x=1, y=0),
        )
        notation = move_to_algebraic(move, BoardType.SQUARE8)
        assert notation == "C a1-c1 xb1"

    def test_skip_placement(self):
        """Test skip placement notation."""
        move = self.create_move(MoveType.SKIP_PLACEMENT, Position(x=0, y=0))
        notation = move_to_algebraic(move, BoardType.SQUARE8)
        assert notation == "SP"

    def test_swap_sides(self):
        """Test swap sides notation."""
        move = self.create_move(MoveType.SWAP_SIDES, Position(x=0, y=0))
        notation = move_to_algebraic(move, BoardType.SQUARE8)
        assert notation == "SW"

    def test_hex_placement(self):
        """Test hex board placement notation."""
        move = self.create_move(MoveType.PLACE_RING, Position(x=3, y=-2, z=-1))
        notation = move_to_algebraic(move, BoardType.HEXAGONAL)
        assert notation == "P 3.-2"

    def test_hex_movement(self):
        """Test hex board movement notation."""
        move = self.create_move(
            MoveType.MOVE_STACK,
            to=Position(x=1, y=0, z=-1),
            from_pos=Position(x=0, y=0, z=0),
        )
        notation = move_to_algebraic(move, BoardType.HEXAGONAL)
        assert notation == "M 0.0-1.0"


class TestMoveTypeMapping:
    """Test move type <-> code mapping."""

    def test_all_move_types_have_codes(self):
        """Ensure all major move types have codes."""
        required_types = [
            MoveType.PLACE_RING,
            MoveType.SKIP_PLACEMENT,
            MoveType.MOVE_STACK,
            MoveType.OVERTAKING_CAPTURE,
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.ELIMINATE_RINGS_FROM_STACK,
        ]
        for mt in required_types:
            assert mt in MOVE_TYPE_TO_CODE, f"{mt} missing from MOVE_TYPE_TO_CODE"


class TestPGNGeneration:
    """Test PGN game record generation."""

    def test_generate_pgn(self):
        """Test basic PGN generation."""
        moves = [
            Move(
                id="m1",
                type=MoveType.PLACE_RING,
                player=1,
                to=Position(x=3, y=3),
                timestamp=datetime.now(),
                think_time=100,
                move_number=0,
            ),
            Move(
                id="m2",
                type=MoveType.PLACE_RING,
                player=2,
                to=Position(x=4, y=4),
                timestamp=datetime.now(),
                think_time=100,
                move_number=1,
            ),
        ]

        metadata = {
            "game_id": "test-001",
            "player1": "Alice",
            "player2": "Bob",
            "winner": 1,
            "termination": "ring_elimination",
            "rng_seed": 42,
        }

        pgn = game_to_pgn(moves, metadata, BoardType.SQUARE8)

        assert '[Game "RingRift"]' in pgn
        assert '[Board "square8"]' in pgn
        assert '[Player1 "Alice"]' in pgn
        assert '[Player2 "Bob"]' in pgn
        assert '[Result "1-0"]' in pgn
        assert "P d4" in pgn
        assert "P e5" in pgn

    def test_pgn_parse_roundtrip(self):
        """Test parsing PGN back to metadata and moves."""
        from app.notation.algebraic import parse_pgn

        pgn_text = """[Game "RingRift"]
[Board "square8"]
[Player1 "HeuristicAI"]
[Player2 "RandomAI"]
[Result "1-0"]

1. P d4      P e5
2. P c3      P f6

1-0"""

        metadata, move_notations = parse_pgn(pgn_text)

        assert metadata["game"] == "RingRift"
        assert metadata["board"] == "square8"
        assert metadata["result"] == "1-0"
        # Parser splits moves into individual tokens (code + position pairs)
        assert len(move_notations) == 8  # P, d4, P, e5, P, c3, P, f6
        assert move_notations[0] == "P"
        assert move_notations[1] == "d4"
        assert move_notations[2] == "P"
        assert move_notations[3] == "e5"
