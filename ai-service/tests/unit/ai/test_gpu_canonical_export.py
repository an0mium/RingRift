"""Unit tests for GPU canonical export translation module.

Tests cover:
1. GPU to canonical move type mapping
2. GPU to canonical phase mapping
3. Canvas to cube coordinate conversion
4. Move conversion functions
5. Canonical move sequence validation
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from app.ai.gpu_canonical_export import (
    _canvas_to_cube_coords,
    convert_gpu_move_to_canonical,
    gpu_move_type_to_canonical,
    gpu_phase_to_canonical,
    validate_canonical_move_sequence,
)
from app.ai.gpu_game_types import GamePhase, MoveType


class TestGpuMoveTypeToCanonical(unittest.TestCase):
    """Test GPU MoveType to canonical string conversion."""

    def test_placement_move_type(self):
        """Test PLACEMENT converts to place_ring."""
        result = gpu_move_type_to_canonical(MoveType.PLACEMENT)
        self.assertEqual(result, "place_ring")

    def test_movement_move_type(self):
        """Test MOVEMENT converts to move_stack."""
        result = gpu_move_type_to_canonical(MoveType.MOVEMENT)
        self.assertEqual(result, "move_stack")

    def test_capture_move_type(self):
        """Test CAPTURE converts to overtaking_capture."""
        result = gpu_move_type_to_canonical(MoveType.CAPTURE)
        self.assertEqual(result, "overtaking_capture")

    def test_overtaking_capture_move_type(self):
        """Test canonical OVERTAKING_CAPTURE."""
        result = gpu_move_type_to_canonical(MoveType.OVERTAKING_CAPTURE)
        self.assertEqual(result, "overtaking_capture")

    def test_skip_placement_move_type(self):
        """Test SKIP_PLACEMENT converts correctly."""
        result = gpu_move_type_to_canonical(MoveType.SKIP_PLACEMENT)
        self.assertEqual(result, "skip_placement")

    def test_forced_elimination_move_type(self):
        """Test FORCED_ELIMINATION converts correctly."""
        result = gpu_move_type_to_canonical(MoveType.FORCED_ELIMINATION)
        self.assertEqual(result, "forced_elimination")

    def test_no_placement_action_move_type(self):
        """Test NO_PLACEMENT_ACTION converts correctly."""
        result = gpu_move_type_to_canonical(MoveType.NO_PLACEMENT_ACTION)
        self.assertEqual(result, "no_placement_action")

    def test_no_movement_action_move_type(self):
        """Test NO_MOVEMENT_ACTION converts correctly."""
        result = gpu_move_type_to_canonical(MoveType.NO_MOVEMENT_ACTION)
        self.assertEqual(result, "no_movement_action")

    def test_invalid_move_type_returns_unknown(self):
        """Test invalid move type returns unknown."""
        result = gpu_move_type_to_canonical(-999)
        self.assertEqual(result, "unknown")

    def test_line_formation_move_type(self):
        """Test LINE_FORMATION converts to process_line."""
        result = gpu_move_type_to_canonical(MoveType.LINE_FORMATION)
        self.assertEqual(result, "process_line")


class TestGpuPhaseToCanonical(unittest.TestCase):
    """Test GPU GamePhase to canonical string conversion."""

    def test_ring_placement_phase(self):
        """Test RING_PLACEMENT converts correctly."""
        result = gpu_phase_to_canonical(GamePhase.RING_PLACEMENT)
        self.assertEqual(result, "ring_placement")

    def test_movement_phase(self):
        """Test MOVEMENT converts correctly."""
        result = gpu_phase_to_canonical(GamePhase.MOVEMENT)
        self.assertEqual(result, "movement")

    def test_capture_phase(self):
        """Test CAPTURE converts correctly."""
        result = gpu_phase_to_canonical(GamePhase.CAPTURE)
        self.assertEqual(result, "capture")

    def test_chain_capture_phase(self):
        """Test CHAIN_CAPTURE converts correctly."""
        result = gpu_phase_to_canonical(GamePhase.CHAIN_CAPTURE)
        self.assertEqual(result, "chain_capture")

    def test_line_processing_phase(self):
        """Test LINE_PROCESSING converts correctly."""
        result = gpu_phase_to_canonical(GamePhase.LINE_PROCESSING)
        self.assertEqual(result, "line_processing")

    def test_territory_processing_phase(self):
        """Test TERRITORY_PROCESSING converts correctly."""
        result = gpu_phase_to_canonical(GamePhase.TERRITORY_PROCESSING)
        self.assertEqual(result, "territory_processing")

    def test_game_over_phase(self):
        """Test GAME_OVER converts correctly."""
        result = gpu_phase_to_canonical(GamePhase.GAME_OVER)
        self.assertEqual(result, "game_over")

    def test_forced_elimination_phase(self):
        """Test FORCED_ELIMINATION converts correctly."""
        result = gpu_phase_to_canonical(GamePhase.FORCED_ELIMINATION)
        self.assertEqual(result, "forced_elimination")

    def test_invalid_phase_returns_default(self):
        """Test invalid phase returns ring_placement as default."""
        result = gpu_phase_to_canonical(-999)
        self.assertEqual(result, "ring_placement")

    def test_recovery_phase_maps_to_movement(self):
        """Test RECOVERY maps to movement phase."""
        result = gpu_phase_to_canonical(GamePhase.RECOVERY)
        self.assertEqual(result, "movement")


class TestCanvasToCubeCoords(unittest.TestCase):
    """Test canvas to cube coordinate conversion."""

    def test_square_board_identity(self):
        """Test square boards keep coords unchanged."""
        result = _canvas_to_cube_coords(3, 4, "square8")
        self.assertEqual(result, {"x": 4, "y": 3})

    def test_square19_board_identity(self):
        """Test square19 boards keep coords unchanged."""
        result = _canvas_to_cube_coords(10, 15, "square19")
        self.assertEqual(result, {"x": 15, "y": 10})

    def test_hex8_center_is_origin(self):
        """Test hex8 center (4,4) maps to (0,0,0)."""
        result = _canvas_to_cube_coords(4, 4, "hex8")
        self.assertEqual(result, {"x": 0, "y": 0, "z": 0})

    def test_hex8_offset_position(self):
        """Test hex8 offset position converts correctly."""
        # (6, 5) -> cube (5-4, 6-4, -1-2) = (1, 2, -3)
        result = _canvas_to_cube_coords(6, 5, "hex8")
        self.assertEqual(result, {"x": 1, "y": 2, "z": -3})

    def test_hexagonal_center_is_origin(self):
        """Test hexagonal center (12,12) maps to (0,0,0)."""
        result = _canvas_to_cube_coords(12, 12, "hexagonal")
        self.assertEqual(result, {"x": 0, "y": 0, "z": 0})

    def test_hexagonal_offset_position(self):
        """Test hexagonal offset position converts correctly."""
        # (14, 13) -> cube (13-12, 14-12, -1-2) = (1, 2, -3)
        result = _canvas_to_cube_coords(14, 13, "hexagonal")
        self.assertEqual(result, {"x": 1, "y": 2, "z": -3})

    def test_cube_coords_sum_to_zero(self):
        """Test that hex cube coords always sum to zero."""
        for row in range(9):
            for col in range(9):
                result = _canvas_to_cube_coords(row, col, "hex8")
                total = result["x"] + result["y"] + result["z"]
                self.assertEqual(total, 0, f"Failed at ({row}, {col})")


class TestConvertGpuMoveToCanonical(unittest.TestCase):
    """Test GPU move to canonical format conversion."""

    def test_placement_move(self):
        """Test placement move conversion."""
        result = convert_gpu_move_to_canonical(
            move_type=MoveType.PLACEMENT,
            player=1,
            from_y=-1,
            from_x=-1,
            to_y=3,
            to_x=4,
            phase=GamePhase.RING_PLACEMENT,
            board_type="square8",
        )

        self.assertEqual(result["type"], "place_ring")
        self.assertEqual(result["player"], 1)
        self.assertEqual(result["phase"], "ring_placement")
        self.assertNotIn("from", result)  # from should not be present
        self.assertEqual(result["to"], {"x": 4, "y": 3})

    def test_movement_move(self):
        """Test movement move conversion."""
        result = convert_gpu_move_to_canonical(
            move_type=MoveType.MOVEMENT,
            player=2,
            from_y=2,
            from_x=3,
            to_y=5,
            to_x=6,
            phase=GamePhase.MOVEMENT,
            board_type="square8",
        )

        self.assertEqual(result["type"], "move_stack")
        self.assertEqual(result["player"], 2)
        self.assertEqual(result["phase"], "movement")
        self.assertEqual(result["from"], {"x": 3, "y": 2})
        self.assertEqual(result["to"], {"x": 6, "y": 5})

    def test_capture_move_with_target(self):
        """Test capture move with capture target."""
        result = convert_gpu_move_to_canonical(
            move_type=MoveType.OVERTAKING_CAPTURE,
            player=1,
            from_y=2,
            from_x=2,
            to_y=4,
            to_x=4,
            phase=GamePhase.CAPTURE,
            board_type="square8",
            capture_target_y=3,
            capture_target_x=3,
        )

        self.assertEqual(result["type"], "overtaking_capture")
        self.assertEqual(result["captureTarget"], {"x": 3, "y": 3})

    def test_no_capture_target_when_invalid(self):
        """Test captureTarget not present when coordinates are -1."""
        result = convert_gpu_move_to_canonical(
            move_type=MoveType.PLACEMENT,
            player=1,
            from_y=-1,
            from_x=-1,
            to_y=3,
            to_x=3,
            phase=GamePhase.RING_PLACEMENT,
            capture_target_y=-1,
            capture_target_x=-1,
        )

        self.assertNotIn("captureTarget", result)

    def test_hex_board_coordinate_conversion(self):
        """Test hex board converts to cube coordinates."""
        # Place at hex8 center
        result = convert_gpu_move_to_canonical(
            move_type=MoveType.PLACEMENT,
            player=1,
            from_y=-1,
            from_x=-1,
            to_y=4,  # Center in hex8
            to_x=4,
            phase=GamePhase.RING_PLACEMENT,
            board_type="hex8",
        )

        self.assertEqual(result["to"], {"x": 0, "y": 0, "z": 0})


class TestValidateCanonicalMoveSequence(unittest.TestCase):
    """Test canonical move sequence validation."""

    def test_empty_sequence_is_valid(self):
        """Test empty move sequence is valid."""
        is_valid, errors = validate_canonical_move_sequence([])
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_valid_placement_move(self):
        """Test valid placement move sequence."""
        moves = [
            {"type": "place_ring", "player": 1, "phase": "ring_placement"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_valid_movement_sequence(self):
        """Test valid movement sequence."""
        moves = [
            {"type": "place_ring", "player": 1, "phase": "ring_placement"},
            {"type": "move_stack", "player": 2, "phase": "movement"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_invalid_move_type_for_phase(self):
        """Test detecting invalid move type for phase."""
        moves = [
            {"type": "move_stack", "player": 1, "phase": "ring_placement"},  # Invalid!
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 1)
        self.assertIn("not valid in ring_placement", errors[0])

    def test_invalid_player_number(self):
        """Test detecting invalid player number."""
        moves = [
            {"type": "place_ring", "player": 5, "phase": "ring_placement"},  # Invalid for 2p
        ]
        is_valid, errors = validate_canonical_move_sequence(moves, num_players=2)
        self.assertFalse(is_valid)
        self.assertIn("Invalid player 5", errors[0])

    def test_missing_phase(self):
        """Test detecting missing phase."""
        moves = [
            {"type": "place_ring", "player": 1},  # Missing phase
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertFalse(is_valid)
        self.assertIn("missing phase", errors[0])

    def test_capture_phase_moves(self):
        """Test valid capture phase moves."""
        moves = [
            {"type": "overtaking_capture", "player": 1, "phase": "capture"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertTrue(is_valid)

    def test_skip_capture_valid(self):
        """Test skip_capture is valid in capture phase."""
        moves = [
            {"type": "skip_capture", "player": 1, "phase": "capture"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertTrue(is_valid)

    def test_line_processing_moves(self):
        """Test valid line processing moves."""
        moves = [
            {"type": "process_line", "player": 1, "phase": "line_processing"},
            {"type": "choose_line_option", "player": 1, "phase": "line_processing"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertTrue(is_valid)

    def test_territory_processing_moves(self):
        """Test valid territory processing moves."""
        moves = [
            {"type": "choose_territory_option", "player": 1, "phase": "territory_processing"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertTrue(is_valid)

    def test_forced_elimination_phase(self):
        """Test forced elimination phase moves."""
        moves = [
            {"type": "forced_elimination", "player": 1, "phase": "forced_elimination"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertTrue(is_valid)

    def test_recovery_in_movement_phase(self):
        """Test recovery moves valid in movement phase."""
        moves = [
            {"type": "recovery_slide", "player": 1, "phase": "movement"},
            {"type": "skip_recovery", "player": 1, "phase": "movement"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertTrue(is_valid)

    def test_swap_sides_valid_in_ring_placement(self):
        """Test swap_sides valid in ring_placement."""
        moves = [
            {"type": "swap_sides", "player": 1, "phase": "ring_placement"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves)
        self.assertTrue(is_valid)

    def test_4_player_validation(self):
        """Test player validation for 4 players."""
        moves = [
            {"type": "place_ring", "player": 4, "phase": "ring_placement"},
        ]
        is_valid, errors = validate_canonical_move_sequence(moves, num_players=4)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])


class TestConvertGpuHistoryToCanonical(unittest.TestCase):
    """Test GPU history to canonical conversion."""

    def test_conversion_requires_batch_game_state(self):
        """Test that conversion works with BatchGameState."""
        try:
            from app.ai.gpu_batch_state import BatchGameState
            from app.ai.gpu_canonical_export import convert_gpu_history_to_canonical

            # Create minimal batch state
            batch = BatchGameState.create_batch(
                batch_size=1,
                board_size=8,
                num_players=2,
                device=torch.device("cpu"),
            )

            # Initialize move history with one placement move
            batch.move_history[0, 0, 0] = MoveType.PLACEMENT  # move_type
            batch.move_history[0, 0, 1] = 1  # player
            batch.move_history[0, 0, 2] = -1  # from_y
            batch.move_history[0, 0, 3] = -1  # from_x
            batch.move_history[0, 0, 4] = 3  # to_y
            batch.move_history[0, 0, 5] = 4  # to_x
            batch.move_history[0, 0, 6] = GamePhase.RING_PLACEMENT  # phase

            moves = convert_gpu_history_to_canonical(batch, game_idx=0, board_type="square8")

            self.assertEqual(len(moves), 1)
            self.assertEqual(moves[0]["type"], "place_ring")
            self.assertEqual(moves[0]["player"], 1)
            self.assertEqual(moves[0]["to"], {"x": 4, "y": 3})
        except ImportError:
            self.skipTest("gpu_batch_state not available")


class TestExportGameToCanonicalDict(unittest.TestCase):
    """Test full game export to canonical dictionary."""

    def test_export_creates_valid_game_dict(self):
        """Test export creates valid game dictionary."""
        try:
            from app.ai.gpu_batch_state import BatchGameState
            from app.ai.gpu_canonical_export import export_game_to_canonical_dict

            # Create minimal batch state
            batch = BatchGameState.create_batch(
                batch_size=1,
                board_size=8,
                num_players=2,
                device=torch.device("cpu"),
            )

            # Set winner
            batch.winner[0] = 1

            # Add a placement move
            batch.move_history[0, 0, 0] = MoveType.PLACEMENT
            batch.move_history[0, 0, 1] = 1
            batch.move_history[0, 0, 2] = -1
            batch.move_history[0, 0, 3] = -1
            batch.move_history[0, 0, 4] = 3
            batch.move_history[0, 0, 5] = 4
            batch.move_history[0, 0, 6] = GamePhase.RING_PLACEMENT

            game_dict = export_game_to_canonical_dict(
                batch, game_idx=0, board_type="square8", num_players=2
            )

            # Check required fields
            self.assertEqual(game_dict["board_type"], "square8")
            self.assertEqual(game_dict["num_players"], 2)
            self.assertEqual(game_dict["winner"], 1)
            self.assertEqual(game_dict["status"], "completed")
            self.assertIn("moves", game_dict)
            self.assertEqual(len(game_dict["moves"]), 1)
            self.assertIn("game_id", game_dict)
            self.assertIn("timestamp", game_dict)
        except ImportError:
            self.skipTest("gpu_batch_state not available")


if __name__ == "__main__":
    unittest.main()
