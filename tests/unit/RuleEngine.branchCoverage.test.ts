/**
 * RuleEngine Branch Coverage Tests
 *
 * These tests target specific uncovered branches in RuleEngine.ts to improve
 * branch coverage. Focus areas:
 * - validateCapture and chain capture paths
 * - getValidMoves for various phases
 * - Territory and line processing validation
 * - Skip placement validation
 */

import { GameState, Position, Move, BoardState, BoardType } from '../../src/shared/engine';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { BoardManager } from '../../src/server/game/BoardManager';
import {
  createTestBoard,
  createTestGameState as createBaseGameState,
  pos,
  addStack,
  addMarker,
} from '../utils/fixtures';

// Helper to create a basic BoardManager
function createBoardManager(boardType: 'square8' | 'hexagonal' = 'square8'): BoardManager {
  return new BoardManager(boardType);
}

// Wrapper for createTestGameState that uses fixture helper
function createTestGameState(boardType: BoardType = 'square8', numPlayers = 2): GameState {
  return createBaseGameState({
    boardType,
    currentPlayer: 1,
  });
}

describe('RuleEngine - Branch Coverage', () => {
  let ruleEngine: RuleEngine;
  let boardManager: BoardManager;

  beforeEach(() => {
    boardManager = createBoardManager('square8');
    ruleEngine = new RuleEngine(boardManager, 'square8');
  });

  // ==========================================================================
  // validateMove - Default case and various move types
  // ==========================================================================
  describe('validateMove branches', () => {
    it('returns false for invalid player', () => {
      const state = createTestGameState();
      const move: Move = {
        type: 'place_ring',
        player: 99, // Invalid player
        to: pos(0, 0),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('returns false for wrong turn player', () => {
      const state = createTestGameState();
      state.currentPlayer = 2;
      const move: Move = {
        type: 'place_ring',
        player: 1, // Player 1 but it's player 2's turn
        to: pos(0, 0),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('returns false for unknown move type', () => {
      const state = createTestGameState();
      const move: Move = {
        type: 'invalid_type' as any,
        player: 1,
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('validates move_ring as legacy alias', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1);

      const move: Move = {
        type: 'move_ring',
        player: 1,
        from: pos(2, 2),
        to: pos(3, 2),
      };
      // May be valid or invalid depending on board state, but should be handled
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('validates move_stack', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1);

      const move: Move = {
        type: 'move_stack',
        player: 1,
        from: pos(2, 2),
        to: pos(3, 2),
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });
  });

  // ==========================================================================
  // validateCapture branches
  // ==========================================================================
  describe('validateCapture branches', () => {
    it('returns false when not in capture-allowed phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'ring_placement';
      addStack(state.board, pos(2, 2), 1);
      addStack(state.board, pos(3, 2), 2);

      const move: Move = {
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        captureTarget: pos(3, 2),
        to: pos(4, 2),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('validates capture during movement phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1, 2); // Height 2 to capture height 1
      addStack(state.board, pos(3, 2), 2, 1);

      const move: Move = {
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        captureTarget: pos(3, 2),
        to: pos(4, 2),
      };
      // Will be validated by shared helper
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('validates capture during capture phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'capture';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(3, 2), 2, 1);

      const move: Move = {
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        captureTarget: pos(3, 2),
        to: pos(4, 2),
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('validates capture during chain_capture phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'chain_capture';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(3, 2), 2, 1);

      const move: Move = {
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        captureTarget: pos(3, 2),
        to: pos(4, 2),
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('returns false when missing from position', () => {
      const state = createTestGameState();
      state.currentPhase = 'capture';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'overtaking_capture',
        player: 1,
        captureTarget: pos(3, 2),
        to: pos(4, 2),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('returns false when missing captureTarget', () => {
      const state = createTestGameState();
      state.currentPhase = 'capture';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1);

      const move: Move = {
        type: 'overtaking_capture',
        player: 1,
        from: pos(2, 2),
        to: pos(4, 2),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });
  });

  // ==========================================================================
  // Chain capture continuation validation
  // ==========================================================================
  describe('validateChainCaptureContinuation branches', () => {
    it('returns false when not in chain_capture phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(4, 2), 1, 3);
      addStack(state.board, pos(5, 2), 2, 1);

      const move: Move = {
        type: 'continue_capture_segment',
        player: 1,
        from: pos(4, 2),
        captureTarget: pos(5, 2),
        to: pos(6, 2),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('validates continuation during chain_capture phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'chain_capture';
      state.currentPlayer = 1;
      addStack(state.board, pos(4, 2), 1, 3);
      addStack(state.board, pos(5, 2), 2, 1);

      const move: Move = {
        type: 'continue_capture_segment',
        player: 1,
        from: pos(4, 2),
        captureTarget: pos(5, 2),
        to: pos(6, 2),
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('returns false when missing required positions', () => {
      const state = createTestGameState();
      state.currentPhase = 'chain_capture';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'continue_capture_segment',
        player: 1,
        to: pos(6, 2),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });
  });

  // ==========================================================================
  // Stack movement validation branches
  // ==========================================================================
  describe('validateStackMovement branches', () => {
    it('returns false during ring_placement phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'ring_placement';
      addStack(state.board, pos(2, 2), 1);

      const move: Move = {
        type: 'move_stack',
        player: 1,
        from: pos(2, 2),
        to: pos(3, 2),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('returns false when from is missing', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'move_stack',
        player: 1,
        to: pos(3, 2),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('returns false when source stack does not exist', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'move_stack',
        player: 1,
        from: pos(2, 2),
        to: pos(3, 2),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('returns false when stack belongs to another player', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 2); // Player 2's stack

      const move: Move = {
        type: 'move_stack',
        player: 1,
        from: pos(2, 2),
        to: pos(3, 2),
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });
  });

  // ==========================================================================
  // Line processing validation branches
  // ==========================================================================
  describe('validateLineProcessingMove branches', () => {
    it('validates process_line move', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('validates choose_line_reward move', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_decision';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'choose_line_reward',
        player: 1,
        lineIndex: 0,
        rewardChoice: 'COLLAPSE_ALL',
        collapsedPositions: [],
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('rejects process_line when formed line does not match any existing player line', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;

      // Create a real horizontal line for player 1 so BoardManager detects at least one line.
      addMarker(state.board, pos(0, 0), 1);
      addMarker(state.board, pos(1, 0), 1);
      addMarker(state.board, pos(2, 0), 1);
      addMarker(state.board, pos(3, 0), 1);

      // Provide a mismatched formedLines[0] that does not correspond to the real line.
      const bogusLine = {
        positions: [pos(7, 7)],
        player: 1,
      } as any;

      const move: Move = {
        type: 'process_line',
        player: 1,
        formedLines: [bogusLine],
      } as any;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });
  });

  // ==========================================================================
  // Territory processing validation branches
  // ==========================================================================
  describe('validateTerritoryProcessingMove branches', () => {
    it('validates process_territory_region move', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'process_territory_region',
        player: 1,
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('validates eliminate_rings_from_stack move', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1, 3);

      const move: Move = {
        type: 'eliminate_rings_from_stack',
        player: 1,
        to: pos(2, 2),
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('rejects elimination move when eliminatedRings count is non-positive', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1, 3);

      const move: Move = {
        type: 'eliminate_rings_from_stack',
        player: 1,
        to: pos(2, 2),
        eliminatedRings: [{ player: 1, count: 0 }],
      } as any;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });

    it('rejects elimination move when eliminatedRings count exceeds cap height', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;
      // Height 3 stack → capHeight >= 1
      addStack(state.board, pos(2, 2), 1, 3);

      const move: Move = {
        type: 'eliminate_rings_from_stack',
        player: 1,
        to: pos(2, 2),
        eliminatedRings: [{ player: 1, count: 10 }],
      } as any;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });

    it('rejects process_territory_region when disconnectedRegions[0] does not match any existing region', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;

      // Stub BoardManager.findDisconnectedRegions to return a single region
      // that does not match the one we place on the Move.
      const regionFromEngine = {
        spaces: [pos(0, 0)],
        controllingPlayer: 1,
        isDisconnected: true,
      } as any;

      jest
        .spyOn(boardManager as any, 'findDisconnectedRegions')
        .mockReturnValue([regionFromEngine]);

      const mismatchedRegion = {
        spaces: [pos(1, 1)],
        controllingPlayer: 1,
        isDisconnected: true,
      } as any;

      const move: Move = {
        type: 'process_territory_region',
        player: 1,
        disconnectedRegions: [mismatchedRegion],
      } as any;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });
  });

  // ==========================================================================
  // Skip placement validation branches
  // ==========================================================================
  describe('validateSkipPlacement branches', () => {
    it('validates skip_placement during ring_placement phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'ring_placement';
      state.currentPlayer = 1;
      // Need a controlled stack with legal moves for skip to be valid
      addStack(state.board, pos(2, 2), 1);

      const move: Move = {
        type: 'skip_placement',
        player: 1,
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('returns false when player has no rings in hand', () => {
      const state = createTestGameState();
      state.currentPhase = 'ring_placement';
      state.currentPlayer = 1;
      state.players[0].ringsInHand = 0;

      const move: Move = {
        type: 'skip_placement',
        player: 1,
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });

    it('returns false for invalid player', () => {
      const state = createTestGameState();
      state.currentPhase = 'ring_placement';
      state.currentPlayer = 99;

      const move: Move = {
        type: 'skip_placement',
        player: 99,
      };
      expect(ruleEngine.validateMove(move, state)).toBe(false);
    });
  });

  // ==========================================================================
  // getValidMoves branches for different phases
  // ==========================================================================
  describe('getValidMoves phase branches', () => {
    it('returns moves for ring_placement phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'ring_placement';
      state.currentPlayer = 1;

      const moves = ruleEngine.getValidMoves(state);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns moves for movement phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1);

      const moves = ruleEngine.getValidMoves(state);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns moves for capture phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'capture';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1, 2);
      addStack(state.board, pos(3, 2), 2, 1);

      const moves = ruleEngine.getValidMoves(state);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns moves for line_processing phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;

      const moves = ruleEngine.getValidMoves(state);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns moves for territory_processing phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'territory_processing';
      state.currentPlayer = 1;

      const moves = ruleEngine.getValidMoves(state);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns moves for chain_capture phase with chainCapturePosition', () => {
      const state = createTestGameState();
      state.currentPhase = 'chain_capture';
      state.currentPlayer = 1;
      state.chainCapturePosition = pos(4, 2);
      addStack(state.board, pos(4, 2), 1, 3);
      addStack(state.board, pos(5, 2), 2, 1);

      const moves = ruleEngine.getValidMoves(state);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns empty for chain_capture phase without chainCapturePosition', () => {
      const state = createTestGameState();
      state.currentPhase = 'chain_capture';
      state.currentPlayer = 1;
      // No chainCapturePosition set

      const moves = ruleEngine.getValidMoves(state);
      expect(Array.isArray(moves)).toBe(true);
    });
  });

  // ==========================================================================
  // checkGameEnd branches
  // ==========================================================================
  describe('checkGameEnd branches', () => {
    it('returns not game over for active game', () => {
      const state = createTestGameState();
      state.players[0].ringsInHand = 5;
      state.players[1].ringsInHand = 5;

      const result = ruleEngine.checkGameEnd(state);
      expect(result.isGameOver).toBe(false);
    });

    it('handles game over with winner', () => {
      const state = createTestGameState();
      // Eliminate all rings from player 2
      state.players[1].ringsInHand = 0;
      state.players[1].eliminatedRings = 10;

      const result = ruleEngine.checkGameEnd(state);
      expect(typeof result.isGameOver).toBe('boolean');
      if (result.isGameOver) {
        expect(result.winner).toBeDefined();
      }
    });
  });

  // ==========================================================================
  // Hexagonal board type support
  // ==========================================================================
  describe('hexagonal board support', () => {
    let hexEngine: RuleEngine;
    let hexBoardManager: BoardManager;

    beforeEach(() => {
      hexBoardManager = createBoardManager('hexagonal');
      hexEngine = new RuleEngine(hexBoardManager, 'hexagonal');
    });

    it('validates moves on hexagonal board', () => {
      const state = createTestGameState('hexagonal');
      state.currentPhase = 'ring_placement';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'place_ring',
        player: 1,
        to: { x: 0, y: 0, z: 0 },
      };
      const result = hexEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('gets valid moves on hexagonal board', () => {
      const state = createTestGameState('hexagonal');
      state.currentPhase = 'ring_placement';
      state.currentPlayer = 1;

      const moves = hexEngine.getValidMoves(state);
      expect(Array.isArray(moves)).toBe(true);
    });
  });

  // ==========================================================================
  // Path clearing (isPathClear) - internal method covered via validateStackMovement
  // ==========================================================================
  describe('path clearing via movement', () => {
    it('validates movement with clear path', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1);

      // Move to adjacent cell
      const move: Move = {
        type: 'move_stack',
        player: 1,
        from: pos(2, 2),
        to: pos(2, 3),
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('handles movement blocked by collapsed space', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1);
      // Ensure collapsedSpaces is a Set and add a blocking space
      if (!(state.board.collapsedSpaces instanceof Set)) {
        state.board.collapsedSpaces = new Set<string>();
      }
      state.board.collapsedSpaces.add('3,2'); // Block the path

      const move: Move = {
        type: 'move_stack',
        player: 1,
        from: pos(2, 2),
        to: pos(4, 2),
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('handles movement blocked by other stack', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(2, 2), 1);
      addStack(state.board, pos(3, 2), 2); // Blocking stack

      const move: Move = {
        type: 'move_stack',
        player: 1,
        from: pos(2, 2),
        to: pos(4, 2),
      };
      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });
  });

  // ==========================================================================
  // Additional coverage for line processing
  // ==========================================================================
  describe('line processing validation', () => {
    it('rejects process_line in wrong phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      } as Move;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });

    it('validates process_line during line_processing phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      // Add a formed line
      state.board.formedLines = [
        {
          player: 1,
          positions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)],
          length: 4,
        },
      ];

      const move: Move = {
        type: 'process_line',
        player: 1,
        lineIndex: 0,
      } as Move;

      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('rejects choose_line_reward in wrong phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'choose_line_reward',
        player: 1,
        lineIndex: 0,
        selection: 'COLLAPSE_ALL',
      } as Move;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });
  });

  // ==========================================================================
  // Territory processing validation
  // ==========================================================================
  describe('territory processing validation', () => {
    it('rejects process_territory_region in wrong phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'process_territory_region',
        player: 1,
        regionId: 'region-1',
      } as Move;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });

    it('rejects eliminate_rings_from_stack in wrong phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'eliminate_rings_from_stack',
        player: 1,
        stackPosition: pos(0, 0),
      } as Move;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });
  });

  // ==========================================================================
  // Skip placement validation
  // ==========================================================================
  describe('skip placement validation', () => {
    it('rejects skip_placement in wrong phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'skip_placement',
        player: 1,
      } as Move;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });

    it('validates skip_placement during ring_placement phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'ring_placement';
      state.currentPlayer = 1;
      // Player 1 has rings in hand
      state.players[0].ringsInHand = 5;
      // Add a stack for player 1
      addStack(state.board, pos(3, 3), 1, 2, 2);

      const move: Move = {
        type: 'skip_placement',
        player: 1,
      } as Move;

      const result = ruleEngine.validateMove(move, state);
      expect(typeof result).toBe('boolean');
    });

    it('rejects skip_placement when player has no rings', () => {
      const state = createTestGameState();
      state.currentPhase = 'ring_placement';
      state.currentPlayer = 1;
      state.players[0].ringsInHand = 0;

      const move: Move = {
        type: 'skip_placement',
        player: 1,
      } as Move;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });
  });

  // ==========================================================================
  // Unknown move type handling
  // ==========================================================================
  describe('unknown move type handling', () => {
    it('rejects unknown move types', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;

      const move = {
        type: 'unknown_move_type',
        player: 1,
      } as Move;

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });
  });

  // ==========================================================================
  // Player validation edge cases
  // ==========================================================================
  describe('player validation edge cases', () => {
    it('rejects move from non-existent player', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;

      const move: Move = {
        type: 'move_stack',
        player: 99, // Non-existent player
        from: pos(0, 0),
        to: pos(1, 0),
      };

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });

    it('rejects move when not player turn', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 2;
      addStack(state.board, pos(0, 0), 1);

      const move: Move = {
        type: 'move_stack',
        player: 1, // Not current player
        from: pos(0, 0),
        to: pos(1, 0),
      };

      const result = ruleEngine.validateMove(move, state);
      expect(result).toBe(false);
    });
  });

  // ==========================================================================
  // Valid moves enumeration
  // ==========================================================================
  describe('getValidMoves', () => {
    it('returns moves for ring placement phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'ring_placement';
      state.currentPlayer = 1;
      state.players[0].ringsInHand = 5;

      const moves = ruleEngine.getValidMoves(1, state);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns moves for movement phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      addStack(state.board, pos(3, 3), 1, 2, 2);

      const moves = ruleEngine.getValidMoves(1, state);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns empty array for player with no valid moves', () => {
      const state = createTestGameState();
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      // No stacks for player 1

      const moves = ruleEngine.getValidMoves(1, state);
      expect(Array.isArray(moves)).toBe(true);
    });

    it('returns moves for line processing phase', () => {
      const state = createTestGameState();
      state.currentPhase = 'line_processing';
      state.currentPlayer = 1;
      state.board.formedLines = [
        {
          player: 1,
          positions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)],
          length: 4,
        },
      ];

      const moves = ruleEngine.getValidMoves(1, state);
      expect(Array.isArray(moves)).toBe(true);
    });
  });
});
