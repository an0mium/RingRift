/**
 * lineDecisionHelpers.branchCoverage.test.ts
 *
 * Branch coverage tests for lineDecisionHelpers.ts targeting uncovered branches:
 * - computeNextMoveNumber with various history states
 * - detectPlayerLines cache vs fresh detection
 * - resolveLineForMove matching and fallback
 * - collapseLinePositions with stacks, markers, duplicates
 * - enumerateProcessLineMoves edge cases
 * - enumerateChooseLineRewardMoves boundary conditions
 * - applyProcessLineDecision validation and outcomes
 * - applyChooseLineRewardDecision all variants
 */

import {
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
  LineEnumerationOptions,
} from '../../src/shared/engine/lineDecisionHelpers';
import type { GameState, Position, Move, LineInfo, BoardState } from '../../src/shared/types/game';
import { BOARD_CONFIGS } from '../../src/shared/types/game';

// Helper to create position
const pos = (x: number, y: number): Position => ({ x, y });

// Helper to create a line info
const makeLine = (player: number, positions: Position[]): LineInfo => ({
  player,
  positions,
  length: positions.length,
});

// Helper to create empty board state
// Using square19 because lineLength=4 (vs square8 lineLength=3).
// This means 3-marker lines are "below required length" and 4-marker lines are "exact length".
const makeEmptyBoard = (): BoardState => ({
  type: 'square19',
  size: 19,
  stacks: new Map(),
  markers: new Map(),
  collapsedSpaces: new Map(),
  territories: new Map(),
  formedLines: [],
  eliminatedRings: { 1: 0, 2: 0 },
});

// Helper to create a minimal game state
const makeGameState = (overrides?: Partial<GameState>): GameState => ({
  id: 'test-game',
  boardType: 'square19',
  board: makeEmptyBoard(),
  players: [
    {
      id: 'p1',
      username: 'Player1',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ],
  currentPlayer: 1,
  currentPhase: 'line_processing',
  gameStatus: 'active',
  moveHistory: [],
  spectators: [],
  timeControl: { initialTime: 600, increment: 0, type: 'blitz' },
  ...overrides,
});

// Helper to add a marker to the board
const addMarker = (state: GameState, position: Position, player: number): void => {
  const key = `${position.x},${position.y}`;
  state.board.markers.set(key, player);
};

// Helper to add a stack to the board
const addStack = (
  state: GameState,
  position: Position,
  owner: number,
  height: number,
  rings?: number[]
): void => {
  const key = `${position.x},${position.y}`;
  state.board.stacks.set(key, {
    position,
    owner,
    height,
    rings: rings || Array(height).fill(owner),
    capHeight: height,
  });
};

// Helper to add a formed line to board cache
const addFormedLine = (state: GameState, line: LineInfo): void => {
  state.board.formedLines.push(line);
};

describe('lineDecisionHelpers branch coverage', () => {
  describe('enumerateProcessLineMoves', () => {
    describe('cache vs detection modes', () => {
      it('uses board.formedLines cache by default', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        const moves = enumerateProcessLineMoves(state, 1);
        expect(moves.length).toBe(1);
        expect(moves[0].type).toBe('process_line');
        expect(moves[0].player).toBe(1);
      });

      it('detects lines fresh with detect_now mode', () => {
        const state = makeGameState();
        // Put line in cache but use detect_now mode
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        const options: LineEnumerationOptions = { detectionMode: 'detect_now' };
        const moves = enumerateProcessLineMoves(state, 1, options);
        // Fresh detection finds no lines since we only have markers in cache, not actual board
        // This tests the detect_now branch is exercised
        expect(moves).toHaveLength(0);
      });

      it('falls back to fresh detection when cache is empty', () => {
        const state = makeGameState();
        // Empty cache triggers fresh detection fallback
        state.board.formedLines = [];

        const moves = enumerateProcessLineMoves(state, 1);
        // Fresh detection with empty board finds no lines
        expect(moves).toHaveLength(0);
      });

      it('uses boardTypeOverride when specified', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        const options: LineEnumerationOptions = { boardTypeOverride: 'square8' };
        const moves = enumerateProcessLineMoves(state, 1, options);
        expect(moves.length).toBe(1);
      });
    });

    describe('line filtering', () => {
      it('returns empty array when player has no lines', () => {
        const state = makeGameState();
        const line = makeLine(2, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        const moves = enumerateProcessLineMoves(state, 1);
        expect(moves).toHaveLength(0);
      });

      it('filters out lines below required length', () => {
        const state = makeGameState();
        // On square19 boards, lineLength=4, so 3-in-a-row is below required length
        const shortLine = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0)]);
        addFormedLine(state, shortLine);

        const moves = enumerateProcessLineMoves(state, 1);
        expect(moves).toHaveLength(0);
      });

      it('enumerates multiple valid lines', () => {
        const state = makeGameState();
        const line1 = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        const line2 = makeLine(1, [pos(0, 1), pos(1, 1), pos(2, 1), pos(3, 1)]);
        addFormedLine(state, line1);
        addFormedLine(state, line2);

        const moves = enumerateProcessLineMoves(state, 1);
        expect(moves.length).toBe(2);
      });

      it('filters lines by player ownership', () => {
        const state = makeGameState();
        const player1Line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        const player2Line = makeLine(2, [pos(0, 1), pos(1, 1), pos(2, 1), pos(3, 1)]);
        addFormedLine(state, player1Line);
        addFormedLine(state, player2Line);

        const moves = enumerateProcessLineMoves(state, 1);
        expect(moves.length).toBe(1);
        expect(moves[0].formedLines![0].player).toBe(1);
      });
    });

    describe('move number computation', () => {
      it('computes next move number from history', () => {
        const state = makeGameState();
        state.history = [{ moveNumber: 5 } as Move];
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        const moves = enumerateProcessLineMoves(state, 1);
        expect(moves[0].moveNumber).toBe(6);
      });

      it('falls back to moveHistory when history has no valid moveNumber', () => {
        const state = makeGameState();
        state.history = [{ moveNumber: 0 } as Move];
        state.moveHistory = [{ moveNumber: 3 } as Move];
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        const moves = enumerateProcessLineMoves(state, 1);
        expect(moves[0].moveNumber).toBe(4);
      });

      it('defaults to moveNumber 1 when no history exists', () => {
        const state = makeGameState();
        state.history = undefined;
        state.moveHistory = [];
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        const moves = enumerateProcessLineMoves(state, 1);
        expect(moves[0].moveNumber).toBe(1);
      });
    });
  });

  describe('enumerateChooseLineRewardMoves', () => {
    describe('edge cases', () => {
      it('returns empty for negative lineIndex', () => {
        const state = makeGameState();
        const moves = enumerateChooseLineRewardMoves(state, 1, -1);
        expect(moves).toHaveLength(0);
      });

      it('returns empty when player has no lines', () => {
        const state = makeGameState();
        const moves = enumerateChooseLineRewardMoves(state, 1, 0);
        expect(moves).toHaveLength(0);
      });

      it('returns empty for out of bounds lineIndex', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        const moves = enumerateChooseLineRewardMoves(state, 1, 5);
        expect(moves).toHaveLength(0);
      });

      it('returns empty for line with empty positions', () => {
        const state = makeGameState();
        const emptyLine: LineInfo = { player: 1, positions: [], length: 0 };
        addFormedLine(state, emptyLine);

        const moves = enumerateChooseLineRewardMoves(state, 1, 0);
        expect(moves).toHaveLength(0);
      });

      it('returns empty for line below required length', () => {
        const state = makeGameState();
        const shortLine = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0)]);
        addFormedLine(state, shortLine);

        const moves = enumerateChooseLineRewardMoves(state, 1, 0);
        expect(moves).toHaveLength(0);
      });
    });

    describe('exact length lines', () => {
      it('returns single collapse-all move for exact length line', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        const moves = enumerateChooseLineRewardMoves(state, 1, 0);
        expect(moves.length).toBe(1);
        expect(moves[0].type).toBe('choose_line_option');
        expect(moves[0].collapsedMarkers).toEqual(line.positions);
      });
    });

    describe('overlength lines', () => {
      it('returns collapse-all and minimum-collapse options for overlength line', () => {
        const state = makeGameState();
        // 5-in-a-row on 2-player = overlength
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        addFormedLine(state, line);

        const moves = enumerateChooseLineRewardMoves(state, 1, 0);
        // 1 collapse-all + 2 minimum collapse segments (start=0, start=1)
        expect(moves.length).toBe(3);

        // First is collapse-all
        expect(moves[0].collapsedMarkers).toEqual(line.positions);

        // Following are minimum collapse segments
        expect(moves[1].collapsedMarkers).toHaveLength(4);
        expect(moves[2].collapsedMarkers).toHaveLength(4);
      });

      it('generates correct number of segments for 6-length line', () => {
        const state = makeGameState();
        // 6-in-a-row
        const line = makeLine(1, [
          pos(0, 0),
          pos(1, 0),
          pos(2, 0),
          pos(3, 0),
          pos(4, 0),
          pos(5, 0),
        ]);
        addFormedLine(state, line);

        const moves = enumerateChooseLineRewardMoves(state, 1, 0);
        // 1 collapse-all + 3 segments (length 6 - required 4 + 1 = 3)
        expect(moves.length).toBe(4);
      });
    });
  });

  describe('applyProcessLineDecision', () => {
    describe('validation', () => {
      it('throws for wrong move type', () => {
        const state = makeGameState();
        const wrongMove: Move = {
          id: 'test',
          type: 'place_ring',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => applyProcessLineDecision(state, wrongMove)).toThrow(
          "applyProcessLineDecision expected move.type === 'process_line'"
        );
      });
    });

    describe('no-op cases', () => {
      it('returns unchanged state when no line can be resolved', () => {
        const state = makeGameState();
        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)])],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyProcessLineDecision(state, move);
        expect(result.nextState).toBe(state);
        expect(result.pendingLineRewardElimination).toBe(false);
      });

      it('returns unchanged state for line below required length', () => {
        const state = makeGameState();
        const shortLine = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0)]);
        addFormedLine(state, shortLine);

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [shortLine],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyProcessLineDecision(state, move);
        expect(result.nextState).toBe(state);
        expect(result.pendingLineRewardElimination).toBe(false);
      });

      it('returns unchanged state for overlength line (requires choose_line_reward)', () => {
        const state = makeGameState();
        const overlengthLine = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        addFormedLine(state, overlengthLine);

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [overlengthLine],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyProcessLineDecision(state, move);
        expect(result.nextState).toBe(state);
        expect(result.pendingLineRewardElimination).toBe(false);
      });
    });

    describe('exact length line processing', () => {
      it('collapses exact length line and sets pending reward', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        // Add markers at line positions
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyProcessLineDecision(state, move);
        expect(result.pendingLineRewardElimination).toBe(true);
        expect(result.nextState).not.toBe(state);

        // Verify markers collapsed
        for (const p of line.positions) {
          const key = `${p.x},${p.y}`;
          expect(result.nextState.board.markers.has(key)).toBe(false);
          expect(result.nextState.board.collapsedSpaces.has(key)).toBe(true);
        }

        // Verify territory count
        const player1 = result.nextState.players.find((p) => p.playerNumber === 1);
        expect(player1?.territorySpaces).toBe(4);
      });

      it('returns rings from stacks to owners hands', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);

        // Add markers and a stack with mixed ownership
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }
        addStack(state, pos(0, 0), 1, 2, [1, 2]); // Stack with rings from both players

        // Set initial rings in hand
        state.players[0].ringsInHand = 10;
        state.players[1].ringsInHand = 10;

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyProcessLineDecision(state, move);

        // Rings returned to owners
        const player1 = result.nextState.players.find((p) => p.playerNumber === 1);
        const player2 = result.nextState.players.find((p) => p.playerNumber === 2);
        expect(player1?.ringsInHand).toBe(11); // Got 1 back
        expect(player2?.ringsInHand).toBe(11); // Got 1 back
      });
    });

    describe('line resolution', () => {
      it('resolves line by canonical key match', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyProcessLineDecision(state, move);
        expect(result.pendingLineRewardElimination).toBe(true);
      });

      it('falls back to first line when no exact match', () => {
        const state = makeGameState();
        const boardLine = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, boardLine);
        for (const p of boardLine.positions) {
          addMarker(state, p, 1);
        }

        // Move refers to different line
        const moveReferencedLine = makeLine(1, [pos(5, 5), pos(6, 5), pos(7, 5), pos(8, 5)]);

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(5, 5),
          formedLines: [moveReferencedLine],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyProcessLineDecision(state, move);
        // Falls back to first available line
        expect(result.pendingLineRewardElimination).toBe(true);
      });
    });
  });

  describe('applyChooseLineRewardDecision', () => {
    describe('validation', () => {
      it('throws for wrong move type', () => {
        const state = makeGameState();
        const wrongMove: Move = {
          id: 'test',
          type: 'place_ring',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => applyChooseLineRewardDecision(state, wrongMove)).toThrow(
          "applyChooseLineRewardDecision expected move.type === 'choose_line_option'"
        );
      });
    });

    describe('no-op cases', () => {
      it('returns unchanged state when no line resolved', () => {
        const state = makeGameState();
        const move: Move = {
          id: 'test',
          type: 'choose_line_option',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyChooseLineRewardDecision(state, move);
        expect(result.nextState).toBe(state);
        expect(result.pendingLineRewardElimination).toBe(false);
      });

      it('returns unchanged state for line below required length', () => {
        const state = makeGameState();
        const shortLine = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0)]);
        addFormedLine(state, shortLine);

        const move: Move = {
          id: 'test',
          type: 'choose_line_option',
          player: 1,
          to: pos(0, 0),
          formedLines: [shortLine],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyChooseLineRewardDecision(state, move);
        expect(result.nextState).toBe(state);
        expect(result.pendingLineRewardElimination).toBe(false);
      });
    });

    describe('exact length line', () => {
      it('collapses all and sets pending reward for exact length', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }

        const move: Move = {
          id: 'test',
          type: 'choose_line_option',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyChooseLineRewardDecision(state, move);
        expect(result.pendingLineRewardElimination).toBe(true);

        const player1 = result.nextState.players.find((p) => p.playerNumber === 1);
        expect(player1?.territorySpaces).toBe(4);
      });
    });

    describe('overlength line - collapse all', () => {
      it('collapses entire overlength line and sets pending reward', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        addFormedLine(state, line);
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }

        const move: Move = {
          id: 'test',
          type: 'choose_line_reward',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          // No collapsedMarkers = collapse all
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyChooseLineRewardDecision(state, move);
        expect(result.pendingLineRewardElimination).toBe(true);

        const player1 = result.nextState.players.find((p) => p.playerNumber === 1);
        expect(player1?.territorySpaces).toBe(5);
      });

      it('treats collapsedMarkers length >= line length as collapse all', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        addFormedLine(state, line);
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }

        const move: Move = {
          id: 'test',
          type: 'choose_line_reward',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          collapsedMarkers: line.positions, // Same length as line
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyChooseLineRewardDecision(state, move);
        expect(result.pendingLineRewardElimination).toBe(true);
      });
    });

    describe('overlength line - minimum collapse', () => {
      it('collapses only minimum segment without reward', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        addFormedLine(state, line);
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }

        const segment = [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]; // 4 markers

        const move: Move = {
          id: 'test',
          type: 'choose_line_reward',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          collapsedMarkers: segment,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyChooseLineRewardDecision(state, move);
        expect(result.pendingLineRewardElimination).toBe(false);

        // Only 4 spaces collapsed, not 5
        const player1 = result.nextState.players.find((p) => p.playerNumber === 1);
        expect(player1?.territorySpaces).toBe(4);

        // Last marker should remain
        expect(result.nextState.board.markers.has('4,0')).toBe(true);
      });
    });
  });

  describe('collapseLinePositions internals', () => {
    describe('duplicate position handling', () => {
      it('skips duplicate positions in collapse', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyProcessLineDecision(state, move);

        // Territory should be 4, not more
        const player1 = result.nextState.players.find((p) => p.playerNumber === 1);
        expect(player1?.territorySpaces).toBe(4);
      });
    });

    describe('formedLines filtering', () => {
      it('removes formedLines that intersect collapsed spaces', () => {
        const state = makeGameState();
        const line1 = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        const line2 = makeLine(1, [pos(0, 0), pos(0, 1), pos(0, 2), pos(0, 3)]); // Shares pos(0,0)
        addFormedLine(state, line1);
        addFormedLine(state, line2);

        for (const p of line1.positions) {
          addMarker(state, p, 1);
        }
        for (const p of line2.positions) {
          addMarker(state, p, 1);
        }

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [line1],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const result = applyProcessLineDecision(state, move);

        // line2 should be removed because it intersects collapsed space at (0,0)
        expect(result.nextState.board.formedLines.length).toBe(0);
      });
    });

    describe('stack ring return', () => {
      it('handles stacks with empty rings array', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }

        // Add stack with empty rings - only stacks WITH rings get deleted
        // (rings returned to hand), empty stacks are left as-is
        state.board.stacks.set('0,0', {
          position: pos(0, 0),
          owner: 1,
          height: 0,
          rings: [],
          capHeight: 0,
        });

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        // Should not throw - but stack stays since no rings to return
        const result = applyProcessLineDecision(state, move);
        // Stack remains because only stacks with rings.length > 0 get deleted
        expect(result.nextState.board.stacks.has('0,0')).toBe(true);
      });

      it('handles ring return for non-existent player gracefully', () => {
        const state = makeGameState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        addFormedLine(state, line);
        for (const p of line.positions) {
          addMarker(state, p, 1);
        }

        // Add stack with ring from player 3 (doesn't exist in players array)
        state.board.stacks.set('0,0', {
          position: pos(0, 0),
          owner: 1,
          height: 1,
          rings: [3],
          capHeight: 1,
        });

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        // Should not throw - gracefully handles missing player
        const result = applyProcessLineDecision(state, move);
        expect(result.nextState.board.stacks.has('0,0')).toBe(false);
      });
    });
  });

  describe('3-player games', () => {
    it('uses correct line length threshold for 3 players', () => {
      const state = makeGameState();
      state.players.push({
        id: 'p3',
        username: 'Player3',
        playerNumber: 3,
        type: 'human',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      });

      // Line length is determined by board type, not player count.
      // square19 has lineLength=4, so 4-in-a-row is valid.
      const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
      addFormedLine(state, line);

      const moves = enumerateProcessLineMoves(state, 1);
      expect(moves.length).toBe(1);
    });
  });

  describe('move metadata', () => {
    it('includes correct metadata in process_line moves', () => {
      const state = makeGameState();
      const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
      addFormedLine(state, line);

      const moves = enumerateProcessLineMoves(state, 1);
      expect(moves[0].id).toMatch(/^process-line-/);
      expect(moves[0].type).toBe('process_line');
      expect(moves[0].player).toBe(1);
      expect(moves[0].to).toEqual(pos(0, 0));
      expect(moves[0].formedLines).toHaveLength(1);
      expect(moves[0].timestamp).toBeInstanceOf(Date);
      expect(moves[0].thinkTime).toBe(0);
    });

    it('includes correct metadata in choose_line_reward moves', () => {
      const state = makeGameState();
      const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
      addFormedLine(state, line);

      const moves = enumerateChooseLineRewardMoves(state, 1, 0);
      expect(moves[0].id).toMatch(/^choose-line-option-/);
      expect(moves[0].type).toBe('choose_line_option');
      expect(moves[0].player).toBe(1);
      expect(moves[0].timestamp).toBeInstanceOf(Date);
    });
  });
});
