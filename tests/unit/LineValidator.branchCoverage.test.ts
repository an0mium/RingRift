/**
 * LineValidator.branchCoverage.test.ts
 *
 * Branch coverage tests for LineValidator.ts targeting uncovered branches:
 * - validateProcessLine: phase, turn, line index bounds, line ownership
 * - validateChooseLineReward: all above plus selection validations,
 *   position validations, consecutive check
 */

import {
  validateProcessLine,
  validateChooseLineReward,
} from '../../src/shared/engine/validators/LineValidator';
import type {
  GameState,
  ProcessLineAction,
  ChooseLineRewardAction,
  FormedLine,
} from '../../src/shared/engine/types';
import type { Position, BoardType } from '../../src/shared/types/game';

// Helper to create a position
const pos = (x: number, y: number): Position => ({ x, y });

// Helper to create markers for line positions (RR-CANON-R120 requirement)
function makeMarkersForLines(lines: FormedLine[]): Map<string, number> {
  const markers = new Map<string, number>();
  for (const line of lines) {
    for (const p of line.positions) {
      markers.set(`${p.x},${p.y}`, line.player);
    }
  }
  return markers;
}

// Helper to create a minimal game state for testing
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  // Use square19 board type because lineLength=4, matching our test lines.
  // square8 has lineLength=3, so 4+ position lines would have different validation.
  const formedLines = (overrides.board?.formedLines as FormedLine[]) ?? [];
  const markers = makeMarkersForLines(formedLines);

  const defaultState: GameState = {
    id: 'test-game',
    board: {
      type: 'square19' as BoardType,
      size: 19,
      stacks: new Map(),
      markers,
      collapsedSpaces: new Map(),
      formedLines,
      territories: new Map(),
      eliminatedRings: { 1: 0, 2: 0 },
    },
    players: [
      {
        id: 'p1',
        username: 'Player1',
        playerNumber: 1,
        type: 'human',
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: 10,
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
        ringsInHand: 10,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ],
    currentPlayer: 1,
    currentPhase: 'line_processing',
    gameStatus: 'active',
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    moveHistory: [],
    spectators: [],
    boardType: 'square19',
  };

  return { ...defaultState, ...overrides } as GameState;
}

// Helper to set formedLines AND auto-populate markers (RR-CANON-R120)
function setLinesWithMarkers(state: GameState, lines: FormedLine[]): void {
  state.board.formedLines = lines;
  // Clear and repopulate markers
  state.board.markers = makeMarkersForLines(lines);
}

// Helper to create a formed line
function makeFormedLine(player: number, positions: Position[]): FormedLine {
  return {
    player,
    positions,
    length: positions.length,
  };
}

describe('LineValidator branch coverage', () => {
  describe('validateProcessLine', () => {
    describe('phase check', () => {
      it('rejects when not in line_processing phase', () => {
        const state = makeGameState({ currentPhase: 'movement' });
        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        const result = validateProcessLine(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
      });

      it('accepts when in line_processing phase', () => {
        const state = makeGameState({ currentPhase: 'line_processing' });
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        const result = validateProcessLine(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('turn check', () => {
      it('rejects when it is not the acting player turn', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(2, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 2,
          lineIndex: 0,
        };

        const result = validateProcessLine(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_TURN');
      });
    });

    describe('line index bounds', () => {
      it('rejects negative line index', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: -1,
        };

        const result = validateProcessLine(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_LINE_INDEX');
      });

      it('rejects line index >= formedLines.length', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 5, // Out of bounds
        };

        const result = validateProcessLine(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_LINE_INDEX');
      });

      it('accepts valid line index at boundary', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
          makeFormedLine(1, [pos(0, 1), pos(1, 1), pos(2, 1), pos(3, 1)]),
        ]);
        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 1, // Last valid index
        };

        const result = validateProcessLine(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('line ownership', () => {
      it('rejects processing opponent line', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(2, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        state.currentPlayer = 1;
        const action: ProcessLineAction = {
          type: 'process_line',
          playerId: 1,
          lineIndex: 0,
        };

        const result = validateProcessLine(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_LINE');
      });
    });
  });

  describe('validateChooseLineReward', () => {
    describe('phase check', () => {
      it('rejects when not in line_processing phase', () => {
        const state = makeGameState({ currentPhase: 'capture' });
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_PHASE');
      });
    });

    describe('turn check', () => {
      it('rejects when it is not the acting player turn', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 2, // Not current player
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_TURN');
      });
    });

    describe('line index bounds', () => {
      it('rejects negative line index', () => {
        const state = makeGameState();
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: -1,
          selection: 'COLLAPSE_ALL',
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_LINE_INDEX');
      });

      it('rejects line index >= formedLines.length', () => {
        const state = makeGameState();
        state.board.formedLines = [];
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_LINE_INDEX');
      });
    });

    describe('line ownership', () => {
      it('rejects processing opponent line', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(2, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NOT_YOUR_LINE');
      });
    });

    describe('exact length line validation', () => {
      it('rejects MINIMUM_COLLAPSE for exact length line', () => {
        const state = makeGameState();
        // For square19, minimum line length is 4
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          collapsedPositions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)],
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_SELECTION');
      });

      it('accepts COLLAPSE_ALL for exact length line', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('MINIMUM_COLLAPSE validation', () => {
      it('rejects when collapsedPositions is missing', () => {
        const state = makeGameState();
        // 5-space line (> minimum of 4)
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          // collapsedPositions is missing
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('MISSING_POSITIONS');
      });

      it('rejects when position count is wrong', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          collapsedPositions: [pos(0, 0), pos(1, 0), pos(2, 0)], // Only 3, need 4
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_POSITION_COUNT');
      });

      it('rejects when position is not part of line', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          collapsedPositions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(5, 5)], // (5,5) not in line
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_POSITION');
      });

      it('rejects non-consecutive positions', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          collapsedPositions: [pos(0, 0), pos(1, 0), pos(3, 0), pos(4, 0)], // Gap at pos 2
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NON_CONSECUTIVE');
      });

      it('accepts valid consecutive positions', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          collapsedPositions: [pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)], // Last 4, consecutive
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(true);
      });

      it('accepts positions in any order as long as consecutive in line', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          // Positions provided out of order but consecutive in line
          collapsedPositions: [pos(3, 0), pos(1, 0), pos(2, 0), pos(0, 0)],
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('COLLAPSE_ALL validation', () => {
      it('accepts COLLAPSE_ALL for longer line', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0), pos(5, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(true);
      });

      it('accepts COLLAPSE_ALL without collapsedPositions', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
          // No collapsedPositions needed
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(true);
      });
    });

    describe('edge cases', () => {
      it('handles line with all positions collapsed', () => {
        const state = makeGameState();
        setLinesWithMarkers(state, [
          makeFormedLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]),
        ]);
        const action: ChooseLineRewardAction = {
          type: 'choose_line_option',
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL',
        };

        const result = validateChooseLineReward(state, action);
        expect(result.valid).toBe(true);
      });
    });
  });
});
