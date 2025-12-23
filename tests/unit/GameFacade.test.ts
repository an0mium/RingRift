/**
 * Unit tests for GameFacade utility functions
 *
 * Tests the pure utility functions exported from GameFacade:
 * - extractChainCapturePath
 * - deriveMustMoveFrom
 * - canSubmitMove
 * - canInteract
 *
 * @module tests/unit/GameFacade.test
 */

import type { GameState, Move, Position, BoardState } from '../../src/shared/types/game';
import {
  extractChainCapturePath,
  deriveMustMoveFrom,
  canSubmitMove,
  canInteract,
  type GameFacade,
  type FacadeConnectionStatus,
  type GameFacadeMode,
} from '../../src/client/facades/GameFacade';

// ═══════════════════════════════════════════════════════════════════════════
// TEST FIXTURES
// ═══════════════════════════════════════════════════════════════════════════

function createMinimalGameState(overrides: Partial<GameState> = {}): GameState {
  const minimalBoard: BoardState = {
    type: 'square8',
    size: 8,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
  };

  return {
    gameId: 'test-game',
    gameStatus: 'active',
    currentTurn: 1,
    currentPlayer: 1,
    currentPhase: 'movement',
    board: minimalBoard,
    players: [
      {
        id: 'player-1',
        username: 'Player 1',
        playerNumber: 1,
        type: 'human',
        ringsInHand: 10,
        eliminatedRings: 0,
        lostSpaces: 0,
        createdAt: new Date(),
        updatedAt: new Date(),
      },
      {
        id: 'player-2',
        username: 'Player 2',
        playerNumber: 2,
        type: 'ai',
        ringsInHand: 10,
        eliminatedRings: 0,
        lostSpaces: 0,
        createdAt: new Date(),
        updatedAt: new Date(),
      },
    ],
    history: [],
    moveHistory: [],
    ...overrides,
  } as GameState;
}

function createMockFacade(overrides: Partial<GameFacade> = {}): GameFacade {
  return {
    gameState: createMinimalGameState(),
    validMoves: [],
    victoryState: null,
    gameEndExplanation: null,
    mode: 'backend' as GameFacadeMode,
    connectionStatus: 'connected' as FacadeConnectionStatus,
    isPlayer: true,
    isMyTurn: true,
    boardType: 'square8',
    currentUserId: 'player-1',
    decisionState: {
      pendingChoice: null,
      choiceDeadline: null,
      choiceTimeRemainingMs: null,
    },
    chainCaptureState: null,
    mustMoveFrom: undefined,
    players: [],
    submitMove: jest.fn(),
    respondToChoice: jest.fn(),
    ...overrides,
  };
}

function createMove(overrides: Partial<Move> = {}): Move {
  return {
    type: 'move_stack',
    from: { x: 0, y: 0 },
    to: { x: 1, y: 1 },
    player: 1,
    turn: 1,
    phase: 'movement',
    timestamp: Date.now(),
    ...overrides,
  } as Move;
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS: extractChainCapturePath
// ═══════════════════════════════════════════════════════════════════════════

describe('extractChainCapturePath', () => {
  it('returns undefined when not in chain_capture phase', () => {
    const gameState = createMinimalGameState({ currentPhase: 'movement' });
    expect(extractChainCapturePath(gameState)).toBeUndefined();
  });

  it('returns undefined when in chain_capture but no move history', () => {
    const gameState = createMinimalGameState({
      currentPhase: 'chain_capture',
      moveHistory: [],
    });
    expect(extractChainCapturePath(gameState)).toBeUndefined();
  });

  it('returns undefined when moveHistory is undefined', () => {
    const gameState = createMinimalGameState({
      currentPhase: 'chain_capture',
      moveHistory: undefined,
    });
    expect(extractChainCapturePath(gameState)).toBeUndefined();
  });

  it('extracts path from single overtaking_capture move', () => {
    const gameState = createMinimalGameState({
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      moveHistory: [
        createMove({
          type: 'overtaking_capture',
          from: { x: 0, y: 0 },
          to: { x: 2, y: 2 },
          player: 1,
        }),
      ],
    });

    const path = extractChainCapturePath(gameState);
    expect(path).toEqual([
      { x: 0, y: 0 },
      { x: 2, y: 2 },
    ]);
  });

  it('extracts path from multiple chain capture moves', () => {
    const gameState = createMinimalGameState({
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      moveHistory: [
        createMove({
          type: 'overtaking_capture',
          from: { x: 0, y: 0 },
          to: { x: 2, y: 2 },
          player: 1,
        }),
        createMove({
          type: 'continue_capture_segment',
          from: { x: 2, y: 2 },
          to: { x: 4, y: 4 },
          player: 1,
        }),
        createMove({
          type: 'continue_capture_segment',
          from: { x: 4, y: 4 },
          to: { x: 6, y: 6 },
          player: 1,
        }),
      ],
    });

    const path = extractChainCapturePath(gameState);
    expect(path).toEqual([
      { x: 0, y: 0 },
      { x: 2, y: 2 },
      { x: 4, y: 4 },
      { x: 6, y: 6 },
    ]);
  });

  it('stops at move from different player', () => {
    const gameState = createMinimalGameState({
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      moveHistory: [
        createMove({
          type: 'move_stack',
          from: { x: 1, y: 1 },
          to: { x: 2, y: 2 },
          player: 2,
        }),
        createMove({
          type: 'overtaking_capture',
          from: { x: 0, y: 0 },
          to: { x: 2, y: 2 },
          player: 1,
        }),
        createMove({
          type: 'continue_capture_segment',
          from: { x: 2, y: 2 },
          to: { x: 4, y: 4 },
          player: 1,
        }),
      ],
    });

    const path = extractChainCapturePath(gameState);
    expect(path).toEqual([
      { x: 0, y: 0 },
      { x: 2, y: 2 },
      { x: 4, y: 4 },
    ]);
  });

  it('stops at non-capture move', () => {
    const gameState = createMinimalGameState({
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      moveHistory: [
        createMove({
          type: 'move_stack',
          from: { x: 1, y: 1 },
          to: { x: 2, y: 2 },
          player: 1,
        }),
        createMove({
          type: 'overtaking_capture',
          from: { x: 2, y: 2 },
          to: { x: 4, y: 4 },
          player: 1,
        }),
      ],
    });

    const path = extractChainCapturePath(gameState);
    expect(path).toEqual([
      { x: 2, y: 2 },
      { x: 4, y: 4 },
    ]);
  });

  it('returns undefined when path has fewer than 2 positions', () => {
    const gameState = createMinimalGameState({
      currentPhase: 'chain_capture',
      currentPlayer: 1,
      moveHistory: [
        createMove({
          type: 'move_stack',
          from: { x: 0, y: 0 },
          to: { x: 1, y: 1 },
          player: 1,
        }),
      ],
    });

    expect(extractChainCapturePath(gameState)).toBeUndefined();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// TESTS: deriveMustMoveFrom
// ═══════════════════════════════════════════════════════════════════════════

describe('deriveMustMoveFrom', () => {
  it('returns undefined when validMoves is empty', () => {
    const gameState = createMinimalGameState({ currentPhase: 'movement' });
    expect(deriveMustMoveFrom([], gameState)).toBeUndefined();
  });

  it('returns undefined when validMoves is undefined', () => {
    const gameState = createMinimalGameState({ currentPhase: 'movement' });
    expect(deriveMustMoveFrom(undefined as unknown as Move[], gameState)).toBeUndefined();
  });

  it('returns undefined when gameState is null', () => {
    const validMoves = [createMove({ from: { x: 0, y: 0 }, to: { x: 1, y: 1 } })];
    expect(deriveMustMoveFrom(validMoves, null)).toBeUndefined();
  });

  it('returns undefined when not in movement or capture phase', () => {
    const gameState = createMinimalGameState({ currentPhase: 'ring_placement' });
    const validMoves = [createMove({ from: { x: 0, y: 0 }, to: { x: 1, y: 1 } })];
    expect(deriveMustMoveFrom(validMoves, gameState)).toBeUndefined();
  });

  it('returns undefined in line_processing phase', () => {
    const gameState = createMinimalGameState({ currentPhase: 'line_processing' });
    const validMoves = [createMove({ from: { x: 0, y: 0 }, to: { x: 1, y: 1 } })];
    expect(deriveMustMoveFrom(validMoves, gameState)).toBeUndefined();
  });

  it('returns undefined when moves have different origins', () => {
    const gameState = createMinimalGameState({ currentPhase: 'movement' });
    const validMoves = [
      createMove({ type: 'move_stack', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } }),
      createMove({ type: 'move_stack', from: { x: 2, y: 2 }, to: { x: 3, y: 3 } }),
    ];
    expect(deriveMustMoveFrom(validMoves, gameState)).toBeUndefined();
  });

  it('returns the origin when all move_stack moves share the same from position', () => {
    const gameState = createMinimalGameState({ currentPhase: 'movement' });
    const validMoves = [
      createMove({ type: 'move_stack', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } }),
      createMove({ type: 'move_stack', from: { x: 0, y: 0 }, to: { x: 0, y: 1 } }),
      createMove({ type: 'move_stack', from: { x: 0, y: 0 }, to: { x: 1, y: 0 } }),
    ];
    expect(deriveMustMoveFrom(validMoves, gameState)).toEqual({ x: 0, y: 0 });
  });

  it('returns the origin when all overtaking_capture moves share the same from position', () => {
    const gameState = createMinimalGameState({ currentPhase: 'capture' });
    const validMoves = [
      createMove({ type: 'overtaking_capture', from: { x: 2, y: 2 }, to: { x: 4, y: 4 } }),
      createMove({ type: 'overtaking_capture', from: { x: 2, y: 2 }, to: { x: 0, y: 0 } }),
    ];
    expect(deriveMustMoveFrom(validMoves, gameState)).toEqual({ x: 2, y: 2 });
  });

  it('ignores moves without from position', () => {
    const gameState = createMinimalGameState({ currentPhase: 'movement' });
    const validMoves = [
      createMove({ type: 'move_stack', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } }),
      createMove({ type: 'skip_capture', from: undefined, to: { x: 0, y: 0 } }),
    ];
    expect(deriveMustMoveFrom(validMoves, gameState)).toEqual({ x: 0, y: 0 });
  });

  it('ignores non-movement/capture move types', () => {
    const gameState = createMinimalGameState({ currentPhase: 'movement' });
    const validMoves = [
      createMove({ type: 'move_stack', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } }),
      createMove({ type: 'place_ring', from: { x: 5, y: 5 }, to: { x: 5, y: 5 } }),
    ];
    expect(deriveMustMoveFrom(validMoves, gameState)).toEqual({ x: 0, y: 0 });
  });

  it('returns undefined when no movement/capture moves have from', () => {
    const gameState = createMinimalGameState({ currentPhase: 'movement' });
    const validMoves = [createMove({ type: 'skip_capture', from: undefined, to: { x: 0, y: 0 } })];
    expect(deriveMustMoveFrom(validMoves, gameState)).toBeUndefined();
  });

  it('handles positions with z coordinate', () => {
    const gameState = createMinimalGameState({ currentPhase: 'movement' });
    const validMoves = [
      createMove({ type: 'move_stack', from: { x: 0, y: 0, z: 1 }, to: { x: 1, y: 1 } }),
      createMove({ type: 'move_stack', from: { x: 0, y: 0, z: 1 }, to: { x: 0, y: 1 } }),
    ];
    expect(deriveMustMoveFrom(validMoves, gameState)).toEqual({ x: 0, y: 0, z: 1 });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// TESTS: canSubmitMove
// ═══════════════════════════════════════════════════════════════════════════

describe('canSubmitMove', () => {
  it('returns false when gameState is null', () => {
    const facade = createMockFacade({ gameState: null });
    expect(canSubmitMove(facade)).toBe(false);
  });

  it('returns false when game status is not active', () => {
    const inactiveGameState = createMinimalGameState({ gameStatus: 'finished' });
    const facade = createMockFacade({ gameState: inactiveGameState });
    expect(canSubmitMove(facade)).toBe(false);
  });

  it('returns false when user is not a player', () => {
    const facade = createMockFacade({ isPlayer: false });
    expect(canSubmitMove(facade)).toBe(false);
  });

  it('returns false when disconnected', () => {
    const facade = createMockFacade({ connectionStatus: 'disconnected' });
    expect(canSubmitMove(facade)).toBe(false);
  });

  it('returns true when all conditions are met', () => {
    const facade = createMockFacade({
      gameState: createMinimalGameState({ gameStatus: 'active' }),
      isPlayer: true,
      connectionStatus: 'connected',
    });
    expect(canSubmitMove(facade)).toBe(true);
  });

  it('returns true when connecting (not disconnected)', () => {
    const facade = createMockFacade({ connectionStatus: 'connecting' });
    expect(canSubmitMove(facade)).toBe(true);
  });

  it('returns true when reconnecting', () => {
    const facade = createMockFacade({ connectionStatus: 'reconnecting' });
    expect(canSubmitMove(facade)).toBe(true);
  });

  it('returns true for local-only mode', () => {
    const facade = createMockFacade({ connectionStatus: 'local-only' });
    expect(canSubmitMove(facade)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// TESTS: canInteract
// ═══════════════════════════════════════════════════════════════════════════

describe('canInteract', () => {
  it('returns false when gameState is null', () => {
    const facade = createMockFacade({ gameState: null });
    expect(canInteract(facade)).toBe(false);
  });

  it('returns false for spectator mode', () => {
    const facade = createMockFacade({ mode: 'spectator' });
    expect(canInteract(facade)).toBe(false);
  });

  it('returns false when disconnected in backend mode', () => {
    const facade = createMockFacade({
      mode: 'backend',
      connectionStatus: 'disconnected',
    });
    expect(canInteract(facade)).toBe(false);
  });

  it('returns true when disconnected in sandbox mode', () => {
    const facade = createMockFacade({
      mode: 'sandbox',
      connectionStatus: 'disconnected',
    });
    expect(canInteract(facade)).toBe(true);
  });

  it('returns true for normal backend mode with active connection', () => {
    const facade = createMockFacade({
      mode: 'backend',
      connectionStatus: 'connected',
    });
    expect(canInteract(facade)).toBe(true);
  });

  it('returns true for sandbox mode with local-only connection', () => {
    const facade = createMockFacade({
      mode: 'sandbox',
      connectionStatus: 'local-only',
    });
    expect(canInteract(facade)).toBe(true);
  });

  it('returns true for backend mode when reconnecting', () => {
    const facade = createMockFacade({
      mode: 'backend',
      connectionStatus: 'reconnecting',
    });
    expect(canInteract(facade)).toBe(true);
  });
});
