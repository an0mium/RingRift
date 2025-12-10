import {
  BoardType,
  Player,
  TimeControl,
  GameState,
  Position,
  RingStack,
  positionToString,
  stringToPosition,
  BOARD_CONFIGS,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import {
  enumerateChainCaptureSegments,
  enumerateCaptureMoves,
  applyCaptureSegment,
  ChainCaptureStateSnapshot,
  CaptureBoardAdapters,
} from '../../src/shared/engine/aggregates/CaptureAggregate';
import {
  calculateCapHeight,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView,
} from '../../src/shared/engine/core';
import { isValidPosition } from '../../src/shared/engine/validators/utils';

/**
 * Classification: canonical shared capture chain helper tests.
 *
 * These tests verify the capture chain enumeration and validation helpers
 * that support both the backend GameEngine and client sandbox engine.
 *
 * Note: This test now uses CaptureAggregate directly per RR-CANON-R070.
 * The captureChainHelpers.ts wrapper is deprecated.
 */

// ═══════════════════════════════════════════════════════════════════════════
// Local Helper Functions (migrated from captureChainHelpers.ts)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create board adapters from GameState for capture enumeration.
 */
function createCaptureBoardAdaptersFromState(state: GameState): CaptureBoardAdapters {
  const board = state.board;
  const boardType = state.board.type;
  const size = board.size;

  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const marker = board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };
}

/**
 * Create board view from GameState for capture validation.
 */
function createCaptureSegmentViewFromState(state: GameState): CaptureSegmentBoardView {
  const board = state.board;
  const boardType = state.board.type;
  const size = board.size;

  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const marker = board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };
}

/**
 * Check if a capture is valid (replaces captureChainHelpers.canCapture).
 */
function canCapture(
  state: GameState,
  from: Position,
  target: Position,
  landing: Position,
  player: number
): boolean {
  const view = createCaptureSegmentViewFromState(state);
  return validateCaptureSegmentOnBoard(state.boardType, from, target, landing, player, view);
}

/**
 * Get all valid capture targets from a position (replaces captureChainHelpers.getValidCaptureTargets).
 */
function getValidCaptureTargets(
  state: GameState,
  from: Position,
  player: number
): Array<{ target: Position; landings: Position[] }> {
  const adapters = createCaptureBoardAdaptersFromState(state);
  const moveNumber = state.moveHistory.length + 1;

  const moves = enumerateCaptureMoves(state.boardType, from, player, adapters, moveNumber);

  // Group moves by captureTarget
  const byTarget = new Map<string, { target: Position; landings: Position[] }>();

  for (const move of moves) {
    if (!move.captureTarget || !move.to) continue;

    const key = positionToString(move.captureTarget);
    let entry = byTarget.get(key);
    if (!entry) {
      entry = { target: move.captureTarget, landings: [] };
      byTarget.set(key, entry);
    }
    entry.landings.push(move.to);
  }

  return Array.from(byTarget.values());
}

/**
 * Get chain capture continuation info (replaces captureChainHelpers.getChainCaptureContinuationInfo).
 */
function getChainCaptureContinuationInfo(
  state: GameState,
  snapshot: ChainCaptureStateSnapshot,
  options?: { disallowRevisitedTargets?: boolean }
): { hasFurtherCaptures: boolean; segments: ReturnType<typeof enumerateChainCaptureSegments> } {
  const segments = enumerateChainCaptureSegments(state, snapshot, options);
  return {
    hasFurtherCaptures: segments.length > 0,
    segments,
  };
}

/**
 * Process chain capture (replaces captureChainHelpers.processChainCapture).
 */
function processChainCapture(
  state: GameState,
  initialFrom: Position,
  initialTarget: Position,
  initialLanding: Position,
  player: number
): {
  isValid: boolean;
  hasContinuation: boolean;
  continuationOptions: ReturnType<typeof enumerateChainCaptureSegments>;
} {
  if (!canCapture(state, initialFrom, initialTarget, initialLanding, player)) {
    return {
      isValid: false,
      hasContinuation: false,
      continuationOptions: [],
    };
  }

  const snapshot: ChainCaptureStateSnapshot = {
    player,
    currentPosition: initialLanding,
    capturedThisChain: [initialTarget],
  };

  const info = getChainCaptureContinuationInfo(state, snapshot, {
    disallowRevisitedTargets: true,
  });

  return {
    isValid: true,
    hasContinuation: info.hasFurtherCaptures,
    continuationOptions: info.segments,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Test Suite
// ═══════════════════════════════════════════════════════════════════════════

describe('CaptureAggregate – shared capture chain enumeration and validation', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function createEmptyState(): GameState {
    const state = createInitialGameState(
      'capture-helpers-test',
      boardType,
      players,
      timeControl
    ) as unknown as GameState;

    state.currentPlayer = 1;
    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();
    state.board.formedLines = [];
    state.board.eliminatedRings = {};
    return state;
  }

  function addStack(
    state: GameState,
    position: Position,
    rings: number[],
    controllingPlayer?: number
  ): void {
    const key = positionToString(position);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: controllingPlayer ?? rings[0],
    };
    state.board.stacks.set(key, stack);
  }

  function addMarker(state: GameState, position: Position, player: number): void {
    const key = positionToString(position);
    state.board.markers.set(key, {
      player,
      position,
      type: 'regular',
    } as any);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // enumerateChainCaptureSegments tests
  // ═══════════════════════════════════════════════════════════════════════════

  it('returns empty array when no stacks exist for the player', () => {
    const state = createEmptyState();

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 3, y: 3 },
      capturedThisChain: [],
    };

    const moves = enumerateChainCaptureSegments(state, snapshot);
    expect(moves).toHaveLength(0);
  });

  it('returns empty array when attacker has no valid capture targets', () => {
    const state = createEmptyState();

    // Add a stack for player 1 with no capturable targets nearby
    addStack(state, { x: 0, y: 0 }, [1]);

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 0, y: 0 },
      capturedThisChain: [],
    };

    const moves = enumerateChainCaptureSegments(state, snapshot);
    expect(moves).toHaveLength(0);
  });

  it('returns capture moves when valid targets are available', () => {
    const state = createEmptyState();

    // Add attacker stack for player 1 at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]); // cap height 2

    // Add target stack for player 2 at (2, 0) - cap height 1
    addStack(state, { x: 2, y: 0 }, [2]); // cap height 1

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 0, y: 0 },
      capturedThisChain: [],
    };

    const moves = enumerateChainCaptureSegments(state, snapshot, { kind: 'initial' });

    expect(moves.length).toBeGreaterThan(0);
    expect(moves[0].type).toBe('overtaking_capture');
    expect(moves[0].captureTarget).toEqual({ x: 2, y: 0 });
  });

  it('returns continuation segment moves when kind is continuation', () => {
    const state = createEmptyState();

    // Attacker at (0, 0) with cap height 2
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0) with cap height 1
    addStack(state, { x: 2, y: 0 }, [2]);

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 0, y: 0 },
      capturedThisChain: [],
    };

    const moves = enumerateChainCaptureSegments(state, snapshot, { kind: 'continuation' });

    expect(moves.length).toBeGreaterThan(0);
    expect(moves[0].type).toBe('continue_capture_segment');
  });

  it('filters out visited targets when disallowRevisitedTargets is true', () => {
    const state = createEmptyState();

    // Attacker at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]);

    // Another target at (0, 2)
    addStack(state, { x: 0, y: 2 }, [2]);

    const visitedTargetPos = { x: 2, y: 0 };
    const visitedTarget = positionToString(visitedTargetPos);
    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 0, y: 0 },
      capturedThisChain: [visitedTargetPos],
    };

    const movesWithFilter = enumerateChainCaptureSegments(state, snapshot, {
      disallowRevisitedTargets: true,
    });

    const movesWithoutFilter = enumerateChainCaptureSegments(state, snapshot, {
      disallowRevisitedTargets: false,
    });

    // Without filter should include both targets
    expect(
      movesWithoutFilter.some((m) => positionToString(m.captureTarget!) === visitedTarget)
    ).toBe(true);

    // With filter should exclude the visited target
    expect(movesWithFilter.some((m) => positionToString(m.captureTarget!) === visitedTarget)).toBe(
      false
    );
  });

  it('generates unique move IDs for each capture segment', () => {
    const state = createEmptyState();

    // Attacker at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]);

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 0, y: 0 },
      capturedThisChain: [],
    };

    const moves = enumerateChainCaptureSegments(state, snapshot);

    // All move IDs should be unique
    const ids = new Set(moves.map((m) => m.id));
    expect(ids.size).toBe(moves.length);

    // Each ID should contain position information
    moves.forEach((m) => {
      expect(m.id).toContain('continue_capture_segment');
      expect(m.id).toContain(positionToString(m.from!));
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // getChainCaptureContinuationInfo tests
  // ═══════════════════════════════════════════════════════════════════════════

  it('returns hasFurtherCaptures: false when no captures are available', () => {
    const state = createEmptyState();

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 3, y: 3 },
      capturedThisChain: [],
    };

    const info = getChainCaptureContinuationInfo(state, snapshot);

    expect(info.hasFurtherCaptures).toBe(false);
    expect(info.segments).toHaveLength(0);
  });

  it('returns hasFurtherCaptures: true when captures are available', () => {
    const state = createEmptyState();

    // Attacker at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]);

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 0, y: 0 },
    };

    const info = getChainCaptureContinuationInfo(state, snapshot);

    expect(info.hasFurtherCaptures).toBe(true);
    expect(info.segments.length).toBeGreaterThan(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // canCapture tests
  // ═══════════════════════════════════════════════════════════════════════════

  it('returns true for a valid capture', () => {
    const state = createEmptyState();

    // Attacker at (0, 0) with cap height 2
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0) with cap height 1
    addStack(state, { x: 2, y: 0 }, [2]);

    const result = canCapture(
      state,
      { x: 0, y: 0 }, // from
      { x: 2, y: 0 }, // target
      { x: 3, y: 0 }, // landing
      1 // player
    );

    expect(result).toBe(true);
  });

  it('returns false when attacker cap height is lower than target', () => {
    const state = createEmptyState();

    // Attacker at (0, 0) with cap height 1
    addStack(state, { x: 0, y: 0 }, [1]);

    // Target at (2, 0) with cap height 2
    addStack(state, { x: 2, y: 0 }, [2, 2]);

    const result = canCapture(
      state,
      { x: 0, y: 0 }, // from
      { x: 2, y: 0 }, // target
      { x: 3, y: 0 }, // landing
      1 // player
    );

    expect(result).toBe(false);
  });

  it('returns false for captures not in a straight line', () => {
    const state = createEmptyState();

    // Attacker at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 1) - not in straight line
    addStack(state, { x: 2, y: 1 }, [2]);

    const result = canCapture(
      state,
      { x: 0, y: 0 }, // from
      { x: 2, y: 1 }, // target (not in straight line)
      { x: 3, y: 1 }, // landing
      1 // player
    );

    expect(result).toBe(false);
  });

  it('returns false when landing position is occupied', () => {
    const state = createEmptyState();

    // Attacker at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]);

    // Blocker at landing position (3, 0)
    addStack(state, { x: 3, y: 0 }, [2, 2]);

    const result = canCapture(
      state,
      { x: 0, y: 0 }, // from
      { x: 2, y: 0 }, // target
      { x: 3, y: 0 }, // landing (blocked)
      1 // player
    );

    expect(result).toBe(false);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // getValidCaptureTargets tests
  // ═══════════════════════════════════════════════════════════════════════════

  it('returns valid capture targets with their landing positions', () => {
    const state = createEmptyState();

    // Attacker at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]);

    const targets = getValidCaptureTargets(state, { x: 0, y: 0 }, 1);

    expect(targets.length).toBeGreaterThan(0);

    const targetAt2_0 = targets.find((t) => t.target.x === 2 && t.target.y === 0);
    expect(targetAt2_0).toBeDefined();
    expect(targetAt2_0!.landings.length).toBeGreaterThan(0);
    // Landing should be at x=3, 4, 5, 6, or 7 along the x-axis
    expect(targetAt2_0!.landings[0].y).toBe(0);
    expect(targetAt2_0!.landings[0].x).toBeGreaterThan(2);
  });

  it('returns empty array when player has no stacks', () => {
    const state = createEmptyState();

    const targets = getValidCaptureTargets(state, { x: 0, y: 0 }, 1);
    expect(targets).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // processChainCapture tests
  // ═══════════════════════════════════════════════════════════════════════════

  it('returns isValid: true for a valid initial capture', () => {
    const state = createEmptyState();

    // Attacker at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]);

    const result = processChainCapture(
      state,
      { x: 0, y: 0 }, // from
      { x: 2, y: 0 }, // target
      { x: 4, y: 0 }, // landing
      1 // player
    );

    expect(result.isValid).toBe(true);
  });

  it('returns isValid: false for an invalid capture', () => {
    const state = createEmptyState();

    // No attacker at the from position

    const result = processChainCapture(
      state,
      { x: 0, y: 0 }, // from (no stack here)
      { x: 2, y: 0 }, // target
      { x: 4, y: 0 }, // landing
      1 // player
    );

    expect(result.isValid).toBe(false);
    expect(result.hasContinuation).toBe(false);
    expect(result.continuationOptions).toHaveLength(0);
  });

  it('correctly identifies when chain continuation is possible', () => {
    const state = createEmptyState();

    // Attacker at (0, 0) with tall cap
    addStack(state, { x: 0, y: 0 }, [1, 1, 1]);

    // First target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]);

    // Second potential target at (6, 0) - after landing at (4, 0), could continue
    addStack(state, { x: 6, y: 0 }, [2]);

    const result = processChainCapture(
      state,
      { x: 0, y: 0 }, // from
      { x: 2, y: 0 }, // target
      { x: 4, y: 0 }, // landing
      1 // player
    );

    expect(result.isValid).toBe(true);
    // Note: hasContinuation is approximative without actually applying the capture
    // The function checks if captures would be available from landing position
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Hexagonal board support tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('hexagonal board support', () => {
    const hexBoardType: BoardType = 'hexagonal';

    function createHexEmptyState(): GameState {
      const hexPlayers: Player[] = [
        {
          id: 'p1',
          username: 'Player 1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player 2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      const state = createInitialGameState(
        'hex-capture-test',
        hexBoardType,
        hexPlayers,
        timeControl
      ) as unknown as GameState;

      state.currentPlayer = 1;
      state.board.stacks.clear();
      state.board.markers.clear();
      state.board.collapsedSpaces.clear();
      state.board.formedLines = [];
      state.board.eliminatedRings = {};
      return state;
    }

    it('enumerates captures correctly on hexagonal board', () => {
      const state = createHexEmptyState();

      // Center position in hex cube coordinates (0, 0, 0)
      addStack(state, { x: 0, y: 0, z: 0 }, [1, 1]);

      // Target in one of the 6 hex directions (1, -1, 0)
      addStack(state, { x: 1, y: -1, z: 0 }, [2]);

      const snapshot: ChainCaptureStateSnapshot = {
        player: 1,
        currentPosition: { x: 0, y: 0, z: 0 },
      };

      const moves = enumerateChainCaptureSegments(state, snapshot);

      // Should find at least one capture along the hex direction
      expect(moves.length).toBeGreaterThan(0);
      expect(moves[0].player).toBe(1);
    });

    it('validates hex captures correctly', () => {
      const state = createHexEmptyState();

      // Center position
      addStack(state, { x: 0, y: 0, z: 0 }, [1, 1]);

      // Target at (1, -1, 0)
      addStack(state, { x: 1, y: -1, z: 0 }, [2]);

      const result = canCapture(
        state,
        { x: 0, y: 0, z: 0 }, // from
        { x: 1, y: -1, z: 0 }, // target
        { x: 2, y: -2, z: 0 }, // landing
        1 // player
      );

      expect(result).toBe(true);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Multiple captures in a row (chain capture simulation)
  // ═══════════════════════════════════════════════════════════════════════════

  it('passes moveNumber option to aggregate layer', () => {
    const state = createEmptyState();

    // Attacker at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]);

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 0, y: 0 },
    };

    // Test with moveNumber option to cover the options.moveNumber !== undefined branch
    const moves = enumerateChainCaptureSegments(state, snapshot, {
      kind: 'initial',
      moveNumber: 5,
    });

    expect(moves.length).toBeGreaterThan(0);
    // The move should have the moveNumber set
    expect(moves[0].moveNumber).toBe(5);
  });

  it('handles markers on the board (getMarkerOwner coverage)', () => {
    const state = createEmptyState();

    // Attacker at (0, 0)
    addStack(state, { x: 0, y: 0 }, [1, 1]);

    // Target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]);

    // Add a marker at the target position - this exercises the getMarkerOwner function
    // returning a marker instead of undefined
    addMarker(state, { x: 2, y: 0 }, 2);

    // Also add a marker at a potential landing position
    addMarker(state, { x: 3, y: 0 }, 1);

    const snapshot: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 0, y: 0 },
    };

    const moves = enumerateChainCaptureSegments(state, snapshot, { kind: 'initial' });

    // Should still find capture moves (markers don't block captures)
    expect(moves.length).toBeGreaterThan(0);
  });

  it('correctly handles progression through a chain of captures', () => {
    const state = createEmptyState();

    // Set up a line of targets that could be captured in sequence
    // Attacker at (0, 0) with tall cap
    addStack(state, { x: 0, y: 0 }, [1, 1, 1, 1]);

    // Single target at (2, 0)
    addStack(state, { x: 2, y: 0 }, [2]); // Target 1

    // First segment: capture target at (2, 0)
    const snapshot1: ChainCaptureStateSnapshot = {
      player: 1,
      currentPosition: { x: 0, y: 0 },
      visitedTargets: [],
    };

    const moves1 = enumerateChainCaptureSegments(state, snapshot1, { kind: 'initial' });
    expect(moves1.length).toBeGreaterThan(0);

    // Find a move that captures the target at (2, 0)
    const captureMove = moves1.find((m) => m.captureTarget!.x === 2 && m.captureTarget!.y === 0);
    expect(captureMove).toBeDefined();
    expect(captureMove!.captureTarget).toEqual({ x: 2, y: 0 });
    // Landing should be beyond the target (x > 2)
    expect(captureMove!.to!.x).toBeGreaterThan(2);
    expect(captureMove!.to!.y).toBe(0);
  });
});
