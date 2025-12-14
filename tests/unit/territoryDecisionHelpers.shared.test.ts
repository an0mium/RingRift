import {
  BoardType,
  Player,
  TimeControl,
  GameState,
  Position,
  Territory,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
  enumerateTerritoryEliminationMoves,
  applyEliminateRingsFromStackDecision,
} from '../../src/shared/engine/territoryDecisionHelpers';

describe('territoryDecisionHelpers – shared territory decision enumeration and application', () => {
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

  function createEmptyState(id: string): GameState {
    const state = createInitialGameState(
      id,
      boardType,
      players,
      timeControl
    ) as unknown as GameState;

    state.currentPlayer = 1;
    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();
    state.board.territories = new Map();
    state.board.eliminatedRings = {};
    state.board.formedLines = [];
    state.totalRingsEliminated = 0;
    state.players = state.players.map((p) => ({
      ...p,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));
    return state;
  }

  function snapshotS(state: GameState): number {
    return computeProgressSnapshot(state as any).S;
  }

  it('enumerateProcessTerritoryRegionMoves filters regions by self-elimination prerequisite and encodes geometry', () => {
    const state = createEmptyState('territory-enum');
    const board = state.board;

    // Two stacks for player 1 inside a larger region; only the smaller
    // region that does not cover all stacks satisfies the self-elimination
    // prerequisite. Under the canonical Q23 prerequisite (RR-CANON-R082),
    // at least one stack **outside** the region must be an eligible cap
    // target:
    //   - multicolour stack (stackHeight > capHeight), or
    //   - single-colour stack with height > 1.
    //
    // Here:
    //   - Stack at `a` is a single-ring stack (not eligible).
    //   - Stack at `b` is a height-2 single-colour stack (eligible).
    const a: Position = { x: 0, y: 0 };
    const b: Position = { x: 1, y: 0 };
    const aKey = positionToString(a);
    const bKey = positionToString(b);

    board.stacks.set(aKey, {
      position: a,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    } as any);
    board.stacks.set(bKey, {
      position: b,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as any);

    // Per canonical rules, territory regions must have a valid controllingPlayer
    // (player who owns the border markers that created the disconnection)
    const regionWithOutsideStack: Territory = {
      spaces: [a],
      controllingPlayer: 1,
      isDisconnected: true,
    };

    const regionWithoutOutsideStack: Territory = {
      spaces: [a, b],
      controllingPlayer: 1,
      isDisconnected: true,
    };

    const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
      testOverrideRegions: [regionWithOutsideStack, regionWithoutOutsideStack],
    });

    expect(moves).toHaveLength(1);
    const move = moves[0];
    expect(move.type).toBe('choose_territory_option');
    expect(move.player).toBe(1);
    expect(move.disconnectedRegions).toBeDefined();
    expect(move.disconnectedRegions!.length).toBe(1);

    const region = move.disconnectedRegions![0];
    expect(region.spaces.length).toBe(1);
    expect(region.spaces[0]).toEqual(a);
    expect(move.to).toEqual(a);
  });

  it('applyProcessTerritoryRegionDecision eliminates all internal rings, collapses region, credits gains, and sets pendingSelfElimination', () => {
    const state = createEmptyState('territory-apply-region');
    const board = state.board;

    // Region consists of two interior spaces; stacks inside the region may
    // belong to any player but all eliminations and territory gain are
    // credited to the acting player (player 1).
    const p1a: Position = { x: 0, y: 0 };
    const p2a: Position = { x: 1, y: 0 };
    const outside: Position = { x: 3, y: 3 };

    const p1aKey = positionToString(p1a);
    const p2aKey = positionToString(p2a);
    const outsideKey = positionToString(outside);

    // Two rings for player 1 inside the region.
    board.stacks.set(p1aKey, {
      position: p1a,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as any);

    // Three rings for player 2 inside the same region; these eliminations
    // are still credited to player 1 under the current rules.
    board.stacks.set(p2aKey, {
      position: p2a,
      rings: [2, 2, 2],
      stackHeight: 3,
      capHeight: 3,
      controllingPlayer: 2,
    } as any);

    // Single stack for player 1 outside the region to satisfy the
    // self-elimination prerequisite. Under RR-CANON-R082 the outside stack
    // must be an eligible cap target (height > 1 or multicolour).
    board.stacks.set(outsideKey, {
      position: outside,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as any);

    const region: Territory = {
      spaces: [p1a, p2a],
      // Canonical detector always attributes disconnected regions to the
      // border-marker owner; controllingPlayer 0 is considered non-canonical.
      controllingPlayer: 1,
      isDisconnected: true,
    };

    const move = {
      id: 'process-region-test',
      type: 'choose_territory_option' as const,
      player: 1,
      to: p1a,
      disconnectedRegions: [region],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const beforeS = snapshotS(state);
    const outcome = applyProcessTerritoryRegionDecision(state, move);
    const next = outcome.nextState;
    const afterS = snapshotS(next);

    expect(outcome.pendingSelfElimination).toBe(true);
    expect(outcome.processedRegion.spaces).toHaveLength(2);

    // All stacks inside the region have been eliminated.
    expect(next.board.stacks.has(p1aKey)).toBe(false);
    expect(next.board.stacks.has(p2aKey)).toBe(false);
    // Stack outside the region remains.
    expect(next.board.stacks.has(outsideKey)).toBe(true);

    // Territory gain: all interior spaces (no border markers in this setup).
    const p1After = next.players.find((p) => p.playerNumber === 1)!;
    expect(p1After.territorySpaces).toBe(2);
    const p2After = next.players.find((p) => p.playerNumber === 2)!;
    expect(p2After.territorySpaces).toBe(0);

    // Eliminated rings: 2 (player 1) + 3 (player 2) credited entirely to player 1.
    expect(next.board.eliminatedRings[1]).toBe(5);
    expect(next.totalRingsEliminated).toBe(5);
    expect(p1After.eliminatedRings).toBe(5);

    // Region spaces are now collapsed territory for player 1.
    expect(next.board.collapsedSpaces.get(p1aKey)).toBe(1);
    expect(next.board.collapsedSpaces.get(p2aKey)).toBe(1);

    // S-invariant (markers + collapsed + eliminated) is non-decreasing.
    expect(afterS).toBeGreaterThanOrEqual(beforeS);
  });

  it('enumerateTerritoryEliminationMoves surfaces one elimination per eligible stack and applyEliminateRingsFromStackDecision updates caps and S-invariant', () => {
    const state = createEmptyState('territory-elim');
    const board = state.board;

    const a: Position = { x: 0, y: 0 };
    const b: Position = { x: 1, y: 0 };
    const c: Position = { x: 2, y: 0 };

    const aKey = positionToString(a);
    const bKey = positionToString(b);
    const cKey = positionToString(c);

    // Two stacks controlled by player 1 with non-zero caps.
    board.stacks.set(aKey, {
      position: a,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as any);

    board.stacks.set(bKey, {
      position: b,
      rings: [1, 2, 1],
      stackHeight: 3,
      capHeight: 1, // only the top ring belongs to player 1
      controllingPlayer: 1,
    } as any);

    // Opponent stack should not generate an elimination move for player 1.
    board.stacks.set(cKey, {
      position: c,
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    } as any);

    // Ensure board-level elimination counters start from a known baseline.
    state.board.eliminatedRings = { 1: 0, 2: 0 };
    state.totalRingsEliminated = 0;
    state.players = state.players.map((p) =>
      p.playerNumber === 1 ? { ...p, eliminatedRings: 0 } : p
    );

    const beforeS = snapshotS(state);
    const moves = enumerateTerritoryEliminationMoves(state, 1);

    // Exactly one elimination move per eligible stack controlled by player 1.
    expect(moves.length).toBe(2);
    const ids = moves.map((m) => m.id).sort();
    expect(ids).toEqual([`eliminate-${aKey}`, `eliminate-${bKey}`].sort());

    moves.forEach((m) => {
      expect(m.type).toBe('eliminate_rings_from_stack');
      expect(m.player).toBe(1);
      expect(m.to).toBeDefined();
      expect(m.eliminationFromStack).toBeDefined();
      const snapshot = m.eliminationFromStack!;
      expect(snapshot.capHeight).toBeGreaterThan(0);
      expect(snapshot.totalHeight).toBeGreaterThanOrEqual(snapshot.capHeight);
    });

    // Apply elimination from stack A and verify counters and geometry.
    const aMove = moves.find((m) => positionToString(m.to!) === aKey)!;
    const outcome = applyEliminateRingsFromStackDecision(state, aMove);
    const next = outcome.nextState;
    const afterS = snapshotS(next);

    // Stack at A is fully removed (pure cap).
    expect(next.board.stacks.has(aKey)).toBe(false);

    // Board and player elimination counters updated by the cap height (2).
    expect(next.board.eliminatedRings[1]).toBe(2);
    const p1After = next.players.find((p) => p.playerNumber === 1)!;
    expect(p1After.eliminatedRings).toBe(2);
    expect(next.totalRingsEliminated).toBe(2);

    // Other stacks remain untouched.
    expect(next.board.stacks.has(bKey)).toBe(true);
    expect(next.board.stacks.has(cKey)).toBe(true);

    // S-invariant is non-decreasing after elimination.
    expect(afterS).toBeGreaterThanOrEqual(beforeS);
  });

  /**
   * RR-CANON-R022, R122: Line vs Territory Elimination Distinction Tests
   *
   * These tests verify the canonical distinction between:
   * - Line elimination (RR-CANON-R122): Eliminates exactly ONE ring from ANY controlled stack
   *   (including height-1 standalone rings)
   * - Territory elimination (RR-CANON-R145): Eliminates entire cap from ELIGIBLE stacks only
   *   (multicolor stacks or single-color stacks with height > 1)
   */
  describe('Line vs Territory Elimination Distinction (RR-CANON-R022, R122, R145)', () => {
    it('line elimination context includes height-1 standalone rings as eligible targets', () => {
      const state = createEmptyState('line-elim-height-1');
      const board = state.board;

      const a: Position = { x: 0, y: 0 };
      const aKey = positionToString(a);

      // Single height-1 ring controlled by player 1
      board.stacks.set(aKey, {
        position: a,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      } as any);

      // Line elimination context: height-1 rings ARE eligible (per RR-CANON-R122)
      const lineMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'line',
      });
      expect(lineMoves.length).toBe(1);
      expect(lineMoves[0].eliminationContext).toBe('line');

      // Territory elimination context: height-1 rings ARE now eligible (per RR-CANON-R145)
      const territoryMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'territory',
      });
      expect(territoryMoves.length).toBe(1);
      expect(territoryMoves[0].eliminationContext).toBe('territory');
    });

    it('line elimination removes exactly ONE ring, not entire cap', () => {
      const state = createEmptyState('line-elim-one-ring');
      const board = state.board;

      const a: Position = { x: 0, y: 0 };
      const aKey = positionToString(a);

      // Height-3 stack controlled by player 1
      board.stacks.set(aKey, {
        position: a,
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      } as any);

      state.board.eliminatedRings = { 1: 0 };
      state.totalRingsEliminated = 0;

      const lineMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'line',
      });
      expect(lineMoves.length).toBe(1);

      const lineMove = lineMoves[0];
      expect(lineMove.eliminationContext).toBe('line');
      expect(lineMove.eliminatedRings![0].count).toBe(1); // Only 1 ring listed

      // Apply the line elimination
      const outcome = applyEliminateRingsFromStackDecision(state, lineMove);
      const next = outcome.nextState;

      // Stack should still exist with 2 remaining rings
      expect(next.board.stacks.has(aKey)).toBe(true);
      const remainingStack = next.board.stacks.get(aKey)!;
      expect(remainingStack.stackHeight).toBe(2);
      expect(remainingStack.rings).toEqual([1, 1]);

      // Only 1 ring was eliminated
      expect(next.board.eliminatedRings[1]).toBe(1);
      expect(next.totalRingsEliminated).toBe(1);
    });

    it('territory elimination removes entire cap', () => {
      const state = createEmptyState('territory-elim-full-cap');
      const board = state.board;

      const a: Position = { x: 0, y: 0 };
      const aKey = positionToString(a);

      // Height-3 stack controlled by player 1
      board.stacks.set(aKey, {
        position: a,
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      } as any);

      state.board.eliminatedRings = { 1: 0 };
      state.totalRingsEliminated = 0;

      const territoryMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'territory',
      });
      expect(territoryMoves.length).toBe(1);

      const territoryMove = territoryMoves[0];
      expect(territoryMove.eliminationContext).toBe('territory');

      // Apply the territory elimination
      const outcome = applyEliminateRingsFromStackDecision(state, territoryMove);
      const next = outcome.nextState;

      // Stack should be completely removed (entire cap eliminated)
      expect(next.board.stacks.has(aKey)).toBe(false);

      // All 3 rings were eliminated
      expect(next.board.eliminatedRings[1]).toBe(3);
      expect(next.totalRingsEliminated).toBe(3);
    });

    it('line elimination on multicolor stack removes exactly 1 ring from cap', () => {
      const state = createEmptyState('line-elim-multicolor');
      const board = state.board;

      const a: Position = { x: 0, y: 0 };
      const aKey = positionToString(a);

      // Multicolor stack: player 1 controls cap (2 rings), player 2's ring buried
      board.stacks.set(aKey, {
        position: a,
        rings: [1, 1, 2],
        stackHeight: 3,
        capHeight: 2,
        controllingPlayer: 1,
      } as any);

      state.board.eliminatedRings = { 1: 0, 2: 0 };
      state.totalRingsEliminated = 0;

      const lineMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'line',
      });
      expect(lineMoves.length).toBe(1);

      const lineMove = lineMoves[0];
      const outcome = applyEliminateRingsFromStackDecision(state, lineMove);
      const next = outcome.nextState;

      // Stack should still exist with 2 remaining rings
      expect(next.board.stacks.has(aKey)).toBe(true);
      const remainingStack = next.board.stacks.get(aKey)!;
      expect(remainingStack.stackHeight).toBe(2);
      expect(remainingStack.rings).toEqual([1, 2]); // One ring removed from top

      // Only 1 ring was eliminated
      expect(next.board.eliminatedRings[1]).toBe(1);
      expect(next.totalRingsEliminated).toBe(1);
    });

    it('territory elimination on multicolor stack removes entire cap exposing buried rings', () => {
      const state = createEmptyState('territory-elim-multicolor');
      const board = state.board;

      const a: Position = { x: 0, y: 0 };
      const aKey = positionToString(a);

      // Multicolor stack: player 1 controls cap (2 rings), player 2's ring buried
      board.stacks.set(aKey, {
        position: a,
        rings: [1, 1, 2],
        stackHeight: 3,
        capHeight: 2,
        controllingPlayer: 1,
      } as any);

      state.board.eliminatedRings = { 1: 0, 2: 0 };
      state.totalRingsEliminated = 0;

      const territoryMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'territory',
      });
      expect(territoryMoves.length).toBe(1);

      const territoryMove = territoryMoves[0];
      const outcome = applyEliminateRingsFromStackDecision(state, territoryMove);
      const next = outcome.nextState;

      // Stack should still exist but now controlled by player 2
      expect(next.board.stacks.has(aKey)).toBe(true);
      const remainingStack = next.board.stacks.get(aKey)!;
      expect(remainingStack.stackHeight).toBe(1);
      expect(remainingStack.rings).toEqual([2]); // Only buried ring remains
      expect(remainingStack.controllingPlayer).toBe(2);

      // 2 rings of player 1's cap were eliminated
      expect(next.board.eliminatedRings[1]).toBe(2);
      expect(next.totalRingsEliminated).toBe(2);
    });

    it('forced elimination context allows any stack including height-1', () => {
      const state = createEmptyState('forced-elim-height-1');
      const board = state.board;

      const a: Position = { x: 0, y: 0 };
      const aKey = positionToString(a);

      // Single height-1 ring
      board.stacks.set(aKey, {
        position: a,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      } as any);

      // Forced elimination: height-1 rings ARE eligible (per RR-CANON-R100)
      const forcedMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'forced',
      });
      expect(forcedMoves.length).toBe(1);
      expect(forcedMoves[0].eliminationContext).toBe('forced');
    });

    it('forced elimination removes entire cap like territory elimination', () => {
      const state = createEmptyState('forced-elim-full-cap');
      const board = state.board;

      const a: Position = { x: 0, y: 0 };
      const aKey = positionToString(a);

      // Height-2 stack
      board.stacks.set(aKey, {
        position: a,
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      } as any);

      state.board.eliminatedRings = { 1: 0 };
      state.totalRingsEliminated = 0;

      const forcedMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'forced',
      });
      expect(forcedMoves.length).toBe(1);

      const forcedMove = forcedMoves[0];
      const outcome = applyEliminateRingsFromStackDecision(state, forcedMove);
      const next = outcome.nextState;

      // Stack should be completely removed
      expect(next.board.stacks.has(aKey)).toBe(false);

      // All 2 rings were eliminated
      expect(next.board.eliminatedRings[1]).toBe(2);
      expect(next.totalRingsEliminated).toBe(2);
    });

    it('default elimination context (undefined) behaves like territory elimination', () => {
      const state = createEmptyState('default-elim-context');
      const board = state.board;

      const a: Position = { x: 0, y: 0 };
      const aKey = positionToString(a);

      // Height-1 stack IS eligible with default context (territory rules now allow it)
      board.stacks.set(aKey, {
        position: a,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      } as any);

      // No scope provided means default to territory rules - height-1 now eligible
      const defaultMoves = enumerateTerritoryEliminationMoves(state, 1);
      expect(defaultMoves.length).toBe(1); // Height-1 now eligible
    });

    it('multiple stacks with mixed eligibility are correctly filtered by context', () => {
      const state = createEmptyState('mixed-eligibility');
      const board = state.board;

      const a: Position = { x: 0, y: 0 }; // height-1 standalone
      const b: Position = { x: 1, y: 0 }; // height-2 single color (eligible for both)
      const c: Position = { x: 2, y: 0 }; // multicolor (eligible for both)

      const aKey = positionToString(a);
      const bKey = positionToString(b);
      const cKey = positionToString(c);

      board.stacks.set(aKey, {
        position: a,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      } as any);

      board.stacks.set(bKey, {
        position: b,
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      } as any);

      board.stacks.set(cKey, {
        position: c,
        rings: [1, 2],
        stackHeight: 2,
        capHeight: 1,
        controllingPlayer: 1,
      } as any);

      // Line context: all 3 stacks eligible
      const lineMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'line',
      });
      expect(lineMoves.length).toBe(3);

      // Territory context: all 3 stacks eligible (including height-1 standalone at a)
      const territoryMoves = enumerateTerritoryEliminationMoves(state, 1, {
        eliminationContext: 'territory',
      });
      expect(territoryMoves.length).toBe(3);
      const territoryKeys = territoryMoves.map((m) => positionToString(m.to!)).sort();
      expect(territoryKeys).toEqual([aKey, bKey, cKey].sort());
    });
  });
});
