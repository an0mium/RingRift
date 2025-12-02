import {
  chooseLocalMoveFromCandidates,
  type LocalAIRng,
} from '../../src/shared/engine/localAIMoveSelection';
import type { GameState, Move, RingStack } from '../../src/shared/types/game';

function makeSeededRng(seed: number): LocalAIRng {
  let s = seed >>> 0;
  return () => {
    // Simple LCG: Numerical Recipes parameters
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function makeBaseGameState(): GameState {
  const now = new Date();
  return {
    id: 'swap-test-game',
    boardType: 'square8',
    board: {
      stacks: {} as unknown as Map<string, RingStack>, // overridden per test via `as any`
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8',
    } as any,
    players: [
      {
        id: 'p1',
        username: 'P1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'P2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ],
    currentPhase: 'movement',
    currentPlayer: 2,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 600, increment: 0, type: 'rapid' },
    spectators: [],
    gameStatus: 'active',
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 0,
    territoryVictoryThreshold: 0,
  };
}

function makeSwapCandidate(player: number = 2): Move {
  const now = new Date();
  return {
    id: 'swap-move',
    type: 'swap_sides',
    player,
    from: undefined,
    to: { x: 0, y: 0 },
    timestamp: now,
    thinkTime: 0,
    moveNumber: 1,
  };
}

function makeSimpleMoveCandidate(): Move {
  const now = new Date();
  return {
    id: 'simple-move',
    type: 'move_stack',
    player: 2,
    from: { x: 0, y: 1 },
    to: { x: 0, y: 2 },
    timestamp: now,
    thinkTime: 0,
    moveNumber: 1,
  };
}

describe('localAIMoveSelection swap behaviour', () => {
  it('prefers swap_sides for a strong center opening', () => {
    const state = makeBaseGameState();

    // Simulate a strong center opening: P1 stack in the 8x8 center cluster.
    const centerStack: RingStack = {
      position: { x: 3, y: 3 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    };

    (state.board as any).stacks = {
      '3,3': centerStack,
    };

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const rng = makeSeededRng(1234);

    const selected = chooseLocalMoveFromCandidates(2, state as any, [swapMove, simpleMove], rng, 0);

    expect(selected).toBe(swapMove);
  });

  it('declines swap_sides when opening provides no P1 stacks (neutral/weak opening)', () => {
    const state = makeBaseGameState();

    // No P1 stacks on board – evaluateSwapOpportunity returns 0 and
    // chooseLocalMoveFromCandidates must fall back to non-swap moves.
    (state.board as any).stacks = {};

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const rng = makeSeededRng(42);

    const selected = chooseLocalMoveFromCandidates(2, state as any, [swapMove, simpleMove], rng, 0);

    expect(selected).toBe(simpleMove);
  });

  it('is deterministic for randomness = 0 with a fixed LocalAIRng', () => {
    const state = makeBaseGameState();

    // Use a modest but clearly positive opening so swap is preferred.
    const centerStack: RingStack = {
      position: { x: 4, y: 4 },
      rings: [1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    };

    (state.board as any).stacks = {
      '4,4': centerStack,
    };

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const candidates: Move[] = [swapMove, simpleMove];

    const rng1 = makeSeededRng(999);
    const rng2 = makeSeededRng(999);

    const selected1 = chooseLocalMoveFromCandidates(2, state as any, [...candidates], rng1, 0);
    const selected2 = chooseLocalMoveFromCandidates(2, state as any, [...candidates], rng2, 0);

    expect(selected1?.id).toBe(selected2?.id);
  });

  it('with randomness > 0 and borderline opening, sometimes swaps and sometimes declines', () => {
    const state = makeBaseGameState();

    // Construct a borderline opening: P1 stack with zero height in a
    // non-center, non-adjacent position so the base swap value is 0.0.
    const borderlineStack: RingStack = {
      position: { x: 0, y: 0 },
      rings: [],
      stackHeight: 0,
      capHeight: 0,
      controllingPlayer: 1,
    };

    (state.board as any).stacks = {
      '0,0': borderlineStack,
    };

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const candidates: Move[] = [swapMove, simpleMove];

    const originalRandom = Math.random;
    try {
      // Seeded Math.random so evaluateSwapOpportunity's noise term is
      // reproducible and covers both positive and negative values over
      // multiple trials.
      let s = 123456789;
      Math.random = () => {
        s = (1103515245 * s + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
      };

      let swapCount = 0;
      let nonSwapCount = 0;

      for (let i = 0; i < 50; i++) {
        const rng = makeSeededRng(100 + i);
        const selected = chooseLocalMoveFromCandidates(
          2,
          state as any,
          [...candidates],
          rng,
          2.0 // large enough to produce both positive and negative swapValue
        );

        if (selected?.id === swapMove.id) {
          swapCount += 1;
        } else if (selected?.id === simpleMove.id) {
          nonSwapCount += 1;
        }
      }

      expect(swapCount).toBeGreaterThan(0);
      expect(nonSwapCount).toBeGreaterThan(0);
    } finally {
      Math.random = originalRandom;
    }
  });
});
