/**
 * turnLogic.branchCoverage.test.ts
 *
 * Branch coverage tests for turnLogic.ts targeting uncovered branches:
 * - Game status check (not active early return)
 * - Phase switch cases (ring_placement, movement, capture, chain_capture, line_processing, territory_processing, default)
 * - ring_placement: canMove || canCapture ternary
 * - territory_processing: forced elimination, game ended after elimination, player skip loop, starting phase determination
 */

import {
  advanceTurnAndPhase,
  type TurnLogicDelegates,
  type PerTurnState,
} from '../../src/shared/engine/turnLogic';
import type { GameState, Position, GamePhase } from '../../src/shared/types/game';

// Helper to create a position
const pos = (x: number, y: number): Position => ({ x, y });

// Helper to create a minimal GameState
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  return {
    id: 'test-game',
    boardType: 'square8',
    board: {
      type: 'square8',
      size: 8,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      formedLines: [],
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
    currentPhase: 'ring_placement',
    moveHistory: [],
    history: [],
    gameStatus: 'active',
    winner: undefined,
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    spectators: [],
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 15,
    territoryVictoryThreshold: 8,
    ...overrides,
  } as GameState;
}

// Helper to create a minimal PerTurnState
function makeTurnState(overrides: Partial<PerTurnState> = {}): PerTurnState {
  return {
    hasPlacedThisTurn: false,
    mustMoveFromStackKey: undefined,
    ...overrides,
  };
}

// Mock delegates factory
function makeDelegates(overrides: Partial<TurnLogicDelegates> = {}): TurnLogicDelegates {
  return {
    getPlayerStacks: jest.fn().mockReturnValue([]),
    hasAnyPlacement: jest.fn().mockReturnValue(true),
    hasAnyMovement: jest.fn().mockReturnValue(true),
    hasAnyCapture: jest.fn().mockReturnValue(false),
    applyForcedElimination: jest.fn((state) => state),
    getNextPlayerNumber: jest.fn((state, current) => (current % state.players.length) + 1),
    ...overrides,
  };
}

describe('turnLogic branch coverage', () => {
  describe('game status check', () => {
    it('returns unchanged state when game is completed', () => {
      const state = makeGameState({ gameStatus: 'completed' });
      const turn = makeTurnState();
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState).toBe(state);
      expect(result.nextTurn).toBe(turn);
    });

    it('returns unchanged state when game is abandoned', () => {
      const state = makeGameState({ gameStatus: 'abandoned' });
      const turn = makeTurnState();
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState).toBe(state);
      expect(result.nextTurn).toBe(turn);
    });

    it('processes state when game is active', () => {
      const state = makeGameState({ gameStatus: 'active', currentPhase: 'line_processing' });
      const turn = makeTurnState();
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPhase).toBe('territory_processing');
    });
  });

  describe('ring_placement phase', () => {
    it('transitions to movement when player can move', () => {
      const state = makeGameState({ currentPhase: 'ring_placement' });
      const turn = makeTurnState();
      const delegates = makeDelegates({
        hasAnyMovement: jest.fn().mockReturnValue(true),
        hasAnyCapture: jest.fn().mockReturnValue(false),
      });

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPhase).toBe('movement');
    });

    it('transitions to movement when player can capture', () => {
      const state = makeGameState({ currentPhase: 'ring_placement' });
      const turn = makeTurnState();
      const delegates = makeDelegates({
        hasAnyMovement: jest.fn().mockReturnValue(false),
        hasAnyCapture: jest.fn().mockReturnValue(true),
      });

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPhase).toBe('movement');
    });

    it('transitions to movement when player can both move and capture', () => {
      const state = makeGameState({ currentPhase: 'ring_placement' });
      const turn = makeTurnState();
      const delegates = makeDelegates({
        hasAnyMovement: jest.fn().mockReturnValue(true),
        hasAnyCapture: jest.fn().mockReturnValue(true),
      });

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPhase).toBe('movement');
    });

    it('transitions to line_processing when player cannot move or capture', () => {
      const state = makeGameState({ currentPhase: 'ring_placement' });
      const turn = makeTurnState();
      const delegates = makeDelegates({
        hasAnyMovement: jest.fn().mockReturnValue(false),
        hasAnyCapture: jest.fn().mockReturnValue(false),
      });

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPhase).toBe('line_processing');
    });
  });

  describe('movement phase', () => {
    it('transitions to line_processing after movement', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      const turn = makeTurnState();
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPhase).toBe('line_processing');
    });
  });

  describe('capture phase', () => {
    it('transitions to line_processing after capture', () => {
      const state = makeGameState({ currentPhase: 'capture' });
      const turn = makeTurnState();
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPhase).toBe('line_processing');
    });
  });

  describe('chain_capture phase', () => {
    it('transitions to line_processing after chain_capture', () => {
      const state = makeGameState({ currentPhase: 'chain_capture' });
      const turn = makeTurnState();
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPhase).toBe('line_processing');
    });
  });

  describe('line_processing phase', () => {
    it('transitions to territory_processing after line_processing', () => {
      const state = makeGameState({ currentPhase: 'line_processing' });
      const turn = makeTurnState();
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPhase).toBe('territory_processing');
    });
  });

  describe('territory_processing phase', () => {
    describe('forced elimination', () => {
      it('triggers forced elimination when player has stacks but no actions', () => {
        const state = makeGameState({ currentPhase: 'territory_processing', currentPlayer: 1 });
        const turn = makeTurnState();
        const applyForcedElimination = jest.fn((s) => ({ ...s, gameStatus: 'active' }));
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn((_, player) =>
            player === 2 ? [{ position: pos(0, 0), stackHeight: 1 }] : []
          ),
          hasAnyPlacement: jest.fn().mockReturnValue(false),
          hasAnyMovement: jest.fn().mockReturnValue(false),
          hasAnyCapture: jest.fn().mockReturnValue(false),
          applyForcedElimination,
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(applyForcedElimination).toHaveBeenCalledWith(expect.anything(), 2);
        expect(result.nextState.currentPhase).toBe('movement');
      });

      it('ends game when forced elimination results in game over', () => {
        const state = makeGameState({ currentPhase: 'territory_processing', currentPlayer: 1 });
        const turn = makeTurnState();
        const applyForcedElimination = jest.fn((s) => ({
          ...s,
          gameStatus: 'completed',
          winner: 1,
        }));
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn((_, player) =>
            player === 2 ? [{ position: pos(0, 0), stackHeight: 1 }] : []
          ),
          hasAnyPlacement: jest.fn().mockReturnValue(false),
          hasAnyMovement: jest.fn().mockReturnValue(false),
          hasAnyCapture: jest.fn().mockReturnValue(false),
          applyForcedElimination,
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.gameStatus).toBe('completed');
        expect(result.nextTurn.hasPlacedThisTurn).toBe(false);
      });
    });

    describe('normal turn progression', () => {
      it('passes turn to next player', () => {
        const state = makeGameState({ currentPhase: 'territory_processing', currentPlayer: 1 });
        const turn = makeTurnState();
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyPlacement: jest.fn().mockReturnValue(true),
          getNextPlayerNumber: jest.fn().mockReturnValue(2),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPlayer).toBe(2);
      });

      it('resets hasPlacedThisTurn for new player', () => {
        const state = makeGameState({ currentPhase: 'territory_processing', currentPlayer: 1 });
        const turn = makeTurnState({ hasPlacedThisTurn: true });
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyPlacement: jest.fn().mockReturnValue(true),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextTurn.hasPlacedThisTurn).toBe(false);
      });

      it('clears mustMoveFromStackKey for new player', () => {
        const state = makeGameState({ currentPhase: 'territory_processing', currentPlayer: 1 });
        const turn = makeTurnState({ mustMoveFromStackKey: '0,0' });
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyPlacement: jest.fn().mockReturnValue(true),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextTurn.mustMoveFromStackKey).toBeUndefined();
      });
    });

    describe('player skipping', () => {
      it('skips player with no stacks and no rings in hand', () => {
        const state = makeGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
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
              ringsInHand: 0, // No rings in hand
              eliminatedRings: 10,
              territorySpaces: 0,
            },
            {
              id: 'p3',
              username: 'Player3',
              playerNumber: 3,
              type: 'human',
              isReady: true,
              timeRemaining: 600000,
              ringsInHand: 8,
              eliminatedRings: 0,
              territorySpaces: 0,
            },
          ],
        });
        const turn = makeTurnState();
        let skipCount = 0;
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn((_, player) => {
            if (player === 2) return []; // Player 2 has no stacks
            return [{ position: pos(0, 0), stackHeight: 1 }];
          }),
          hasAnyPlacement: jest.fn().mockReturnValue(true),
          getNextPlayerNumber: jest.fn((_, current) => {
            if (current === 1) return 2;
            if (current === 2) {
              skipCount++;
              return 3;
            }
            return 1;
          }),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        // Player 2 should be skipped, player 3 should be active
        expect(result.nextState.currentPlayer).toBe(3);
      });

      it('does not skip player with stacks but no rings', () => {
        const state = makeGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
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
              ringsInHand: 0, // No rings in hand but has stacks
              eliminatedRings: 10,
              territorySpaces: 0,
            },
          ],
        });
        const turn = makeTurnState();
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([{ position: pos(0, 0), stackHeight: 1 }]),
          hasAnyPlacement: jest.fn().mockReturnValue(false),
          hasAnyMovement: jest.fn().mockReturnValue(true),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        // Player 2 should be active with movement phase (no rings)
        expect(result.nextState.currentPlayer).toBe(2);
        expect(result.nextState.currentPhase).toBe('movement');
      });

      it('breaks from loop when player not found', () => {
        const state = makeGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
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
          ],
        });
        const turn = makeTurnState();
        // Return a player number that doesn't exist
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyPlacement: jest.fn().mockReturnValue(true),
          getNextPlayerNumber: jest.fn().mockReturnValue(99), // Non-existent player
        });

        // Should not throw, just break from loop
        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState).toBeDefined();
      });

      it('handles max skips guard (prevents infinite loop)', () => {
        // Create a state where all players would be skipped
        const state = makeGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
          players: [
            {
              id: 'p1',
              username: 'Player1',
              playerNumber: 1,
              type: 'human',
              isReady: true,
              timeRemaining: 600000,
              ringsInHand: 0,
              eliminatedRings: 10,
              territorySpaces: 0,
            },
            {
              id: 'p2',
              username: 'Player2',
              playerNumber: 2,
              type: 'human',
              isReady: true,
              timeRemaining: 600000,
              ringsInHand: 0,
              eliminatedRings: 10,
              territorySpaces: 0,
            },
          ],
        });
        const turn = makeTurnState();
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]), // All players have no stacks
          hasAnyPlacement: jest.fn().mockReturnValue(false),
        });

        // Should complete without infinite loop
        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState).toBeDefined();
      });
    });

    describe('starting phase determination', () => {
      it('starts next player in ring_placement when they have rings', () => {
        const state = makeGameState({ currentPhase: 'territory_processing', currentPlayer: 1 });
        const turn = makeTurnState();
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyPlacement: jest.fn().mockReturnValue(true),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPhase).toBe('ring_placement');
      });

      it('starts next player in movement when they have no rings', () => {
        const state = makeGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
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
              ringsInHand: 0, // No rings
              eliminatedRings: 10,
              territorySpaces: 0,
            },
          ],
        });
        const turn = makeTurnState();
        const delegates = makeDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([{ position: pos(0, 0), stackHeight: 1 }]),
          hasAnyPlacement: jest.fn().mockReturnValue(false),
          hasAnyMovement: jest.fn().mockReturnValue(true),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPlayer).toBe(2);
        expect(result.nextState.currentPhase).toBe('movement');
      });
    });
  });

  describe('default phase (unrecognized)', () => {
    it('returns unchanged state for unrecognized phase', () => {
      const state = makeGameState({ currentPhase: 'unknown_phase' as GamePhase });
      const turn = makeTurnState();
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState).toBe(state);
      expect(result.nextTurn).toBe(turn);
    });
  });

  describe('state immutability', () => {
    it('does not mutate original state', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      const originalPhase = state.currentPhase;
      const turn = makeTurnState();
      const delegates = makeDelegates();

      advanceTurnAndPhase(state, turn, delegates);

      expect(state.currentPhase).toBe(originalPhase);
    });

    it('returns shallow clone with updated fields', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      const turn = makeTurnState();
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState).not.toBe(state);
      expect(result.nextState.currentPhase).toBe('line_processing');
      expect(result.nextState.id).toBe(state.id);
    });
  });

  describe('turn state propagation', () => {
    it('preserves turn state during non-boundary phases', () => {
      const state = makeGameState({ currentPhase: 'movement' });
      const turn = makeTurnState({ hasPlacedThisTurn: true, mustMoveFromStackKey: '1,1' });
      const delegates = makeDelegates();

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextTurn.hasPlacedThisTurn).toBe(true);
      expect(result.nextTurn.mustMoveFromStackKey).toBe('1,1');
    });
  });

  describe('edge cases', () => {
    it('handles single-player state', () => {
      const state = makeGameState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
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
        ],
      });
      const turn = makeTurnState();
      const delegates = makeDelegates({
        getPlayerStacks: jest.fn().mockReturnValue([]),
        hasAnyPlacement: jest.fn().mockReturnValue(true),
        getNextPlayerNumber: jest.fn().mockReturnValue(1), // Back to self
      });

      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState.currentPlayer).toBe(1);
    });

    it('handles empty players array', () => {
      const state = makeGameState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        players: [],
      });
      const turn = makeTurnState();
      const delegates = makeDelegates({
        getPlayerStacks: jest.fn().mockReturnValue([]),
        hasAnyPlacement: jest.fn().mockReturnValue(false),
        getNextPlayerNumber: jest.fn().mockReturnValue(1),
      });

      // Should not throw
      const result = advanceTurnAndPhase(state, turn, delegates);

      expect(result.nextState).toBeDefined();
    });
  });
});
