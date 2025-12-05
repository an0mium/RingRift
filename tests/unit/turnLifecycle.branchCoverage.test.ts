/**
 * turnLifecycle.branchCoverage.test.ts
 *
 * Branch coverage tests for turnLifecycle.ts targeting uncovered branches:
 * - isInteractivePhase: all phase types
 * - advanceFromMovementBoundary: hook invocation, game status checks
 * - startInteractiveTurnForCurrentPlayer: loop iterations, forced elimination detection, early returns
 */

import {
  advanceFromMovementBoundary,
  startInteractiveTurnForCurrentPlayer,
  TurnLifecycleContext,
  TurnLifecycleDeps,
  TurnLifecycleHooks,
} from '../../src/shared/engine/turnLifecycle';
import type { GameState, GamePhase } from '../../src/shared/types/game';
import type { PerTurnState, TurnLogicDelegates } from '../../src/shared/engine/turnLogic';

// Mock the turnLogic module
jest.mock('../../src/shared/engine/turnLogic', () => ({
  advanceTurnAndPhase: jest.fn(),
}));

import { advanceTurnAndPhase } from '../../src/shared/engine/turnLogic';

const mockAdvanceTurnAndPhase = advanceTurnAndPhase as jest.MockedFunction<
  typeof advanceTurnAndPhase
>;

// Helper to create a minimal game state
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultState: GameState = {
    id: 'test-game',
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
    currentPhase: 'ring_placement' as GamePhase,
    gameStatus: 'active',
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    moveHistory: [],
    spectators: [],
    boardType: 'square8',
  };

  return { ...defaultState, ...overrides } as GameState;
}

// Helper to create mock delegates
function makeMockDelegates(): TurnLogicDelegates {
  return {
    getPlayerStacks: jest.fn().mockReturnValue([]),
    hasAnyPlacement: jest.fn().mockReturnValue(true),
    hasAnyMovement: jest.fn().mockReturnValue(true),
    hasAnyCapture: jest.fn().mockReturnValue(false),
    applyForcedElimination: jest.fn(),
    getNextPlayerNumber: jest.fn().mockReturnValue(2),
  };
}

// Helper to create per-turn state
function makePerTurnState(overrides: Partial<PerTurnState> = {}): PerTurnState {
  return {
    hasPlacedThisTurn: false,
    mustMoveFromStackKey: undefined,
    ...overrides,
  };
}

describe('turnLifecycle branch coverage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('advanceFromMovementBoundary', () => {
    it('calls advanceTurnAndPhase with correct arguments', () => {
      const state = makeGameState();
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();

      const nextState = makeGameState({ currentPhase: 'ring_placement' });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      advanceFromMovementBoundary(ctx, deps);

      expect(mockAdvanceTurnAndPhase).toHaveBeenCalledWith(state, perTurn, delegates);
    });

    it('invokes onStartInteractiveTurn when transitioning to interactive phase', () => {
      const state = makeGameState({ currentPhase: 'territory_processing' });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onStartInteractiveTurn = jest.fn();

      const nextState = makeGameState({
        currentPhase: 'ring_placement',
        gameStatus: 'active',
      });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onStartInteractiveTurn },
      };

      advanceFromMovementBoundary(ctx, deps);

      expect(onStartInteractiveTurn).toHaveBeenCalledWith(nextState, {
        hasPlacedThisTurn: false,
        mustMoveFromStackKey: undefined,
      });
    });

    it('invokes onStartInteractiveTurn for movement phase', () => {
      const state = makeGameState({ currentPhase: 'territory_processing' });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onStartInteractiveTurn = jest.fn();

      const nextState = makeGameState({
        currentPhase: 'movement',
        gameStatus: 'active',
      });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: true, mustMoveFromStackKey: '0,0' },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onStartInteractiveTurn },
      };

      advanceFromMovementBoundary(ctx, deps);

      expect(onStartInteractiveTurn).toHaveBeenCalled();
    });

    it('invokes onStartInteractiveTurn for capture phase', () => {
      const state = makeGameState({ currentPhase: 'territory_processing' });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onStartInteractiveTurn = jest.fn();

      const nextState = makeGameState({
        currentPhase: 'capture',
        gameStatus: 'active',
      });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onStartInteractiveTurn },
      };

      advanceFromMovementBoundary(ctx, deps);

      expect(onStartInteractiveTurn).toHaveBeenCalled();
    });

    it('invokes onStartInteractiveTurn for chain_capture phase', () => {
      const state = makeGameState({ currentPhase: 'territory_processing' });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onStartInteractiveTurn = jest.fn();

      const nextState = makeGameState({
        currentPhase: 'chain_capture',
        gameStatus: 'active',
      });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onStartInteractiveTurn },
      };

      advanceFromMovementBoundary(ctx, deps);

      expect(onStartInteractiveTurn).toHaveBeenCalled();
    });

    it('does not invoke hook when game is completed', () => {
      const state = makeGameState({ gameStatus: 'active' });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onStartInteractiveTurn = jest.fn();

      const nextState = makeGameState({
        currentPhase: 'ring_placement',
        gameStatus: 'completed',
      });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onStartInteractiveTurn },
      };

      advanceFromMovementBoundary(ctx, deps);

      expect(onStartInteractiveTurn).not.toHaveBeenCalled();
    });

    it('does not invoke hook for non-interactive phase', () => {
      const state = makeGameState({ currentPhase: 'territory_processing' });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onStartInteractiveTurn = jest.fn();

      const nextState = makeGameState({
        currentPhase: 'line_processing',
        gameStatus: 'active',
      });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onStartInteractiveTurn },
      };

      advanceFromMovementBoundary(ctx, deps);

      expect(onStartInteractiveTurn).not.toHaveBeenCalled();
    });

    it('works without hooks provided', () => {
      const state = makeGameState({ currentPhase: 'territory_processing' });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();

      const nextState = makeGameState({
        currentPhase: 'ring_placement',
        gameStatus: 'active',
      });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      const result = advanceFromMovementBoundary(ctx, deps);

      expect(result.state).toBe(nextState);
    });

    it('returns updated context with new state and turn', () => {
      const state = makeGameState();
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();

      const nextState = makeGameState({ currentPlayer: 2 });
      const nextTurn = { hasPlacedThisTurn: true, mustMoveFromStackKey: '1,1' };
      mockAdvanceTurnAndPhase.mockReturnValue({ nextState, nextTurn });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      const result = advanceFromMovementBoundary(ctx, deps);

      expect(result.state).toBe(nextState);
      expect(result.perTurn).toBe(nextTurn);
    });
  });

  describe('startInteractiveTurnForCurrentPlayer', () => {
    it('resets per-turn flags at the start', () => {
      const state = makeGameState({ gameStatus: 'active', currentPhase: 'ring_placement' });
      const perTurn = makePerTurnState({
        hasPlacedThisTurn: true,
        mustMoveFromStackKey: '0,0',
      });
      const delegates = makeMockDelegates();

      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState: state,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      const result = startInteractiveTurnForCurrentPlayer(ctx, deps);

      // The function should have been called with reset flags
      expect(mockAdvanceTurnAndPhase).toHaveBeenCalledWith(
        state,
        { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
        delegates
      );
    });

    it('returns early when game is not active at start', () => {
      const state = makeGameState({ gameStatus: 'completed' });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      const result = startInteractiveTurnForCurrentPlayer(ctx, deps);

      expect(result.state.gameStatus).toBe('completed');
      expect(mockAdvanceTurnAndPhase).not.toHaveBeenCalled();
    });

    it('returns early when advanceTurnAndPhase ends the game', () => {
      const state = makeGameState({ gameStatus: 'active', currentPhase: 'territory_processing' });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();

      const completedState = makeGameState({ gameStatus: 'completed' });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState: completedState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      const result = startInteractiveTurnForCurrentPlayer(ctx, deps);

      expect(result.state.gameStatus).toBe('completed');
    });

    it('calls onStartInteractiveTurn when reaching interactive phase', () => {
      const state = makeGameState({
        gameStatus: 'active',
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onStartInteractiveTurn = jest.fn();

      const interactiveState = makeGameState({
        gameStatus: 'active',
        currentPhase: 'ring_placement',
        currentPlayer: 1,
      });
      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState: interactiveState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onStartInteractiveTurn },
      };

      startInteractiveTurnForCurrentPlayer(ctx, deps);

      expect(onStartInteractiveTurn).toHaveBeenCalledWith(interactiveState, {
        hasPlacedThisTurn: false,
        mustMoveFromStackKey: undefined,
      });
    });

    it('detects and reports forced elimination', () => {
      const state = makeGameState({
        gameStatus: 'active',
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onForcedElimination = jest.fn();
      const onStartInteractiveTurn = jest.fn();

      // First call: forced elimination occurs (player changes, was in territory_processing)
      const stateAfterElimination = makeGameState({
        gameStatus: 'active',
        currentPhase: 'ring_placement',
        currentPlayer: 2,
      });

      mockAdvanceTurnAndPhase.mockReturnValueOnce({
        nextState: stateAfterElimination,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onForcedElimination, onStartInteractiveTurn },
      };

      startInteractiveTurnForCurrentPlayer(ctx, deps);

      expect(onForcedElimination).toHaveBeenCalledWith(stateAfterElimination, 1);
    });

    it('loops through multiple players when skipping', () => {
      const state = makeGameState({
        gameStatus: 'active',
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();

      // First call: still in territory_processing (player 2)
      const territoryState = makeGameState({
        gameStatus: 'active',
        currentPhase: 'territory_processing',
        currentPlayer: 2,
      });

      // Second call: reaches interactive phase
      const interactiveState = makeGameState({
        gameStatus: 'active',
        currentPhase: 'ring_placement',
        currentPlayer: 2,
      });

      mockAdvanceTurnAndPhase
        .mockReturnValueOnce({
          nextState: territoryState,
          nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
        })
        .mockReturnValueOnce({
          nextState: interactiveState,
          nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
        });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      const result = startInteractiveTurnForCurrentPlayer(ctx, deps);

      expect(mockAdvanceTurnAndPhase).toHaveBeenCalledTimes(2);
      expect(result.state.currentPhase).toBe('ring_placement');
    });

    it('handles game ending mid-loop', () => {
      const state = makeGameState({
        gameStatus: 'active',
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();

      // First call: still active
      const territoryState = makeGameState({
        gameStatus: 'active',
        currentPhase: 'territory_processing',
        currentPlayer: 2,
      });

      // Second call: game ends
      const completedState = makeGameState({
        gameStatus: 'completed',
        currentPhase: 'territory_processing',
        currentPlayer: 2,
      });

      mockAdvanceTurnAndPhase
        .mockReturnValueOnce({
          nextState: territoryState,
          nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
        })
        .mockReturnValueOnce({
          nextState: completedState,
          nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
        });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      const result = startInteractiveTurnForCurrentPlayer(ctx, deps);

      expect(result.state.gameStatus).toBe('completed');
    });

    it('stops at max iterations to prevent infinite loops', () => {
      const state = makeGameState({
        gameStatus: 'active',
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();

      // Always return territory_processing to force max iterations
      const territoryState = makeGameState({
        gameStatus: 'active',
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });

      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState: territoryState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      startInteractiveTurnForCurrentPlayer(ctx, deps);

      // Max iterations = players.length * 2 = 4
      expect(mockAdvanceTurnAndPhase).toHaveBeenCalledTimes(4);
    });

    it('works without hooks provided', () => {
      const state = makeGameState({
        gameStatus: 'active',
        currentPhase: 'ring_placement',
      });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();

      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState: state,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = { delegates };

      const result = startInteractiveTurnForCurrentPlayer(ctx, deps);

      expect(result.state).toBe(state);
    });

    it('does not call onForcedElimination when phase changes but not from territory_processing', () => {
      const state = makeGameState({
        gameStatus: 'active',
        currentPhase: 'ring_placement',
        currentPlayer: 1,
      });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onForcedElimination = jest.fn();

      // Phase changes but original phase was ring_placement, not territory_processing
      const nextState = makeGameState({
        gameStatus: 'active',
        currentPhase: 'movement',
        currentPlayer: 1,
      });

      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: true, mustMoveFromStackKey: '0,0' },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onForcedElimination },
      };

      startInteractiveTurnForCurrentPlayer(ctx, deps);

      expect(onForcedElimination).not.toHaveBeenCalled();
    });

    it('does not call onForcedElimination when starting from non-territory phase', () => {
      const state = makeGameState({
        gameStatus: 'active',
        currentPhase: 'ring_placement',
        currentPlayer: 1,
      });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onForcedElimination = jest.fn();

      // Player changes but NOT from territory_processing, so not forced elimination
      const nextState = makeGameState({
        gameStatus: 'active',
        currentPhase: 'movement',
        currentPlayer: 2,
      });

      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: true, mustMoveFromStackKey: '0,0' },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onForcedElimination },
      };

      startInteractiveTurnForCurrentPlayer(ctx, deps);

      // The beforePhase is ring_placement (after reset), not territory_processing
      // So forced elimination is not detected
      expect(onForcedElimination).not.toHaveBeenCalled();
    });

    it('handles hooks with only onForcedElimination defined', () => {
      const state = makeGameState({
        gameStatus: 'active',
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const perTurn = makePerTurnState();
      const delegates = makeMockDelegates();
      const onForcedElimination = jest.fn();

      const nextState = makeGameState({
        gameStatus: 'active',
        currentPhase: 'ring_placement',
        currentPlayer: 2,
      });

      mockAdvanceTurnAndPhase.mockReturnValue({
        nextState,
        nextTurn: { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined },
      });

      const ctx: TurnLifecycleContext = { state, perTurn };
      const deps: TurnLifecycleDeps = {
        delegates,
        hooks: { onForcedElimination }, // No onStartInteractiveTurn
      };

      startInteractiveTurnForCurrentPlayer(ctx, deps);

      expect(onForcedElimination).toHaveBeenCalledWith(nextState, 1);
    });
  });
});
