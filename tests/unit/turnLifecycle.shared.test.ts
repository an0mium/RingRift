import {
  advanceFromMovementBoundary,
  startInteractiveTurnForCurrentPlayer,
} from '../../src/shared/engine/turnLifecycle';
import type {
  TurnLifecycleContext,
  TurnLifecycleDeps,
} from '../../src/shared/engine/turnLifecycle';
import type { GameState, GamePhase } from '../../src/shared/types/game';
import type { PerTurnState } from '../../src/shared/engine/turnLogic';

// We mock advanceTurnAndPhase so we can drive turnLifecycle behaviour
// without depending on the full rules engine.
jest.mock('../../src/shared/engine/turnLogic', () => {
  return {
    advanceTurnAndPhase: jest.fn(),
  };
});

const { advanceTurnAndPhase } = require('../../src/shared/engine/turnLogic') as {
  advanceTurnAndPhase: jest.Mock;
};

function createState(overrides: Partial<GameState> = {}): GameState {
  return {
    id: 'test',
    boardType: 'square8',
    players: [{ playerNumber: 1 } as any, { playerNumber: 2 } as any],
    currentPlayer: 1,
    currentPhase: 'territory_processing' as GamePhase,
    gameStatus: 'active',
    board: {} as any,
    history: [] as any,
    ...overrides,
  } as GameState;
}

function createPerTurn(overrides: Partial<PerTurnState> = {}): PerTurnState {
  return {
    hasPlacedThisTurn: false,
    mustMoveFromStackKey: undefined,
    ...overrides,
  } as PerTurnState;
}

describe('shared/engine/turnLifecycle', () => {
  beforeEach(() => {
    advanceTurnAndPhase.mockReset();
  });

  describe('advanceFromMovementBoundary', () => {
    it('returns next state/turn and invokes onStartInteractiveTurn when next phase is interactive & active', () => {
      const state = createState({
        currentPhase: 'territory_processing',
      });
      const perTurn = createPerTurn();

      const nextState = createState({
        currentPhase: 'movement',
        gameStatus: 'active',
        currentPlayer: 1,
      });
      const nextTurn = createPerTurn();

      advanceTurnAndPhase.mockReturnValue({ nextState, nextTurn });

      const onStartInteractiveTurn = jest.fn();
      const deps: TurnLifecycleDeps = {
        delegates: {} as any,
        hooks: { onStartInteractiveTurn },
      };

      const result = advanceFromMovementBoundary({ state, perTurn }, deps);

      expect(result).toEqual({ state: nextState, perTurn: nextTurn });
      expect(onStartInteractiveTurn).toHaveBeenCalledTimes(1);
      expect(onStartInteractiveTurn).toHaveBeenCalledWith(nextState, nextTurn);
    });

    it('does not call onStartInteractiveTurn when game is not active or phase is not interactive', () => {
      const state = createState({ currentPhase: 'territory_processing' });
      const perTurn = createPerTurn();

      const nonInteractiveNext = createState({
        currentPhase: 'territory_processing',
        gameStatus: 'active',
      });
      advanceTurnAndPhase.mockReturnValue({ nextState: nonInteractiveNext, nextTurn: perTurn });

      const onStartInteractiveTurn = jest.fn();
      const deps: TurnLifecycleDeps = {
        delegates: {} as any,
        hooks: { onStartInteractiveTurn },
      };

      const result = advanceFromMovementBoundary({ state, perTurn }, deps);

      expect(result.state).toBe(nonInteractiveNext);
      expect(onStartInteractiveTurn).not.toHaveBeenCalled();
    });
  });

  describe('startInteractiveTurnForCurrentPlayer', () => {
    it('resets per-turn flags and starts an interactive phase, calling onStartInteractiveTurn', () => {
      const state = createState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const perTurn = createPerTurn({ hasPlacedThisTurn: true, mustMoveFromStackKey: 'stack-1' });

      const nextState = createState({
        currentPhase: 'movement',
        currentPlayer: 1,
      });
      const nextTurn = createPerTurn();

      advanceTurnAndPhase.mockReturnValue({ nextState, nextTurn });

      const onStartInteractiveTurn = jest.fn();
      const deps: TurnLifecycleDeps = {
        delegates: {} as any,
        hooks: { onStartInteractiveTurn },
      };

      const result = startInteractiveTurnForCurrentPlayer({ state, perTurn }, deps);

      // Should return the values from advanceTurnAndPhase
      expect(result).toEqual({ state: nextState, perTurn: nextTurn });

      // Per-turn flags passed into advanceTurnAndPhase should have been reset
      const callArgs = advanceTurnAndPhase.mock.calls[0];
      const passedPerTurn = callArgs[1] as PerTurnState;
      expect(passedPerTurn.hasPlacedThisTurn).toBe(false);
      expect(passedPerTurn.mustMoveFromStackKey).toBeUndefined();

      // Interactive-start hook is called
      expect(onStartInteractiveTurn).toHaveBeenCalledTimes(1);
      expect(onStartInteractiveTurn).toHaveBeenCalledWith(nextState, nextTurn);
    });

    it('invokes onForcedElimination when phase advances from territory_processing while game remains active', () => {
      const state = createState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const perTurn = createPerTurn();

      const nextState = createState({
        currentPhase: 'capture', // interactive phase
        currentPlayer: 2, // player changed -> forced elimination of player 1
        gameStatus: 'active',
      });
      const nextTurn = createPerTurn();

      advanceTurnAndPhase.mockReturnValue({ nextState, nextTurn });

      const onForcedElimination = jest.fn();
      const onStartInteractiveTurn = jest.fn();
      const deps: TurnLifecycleDeps = {
        delegates: {} as any,
        hooks: { onForcedElimination, onStartInteractiveTurn },
      };

      const result = startInteractiveTurnForCurrentPlayer({ state, perTurn }, deps);

      expect(result).toEqual({ state: nextState, perTurn: nextTurn });

      // Forced elimination hook should be called once with the eliminated player (beforePlayer = 1)
      expect(onForcedElimination).toHaveBeenCalledTimes(1);
      expect(onForcedElimination).toHaveBeenCalledWith(nextState, 1);

      // Interactive-start hook should also be invoked for the new interactive phase
      expect(onStartInteractiveTurn).toHaveBeenCalledTimes(1);
      expect(onStartInteractiveTurn).toHaveBeenCalledWith(nextState, nextTurn);
    });

    it('returns after maxIterations when game stays active but never reaches an interactive phase', () => {
      const state = createState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
        players: [{ playerNumber: 1 } as any], // maxIterations = 2
      });
      const perTurn = createPerTurn();

      // Always return the same non-interactive active state; this will
      // cause the loop to hit the maxIterations safety net.
      advanceTurnAndPhase.mockImplementation((s: GameState, t: PerTurnState) => ({
        nextState: { ...s, gameStatus: 'active', currentPhase: 'territory_processing' },
        nextTurn: t,
      }));

      const onForcedElimination = jest.fn();
      const onStartInteractiveTurn = jest.fn();

      const deps: TurnLifecycleDeps = {
        delegates: {} as any,
        hooks: { onForcedElimination, onStartInteractiveTurn },
      };

      const result = startInteractiveTurnForCurrentPlayer({ state, perTurn }, deps);

      // Since advanceTurnAndPhase never produces an interactive phase or
      // game end, the function returns the last state/perTurn after hitting
      // the maxIterations cap.
      expect(result.state.gameStatus).toBe('active');
      expect(result.state.currentPhase).toBe('territory_processing');

      // Called exactly maxIterations times (players.length * 2 = 2)
      expect(advanceTurnAndPhase).toHaveBeenCalledTimes(2);

      // No hooks should have been invoked.
      expect(onForcedElimination).not.toHaveBeenCalled();
      expect(onStartInteractiveTurn).not.toHaveBeenCalled();
    });

    it('returns immediately when the game becomes non-active inside the loop', () => {
      const state = createState({
        currentPhase: 'territory_processing',
        currentPlayer: 1,
      });
      const perTurn = createPerTurn();

      const nextState = createState({
        gameStatus: 'completed',
        currentPhase: 'territory_processing',
      });

      advanceTurnAndPhase.mockReturnValue({ nextState, nextTurn: perTurn });

      const onForcedElimination = jest.fn();
      const onStartInteractiveTurn = jest.fn();

      const deps: TurnLifecycleDeps = {
        delegates: {} as any,
        hooks: { onForcedElimination, onStartInteractiveTurn },
      };

      const result = startInteractiveTurnForCurrentPlayer({ state, perTurn }, deps);

      expect(result.state).toBe(nextState);
      expect(onForcedElimination).not.toHaveBeenCalled();
      expect(onStartInteractiveTurn).not.toHaveBeenCalled();
    });
  });
});
