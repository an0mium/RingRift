/**
 * TurnLogic Unit Tests
 *
 * Tests for the shared turn/phase progression logic in turnLogic.ts.
 * Covers all phase transitions and edge cases including:
 * - Phase advancement from ring_placement, movement, capture, chain_capture
 * - Territory processing with player rotation
 * - Forced elimination scenarios
 * - Skipping players with no material
 * - Unknown phase handling
 */

import {
  advanceTurnAndPhase,
  type TurnLogicDelegates,
  type PerTurnState,
} from '../../src/shared/engine/turnLogic';
import type { GameState, GamePhase, Position } from '../../src/shared/types/game';

describe('turnLogic', () => {
  // Helper to create a minimal game state
  const createGameState = (overrides: Partial<GameState> = {}): GameState =>
    ({
      id: 'test-game',
      boardType: 'square8',
      board: {
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: { 1: 0, 2: 0 },
      },
      players: [
        {
          id: 'p1',
          username: 'Player1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
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
      timeControl: { initialTime: 600, increment: 5, type: 'rapid' },
      spectators: [],
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: true,
      maxPlayers: 2,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 19,
      territoryVictoryThreshold: 33,
      ...overrides,
    }) as GameState;

  // Helper to create default per-turn state
  const createTurnState = (overrides: Partial<PerTurnState> = {}): PerTurnState => ({
    hasPlacedThisTurn: false,
    mustMoveFromStackKey: undefined,
    ...overrides,
  });

  // Helper to create mock delegates
  const createDelegates = (overrides: Partial<TurnLogicDelegates> = {}): TurnLogicDelegates => ({
    getPlayerStacks: jest.fn().mockReturnValue([]),
    hasAnyPlacement: jest.fn().mockReturnValue(true),
    hasAnyMovement: jest.fn().mockReturnValue(true),
    hasAnyCapture: jest.fn().mockReturnValue(false),
    applyForcedElimination: jest.fn((state) => state),
    getNextPlayerNumber: jest.fn((state, current) => (current === 1 ? 2 : 1)),
    ...overrides,
  });

  describe('advanceTurnAndPhase', () => {
    describe('inactive game', () => {
      it('should return unchanged state when gameStatus is finished', () => {
        const state = createGameState({ gameStatus: 'finished' });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState).toBe(state);
        expect(result.nextTurn).toBe(turn);
      });

      it('should return unchanged state when gameStatus is waiting', () => {
        const state = createGameState({ gameStatus: 'waiting' });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState).toBe(state);
        expect(result.nextTurn).toBe(turn);
      });

      it('should return unchanged state when gameStatus is paused', () => {
        const state = createGameState({ gameStatus: 'paused' });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState).toBe(state);
        expect(result.nextTurn).toBe(turn);
      });
    });

    describe('ring_placement phase', () => {
      // Per RR-CANON-R075: All phases must be visited with explicit moves.
      // advanceTurnAndPhase always transitions to movement phase - if no moves
      // exist, the player must emit no_movement_action to advance. No silent
      // phase skipping is permitted.

      it('should always advance to movement phase (explicit phase progression)', () => {
        const state = createGameState({ currentPhase: 'ring_placement' });
        const turn = createTurnState({ hasPlacedThisTurn: true });
        const delegates = createDelegates({
          hasAnyMovement: jest.fn().mockReturnValue(true),
          hasAnyCapture: jest.fn().mockReturnValue(false),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        // Per RR-CANON-R075: Always goes to movement, no silent skipping
        expect(result.nextState.currentPhase).toBe('movement');
        // Note: delegate not called since movement is always the next phase
      });

      it('should advance to movement phase even when no movement is available', () => {
        const state = createGameState({ currentPhase: 'ring_placement' });
        const turn = createTurnState({ hasPlacedThisTurn: true });
        const delegates = createDelegates({
          hasAnyMovement: jest.fn().mockReturnValue(false),
          hasAnyCapture: jest.fn().mockReturnValue(false),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        // Per RR-CANON-R075: Always goes to movement phase - player must then
        // emit no_movement_action to advance to line_processing
        expect(result.nextState.currentPhase).toBe('movement');
      });
    });

    describe('movement phase', () => {
      it('should advance to line_processing phase', () => {
        const state = createGameState({ currentPhase: 'movement' });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPhase).toBe('line_processing');
      });
    });

    describe('capture phase', () => {
      it('should advance to line_processing phase', () => {
        const state = createGameState({ currentPhase: 'capture' });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPhase).toBe('line_processing');
      });
    });

    describe('chain_capture phase', () => {
      it('should advance to line_processing phase', () => {
        const state = createGameState({ currentPhase: 'chain_capture' });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPhase).toBe('line_processing');
      });
    });

    describe('line_processing phase', () => {
      it('should advance to territory_processing phase', () => {
        const state = createGameState({ currentPhase: 'line_processing' });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPhase).toBe('territory_processing');
      });
    });

    describe('territory_processing phase', () => {
      it('should rotate to next player with ring_placement phase', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
        });
        const turn = createTurnState();
        const delegates = createDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyPlacement: jest.fn().mockReturnValue(true),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPlayer).toBe(2);
        expect(result.nextState.currentPhase).toBe('ring_placement');
        expect(result.nextTurn.hasPlacedThisTurn).toBe(false);
        expect(result.nextTurn.mustMoveFromStackKey).toBeUndefined();
      });

      it('should start next player in movement phase when they have no rings', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
          players: [
            { ...createGameState().players[0], playerNumber: 1, ringsInHand: 18 },
            { ...createGameState().players[1], playerNumber: 2, ringsInHand: 0 },
          ],
        });
        const turn = createTurnState();
        const delegates = createDelegates({
          getPlayerStacks: jest
            .fn()
            .mockReturnValue([{ position: { x: 0, y: 0 }, stackHeight: 1 }]),
          hasAnyPlacement: jest.fn().mockReturnValue(false),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPlayer).toBe(2);
        expect(result.nextState.currentPhase).toBe('movement');
      });

      it('should trigger forced elimination when player has stacks but no actions', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
        });
        const turn = createTurnState();

        const applyForcedElimination = jest.fn((s) => ({
          ...s,
          currentPhase: 'movement' as GamePhase,
        }));
        const delegates = createDelegates({
          getPlayerStacks: jest
            .fn()
            .mockReturnValue([{ position: { x: 0, y: 0 }, stackHeight: 1 }]),
          hasAnyPlacement: jest.fn().mockReturnValue(false),
          hasAnyMovement: jest.fn().mockReturnValue(false),
          hasAnyCapture: jest.fn().mockReturnValue(false),
          applyForcedElimination,
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(applyForcedElimination).toHaveBeenCalledWith(expect.any(Object), 2);
        expect(result.nextState.currentPhase).toBe('movement');
      });

      it('should end game when forced elimination leads to game over', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
        });
        const turn = createTurnState();

        const applyForcedElimination = jest.fn((s) => ({
          ...s,
          gameStatus: 'finished' as const,
          winner: 1,
        }));
        const delegates = createDelegates({
          getPlayerStacks: jest
            .fn()
            .mockReturnValue([{ position: { x: 0, y: 0 }, stackHeight: 1 }]),
          hasAnyPlacement: jest.fn().mockReturnValue(false),
          hasAnyMovement: jest.fn().mockReturnValue(false),
          hasAnyCapture: jest.fn().mockReturnValue(false),
          applyForcedElimination,
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.gameStatus).toBe('finished');
        expect(result.nextState.winner).toBe(1);
        expect(result.nextTurn.hasPlacedThisTurn).toBe(false);
      });

      it('should skip player with no stacks and no rings', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
          players: [
            { ...createGameState().players[0], playerNumber: 1, ringsInHand: 18 },
            { ...createGameState().players[1], playerNumber: 2, ringsInHand: 0 },
          ],
        });
        const turn = createTurnState();

        const getPlayerStacks = jest
          .fn()
          .mockImplementation((_state: GameState, player: number) =>
            player === 1 ? [{ position: { x: 0, y: 0 }, stackHeight: 1 }] : []
          );

        const delegates = createDelegates({
          getPlayerStacks,
          hasAnyPlacement: jest.fn().mockReturnValue(true),
          getNextPlayerNumber: jest.fn().mockReturnValue(1), // Player 2 -> Player 1
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        // Should skip player 2 and go back to player 1
        expect(result.nextState.currentPlayer).toBe(1);
        expect(result.nextState.currentPhase).toBe('ring_placement');
      });

      it('should handle missing player gracefully', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
          players: [], // Empty players array (edge case)
        });
        const turn = createTurnState();
        const delegates = createDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          getNextPlayerNumber: jest.fn().mockReturnValue(99), // Non-existent player
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        // Should break out of the while loop without crashing
        expect(result.nextState).toBeDefined();
      });

      it('should limit skip iterations to prevent infinite loops', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
          players: [
            { ...createGameState().players[0], playerNumber: 1, ringsInHand: 0 },
            { ...createGameState().players[1], playerNumber: 2, ringsInHand: 0 },
          ],
        });
        const turn = createTurnState();

        // Both players have no material
        const delegates = createDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyPlacement: jest.fn().mockReturnValue(false),
          getNextPlayerNumber: jest.fn((_, current) => (current === 1 ? 2 : 1)),
        });

        // Should not hang - the loop has maxSkips = players.length
        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState).toBeDefined();
      });
    });

    describe('unknown phase', () => {
      it('should return unchanged state for unknown phases', () => {
        const state = createGameState({
          currentPhase: 'some_unknown_phase' as GamePhase,
        });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPhase).toBe('some_unknown_phase');
        expect(result.nextTurn).toEqual(turn);
      });

      it('should preserve state for game_over phase', () => {
        const state = createGameState({
          currentPhase: 'game_over',
        });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPhase).toBe('game_over');
      });
    });

    describe('state cloning', () => {
      it('should not mutate the original state', () => {
        const state = createGameState({ currentPhase: 'movement' });
        const turn = createTurnState();
        const delegates = createDelegates();

        advanceTurnAndPhase(state, turn, delegates);

        expect(state.currentPhase).toBe('movement'); // Original unchanged
      });

      it('should return a new state object', () => {
        const state = createGameState({ currentPhase: 'movement' });
        const turn = createTurnState();
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState).not.toBe(state);
      });
    });

    describe('PerTurnState handling', () => {
      it('should preserve turn state through most phases', () => {
        const state = createGameState({ currentPhase: 'movement' });
        const turn = createTurnState({ hasPlacedThisTurn: true, mustMoveFromStackKey: '3,3' });
        const delegates = createDelegates();

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextTurn.hasPlacedThisTurn).toBe(true);
        expect(result.nextTurn.mustMoveFromStackKey).toBe('3,3');
      });

      it('should reset turn state on player rotation', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
        });
        const turn = createTurnState({ hasPlacedThisTurn: true, mustMoveFromStackKey: '3,3' });
        const delegates = createDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyPlacement: jest.fn().mockReturnValue(true),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextTurn.hasPlacedThisTurn).toBe(false);
        expect(result.nextTurn.mustMoveFromStackKey).toBeUndefined();
      });
    });

    describe('3+ player games', () => {
      it('should correctly rotate through 3 players', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
          players: [
            { ...createGameState().players[0], playerNumber: 1, ringsInHand: 18 },
            { ...createGameState().players[1], playerNumber: 2, ringsInHand: 18 },
            { ...createGameState().players[0], playerNumber: 3, ringsInHand: 18 },
          ],
        });
        const turn = createTurnState();
        const delegates = createDelegates({
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyPlacement: jest.fn().mockReturnValue(true),
          getNextPlayerNumber: jest.fn().mockReturnValue(2),
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        expect(result.nextState.currentPlayer).toBe(2);
      });

      it('should skip eliminated player in 3-player game', () => {
        const state = createGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
          players: [
            { ...createGameState().players[0], playerNumber: 1, ringsInHand: 18 },
            { ...createGameState().players[1], playerNumber: 2, ringsInHand: 0 }, // Eliminated
            { ...createGameState().players[0], playerNumber: 3, ringsInHand: 18 },
          ],
        });
        const turn = createTurnState();

        let currentPlayer = 2; // Start with player 2
        const getNextPlayerNumber = jest.fn().mockImplementation(() => {
          if (currentPlayer === 2) {
            currentPlayer = 3;
            return 3;
          }
          return 1;
        });

        const getPlayerStacks = jest
          .fn()
          .mockImplementation((_state: GameState, player: number) =>
            player === 2 ? [] : [{ position: { x: 0, y: 0 }, stackHeight: 1 }]
          );

        const delegates = createDelegates({
          getPlayerStacks,
          hasAnyPlacement: jest.fn().mockReturnValue(true),
          getNextPlayerNumber,
        });

        const result = advanceTurnAndPhase(state, turn, delegates);

        // Should skip player 2 and land on player 3
        expect(result.nextState.currentPlayer).toBe(3);
      });
    });
  });
});
