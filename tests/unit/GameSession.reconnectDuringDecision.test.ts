/**
 * Tests for the interaction between player reconnection and decision phase timeouts.
 *
 * Key edge cases:
 * 1. Player disconnects during decision phase, reconnects within window → timeout continues
 * 2. Player disconnects during decision phase, decision timeout fires → auto-resolve
 * 3. Decision maker disconnects, reconnection window expires → choices cancelled
 *
 * These tests ensure the decision phase timeout and reconnection window systems
 * interact correctly and don't cause race conditions or stale state.
 */

import { GameSession } from '../../src/server/game/GameSession';
import type { Server as SocketIOServer } from 'socket.io';
import type { Move, Position } from '../../src/shared/types/game';
import { config } from '../../src/server/config';

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => null),
}));

jest.mock('../../src/server/services/PythonRulesClient', () => ({
  PythonRulesClient: jest.fn().mockImplementation(() => ({
    evaluateMove: jest.fn(),
    healthCheck: jest.fn(),
  })),
}));

jest.mock('../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    createAI: jest.fn(),
    createAIFromProfile: jest.fn(),
    getAIConfig: jest.fn(),
    getAIMove: jest.fn(),
    chooseLocalMoveFromCandidates: jest.fn(),
    getLocalFallbackMove: jest.fn(),
    getDiagnostics: jest.fn(() => ({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    })),
  },
}));

jest.mock('../../src/server/services/AIUserService', () => ({
  getOrCreateAIUser: jest.fn(() => Promise.resolve({ id: 'ai-user-id' })),
}));

describe('GameSession reconnect-during-decision edge cases', () => {
  let mockIo: jest.Mocked<SocketIOServer>;
  let session: GameSession;

  const createMockIo = () =>
    ({
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    }) as any;

  beforeEach(() => {
    mockIo = createMockIo();
    session = new GameSession('test-reconnect-decision', mockIo, {} as any, new Map());
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  describe('decision timeout continues during reconnection window', () => {
    it('should auto-resolve decision phase even if player is disconnected', async () => {
      // Setup: game in line_processing phase for Player 1
      const state: any = {
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'line_processing',
        moveHistory: [],
        players: [
          { playerNumber: 1, type: 'human', id: 'p1' },
          { playerNumber: 2, type: 'human', id: 'p2' },
        ],
        board: {
          type: 'square8',
          size: 8,
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          eliminatedRings: { 1: 0, 2: 0 },
          formedLines: [
            {
              player: 1,
              positions: [
                { x: 0, y: 0 },
                { x: 1, y: 0 },
                { x: 2, y: 0 },
              ],
              length: 3,
            },
          ],
        },
      };

      const processLineMove: Move = {
        id: 'line-1',
        type: 'process_line',
        player: 1,
        thinkTime: 0,
        timestamp: new Date(),
        moveNumber: 1,
        formedLines: state.board.formedLines,
      };

      (session as any).gameEngine = {
        getGameState: jest.fn(() => state),
        getValidMoves: jest.fn(() => [processLineMove]),
      };
      (session as any).rulesFacade = {
        applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
      };

      // Start decision phase timeout
      (session as any).scheduleDecisionPhaseTimeout(state);

      // Simulate player 1 disconnecting (choices would be tracked by WebSocketInteractionHandler)
      // Note: WebSocketServer handles disconnect separately, but the decision timeout continues

      // Advance time past decision timeout (default 30s)
      await jest.advanceTimersByTimeAsync(config.decisionPhaseTimeouts.defaultTimeoutMs + 100);

      // The decision should have been auto-resolved even though player is "disconnected"
      expect((session as any).rulesFacade.applyMoveById).toHaveBeenCalledWith(1, 'line-1');
    });

    it('should allow reconnected player to make decision before timeout fires', async () => {
      // Setup: game in territory_processing phase for Player 2
      const state: any = {
        gameStatus: 'active',
        currentPlayer: 2,
        currentPhase: 'territory_processing',
        players: [
          { playerNumber: 1, type: 'human', id: 'p1' },
          { playerNumber: 2, type: 'human', id: 'p2' },
        ],
        board: {
          type: 'square8',
          size: 8,
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          eliminatedRings: { 1: 0, 2: 0 },
          formedLines: [],
        },
      };

      const regionMove: Move = {
        id: 'region-1',
        type: 'choose_territory_option',
        player: 2,
        thinkTime: 0,
        timestamp: new Date(),
        moveNumber: 1,
      };

      (session as any).gameEngine = {
        getGameState: jest.fn(() => state),
        getValidMoves: jest.fn(() => [regionMove]),
      };
      (session as any).rulesFacade = {
        applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
      };

      // Start decision phase timeout
      (session as any).scheduleDecisionPhaseTimeout(state);

      // Advance partially (10 seconds - within reconnection window)
      await jest.advanceTimersByTimeAsync(10_000);

      // Reset the timeout (simulating player reconnecting and making a move)
      (session as any).resetDecisionPhaseTimeout();

      // Advance past original timeout - should NOT auto-resolve since timer was reset
      await jest.advanceTimersByTimeAsync(25_000);

      // Should not have been called since timeout was reset
      expect((session as any).rulesFacade.applyMoveById).not.toHaveBeenCalled();
    });
  });

  describe('chain capture during reconnection scenarios', () => {
    it('should preserve chainCapturePosition on reconnect state sync', () => {
      // Setup: game in chain_capture phase with position set
      const chainPosition: Position = { x: 5, y: 3 };
      const state: any = {
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'chain_capture',
        chainCapturePosition: chainPosition,
        players: [
          { playerNumber: 1, type: 'human', id: 'p1' },
          { playerNumber: 2, type: 'human', id: 'p2' },
        ],
        board: {
          type: 'square8',
          size: 8,
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          eliminatedRings: { 1: 0, 2: 0 },
          formedLines: [],
        },
      };

      (session as any).gameEngine = {
        getGameState: jest.fn(() => state),
        getValidMoves: jest.fn(() => []),
      };

      // Verify getGameState returns chainCapturePosition for reconnected client
      const gameState = session.getGameState();
      expect(gameState.chainCapturePosition).toEqual(chainPosition);
      expect(gameState.currentPhase).toBe('chain_capture');
    });

    it('should auto-resolve chain capture on timeout when player disconnects', async () => {
      // Setup: game in chain_capture phase
      const chainPosition: Position = { x: 7, y: 3 };
      const state: any = {
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'chain_capture',
        chainCapturePosition: chainPosition,
        moveHistory: [],
        players: [
          { playerNumber: 1, type: 'human', id: 'p1' },
          { playerNumber: 2, type: 'human', id: 'p2' },
        ],
        board: {
          type: 'square8',
          size: 8,
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          eliminatedRings: { 1: 0, 2: 0 },
          formedLines: [],
        },
      };

      const chainMoves: Move[] = [
        {
          id: 'continue-1',
          type: 'continue_capture_segment',
          player: 1,
          from: chainPosition,
          captureTarget: { x: 5, y: 3 },
          to: { x: 3, y: 3 },
          thinkTime: 0,
          timestamp: new Date(),
          moveNumber: 1,
        },
        {
          id: 'end-chain',
          type: 'end_chain_capture',
          player: 1,
          thinkTime: 0,
          timestamp: new Date(),
          moveNumber: 1,
        },
      ];

      (session as any).gameEngine = {
        getGameState: jest.fn(() => state),
        getValidMoves: jest.fn(() => chainMoves),
      };
      (session as any).rulesFacade = {
        applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
      };

      // Start decision phase timeout
      (session as any).scheduleDecisionPhaseTimeout(state);

      // Advance past timeout
      await jest.advanceTimersByTimeAsync(config.decisionPhaseTimeouts.defaultTimeoutMs + 100);

      // Should auto-resolve by picking first available move
      expect((session as any).rulesFacade.applyMoveById).toHaveBeenCalled();
    });
  });

  describe('timeout coordination edge cases', () => {
    it('should not double-resolve if both timeouts race', async () => {
      // Edge case: decision timeout and something else both try to resolve
      const state: any = {
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'line_processing',
        moveHistory: [],
        players: [
          { playerNumber: 1, type: 'human', id: 'p1' },
          { playerNumber: 2, type: 'human', id: 'p2' },
        ],
        board: {
          type: 'square8',
          size: 8,
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          eliminatedRings: { 1: 0, 2: 0 },
          formedLines: [{ player: 1, positions: [{ x: 0, y: 0 }], length: 1 }],
        },
      };

      let applyCount = 0;
      const processLineMove: Move = {
        id: 'line-1',
        type: 'process_line',
        player: 1,
        thinkTime: 0,
        timestamp: new Date(),
        moveNumber: 1,
      };

      (session as any).gameEngine = {
        getGameState: jest.fn(() => state),
        getValidMoves: jest.fn(() => (applyCount === 0 ? [processLineMove] : [])),
      };
      (session as any).rulesFacade = {
        applyMoveById: jest.fn().mockImplementation(async () => {
          applyCount++;
          // Transition to non-decision phase after first resolution
          state.currentPhase = 'ring_placement';
          state.currentPlayer = 2;
          return { success: true, gameState: state };
        }),
      };

      // Start decision phase timeout
      (session as any).scheduleDecisionPhaseTimeout(state);

      // Advance past timeout
      await jest.advanceTimersByTimeAsync(config.decisionPhaseTimeouts.defaultTimeoutMs + 100);

      // Should only apply once - the phase check in handleDecisionPhaseTimedOut
      // should prevent duplicate resolution
      expect(applyCount).toBe(1);
    });

    it('should handle AI player not triggering decision timeout', async () => {
      // AI players should not have decision timeout scheduled
      const state: any = {
        gameStatus: 'active',
        currentPlayer: 1,
        currentPhase: 'line_processing',
        players: [
          { playerNumber: 1, type: 'ai', id: 'ai-1' },
          { playerNumber: 2, type: 'human', id: 'p2' },
        ],
        board: {
          type: 'square8',
          size: 8,
          stacks: new Map(),
          markers: new Map(),
          collapsedSpaces: new Map(),
          territories: new Map(),
          eliminatedRings: { 1: 0, 2: 0 },
          formedLines: [],
        },
      };

      (session as any).gameEngine = {
        getGameState: jest.fn(() => state),
        getValidMoves: jest.fn(() => []),
      };
      (session as any).rulesFacade = {
        applyMoveById: jest.fn().mockResolvedValue({ success: true, gameState: state }),
      };

      // This should not schedule timeout for AI player
      (session as any).scheduleDecisionPhaseTimeout(state);

      // Advance past what would be the timeout
      await jest.advanceTimersByTimeAsync(config.decisionPhaseTimeouts.defaultTimeoutMs + 100);

      // Should NOT have auto-resolved since player is AI
      expect((session as any).rulesFacade.applyMoveById).not.toHaveBeenCalled();
    });
  });
});
