/**
 * Focused branch coverage tests for GameSession AI timeout handling,
 * orchestrator shadow path, and analysis-mode position evaluation.
 */

import type { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import type { GameState, Move } from '../../src/shared/types/game';
import type { RulesResult } from '../../src/server/game/RulesBackendFacade';

// Mocks
jest.mock('../../src/shared/utils/timeout', () => ({
  runWithTimeout: jest.fn(),
}));

jest.mock('../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    getAIConfig: jest.fn(),
    createAI: jest.fn(),
    createAIFromProfile: jest.fn(),
    getAIMove: jest.fn(),
    getLocalFallbackMove: jest.fn(),
    chooseLocalMoveFromCandidates: jest.fn(),
    getDiagnostics: jest.fn(() => ({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    })),
  },
}));

jest.mock('../../src/server/services/AIServiceClient', () => ({
  getAIServiceClient: jest.fn(),
}));

jest.mock('../../src/server/services/ShadowModeComparator', () => ({
  shadowComparator: {
    compare: jest.fn(),
  },
}));

jest.mock('../../src/server/game/turn/TurnEngineAdapter', () => ({
  createSimpleAdapter: jest.fn(),
  createAutoSelectDecisionHandler: jest.fn(),
}));

jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../src/server/config', () => ({
  config: {
    aiService: { requestTimeoutMs: 1234 },
    featureFlags: {
      analysisMode: {
        enabled: false,
      },
    },
    decisionPhaseTimeouts: {
      defaultTimeoutMs: 30000,
      warningBeforeTimeoutMs: 5000,
      extensionMs: 15000,
    },
    isTest: true,
    isDevelopment: true,
  },
}));

const { runWithTimeout } = require('../../src/shared/utils/timeout') as {
  runWithTimeout: jest.Mock;
};

const { globalAIEngine } = require('../../src/server/game/ai/AIEngine') as {
  globalAIEngine: {
    getAIMove: jest.Mock;
    getAIConfig: jest.Mock;
    createAI: jest.Mock;
  };
};

const { getAIServiceClient } = require('../../src/server/services/AIServiceClient') as {
  getAIServiceClient: jest.Mock;
};

const { shadowComparator } = require('../../src/server/services/ShadowModeComparator') as {
  shadowComparator: { compare: jest.Mock };
};

const turnAdapterModule = require('../../src/server/game/turn/TurnEngineAdapter') as {
  createSimpleAdapter: jest.Mock;
  createAutoSelectDecisionHandler: jest.Mock;
};

const { logger } = require('../../src/server/utils/logger');
const { config } = require('../../src/server/config');

const createMockIo = (): jest.Mocked<SocketIOServer> =>
  ({
    to: jest.fn().mockReturnThis(),
    emit: jest.fn(),
    sockets: {
      adapter: { rooms: new Map() },
      sockets: new Map(),
    },
  }) as any;

const createMinimalState = (): GameState =>
  ({
    id: 'game-1',
    boardType: 'square8',
    gameStatus: 'active',
    currentPlayer: 1,
    currentPhase: 'ring_placement',
    players: [],
    spectators: [],
    moveHistory: [],
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
    timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
  }) as any;

describe('GameSession.getAIMoveWithTimeout branches', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('passes timeoutMs through to runWithTimeout without adding extra delay layers', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    const fakeMove: Move = {
      id: 'm-timeout-prop',
      type: 'place_ring',
      player: 1,
      to: { x: 1, y: 1 },
      moveNumber: 1,
      timestamp: new Date(),
      thinkTime: 0,
    };

    // Ensure globalAIEngine.getAIMove is actually called by the operation.
    globalAIEngine.getAIMove.mockResolvedValueOnce(fakeMove);

    runWithTimeout.mockImplementationOnce(async (operation, options) => {
      // The timeout budget should be exactly what the caller provided.
      expect(options.timeoutMs).toBe(1234);

      const value = await operation();
      return {
        kind: 'ok',
        value,
        durationMs: 1,
      };
    });

    const state = createMinimalState();
    const result = await (session as any).getAIMoveWithTimeout(1, state, 1234);

    // getAIMove signature: (playerNumber, gameState, rng?, options?)
    expect(globalAIEngine.getAIMove).toHaveBeenCalledWith(1, state, undefined, undefined);
    expect(result).toBe(fakeMove);
  });

  it('returns move when runWithTimeout resolves with ok value', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    const fakeMove: Move = {
      id: 'm1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      moveNumber: 1,
      timestamp: new Date(),
      thinkTime: 0,
    };

    runWithTimeout.mockResolvedValueOnce({ kind: 'ok', value: fakeMove });

    const state = createMinimalState();
    const result = await (session as any).getAIMoveWithTimeout(1, state, 1000);

    expect(result).toBe(fakeMove);
  });

  it('returns null when runWithTimeout resolves with ok and null value', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    runWithTimeout.mockResolvedValueOnce({ kind: 'ok', value: null });

    const state = createMinimalState();
    const result = await (session as any).getAIMoveWithTimeout(1, state, 1000);

    expect(result).toBeNull();
  });

  it('throws timeout error when runWithTimeout reports timeout', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    runWithTimeout.mockResolvedValueOnce({ kind: 'timeout' });

    const state = createMinimalState();

    await expect((session as any).getAIMoveWithTimeout(1, state, 1000)).rejects.toMatchObject({
      message: 'AI request timeout',
      isTimeout: true,
    });
  });

  it('throws cancellation error when runWithTimeout reports canceled', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    runWithTimeout.mockResolvedValueOnce({
      kind: 'canceled',
      cancellationReason: 'manual',
    });

    const state = createMinimalState();

    await expect((session as any).getAIMoveWithTimeout(1, state, 1000)).rejects.toMatchObject({
      message: 'AI request canceled',
      cancellationReason: 'manual',
    });
  });

  it('throws guard error for unexpected TimedOperationResult kind', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    runWithTimeout.mockResolvedValueOnce({ kind: 'unexpected_kind' });

    const state = createMinimalState();

    await expect((session as any).getAIMoveWithTimeout(1, state, 1000)).rejects.toThrow(
      'Unhandled TimedOperationResult outcome in getAIMoveWithTimeout'
    );
  });

  it('derives aiRequestTimeoutMs from config.aiService.requestTimeoutMs', () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    // Private field, but stable for tests: host-level AI timeout budget.
    expect((session as any).aiRequestTimeoutMs).toBe(config.aiService.requestTimeoutMs);
  });
});

describe('GameSession analysis mode and position evaluation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('isAnalysisModeEnabled reflects feature flag', () => {
    const io = createMockIo();

    config.featureFlags.analysisMode.enabled = false;
    const sessionDisabled = new GameSession('game-1', io, {} as any, new Map());
    expect((sessionDisabled as any).isAnalysisModeEnabled()).toBe(false);

    config.featureFlags.analysisMode.enabled = true;
    const sessionEnabled = new GameSession('game-2', io, {} as any, new Map());
    expect((sessionEnabled as any).isAnalysisModeEnabled()).toBe(true);
  });

  it('evaluateAndBroadcastPosition emits position_evaluation on success', async () => {
    const io = createMockIo();
    const session = new GameSession('game-1', io, {} as any, new Map());

    const evaluatePositionMulti = jest.fn().mockResolvedValue({
      move_number: 5,
      board_type: 'square8',
      per_player: { 1: { score: 0.1 }, 2: { score: -0.1 } },
      engine_profile: { name: 'test-profile' },
      evaluation_scale: 'centipawns',
      generated_at: '2025-01-01T00:00:00.000Z',
    });

    getAIServiceClient.mockReturnValue({ evaluatePositionMulti });

    const state = createMinimalState();

    await (session as any).evaluateAndBroadcastPosition(state);

    expect(getAIServiceClient).toHaveBeenCalled();
    expect(evaluatePositionMulti).toHaveBeenCalledWith(state);
    expect(io.to).toHaveBeenCalledWith('game-1');
    expect(io.emit).toHaveBeenCalledWith(
      'position_evaluation',
      expect.objectContaining({
        type: 'position_evaluation',
        data: expect.objectContaining({
          gameId: 'game-1',
          moveNumber: 5,
          boardType: 'square8',
        }),
      })
    );
  });

  it('evaluateAndBroadcastPosition logs warning when AI service fails', async () => {
    const io = createMockIo();
    const session = new GameSession('game-err', io, {} as any, new Map());

    const evaluatePositionMulti = jest.fn().mockRejectedValue(new Error('service down'));
    getAIServiceClient.mockReturnValue({ evaluatePositionMulti });

    const state = createMinimalState();

    await (session as any).evaluateAndBroadcastPosition(state);

    expect(logger.warn).toHaveBeenCalledWith(
      'Failed to emit position evaluation',
      expect.objectContaining({
        gameId: 'game-err',
        error: 'service down',
      })
    );
    expect(io.emit).not.toHaveBeenCalledWith('position_evaluation', expect.anything());
  });
});

describe('GameSession orchestrator shadow path', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('applyMoveWithOrchestratorShadow delegates through shadow comparator', async () => {
    const io = createMockIo();
    const session = new GameSession('shadow-game', io, {} as any, new Map());

    const legacyResult: RulesResult = {
      success: true,
      gameState: createMinimalState(),
    } as any;

    const shadowResult: RulesResult = {
      success: true,
      gameState: { ...createMinimalState(), currentPlayer: 2 },
    } as any;

    (session as any).rulesFacade = {
      applyMove: jest.fn().mockResolvedValue(legacyResult),
    };

    const preState = createMinimalState();
    const engineMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      player: 1,
      type: 'place_ring',
      to: { x: 0, y: 0 },
      thinkTime: 0,
    };

    shadowComparator.compare.mockImplementation(
      async (
        _gameId: string,
        _moveNumber: number,
        legacyFn: () => Promise<RulesResult>,
        orchestratorFn: () => Promise<RulesResult>
      ) => {
        const _legacy = await legacyFn();
        const shadow = await orchestratorFn();
        return { result: shadow };
      }
    );

    // Replace runOrchestratorShadow to isolate this test from adapter internals.
    const orchestratorSpy = jest
      .spyOn(session as any, 'runOrchestratorShadow')
      .mockResolvedValue(shadowResult);

    const result = await (session as any).applyMoveWithOrchestratorShadow(preState, engineMove);

    expect((session as any).rulesFacade.applyMove).toHaveBeenCalled();
    expect(shadowComparator.compare).toHaveBeenCalled();
    expect(orchestratorSpy).toHaveBeenCalled();
    expect(result).toEqual(shadowResult);
  });

  it('runOrchestratorShadow builds RulesResult from adapter output', async () => {
    const io = createMockIo();
    const session = new GameSession('shadow-game-2', io, {} as any, new Map());

    const preState = createMinimalState();
    const engineMove: Omit<Move, 'id' | 'timestamp' | 'moveNumber'> = {
      player: 1,
      type: 'place_ring',
      to: { x: 1, y: 1 },
      thinkTime: 0,
    };

    const adapterProcessMove = jest.fn().mockResolvedValue({
      success: false,
      error: 'shadow mismatch',
      victoryResult: {
        winner: 1,
        reason: 'ring_elimination',
        finalScore: {
          ringsEliminated: { 1: 5, 2: 10 },
          territorySpaces: { 1: 0, 2: 0 },
          ringsRemaining: { 1: 13, 2: 8 },
        },
      },
    });

    const nextState: GameState = {
      ...preState,
      currentPlayer: 2,
    } as GameState;

    turnAdapterModule.createAutoSelectDecisionHandler.mockReturnValue({});
    turnAdapterModule.createSimpleAdapter.mockReturnValue({
      adapter: { processMove: adapterProcessMove },
      getState: jest.fn(() => nextState),
    });

    const result: RulesResult = await (session as any).runOrchestratorShadow(
      preState,
      engineMove,
      3
    );

    expect(turnAdapterModule.createAutoSelectDecisionHandler).toHaveBeenCalled();
    expect(turnAdapterModule.createSimpleAdapter).toHaveBeenCalledWith(preState, expect.anything());
    expect(adapterProcessMove).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'shadow-3',
        moveNumber: 3,
        type: 'place_ring',
      })
    );
    expect(result.success).toBe(false);
    expect(result.error).toBe('shadow mismatch');
    expect(result.gameResult).toEqual(
      expect.objectContaining({
        winner: 1,
        reason: 'ring_elimination',
      })
    );
    expect(result.gameState).toBe(nextState);
  });
});
