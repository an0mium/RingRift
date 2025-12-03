/**
 * Focused metrics tests for GameSession abnormal termination.
 *
 * Verifies that terminate() records game session status transitions and
 * abnormal termination counters for active games, and skips them for
 * non-active games.
 *
 * Spec anchor: docs/P18.3-1_DECISION_LIFECYCLE_SPEC.md (§2.4 / §4.3).
 */

import { GameSession } from '../../src/server/game/GameSession';
import { getMetricsService } from '../../src/server/services/MetricsService';

// Stub MetricsService so we can observe calls to the relevant helpers
// without touching the real Prometheus registry.
jest.mock('../../src/server/services/MetricsService', () => {
  const metrics = {
    recordGameSessionStatusTransition: jest.fn(),
    updateGameSessionStatusCurrent: jest.fn(),
    recordAbnormalTermination: jest.fn(),
    recordAITurnRequestTerminal: jest.fn(),
    recordMoveRejected: jest.fn(),
  };

  return {
    __esModule: true,
    getMetricsService: () => metrics,
  };
});

// Database is not used in these tests (initialize() is never called), but
// provide a lightweight stub to avoid accidental usage.
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => ({
    game: {
      findUnique: jest.fn(),
      update: jest.fn(),
    },
  })),
}));

const createMockIo = () =>
  ({
    to: jest.fn().mockReturnThis(),
    emit: jest.fn(),
    sockets: {
      adapter: { rooms: new Map() },
      sockets: new Map(),
    },
  }) as any;

describe('GameSession abnormal termination metrics', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('records status transition and abnormal termination for active game', () => {
    const io = createMockIo();
    const session = new GameSession('metrics-game', io, {} as any, new Map());

    const activeState: any = {
      id: 'metrics-game',
      gameStatus: 'active',
      boardType: 'square8',
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      players: [],
      spectators: [],
      moveHistory: [],
      history: [],
      rngSeed: 1,
      board: {} as any,
      isRated: true,
    };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => activeState),
    };

    session.terminate('disconnect_timeout');

    const metrics = getMetricsService() as any;
    expect(metrics.recordGameSessionStatusTransition).toHaveBeenCalledWith(
      'active_turn',
      'abandoned'
    );
    expect(metrics.recordAbnormalTermination).toHaveBeenCalledWith('disconnect_timeout');
  });

  it('does not record abnormal termination metrics when game is not active', () => {
    const io = createMockIo();
    const session = new GameSession('metrics-game', io, {} as any, new Map());

    const completedState: any = {
      id: 'metrics-game',
      gameStatus: 'completed',
      boardType: 'square8',
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      players: [],
      spectators: [],
      moveHistory: [],
      history: [],
      rngSeed: 1,
      board: {} as any,
      isRated: true,
    };

    (session as any).gameEngine = {
      getGameState: jest.fn(() => completedState),
    };

    session.terminate('disconnect_timeout');

    const metrics = getMetricsService() as any;
    expect(metrics.recordGameSessionStatusTransition).not.toHaveBeenCalled();
    expect(metrics.recordAbnormalTermination).not.toHaveBeenCalled();
  });
});
