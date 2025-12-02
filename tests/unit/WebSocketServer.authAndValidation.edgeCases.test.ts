import { Server as HTTPServer } from 'http';
import { WebSocketServer, type AuthenticatedSocket } from '../../src/server/websocket/server';

// Mock socket.io to avoid wsEngine constructor issues in Jest
type MiddlewareFn = (socket: any, next: (err?: Error) => void) => void;
let capturedMiddleware: MiddlewareFn | null = null;
let connectionHandler: ((socket: any) => void) | null = null;

jest.mock('socket.io', () => {
  class FakeSocketIOServer {
    public _middleware: MiddlewareFn[] = [];
    public _events: Record<string, (socket: any) => void> = {};

    public use = jest.fn((fn: MiddlewareFn) => {
      this._middleware.push(fn);
      capturedMiddleware = fn;
    });

    public on = jest.fn((event: string, handler: (socket: any) => void) => {
      this._events[event] = handler;
      if (event === 'connection') {
        connectionHandler = handler;
      }
    });

    public to = jest.fn(() => ({
      emit: jest.fn(),
    }));

    public emit = jest.fn();

    public sockets = {
      adapter: {
        rooms: new Map<string, Set<string>>(),
      },
      sockets: new Map<string, any>(),
    };

    constructor(..._args: any[]) {}
  }

  return {
    Server: FakeSocketIOServer,
  };
});

jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => null),
}));

jest.mock('../../src/server/cache/redis', () => ({
  getRedisClient: jest.fn(() => null),
}));

jest.mock('../../src/server/middleware/auth', () => ({
  verifyToken: jest.fn(() => ({
    userId: 'user-1',
    tokenVersion: 1,
  })),
  validateUser: jest.fn(async () => ({
    id: 'user-1',
    username: 'User One',
  })),
}));

jest.mock('../../src/server/services/MetricsService', () => ({
  getMetricsService: () => ({
    incWebSocketConnections: jest.fn(),
    decWebSocketConnections: jest.fn(),
    recordWebsocketReconnection: jest.fn(),
    recordMoveRejected: jest.fn(),
  }),
}));

jest.mock('../../src/server/game/GameSessionManager', () => ({
  GameSessionManager: jest.fn().mockImplementation(() => ({
    withGameLock: (_gameId: string, fn: () => Promise<void>) => fn(),
    getOrCreateSession: async () => ({
      getGameState: () => ({
        gameStatus: 'active',
        boardType: 'square8',
        currentPlayer: 1,
        currentPhase: 'movement',
        players: [
          {
            playerNumber: 1,
            id: 'user-1',
            type: 'human',
          },
        ],
        spectators: [],
        moveHistory: [],
      }),
      handlePlayerMove: jest.fn(),
      getValidMoves: jest.fn(() => []),
      getInteractionHandler: () => ({
        handleChoiceResponse: jest.fn(),
      }),
    }),
  })),
}));

describe('WebSocketServer auth and validation edge cases', () => {
  const { logger } = require('../../src/server/utils/logger');

  beforeEach(() => {
    jest.clearAllMocks();
    capturedMiddleware = null;
    connectionHandler = null;
  });

  it('emits ACCESS_DENIED error when authentication fails in middleware', (done) => {
    const httpServer = new HTTPServer();
    const _wss = new WebSocketServer(httpServer);

    // The middleware should have been captured via jest.mock
    if (!capturedMiddleware) {
      done(new Error('No middleware captured'));
      return;
    }

    const socket: Partial<AuthenticatedSocket> = {
      id: 'socket-1',
      handshake: { auth: {}, query: {} } as any,
      emit: jest.fn(),
    };

    capturedMiddleware(socket, (err?: Error) => {
      expect(err).toBeInstanceOf(Error);
      expect(err?.message).toBe('Authentication token required');

      expect(socket.emit).toHaveBeenCalledWith('error', {
        type: 'error',
        code: 'ACCESS_DENIED',
        message: 'Authentication token required',
      });
      done();
    });
  });

  it('emits validation error for invalid player_move payload', async () => {
    const httpServer = new HTTPServer();
    const _wss = new WebSocketServer(httpServer);

    if (!connectionHandler) {
      throw new Error('No connection handler captured');
    }

    const socket: Partial<AuthenticatedSocket> = {
      id: 'socket-2',
      userId: 'user-1',
      emit: jest.fn(),
      on: jest.fn(),
      join: jest.fn(),
      leave: jest.fn(),
    };

    connectionHandler(socket);

    const playerMoveHandler = (socket.on as jest.Mock).mock.calls.find(
      ([event]: [string]) => event === 'player_move'
    )?.[1] as (data: unknown) => Promise<void>;

    await playerMoveHandler({ invalid: true } as any);

    expect(socket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        type: 'error',
        code: 'INVALID_PAYLOAD',
        event: 'player_move',
      })
    );
  });

  it('emits ACCESS_DENIED error when spectator sends player_choice_response', async () => {
    const httpServer = new HTTPServer();
    const wss = new WebSocketServer(httpServer) as any;

    if (!connectionHandler) {
      throw new Error('No connection handler captured');
    }

    const session = {
      getGameState: () => ({
        gameStatus: 'active',
        boardType: 'square8',
        currentPlayer: 1,
        currentPhase: 'movement',
        players: [
          {
            playerNumber: 1,
            id: 'player-1',
            type: 'human',
          },
        ],
        spectators: ['spectator-1'],
        moveHistory: [],
      }),
      getInteractionHandler: () => ({
        handleChoiceResponse: jest.fn(),
      }),
    };

    (wss as any).sessionManager = {
      getSession: jest.fn(() => session),
    };

    const socket: Partial<AuthenticatedSocket> = {
      id: 'socket-3',
      userId: 'spectator-1',
      gameId: 'game-1',
      emit: jest.fn(),
      on: jest.fn(),
      join: jest.fn(),
      leave: jest.fn(),
    };

    connectionHandler(socket);

    const playerChoiceHandler = (socket.on as jest.Mock).mock.calls.find(
      ([event]: [string]) => event === 'player_choice_response'
    )?.[1] as (data: unknown) => void;

    await playerChoiceHandler({
      choiceId: 'choice-1',
      playerNumber: 1,
      selectedOption: null,
    });

    expect(socket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        type: 'error',
        code: 'ACCESS_DENIED',
        event: 'player_choice_response',
        message: 'Spectators cannot respond to player choices',
      })
    );
  });

  it('emits GAME_NOT_FOUND when no active session exists for player_choice_response', async () => {
    const httpServer = new HTTPServer();
    const wss = new WebSocketServer(httpServer) as any;

    if (!connectionHandler) {
      throw new Error('No connection handler captured');
    }

    (wss as any).sessionManager = {
      getSession: jest.fn(() => undefined),
    };

    const socket: Partial<AuthenticatedSocket> = {
      id: 'socket-4',
      userId: 'user-1',
      gameId: 'game-1',
      emit: jest.fn(),
      on: jest.fn(),
      join: jest.fn(),
      leave: jest.fn(),
    };

    connectionHandler(socket);

    const playerChoiceHandler = (socket.on as jest.Mock).mock.calls.find(
      ([event]: [string]) => event === 'player_choice_response'
    )?.[1] as (data: unknown) => void;

    await playerChoiceHandler({
      choiceId: 'choice-2',
      playerNumber: 1,
      selectedOption: null,
    });

    expect(socket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        type: 'error',
        code: 'GAME_NOT_FOUND',
        event: 'player_choice_response',
        message: 'Game not found',
      })
    );
  });

  it('emits INTERNAL_ERROR Game is not active when underlying session reports inactive game', async () => {
    const httpServer = new HTTPServer();
    const wss = new WebSocketServer(httpServer) as any;

    if (!connectionHandler) {
      throw new Error('No connection handler captured');
    }

    (wss as any).sessionManager = {
      withGameLock: async (_gameId: string, fn: () => Promise<void>) => {
        await fn();
      },
      getOrCreateSession: async () => ({
        handlePlayerMove: jest.fn(async () => {
          throw new Error('Game is not active');
        }),
      }),
    };

    const socket: Partial<AuthenticatedSocket> = {
      id: 'socket-5',
      userId: 'user-1',
      gameId: 'game-1',
      emit: jest.fn(),
      on: jest.fn(),
      join: jest.fn(),
      leave: jest.fn(),
    };

    connectionHandler(socket);

    const playerMoveHandler = (socket.on as jest.Mock).mock.calls.find(
      ([event]: [string]) => event === 'player_move'
    )?.[1] as (data: unknown) => Promise<void>;

    await playerMoveHandler({
      gameId: 'game-1',
      move: {
        moveType: 'move_stack',
        position: {
          from: { x: 0, y: 0 },
          to: { x: 1, y: 0 },
        },
        moveNumber: 1,
      },
    } as any);

    expect(socket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        type: 'error',
        code: 'INTERNAL_ERROR',
        event: 'player_move',
        message: 'Game is not active',
      })
    );
  });
});
