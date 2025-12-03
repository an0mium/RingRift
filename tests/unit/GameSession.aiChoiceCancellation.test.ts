import { GameSession } from '../../src/server/game/GameSession';
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';
import { getAIServiceClient } from '../../src/server/services/AIServiceClient';
import { AIInteractionHandler } from '../../src/server/game/ai/AIInteractionHandler';
import type {
  CaptureDirectionChoice,
  LineOrderChoice,
  PlayerChoice,
  Position,
  RegionOrderChoice,
  RingEliminationChoice,
} from '../../src/shared/types/game';

jest.mock('../../src/server/services/AIServiceClient');

describe('GameSession AI choice cancellation wiring', () => {
  it('propagates GameSession.terminate cancellation into AIServiceClient-backed line_order choices', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const recordedTokens: unknown[] = [];

    const fakeClient = {
      getLineOrderChoice: jest.fn(
        async (
          _gameState: any,
          _playerNumber: number,
          _difficulty: number,
          _aiType: any,
          _options: any,
          requestOptions?: { token?: { isCanceled: boolean } }
        ) => {
          if (requestOptions?.token) {
            recordedTokens.push(requestOptions.token);
          }
          // Minimal payload matching AIServiceClient contract
          return {
            selectedOption: _options[0],
            aiType: 'heuristic',
            difficulty: _difficulty,
          };
        }
      ),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const io = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient: any = {
      evaluateMove: jest.fn(),
      healthCheck: jest.fn(),
    };

    const userSockets = new Map<string, string>();
    const session = new GameSession('game-choices', io, pythonClient, userSockets);

    // Use the session's cancellation token for the AI interaction handler.
    const sessionToken = (session as any).sessionCancellationSource.token;
    const handler = new AIInteractionHandler(sessionToken);

    // Configure a service-backed AI profile for player 1.
    globalAIEngine.createAIFromProfile(1, {
      difficulty: 5,
      mode: 'service',
      aiType: 'heuristic',
    });

    const positionsA: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ];
    const positionsB: Position[] = [
      { x: 2, y: 2 },
      { x: 3, y: 3 },
    ];

    const choice: LineOrderChoice = {
      id: 'choice-line-order-1',
      gameId: 'game-choices',
      playerNumber: 1,
      prompt: 'Test line order',
      type: 'line_order',
      options: [
        { moveId: 'm-short', lineId: 'short', markerPositions: positionsA },
        { moveId: 'm-long', lineId: 'long', markerPositions: positionsB },
      ],
    };

    // Cancel the session-scoped token before requesting the AI-backed
    // choice so that downstream callers observe a canceled token. We
    // avoid calling session.terminate() here because that requires a
    // fully initialised gameEngine.
    (session as any).sessionCancellationSource.cancel('session_cleanup');

    await handler.requestChoice(choice as PlayerChoice);

    // The AIInteractionHandler → AIEngine → AIServiceClient chain should
    // observe a canceled token for this choice, allowing the real client
    // to short-circuit HTTP calls via token.throwIfCanceled.
    expect(fakeClient.getLineOrderChoice).toHaveBeenCalledTimes(1);
    expect(recordedTokens.length).toBe(1);
    const token = recordedTokens[0] as { isCanceled: boolean };
    expect(token.isCanceled).toBe(true);
  });

  it('propagates GameSession.terminate cancellation into AIServiceClient-backed region_order choices', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const recordedTokens: unknown[] = [];

    const fakeClient = {
      getRegionOrderChoice: jest.fn(
        async (
          _gameState: any,
          _playerNumber: number,
          _difficulty: number,
          _aiType: any,
          _options: any,
          requestOptions?: { token?: { isCanceled: boolean } }
        ) => {
          if (requestOptions?.token) {
            recordedTokens.push(requestOptions.token);
          }
          return _options[0];
        }
      ),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const io = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient: any = {
      evaluateMove: jest.fn(),
      healthCheck: jest.fn(),
    };

    const userSockets = new Map<string, string>();
    const session = new GameSession('game-region-order', io, pythonClient, userSockets);

    // Stub a minimal gameEngine so GameSession.terminate can safely
    // inspect game status when recording metrics.
    (session as any).gameEngine = {
      getGameState: jest.fn(() => ({
        gameStatus: 'active',
      })),
    };

    // Use the session's cancellation token for the AI interaction handler.
    const sessionToken = (session as any).sessionCancellationSource.token;
    const handler = new AIInteractionHandler(sessionToken);

    // Configure a service-backed AI profile for player 1 so that
    // AIInteractionHandler delegates to AIServiceClient.getRegionOrderChoice.
    globalAIEngine.createAIFromProfile(1, {
      difficulty: 5,
      mode: 'service',
      aiType: 'heuristic',
    });

    const choice: RegionOrderChoice = {
      id: 'choice-region-order-1',
      gameId: 'game-region-order',
      playerNumber: 1,
      prompt: 'Test region order',
      type: 'region_order',
      options: [
        { regionId: 'A', size: 2 },
        { regionId: 'B', size: 3 },
      ],
    };

    // Invoke GameSession.terminate(), which cancels the shared
    // session token before the AI-backed choice is requested.
    session.terminate('session_cleanup');

    await handler.requestChoice(choice as PlayerChoice);

    expect(fakeClient.getRegionOrderChoice).toHaveBeenCalledTimes(1);
    expect(recordedTokens.length).toBe(1);
    const token = recordedTokens[0] as { isCanceled: boolean };
    expect(token.isCanceled).toBe(true);
  });

  it('propagates GameSession.terminate cancellation into AIServiceClient-backed ring_elimination choices', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const recordedTokens: unknown[] = [];

    const fakeClient = {
      getRingEliminationChoice: jest.fn(
        async (
          _gameState: any,
          _playerNumber: number,
          _difficulty: number,
          _aiType: any,
          _options: any,
          requestOptions?: { token?: { isCanceled: boolean } }
        ) => {
          if (requestOptions?.token) {
            recordedTokens.push(requestOptions.token);
          }
          return _options[0];
        }
      ),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const io = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient: any = {
      evaluateMove: jest.fn(),
      healthCheck: jest.fn(),
    };

    const userSockets = new Map<string, string>();
    const session = new GameSession('game-ring-elim', io, pythonClient, userSockets);

    (session as any).gameEngine = {
      getGameState: jest.fn(() => ({
        gameStatus: 'active',
      })),
    };

    const sessionToken = (session as any).sessionCancellationSource.token;
    const handler = new AIInteractionHandler(sessionToken);

    globalAIEngine.createAIFromProfile(1, {
      difficulty: 5,
      mode: 'service',
      aiType: 'heuristic',
    });

    const choice: RingEliminationChoice = {
      id: 'choice-ring-elim-1',
      gameId: 'game-ring-elim',
      playerNumber: 1,
      prompt: 'Test ring elimination',
      type: 'ring_elimination',
      options: [
        { position: { x: 0, y: 0 }, capHeight: 1, totalHeight: 2 },
        { position: { x: 1, y: 1 }, capHeight: 2, totalHeight: 3 },
      ],
    } as any;

    session.terminate('session_cleanup');

    await handler.requestChoice(choice as PlayerChoice);

    expect(fakeClient.getRingEliminationChoice).toHaveBeenCalledTimes(1);
    expect(recordedTokens.length).toBe(1);
    const token = recordedTokens[0] as { isCanceled: boolean };
    expect(token.isCanceled).toBe(true);
  });

  it('propagates GameSession.terminate cancellation into AIServiceClient-backed capture_direction choices', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const recordedTokens: unknown[] = [];

    const fakeClient = {
      getCaptureDirectionChoice: jest.fn(
        async (
          _gameState: any,
          _playerNumber: number,
          _difficulty: number,
          _aiType: any,
          options: CaptureDirectionChoice['options'],
          requestOptions?: { token?: { isCanceled: boolean } }
        ) => {
          if (requestOptions?.token) {
            recordedTokens.push(requestOptions.token);
          }

          return {
            selectedOption: options[0],
            aiType: 'heuristic',
            difficulty: _difficulty,
          };
        }
      ),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const io = {
      to: jest.fn().mockReturnThis(),
      emit: jest.fn(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient: any = {
      evaluateMove: jest.fn(),
      healthCheck: jest.fn(),
    };

    const userSockets = new Map<string, string>();
    const session = new GameSession('game-capture-direction', io, pythonClient, userSockets);

    (session as any).gameEngine = {
      getGameState: jest.fn(() => ({
        gameStatus: 'active',
      })),
    };

    const sessionToken = (session as any).sessionCancellationSource.token;
    const handler = new AIInteractionHandler(sessionToken);

    globalAIEngine.createAIFromProfile(1, {
      difficulty: 5,
      mode: 'service',
      aiType: 'heuristic',
    });

    const choice: CaptureDirectionChoice = {
      id: 'choice-capture-direction-1',
      gameId: 'game-capture-direction',
      playerNumber: 1,
      prompt: 'Test capture direction',
      type: 'capture_direction',
      options: [
        {
          targetPosition: { x: 3, y: 3 },
          landingPosition: { x: 5, y: 5 },
          capturedCapHeight: 2,
        },
        {
          targetPosition: { x: 4, y: 4 },
          landingPosition: { x: 6, y: 6 },
          capturedCapHeight: 3,
        },
      ],
    };

    session.terminate('session_cleanup');

    await handler.requestChoice(choice as PlayerChoice);

    expect(fakeClient.getCaptureDirectionChoice).toHaveBeenCalledTimes(1);
    expect(recordedTokens.length).toBe(1);
    const token = recordedTokens[0] as { isCanceled: boolean };
    expect(token.isCanceled).toBe(true);
  });
});
