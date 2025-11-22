import { WebSocketServer } from '../../src/server/websocket/server';
import { Move, GameState } from '../../src/shared/types/game';

// Jest-hoisted mock state for the Prisma client methods used by
// WebSocketServer.handlePlayerMove. We keep these mocks at the module
// level so individual tests can configure expectations.
const mockFindUnique = jest.fn();
const mockCreateMove = jest.fn();
const mockUpdateGame = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    game: {
      findUnique: mockFindUnique,
      update: mockUpdateGame,
    },
    move: {
      create: mockCreateMove,
    },
  }),
}));

// Minimal Socket.IO "server" stub that records game_state emissions.
class FakeSocketIOServer {
  public toCalls: Array<{ gameId: string; event: string; payload: any }> = [];

  to(gameId: string) {
    return {
      emit: (event: string, payload: any) => {
        this.toCalls.push({ gameId, event, payload });
      },
    };
  }
}

describe('WebSocketServer + RulesBackendFacade integration', () => {
  beforeEach(() => {
    mockFindUnique.mockReset();
    mockCreateMove.mockReset();
    mockUpdateGame.mockReset();
  });

  it('handlePlayerMove delegates to GameSession.handlePlayerMove via GameSessionManager', async () => {
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer as any;

    const fakeIo = new FakeSocketIOServer();
    serverAny.io = fakeIo;

    const gameId = 'game-rules-backend';
    const userId = 'user-1';

    // Lightweight game record: active status so the handler proceeds.
    mockFindUnique.mockResolvedValue({
      id: gameId,
      status: 'active',
      allowSpectators: true,
    } as any);

    const baseState: GameState = {
      id: gameId,
      boardType: 'square8',
      board: {
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      },
      players: [
        {
          id: userId,
          username: 'Human',
          type: 'human',
          playerNumber: 1,
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
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 18,
      totalRingsEliminated: 0,
      victoryThreshold: 10,
      territoryVictoryThreshold: 32,
    };

    const state = { ...baseState };

    // Fake GameSession implementation that captures the canonical move
    // payload that WebSocketServer passes through GameSessionManager.
    const fakeSession: any = {
      handlePlayerMove: jest.fn().mockResolvedValue(undefined),
      getGameState: jest.fn(() => state),
      getValidMoves: jest.fn(() => []),
    };

    const sessionManager: any = serverAny.sessionManager;
    // Short-circuit locking for the test while preserving the contract that
    // handlePlayerMove runs inside withGameLock.
    sessionManager.withGameLock = jest.fn(async (_gameId: string, fn: () => Promise<void>) => {
      await fn();
    });
    sessionManager.getOrCreateSession = jest.fn().mockResolvedValue(fakeSession);

    // Minimal AuthenticatedSocket stub for a human player in the room.
    const fakeSocket: any = {
      userId,
      username: 'Human',
      gameId,
    };

    // Client payload: geometry-based move. WebSocketServer is responsible
    // only for forwarding this to the appropriate GameSession; canonical
    // Move construction and rules application are handled inside the
    // session/RulesBackendFacade layer.
    const clientMove = {
      gameId,
      move: {
        moveNumber: 1,
        position: JSON.stringify({ to: { x: 0, y: 0 } }),
        moveType: 'place_ring',
      },
    };

    await serverAny.handlePlayerMove(fakeSocket, clientMove);

    // Ensure that the move was processed under the per-game lock and that
    // the session was resolved via GameSessionManager.
    expect(sessionManager.withGameLock).toHaveBeenCalledTimes(1);
    expect(sessionManager.withGameLock).toHaveBeenCalledWith(gameId, expect.any(Function));

    expect(sessionManager.getOrCreateSession).toHaveBeenCalledTimes(1);
    expect(sessionManager.getOrCreateSession).toHaveBeenCalledWith(gameId);

    // The GameSession should receive the original socket and move payload.
    expect(fakeSession.handlePlayerMove).toHaveBeenCalledTimes(1);
    const [socketArg, moveArg] = fakeSession.handlePlayerMove.mock.calls[0];
    expect(socketArg).toBe(fakeSocket);
    expect(moveArg).toEqual(clientMove.move);
  });
});
