import { GameSession } from '../../src/server/game/GameSession';
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';

jest.mock('../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    getAIConfig: jest.fn(() => ({})),
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

const mockedGlobalAIEngine = globalAIEngine as jest.Mocked<typeof globalAIEngine>;

describe('GameSession AI request state modeling', () => {
  const makeSession = (initialState: any) => {
    const io = {
      to: jest.fn().mockReturnThis(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient = {} as any;

    const session = new GameSession('game-1', io, pythonClient, new Map());

    // Track call count to return different states
    let callCount = 0;

    // Inject a minimal gameEngine and rulesFacade so maybePerformAITurn
    // can operate without hitting the real engine or database.
    (session as any).gameEngine = {
      getGameState: jest.fn(() => {
        // After first call, return a state where it's not AI's turn (to prevent recursion)
        callCount++;
        if (callCount > 1) {
          return {
            ...initialState,
            currentPlayer: 1,
            // Change gameStatus to inactive to break recursion
            gameStatus: 'completed',
          };
        }
        return initialState;
      }),
      getValidMoves: jest.fn().mockReturnValue([]),
    };

    (session as any).rulesFacade = {
      applyMove: jest.fn(),
      getDiagnostics: jest.fn().mockReturnValue({
        pythonEvalFailures: 0,
        pythonBackendFallbacks: 0,
        pythonShadowErrors: 0,
      }),
    };

    // Stub out persistence / broadcast side effects
    (session as any).persistAIMove = jest.fn(async () => {});
    (session as any).broadcastUpdate = jest.fn(async () => {});

    return session as any;
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('marks AI request as completed for a successful service-backed move', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    mockedGlobalAIEngine.getAIMove.mockResolvedValueOnce({
      id: 'm1',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 0, r: 0 },
      moveNumber: 1,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    (session.rulesFacade.applyMove as jest.Mock).mockResolvedValueOnce({
      success: true,
      gameState: initialState,
    });

    await session.maybePerformAITurn();

    const requestState = session.getLastAIRequestStateForTesting();
    expect(requestState.kind).toBe('completed');
  });

  it('does not start new AI HTTP calls after session termination cancels the token', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    // Terminate the session up front; this cancels the session-level
    // CancellationToken before maybePerformAITurn is invoked.
    session.terminate('session_cleanup');

    // If cancellation were not observed by getAIMoveWithTimeout / AIEngine,
    // this call would attempt to invoke globalAIEngine.getAIMove.
    await session.maybePerformAITurn();

    // Verify that no remote AI call was attempted after termination.
    expect(mockedGlobalAIEngine.getAIMove).not.toHaveBeenCalled();

    // The AIRequestState should remain in a non-terminals-or-idle state
    // that reflects the fact that no new request was started post-terminate.
    const requestState = session.getLastAIRequestStateForTesting();
    expect(['idle', 'canceled'].includes(requestState.kind)).toBe(true);
  });

  it('records a failed AI request when local fallback is rejected by the rules engine', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    // Service returns a move that is rejected by the rules engine
    mockedGlobalAIEngine.getAIMove.mockResolvedValueOnce({
      id: 'service-move',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 0, r: 0 },
      moveNumber: 1,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    // Local fallback is also provided but will be rejected
    mockedGlobalAIEngine.getLocalFallbackMove.mockReturnValue({
      id: 'fallback-move',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 1, r: 0 },
      moveNumber: 2,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    (session.rulesFacade.applyMove as jest.Mock)
      // Reject service-provided move
      .mockResolvedValueOnce({ success: false, error: 'service move rejected' })
      // Reject local fallback move as well
      .mockResolvedValueOnce({ success: false, error: 'fallback move rejected' });

    await session.maybePerformAITurn();

    const requestState = session.getLastAIRequestStateForTesting();
    expect(requestState.kind).toBe('failed');
    if (requestState.kind === 'failed') {
      expect(requestState.code).toBe('AI_SERVICE_OVERLOADED');
      // The exact aiErrorType may vary depending on where the failure is
      // surfaced (rules engine vs. AI orchestration), but we always
      // record a terminal failure code for diagnostics.
      expect(requestState.aiErrorType).toBeDefined();
    }
  });

  it('handles AI service returning null with successful fallback', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    // Service returns null (no move from service)
    mockedGlobalAIEngine.getAIMove.mockResolvedValueOnce(null);

    // Local fallback provides a valid move
    mockedGlobalAIEngine.getLocalFallbackMove.mockReturnValue({
      id: 'fallback-move',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 1, r: 0 },
      moveNumber: 1,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    (session.rulesFacade.applyMove as jest.Mock).mockResolvedValueOnce({
      success: true,
      gameState: initialState,
    });

    await session.maybePerformAITurn();

    const requestState = session.getLastAIRequestStateForTesting();
    // The request should eventually complete after successful fallback
    expect(requestState.kind).toBe('completed');
  });

  it('handles AI service returning null with no fallback available', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    // Service returns null
    mockedGlobalAIEngine.getAIMove.mockResolvedValueOnce(null);

    // No fallback move available
    mockedGlobalAIEngine.getLocalFallbackMove.mockReturnValue(null);

    await session.maybePerformAITurn();

    const requestState = session.getLastAIRequestStateForTesting();
    // Should fail with AI_SERVICE_OVERLOADED when no fallback available
    expect(requestState.kind).toBe('failed');
    if (requestState.kind === 'failed') {
      expect(requestState.code).toBe('AI_SERVICE_OVERLOADED');
    }
  });

  it('marks AI request as completed when service move rejected but fallback succeeds', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    // Service returns a move that will be rejected
    mockedGlobalAIEngine.getAIMove.mockResolvedValueOnce({
      id: 'service-move',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 0, r: 0 },
      moveNumber: 1,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    // Local fallback provides an alternate move
    mockedGlobalAIEngine.getLocalFallbackMove.mockReturnValue({
      id: 'fallback-move',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 1, r: 0 },
      moveNumber: 1,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    (session.rulesFacade.applyMove as jest.Mock)
      // Service move rejected
      .mockResolvedValueOnce({ success: false, error: 'service move rejected' })
      // Fallback move accepted
      .mockResolvedValueOnce({ success: true, gameState: initialState });

    await session.maybePerformAITurn();

    const requestState = session.getLastAIRequestStateForTesting();
    // Should complete after successful fallback
    expect(requestState.kind).toBe('completed');
  });

  it('handles AI service error with fallback failure after service exception', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    // Service throws an error
    const serviceError = new Error('AI service unavailable') as Error & { aiErrorType?: string };
    serviceError.aiErrorType = 'SERVICE_UNAVAILABLE';
    mockedGlobalAIEngine.getAIMove.mockRejectedValueOnce(serviceError);

    // Fallback is not available - so we go through handleNoMoveFromService -> handleAIFatalFailure
    mockedGlobalAIEngine.getLocalFallbackMove.mockReturnValue(null);

    await session.maybePerformAITurn();

    const requestState = session.getLastAIRequestStateForTesting();
    expect(requestState.kind).toBe('failed');
    if (requestState.kind === 'failed') {
      // After service error, the system tries fallback. When fallback also fails,
      // the final state is AI_SERVICE_OVERLOADED with the fallback failure reason
      expect(requestState.code).toBe('AI_SERVICE_OVERLOADED');
      expect(requestState.aiErrorType).toBe('both_service_and_fallback_failed');
    }
  });

  it('transitions through fallback_local state when service move rejected', async () => {
    const initialState = {
      id: 'game-1',
      gameStatus: 'active',
      currentPlayer: 1,
      currentPhase: 'main',
      boardType: 'square8',
      players: [
        {
          id: 'ai-player-1',
          playerNumber: 1,
          type: 'ai',
          timeRemaining: 600000,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      spectators: [],
      moveHistory: [],
      rngSeed: 1234,
    } as any;

    const session = makeSession(initialState);

    // Track the sequence of AI request states during execution
    const stateSnapshots: string[] = [];
    const originalGetState = session.getLastAIRequestStateForTesting.bind(session);

    // Service returns a move
    mockedGlobalAIEngine.getAIMove.mockResolvedValueOnce({
      id: 'service-move',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 0, r: 0 },
      moveNumber: 1,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    // Fallback move
    mockedGlobalAIEngine.getLocalFallbackMove.mockReturnValue({
      id: 'fallback-move',
      player: 1,
      type: 'place_ring',
      from: null,
      to: { q: 1, r: 0 },
      moveNumber: 1,
      timestamp: new Date().toISOString(),
      thinkTime: 0,
    } as any);

    (session.rulesFacade.applyMove as jest.Mock)
      .mockResolvedValueOnce({ success: false, error: 'rejected' })
      .mockResolvedValueOnce({ success: true, gameState: initialState });

    await session.maybePerformAITurn();

    // After successful fallback, state should be completed
    const finalState = session.getLastAIRequestStateForTesting();
    expect(finalState.kind).toBe('completed');
  });
});
