import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameResult,
  GameState,
  Move,
  PlayerChoice,
  PlayerChoiceResponseFor,
} from '../../src/shared/types/game';

/**
 * Regression: sandbox LPS should not prematurely end rich mid-game states.
 *
 * This test replays a real sandbox fixture reported from the UI
 * (square8, rngSeed=412378288, 2p human vs AI) and asserts that
 * Last-Player-Standing victory does not trigger in a state where
 * both players still have rings in hand and legal placements.
 *
 * The fixture is embedded here as a canonical Move sequence so that
 * the shared orchestrator + ClientSandboxEngine drive all turn and
 * LPS tracking exactly as in the browser.
 */
describe('ClientSandboxEngine LPS regression – sandbox fixture replay (square8)', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;
        const options: any[] = (anyChoice.options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;

        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<TChoice>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  /**
   * Moves copied (in minimal canonical form) from the user's
   * ringrift_sandbox_fixture_v1 sample with rngSeed=412378288.
   */
  function buildFixtureMoves(): Move[] {
    const ts = new Date('2025-12-05T00:00:00.000Z');

    const moves: Move[] = [
      {
        id: 'move-3,4-4,4-1',
        type: 'move_stack',
        player: 1,
        from: { x: 3, y: 4 },
        to: { x: 4, y: 4 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: '',
        type: 'place_ring',
        player: 2,
        to: { x: 4, y: 5 },
        placementCount: 2,
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 2,
      },
      {
        id: '',
        type: 'overtaking_capture',
        player: 2,
        from: { x: 4, y: 5 },
        captureTarget: { x: 4, y: 4 },
        to: { x: 4, y: 3 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 3,
      },
      {
        id: 'move-3,5-5,5-4',
        type: 'move_stack',
        player: 1,
        from: { x: 3, y: 5 },
        to: { x: 5, y: 5 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 4,
      },
      {
        id: '',
        type: 'place_ring',
        player: 2,
        to: { x: 1, y: 2 },
        placementCount: 2,
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 5,
      },
      {
        id: '',
        type: 'move_stack',
        player: 2,
        from: { x: 1, y: 2 },
        to: { x: 3, y: 0 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 6,
      },
      {
        id: 'move-5,5-5,3-7',
        type: 'move_stack',
        player: 1,
        from: { x: 5, y: 5 },
        to: { x: 5, y: 3 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 7,
      },
      {
        id: 'capture-5,3-4,3-0,3-8',
        type: 'overtaking_capture',
        player: 1,
        from: { x: 5, y: 3 },
        captureTarget: { x: 4, y: 3 },
        to: { x: 0, y: 3 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 8,
      },
      {
        id: 'continue_capture_segment-0,3-4,3-7,3-8',
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 0, y: 3 },
        captureTarget: { x: 4, y: 3 },
        to: { x: 7, y: 3 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 9,
      },
      {
        id: '',
        type: 'place_ring',
        player: 2,
        to: { x: 7, y: 0 },
        placementCount: 2,
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 10,
      },
      {
        id: '',
        type: 'overtaking_capture',
        player: 2,
        from: { x: 7, y: 0 },
        captureTarget: { x: 4, y: 3 },
        to: { x: 3, y: 4 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 11,
      },
      {
        id: 'capture-2,5-3,4-4,3-12',
        type: 'overtaking_capture',
        player: 1,
        from: { x: 2, y: 5 },
        captureTarget: { x: 3, y: 4 },
        to: { x: 4, y: 3 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 12,
      },
      {
        id: 'continue_capture_segment-4,3-3,4-0,7-12',
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 4, y: 3 },
        captureTarget: { x: 3, y: 4 },
        to: { x: 0, y: 7 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 13,
      },
      {
        id: '',
        type: 'place_ring',
        player: 2,
        to: { x: 7, y: 2 },
        placementCount: 2,
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 14,
      },
      {
        id: '',
        type: 'overtaking_capture',
        player: 2,
        from: { x: 7, y: 2 },
        captureTarget: { x: 7, y: 3 },
        to: { x: 7, y: 7 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 15,
      },
      {
        id: '',
        type: 'continue_capture_segment',
        player: 2,
        from: { x: 7, y: 7 },
        captureTarget: { x: 7, y: 3 },
        to: { x: 7, y: 0 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 16,
      },
      {
        id: 'move-6,5-2,1-17',
        type: 'move_stack',
        player: 1,
        from: { x: 6, y: 5 },
        to: { x: 2, y: 1 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 17,
      },
      {
        id: '',
        type: 'place_ring',
        player: 2,
        to: { x: 3, y: 7 },
        placementCount: 2,
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 18,
      },
      {
        id: '',
        type: 'move_stack',
        player: 2,
        from: { x: 3, y: 0 },
        to: { x: 5, y: 0 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 19,
      },
      {
        id: 'capture-0,7-3,7-6,7-20',
        type: 'overtaking_capture',
        player: 1,
        from: { x: 0, y: 7 },
        captureTarget: { x: 3, y: 7 },
        to: { x: 6, y: 7 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 20,
      },
      {
        id: 'continue_capture_segment-6,7-3,7-1,7-20',
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 6, y: 7 },
        captureTarget: { x: 3, y: 7 },
        to: { x: 1, y: 7 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 21,
      },
      {
        id: '',
        type: 'place_ring',
        player: 2,
        to: { x: 3, y: 3 },
        placementCount: 2,
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 22,
      },
      {
        id: '',
        type: 'move_stack',
        player: 2,
        from: { x: 3, y: 3 },
        to: { x: 1, y: 1 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 23,
      },
      {
        id: 'capture-1,7-1,1-1,0-24',
        type: 'overtaking_capture',
        player: 1,
        from: { x: 1, y: 7 },
        captureTarget: { x: 1, y: 1 },
        to: { x: 1, y: 0 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 24,
      },
      {
        id: '',
        type: 'place_ring',
        player: 2,
        to: { x: 4, y: 0 },
        placementCount: 1,
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 25,
      },
      {
        id: '',
        type: 'overtaking_capture',
        player: 2,
        from: { x: 5, y: 0 },
        captureTarget: { x: 4, y: 0 },
        to: { x: 2, y: 0 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 26,
      },
      {
        id: 'move-3,2-3,4-27',
        type: 'move_stack',
        player: 1,
        from: { x: 3, y: 2 },
        to: { x: 3, y: 4 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 27,
      },
      {
        id: '',
        type: 'place_ring',
        player: 2,
        to: { x: 2, y: 7 },
        placementCount: 2,
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 28,
      },
      {
        id: '',
        type: 'move_stack',
        player: 2,
        from: { x: 2, y: 0 },
        to: { x: 2, y: 4 },
        timestamp: ts,
        thinkTime: 0,
        moveNumber: 29,
      },
    ];

    return moves;
  }

  it('does not end the game via LPS in the reported mid-game state', async () => {
    const engine = createEngine();
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    // Sanity: game starts active with no victory result.
    expect(state.gameStatus).toBe('active');
    expect(engine.getVictoryResult()).toBeNull();

    const moves = buildFixtureMoves();

    for (const move of moves) {
      const before = engine.getGameState();
      expect(before.gameStatus).toBe('active');

      await engine.applyCanonicalMove(move);
    }

    const finalState = engine.getGameState();
    const victory: GameResult | null = engine.getVictoryResult();

    // Regression expectation: this rich mid-game position should
    // still be active; no LPS victory should have fired.
    expect(victory).toBeNull();
    expect(finalState.gameStatus).toBe('active');
    expect(finalState.winner).toBeUndefined();
  });
});
