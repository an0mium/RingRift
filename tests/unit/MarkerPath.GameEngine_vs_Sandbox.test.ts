import {
  BoardState,
  BoardType,
  GameState,
  Player,
  Position,
} from '../../src/shared/types/game';
import { BoardManager } from '../../src/server/game/BoardManager';
import { GameEngine } from '../../src/server/game/GameEngine';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  snapshotFromBoardAndPlayers,
  snapshotsEqual,
  diffSnapshots,
} from '../utils/stateSnapshots';

/**
 * Focused parity test for marker-path semantics along a movement path:
 *
 *   - Backend: GameEngine.processMarkersAlongPath via a direct call on a shared BoardState.
 *   - Sandbox: ClientSandboxEngine.applyMarkerEffectsAlongPath via its internal helper.
 *
 * Both are given the same BoardState with markers and a straight path from
 * `from` to `to`, and the resulting board snapshots are compared to ensure that:
 *
 *   - Own markers along the path collapse to territory, and
 *   - Opponent markers along the path flip to the mover's color,
 *   - No additional differences in markers/collapsedSpaces are introduced.
 */
describe('Marker-path semantics parity (backend vs sandbox)', () => {
  const boardType: BoardType = 'square8';

  function makeDummyPlayers(): Player[] {
    return [
      {
        id: 'p1',
        username: 'P1',
        type: 'ai',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 0,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'P2',
        type: 'ai',
        playerNumber: 2,
        isReady: true,
        timeRemaining: 0,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];
  }

  function cloneBoard(board: BoardState): BoardState {
    return {
      ...board,
      stacks: new Map(board.stacks),
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      territories: new Map(board.territories),
      formedLines: [...board.formedLines],
      eliminatedRings: { ...board.eliminatedRings },
    };
  }

  /**
   * Build a simple board with markers along a vertical path:
   *
   *   from = (2,1)
   *   to   = (2,5)
   *
   * Intermediate cells (2,2), (2,3), (2,4) are on the path.
   *
   * - At (2,3): own marker (player 1) — should collapse to territory.
   * - At (2,4): opponent marker (player 2) — should flip to player 1.
   *
   * No stacks or collapsed spaces are present initially so we isolate
   * marker-path behaviour.
   */
  function buildMarkerPathFixture() {
    const bm = new BoardManager(boardType);
    const board = bm.createBoard();

    const from: Position = { x: 2, y: 1 };
    const to: Position = { x: 2, y: 5 };

    // Own marker at (2,3)
    bm.setMarker({ x: 2, y: 3 }, 1, board);

    // Opponent marker at (2,4)
    bm.setMarker({ x: 2, y: 4 }, 2, board);

    const players = makeDummyPlayers();

    return { board, players, from, to, movingPlayer: 1 as const };
  }

  test('processMarkersAlongPath vs applyMarkerEffectsAlongPathOnBoard', async () => {
    const { board, players, from, to, movingPlayer } = buildMarkerPathFixture();

    // --- Backend path: GameEngine.processMarkersAlongPath on a shared board ---
    const timeControl = { initialTime: 0, increment: 0, type: 'rapid' as const };
    const backendEngine = new GameEngine(
      'marker-path-test',
      boardType,
      players.map((p) => ({ ...p })),
      timeControl,
      false
    );

    const backendAny: any = backendEngine;
    const backendState0: GameState = backendEngine.getGameState();
    const backendBoard = cloneBoard(board);
    const backendPlayers = players.map((p) => ({ ...p }));

    // Mirror full movement semantics: leave a departure marker on the
    // from-space before processing intermediate markers, just as
    // GameEngine.applyMove does for move_stack.
    backendAny.boardManager.setMarker(from, movingPlayer, backendBoard);

    backendAny.gameState = {
      ...backendState0,
      board: backendBoard,
      players: backendPlayers,
      currentPlayer: movingPlayer,
      currentPhase: 'movement',
    } as GameState;

    // Call the private marker-path helper directly.
    backendAny.processMarkersAlongPath(from, to, movingPlayer);

    const backendSnap = snapshotFromBoardAndPlayers(
      'backend-marker-path',
      backendAny.gameState.board as BoardState,
      backendAny.gameState.players as Player[]
    );

    // --- Sandbox path: ClientSandboxEngine.applyMarkerEffectsAlongPath on a clone ---
    const config: SandboxConfig = {
      boardType,
      numPlayers: players.length,
      playerKinds: players.map((p) => p.type as 'human' | 'ai'),
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice(choice: any) {
        const options = ((choice as any).options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;
        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as any;
      },
    };

    const sandboxEngine = new ClientSandboxEngine({
      config,
      interactionHandler: handler,
      traceMode: true,
    });

    const sandboxAny: any = sandboxEngine;
    const sandboxState0: GameState = sandboxEngine.getGameState();
    const sandboxBoard = cloneBoard(board);
    const sandboxPlayers = players.map((p) => ({ ...p }));

    sandboxAny.gameState = {
      ...sandboxState0,
      board: sandboxBoard,
      players: sandboxPlayers,
      currentPlayer: movingPlayer,
      currentPhase: 'movement',
    } as GameState;

    // Use the sandbox engine's internal wrapper, which delegates to
    // applyMarkerEffectsAlongPathOnBoard with the same semantics used in
    // real movement/capture.
    sandboxAny.applyMarkerEffectsAlongPath(from, to, movingPlayer, {});

    const sandboxSnap = snapshotFromBoardAndPlayers(
      'sandbox-marker-path',
      sandboxAny.gameState.board as BoardState,
      sandboxAny.gameState.players as Player[]
    );

    if (!snapshotsEqual(backendSnap, sandboxSnap)) {
      // eslint-disable-next-line no-console
      console.error(
        '[MarkerPath.GameEngine_vs_Sandbox] marker-path mismatch',
        diffSnapshots(backendSnap, sandboxSnap)
      );
    }

    expect(snapshotsEqual(backendSnap, sandboxSnap)).toBe(true);
  });
});