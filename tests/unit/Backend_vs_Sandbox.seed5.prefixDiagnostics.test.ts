import {
  BoardType,
  GameState,
  Move,
} from '../../src/shared/types/game';
import {
  runSandboxAITrace,
  createBackendEngineFromInitialState,
} from '../utils/traces';
import { findMatchingBackendMove } from '../utils/moveMatching';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  snapshotFromGameState,
  snapshotsEqual,
  diffSnapshots,
  ComparableSnapshot,
} from '../utils/stateSnapshots';

/**
 * Generic prefix diagnostics harness for seed=5 trace parity.
 *
 * This test:
 *   1) Generates the sandbox AI trace for square8 / 2p / seed=5.
 *   2) Rebuilds fresh backend + sandbox engines from the trace initial state.
 *   3) For each move index i, replays the canonical move into both engines:
 *        - Backend via getValidMoves + findMatchingBackendMove + makeMove.
 *        - Sandbox via applyCanonicalMove.
 *   4) At each step logs:
 *        - The canonical move and index.
 *        - Compact backend vs sandbox internal flags
 *          (pendingTerritorySelfElimination, pendingLineRewardElimination,
 *           hasPlacedThisTurn, mustMoveFromStackKey).
 *        - Snapshot equality and, for the first mismatch, a structured diff
 *          of board/players/markers/collapsed spaces.
 *
 * This is intentionally diagnostic: it does not currently fail on snapshot
 * mismatches, serving as a step-by-step replay log that narrows down the
 * exact index and shape of backend vs sandbox divergence.
 */
describe('Backend vs Sandbox prefix diagnostics (square8 / 2p / seed=5)', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 5;
  const MAX_STEPS = 60;

  function summariseMarkerAndCollapsedDiff(
    backend: ComparableSnapshot,
    sandbox: ComparableSnapshot
  ): {
    markerKeysOnlyInBackend: string[];
    markerKeysOnlyInSandbox: string[];
    collapsedKeysOnlyInBackend: string[];
    collapsedKeysOnlyInSandbox: string[];
  } {
    const backendMarkerKeys = new Set(backend.markers.map((m) => m.key));
    const sandboxMarkerKeys = new Set(sandbox.markers.map((m) => m.key));
    const backendCollapsedKeys = new Set(backend.collapsedSpaces.map((c) => c.key));
    const sandboxCollapsedKeys = new Set(sandbox.collapsedSpaces.map((c) => c.key));

    const markerKeysOnlyInBackend: string[] = [];
    const markerKeysOnlyInSandbox: string[] = [];
    const collapsedKeysOnlyInBackend: string[] = [];
    const collapsedKeysOnlyInSandbox: string[] = [];

    for (const key of backendMarkerKeys) {
      if (!sandboxMarkerKeys.has(key)) {
        markerKeysOnlyInBackend.push(key);
      }
    }
    for (const key of sandboxMarkerKeys) {
      if (!backendMarkerKeys.has(key)) {
        markerKeysOnlyInSandbox.push(key);
      }
    }

    for (const key of backendCollapsedKeys) {
      if (!sandboxCollapsedKeys.has(key)) {
        collapsedKeysOnlyInBackend.push(key);
      }
    }
    for (const key of sandboxCollapsedKeys) {
      if (!backendCollapsedKeys.has(key)) {
        collapsedKeysOnlyInSandbox.push(key);
      }
    }

    markerKeysOnlyInBackend.sort();
    markerKeysOnlyInSandbox.sort();
    collapsedKeysOnlyInBackend.sort();
    collapsedKeysOnlyInSandbox.sort();

    return {
      markerKeysOnlyInBackend,
      markerKeysOnlyInSandbox,
      collapsedKeysOnlyInBackend,
      collapsedKeysOnlyInSandbox,
    };
  }

  function createSandboxEngineFromInitial(initial: GameState): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType: initial.boardType,
      numPlayers: initial.players.length,
      playerKinds: initial.players
        .slice()
        .sort((a, b) => a.playerNumber - b.playerNumber)
        .map((p) => p.type as 'human' | 'ai'),
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

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler: handler,
      traceMode: true,
    });
    const engineAny: any = engine;
    engineAny.gameState = initial;
    return engine;
  }

  test('logs per-prefix backend vs sandbox state and first snapshot divergence for seed=5', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
    expect(trace.entries.length).toBeGreaterThan(0);

    const moves: Move[] = trace.entries.map((e) => e.action as Move);

    const backendEngine = createBackendEngineFromInitialState(trace.initialState);
    const sandboxEngine = createSandboxEngineFromInitial(trace.initialState);

    let firstMismatchIndex = -1;

    for (let i = 0; i < moves.length; i++) {
      const move = moves[i];

      // Snapshot + internal flags BEFORE applying move i
      const backendStateBefore = backendEngine.getGameState();
      const sandboxStateBefore = sandboxEngine.getGameState();
      const backendAny: any = backendEngine;
      const sandboxAny: any = sandboxEngine;

      // eslint-disable-next-line no-console
      console.log('[Seed5 PrefixDiagnostics] before step', {
        index: i,
        moveNumber: move.moveNumber,
        type: move.type,
        player: move.player,
        backend: {
          currentPlayer: backendStateBefore.currentPlayer,
          currentPhase: backendStateBefore.currentPhase,
          gameStatus: backendStateBefore.gameStatus,
          totalRingsEliminated: backendStateBefore.totalRingsEliminated,
          pendingTerritorySelfElimination:
            backendAny.pendingTerritorySelfElimination === true,
          pendingLineRewardElimination:
            backendAny.pendingLineRewardElimination === true,
          hasPlacedThisTurn: backendAny.hasPlacedThisTurn === true,
          mustMoveFromStackKey: backendAny.mustMoveFromStackKey,
        },
        sandbox: {
          currentPlayer: sandboxStateBefore.currentPlayer,
          currentPhase: sandboxStateBefore.currentPhase,
          gameStatus: sandboxStateBefore.gameStatus,
          totalRingsEliminated: sandboxStateBefore.totalRingsEliminated,
          pendingTerritorySelfElimination:
            sandboxAny._pendingTerritorySelfElimination === true,
          hasPlacedThisTurn: sandboxAny._hasPlacedThisTurn === true,
          mustMoveFromStackKey: sandboxAny._mustMoveFromStackKey,
        },
      });

      // Backend: map sandbox move to canonical backend move
      const backendValidMoves = backendEngine.getValidMoves(backendStateBefore.currentPlayer);
      const matching = findMatchingBackendMove(move, backendValidMoves);

      if (!matching) {
        // eslint-disable-next-line no-console
        console.error('[Seed5 PrefixDiagnostics] no matching backend move', {
          index: i,
          moveNumber: move.moveNumber,
          type: move.type,
          player: move.player,
          backendCurrentPlayer: backendStateBefore.currentPlayer,
          backendCurrentPhase: backendStateBefore.currentPhase,
          backendValidMovesCount: backendValidMoves.length,
        });
        firstMismatchIndex = i;
        break;
      }

      const { id, timestamp, moveNumber, ...payload } = matching as any;
      const backendResult = await backendEngine.makeMove(
        payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );
      if (!backendResult.success) {
        // eslint-disable-next-line no-console
        console.error('[Seed5 PrefixDiagnostics] backend makeMove failed', {
          index: i,
          moveNumber: move.moveNumber,
          type: move.type,
          player: move.player,
          backendMoveNumber: (matching as any).moveNumber,
          error: backendResult.error,
        });
        firstMismatchIndex = i;
        break;
      }

      // Sandbox: apply canonical move directly.
      await sandboxEngine.applyCanonicalMove(move);

      // AFTER applying move i, compare snapshots.
      const backendAfter = backendEngine.getGameState();
      const sandboxAfter = sandboxEngine.getGameState();

      const backendSnap = snapshotFromGameState(`backend-step-${i}`, backendAfter);
      const sandboxSnap = snapshotFromGameState(`sandbox-step-${i}`, sandboxAfter);

      if (!snapshotsEqual(backendSnap, sandboxSnap)) {
        firstMismatchIndex = i;
        const diff = diffSnapshots(backendSnap, sandboxSnap);
        const markerSummary = summariseMarkerAndCollapsedDiff(backendSnap, sandboxSnap);
        // eslint-disable-next-line no-console
        console.error('[Seed5 PrefixDiagnostics] snapshot mismatch after step', {
          index: i,
          moveNumber: move.moveNumber,
          type: move.type,
          player: move.player,
          diff,
          markerSummary,
        });
        break;
      }
    }

    // eslint-disable-next-line no-console
    console.log('[Seed5 PrefixDiagnostics] result', {
      seed,
      totalMoves: moves.length,
      firstMismatchIndex,
    });

    // Diagnostic harness: ensure we at least exercised the trace; do not
    // fail the test on snapshot mismatches here.
    expect(moves.length).toBeGreaterThan(0);
  });
});
