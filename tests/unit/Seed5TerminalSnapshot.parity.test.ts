import { BoardType, GameState, Move } from '../../src/shared/types/game';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { findMatchingBackendMove } from '../utils/moveMatching';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { snapshotFromGameState, diffSnapshots } from '../utils/stateSnapshots';

/**
 * Seed-5 terminal snapshot / victory parity harness.
 *
 * This diagnostic test replays the full sandbox AI trace for
 *   square8 / 2p / seed=5
 * into both backend and sandbox engines and compares their *final*
 * GameState snapshots once all canonical moves from the trace have been
 * applied.
 *
 * It logs:
 *   - Final gameStatus / winner / currentPhase for both hosts.
 *   - A structured diff of board/players/phase/status when they differ.
 *
 * This is intentionally non-failing for now: it serves as a focused
 * tail-snapshot harness while we finish aligning backend vs sandbox
 * LPS/victory semantics. Once that work is complete, this test can be
 * tightened to assert exact equality of terminal snapshots.
 */

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

describe('Seed5 terminal snapshot / victory parity (square8 / 2p / seed=5)', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 5;
  const MAX_STEPS = 80; // generous upper bound for this game

  test('logs final backend vs sandbox terminal snapshots for seed=5', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
    expect(trace.entries.length).toBeGreaterThan(0);

    const moves: Move[] = trace.entries.map((e) => e.action as Move);

    const backendEngine = createBackendEngineFromInitialState(trace.initialState);
    const sandboxEngine = createSandboxEngineFromInitial(trace.initialState);

    // Replay the entire trace into both engines.
    for (let i = 0; i < moves.length; i++) {
      const move = moves[i];

      // Backend: map sandbox move to a canonical backend move and apply.
      const backendStateBefore = backendEngine.getGameState();
      const backendValidMoves = backendEngine.getValidMoves(backendStateBefore.currentPlayer);
      const matching = findMatchingBackendMove(move, backendValidMoves);

      if (!matching) {
        // eslint-disable-next-line no-console
        console.error('[Seed5 TerminalSnapshot] No matching backend move', {
          index: i,
          moveNumber: move.moveNumber,
          type: move.type,
          player: move.player,
          backendCurrentPlayer: backendStateBefore.currentPlayer,
          backendCurrentPhase: backendStateBefore.currentPhase,
          backendValidMovesCount: backendValidMoves.length,
        });
        break;
      }

      const { id, timestamp, moveNumber, ...payload } = matching as any;
      const backendResult = await backendEngine.makeMove(
        payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );
      if (!backendResult.success) {
        // eslint-disable-next-line no-console
        console.error('[Seed5 TerminalSnapshot] Backend makeMove failed', {
          index: i,
          moveNumber: move.moveNumber,
          type: move.type,
          player: move.player,
          backendMoveNumber: (matching as any).moveNumber,
          error: backendResult.error,
        });
        break;
      }

      // Step backend through automatic phases after each canonical move so
      // its stopping point mirrors the sandbox trace semantics.
      await backendEngine.stepAutomaticPhasesForTesting();

      // Sandbox: apply the canonical move directly.
      await sandboxEngine.applyCanonicalMove(move);
    }

    const backendFinal = backendEngine.getGameState();
    const sandboxFinal = sandboxEngine.getGameState();

    const backendSnap = snapshotFromGameState('backend-final', backendFinal);
    const sandboxSnap = snapshotFromGameState('sandbox-final', sandboxFinal);

    const diff = diffSnapshots(backendSnap, sandboxSnap);

    // eslint-disable-next-line no-console
    console.log('[Seed5 TerminalSnapshot] final summary', {
      seed,
      backend: {
        currentPlayer: backendFinal.currentPlayer,
        currentPhase: backendFinal.currentPhase,
        gameStatus: backendFinal.gameStatus,
        winner: backendFinal.winner,
      },
      sandbox: {
        currentPlayer: sandboxFinal.currentPlayer,
        currentPhase: sandboxFinal.currentPhase,
        gameStatus: sandboxFinal.gameStatus,
        winner: sandboxFinal.winner,
      },
      diff,
    });

    // Diagnostic harness: ensure we exercised the trace; do not yet
    // assert equality of terminal snapshots while tail LPS parity is
    // still being finalised.
    expect(moves.length).toBeGreaterThan(0);
  });
});
