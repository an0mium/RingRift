import { readFileSync, readdirSync, existsSync } from 'fs';
import * as path from 'path';
import type { BoardType, GameState, Move, Position } from '../../src/shared/types/game';
import {
  ClientSandboxEngine,
  type SandboxConfig,
  type SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  getSelfPlayGameService,
  type SelfPlayGameDetail,
} from '../../src/server/services/SelfPlayGameService';

interface StateSummaryFixture {
  move_index: number;
  current_player: number;
  current_phase: string;
  game_status: string;
  state_hash: string;
}

interface ParityFixture {
  db_path: string;
  game_id: string;
  diverged_at: number | null;
  mismatch_kinds: string[];
  mismatch_context: string | null;
  total_moves_python: number;
  total_moves_ts: number;
  python_summary: StateSummaryFixture | null;
  ts_summary: StateSummaryFixture | null;
  canonical_move_index: number | null;
  canonical_move: Move | null;
}

function normalizeRecordedMove(rawMove: Move, fallbackMoveNumber: number): Move {
  const anyMove = rawMove as any;

  const type: Move['type'] =
    anyMove.type === 'forced_elimination' ? 'eliminate_rings_from_stack' : anyMove.type;

  const timestampRaw = anyMove.timestamp;
  const timestamp: Date =
    timestampRaw instanceof Date
      ? timestampRaw
      : typeof timestampRaw === 'string'
        ? new Date(timestampRaw)
        : new Date();

  const from: Position | undefined =
    anyMove.from && typeof anyMove.from === 'object' ? anyMove.from : undefined;

  const moveNumber =
    typeof anyMove.moveNumber === 'number' && Number.isFinite(anyMove.moveNumber)
      ? anyMove.moveNumber
      : fallbackMoveNumber;

  const thinkTime =
    typeof anyMove.thinkTime === 'number'
      ? anyMove.thinkTime
      : typeof anyMove.thinkTimeMs === 'number'
        ? anyMove.thinkTimeMs
        : 0;

  return {
    ...anyMove,
    type,
    from,
    timestamp,
    thinkTime,
    moveNumber,
  } as Move;
}

describe.skip('Python vs TS self-play replay parity (DB fixtures)', () => {
  const fixturesDir = path.join(__dirname, '../../ai-service/parity_fixtures');

  let fixtureFiles: string[] = [];
  try {
    fixtureFiles = readdirSync(fixturesDir).filter((f) => f.endsWith('.json'));
  } catch {
    // No fixtures directory – skip the suite.
  }

  if (fixtureFiles.length === 0) {
    test.skip('No parity fixtures found – run ai-service/scripts/check_ts_python_replay_parity.py with --emit-fixtures-dir first', () => {});
    return;
  }

  const service = getSelfPlayGameService();

  for (const fileName of fixtureFiles) {
    const fixturePath = path.join(fixturesDir, fileName);

    test(`replays canonical move for ${path.basename(fixturePath)} and compares TS vs Python summaries`, async () => {
      const raw = readFileSync(fixturePath, 'utf-8');
      const fixture: ParityFixture = JSON.parse(raw);

      const { db_path: dbPath, game_id: gameId } = fixture;

      if (!existsSync(dbPath)) {
        // Local DB no longer present; treat as a skipped diagnostic in practice.
        return;
      }

      const detail: SelfPlayGameDetail | null = service.getGame(dbPath, gameId);
      expect(detail).not.toBeNull();
      if (!detail) return;

      const rawState = detail.initialState as any;
      const sanitizedState = rawState && typeof rawState === 'object' ? { ...rawState } : rawState;
      if (sanitizedState && Array.isArray(sanitizedState.moveHistory)) {
        sanitizedState.moveHistory = [];
      }
      if (sanitizedState && Array.isArray(sanitizedState.history)) {
        sanitizedState.history = [];
      }

      const config: SandboxConfig = {
        boardType: detail.boardType as BoardType,
        numPlayers: detail.numPlayers,
        playerKinds: Array.from({ length: detail.numPlayers }, () => 'human'),
      };

      const interactionHandler: SandboxInteractionHandler = {
        async requestChoice(choice: any) {
          const options = (choice?.options as any[]) ?? [];
          const selectedOption = options.length > 0 ? options[0] : undefined;
          return {
            choiceId: choice.id,
            playerNumber: choice.playerNumber,
            choiceType: choice.type,
            selectedOption,
          } as any;
        },
      };

      const engine = new ClientSandboxEngine({
        config,
        interactionHandler,
        // Use traceMode so replay semantics (auto-resolution of decision
        // phases, line/territory handling, etc.) match the CLI
        // selfplay-db-ts-replay.ts harness used by the Python parity
        // checker.
        traceMode: true,
      });

      engine.initFromSerializedState(
        sanitizedState as GameState,
        config.playerKinds,
        interactionHandler
      );

      const recordedMoves: Move[] = detail.moves.map((m) =>
        normalizeRecordedMove(m.move as Move, m.moveNumber)
      );

      const targetIndex =
        typeof fixture.canonical_move_index === 'number' && fixture.canonical_move_index >= 0
          ? fixture.canonical_move_index
          : fixture.diverged_at !== null && fixture.diverged_at > 0
            ? Math.min(recordedMoves.length - 1, fixture.diverged_at - 1)
            : 0;

      for (let i = 0; i <= targetIndex && i < recordedMoves.length; i += 1) {
        const move = recordedMoves[i];
        const nextMove =
          i + 1 <= targetIndex && i + 1 < recordedMoves.length ? recordedMoves[i + 1] : null;
        await engine.applyCanonicalMoveForReplay(move, nextMove ?? undefined);
      }

      const tsState = engine.getGameState();
      const tsSummary = {
        current_player: tsState.currentPlayer,
        current_phase: tsState.currentPhase,
        game_status: tsState.gameStatus,
      };

      const pySummary = fixture.python_summary;
      expect(pySummary).not.toBeNull();
      if (!pySummary) {
        return;
      }

      expect(tsSummary.current_player).toBe(pySummary.current_player);
      expect(tsSummary.current_phase).toBe(pySummary.current_phase);
      expect(tsSummary.game_status).toBe(pySummary.game_status);
    });
  }
});
