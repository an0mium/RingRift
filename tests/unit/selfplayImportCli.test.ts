/**
 * Unit tests for the self-play SQLite admin script:
 *   scripts/selfplay-db-ts-replay.ts
 *
 * These tests validate:
 * - CLI argument parsing for the new import mode.
 * - Wiring from import mode into SelfPlayGameService + importSelfPlayGameAsGameRecord.
 * - Correct handling of --limit, --tag, and --dry-run flags.
 */

import type { SelfPlayGameSummary } from '../../src/server/services/SelfPlayGameService';
import {
  main as selfPlayCliMain,
  parseArgs,
  type CliArgs,
} from '../../scripts/selfplay-db-ts-replay';

// Jest hoists jest.mock calls, so factories cannot safely reference
// top-level const/let variables (they would be in the TDZ). Instead we
// define the mocks inside the factory and then access them via the
// imported bindings.

jest.mock('../../src/server/services/SelfPlayGameService', () => ({
  getSelfPlayGameService: jest.fn(),
  importSelfPlayGameAsGameRecord: jest.fn(),
}));

jest.mock('../../src/server/database/connection', () => ({
  connectDatabase: jest.fn(),
  disconnectDatabase: jest.fn(),
}));

import {
  getSelfPlayGameService,
  importSelfPlayGameAsGameRecord,
} from '../../src/server/services/SelfPlayGameService';
import { connectDatabase, disconnectDatabase } from '../../src/server/database/connection';

// Cast imported mocks to jest.Mock for easier usage in tests.
const mockGetSelfPlayGameService = getSelfPlayGameService as unknown as jest.Mock;

const mockImportSelfPlayGameAsGameRecord = importSelfPlayGameAsGameRecord as unknown as jest.Mock;

const mockConnectDatabase = connectDatabase as unknown as jest.Mock;

const mockDisconnectDatabase = disconnectDatabase as unknown as jest.Mock;

describe('selfplay-db-ts-replay CLI - import mode argument parsing', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('parses import mode arguments with limit, tags, and dry-run', () => {
    const argv = [
      '--mode',
      'import',
      '--db',
      '/tmp/selfplay.db',
      '--limit',
      '2',
      '--tag',
      'exp:foo',
      '--tag',
      'exp:bar',
      '--dry-run',
    ];

    const args = parseArgs(argv);
    expect(args).not.toBeNull();

    const typed = args as CliArgs;
    expect(typed.mode).toBe('import');
    expect(typed.dbPath).toBe('/tmp/selfplay.db');
    expect((typed as any).limit).toBe(2);
    expect((typed as any).dryRun).toBe(true);
    expect((typed as any).tags).toEqual(['exp:foo', 'exp:bar']);
  });

  it('defaults to replay mode and requires --game when --mode is omitted', () => {
    const argvMissingGame = ['--db', '/tmp/selfplay.db'];
    const bad = parseArgs(argvMissingGame);
    expect(bad).toBeNull();

    const argvReplay = ['--db', '/tmp/selfplay.db', '--game', 'sp-1'];
    const ok = parseArgs(argvReplay);
    expect(ok).not.toBeNull();

    const typed = ok as CliArgs;
    expect(typed.mode).toBe('replay');
    expect(typed.dbPath).toBe('/tmp/selfplay.db');
    expect((typed as any).gameId).toBe('sp-1');
  });
});

describe('selfplay-db-ts-replay CLI - import mode wiring', () => {
  const originalLog = console.log;
  const originalError = console.error;

  beforeEach(() => {
    jest.clearAllMocks();
    console.log = jest.fn();
    console.error = jest.fn();

    // Ensure disconnectDatabase behaves like an async function returning a
    // Promise so that runImportMode's `await disconnectDatabase().catch(...)`
    // does not throw when using mocked implementations.
    mockDisconnectDatabase.mockResolvedValue(undefined);
  });

  afterEach(() => {
    console.log = originalLog;
    console.error = originalError;
  });

  it('lists completed games and calls importSelfPlayGameAsGameRecord for limited candidates', async () => {
    const createdAt1 = '2024-01-02T00:00:00Z';
    const createdAt2 = '2024-01-01T00:00:00Z';
    const createdAt3 = '2024-01-03T00:00:00Z';

    const summaries: SelfPlayGameSummary[] = [
      {
        gameId: 'g1',
        boardType: 'square8',
        numPlayers: 2,
        winner: 1,
        totalMoves: 10,
        totalTurns: 10,
        createdAt: createdAt1,
        completedAt: '2024-01-02T00:10:00Z',
        source: 'self_play_pipeline',
        terminationReason: 'ring_elimination',
        durationMs: 600000,
      },
      {
        gameId: 'g2',
        boardType: 'square8',
        numPlayers: 2,
        winner: 2,
        totalMoves: 8,
        totalTurns: 8,
        createdAt: createdAt2, // earliest
        completedAt: '2024-01-01T00:08:00Z',
        source: 'self_play_pipeline',
        terminationReason: 'ring_elimination',
        durationMs: 480000,
      },
      {
        gameId: 'g3',
        boardType: 'square8',
        numPlayers: 2,
        winner: null,
        totalMoves: 12,
        totalTurns: 12,
        createdAt: createdAt3,
        completedAt: '2024-01-03T00:12:00Z',
        source: 'self_play_pipeline',
        terminationReason: 'draw',
        durationMs: 720000,
      },
    ];

    const stubService = {
      listGames: jest.fn().mockReturnValue(summaries),
    };

    mockGetSelfPlayGameService.mockReturnValue(stubService);

    // hasExistingSelfPlayImport() uses prisma.game.findFirst; for these tests
    // we always report "not imported yet" so every candidate is attempted.
    const mockFindFirst = jest.fn().mockResolvedValue(null);
    mockConnectDatabase.mockResolvedValue({
      game: {
        findFirst: mockFindFirst,
      },
    });

    mockImportSelfPlayGameAsGameRecord.mockResolvedValue('imported-game-id');

    await selfPlayCliMain([
      '--mode',
      'import',
      '--db',
      '/tmp/selfplay.db',
      '--limit',
      '2',
      '--tag',
      'exp:foo',
    ]);

    // Service should be instantiated and queried for games from this DB.
    expect(mockGetSelfPlayGameService).toHaveBeenCalledTimes(1);
    expect(stubService.listGames).toHaveBeenCalledWith('/tmp/selfplay.db');

    // connectDatabase should be called once for the import session.
    expect(mockConnectDatabase).toHaveBeenCalledTimes(1);

    // Because createdAt2 < createdAt1 < createdAt3 and limit=2,
    // the importer should process games [g2, g1] in that order.
    expect(mockImportSelfPlayGameAsGameRecord).toHaveBeenCalledTimes(2);

    const callOptions = mockImportSelfPlayGameAsGameRecord.mock.calls.map(
      (call) => call[0]
    ) as Array<{
      dbPath: string;
      gameId: string;
      source: string;
      tags: string[];
    }>;

    expect(callOptions[0]).toMatchObject({
      dbPath: '/tmp/selfplay.db',
      gameId: 'g2',
      source: 'self_play',
      tags: ['exp:foo'],
    });

    expect(callOptions[1]).toMatchObject({
      dbPath: '/tmp/selfplay.db',
      gameId: 'g1',
      source: 'self_play',
      tags: ['exp:foo'],
    });
  });

  it('performs a dry run without calling importSelfPlayGameAsGameRecord', async () => {
    const summaries: SelfPlayGameSummary[] = [
      {
        gameId: 'g1',
        boardType: 'square8',
        numPlayers: 2,
        winner: 1,
        totalMoves: 10,
        totalTurns: 10,
        createdAt: '2024-01-02T00:00:00Z',
        completedAt: '2024-01-02T00:10:00Z',
        source: 'self_play_pipeline',
        terminationReason: 'ring_elimination',
        durationMs: 600000,
      },
      {
        gameId: 'g2',
        boardType: 'square8',
        numPlayers: 2,
        winner: 2,
        totalMoves: 8,
        totalTurns: 8,
        createdAt: '2024-01-01T00:00:00Z',
        completedAt: '2024-01-01T00:08:00Z',
        source: 'self_play_pipeline',
        terminationReason: 'ring_elimination',
        durationMs: 480000,
      },
    ];

    const stubService = {
      listGames: jest.fn().mockReturnValue(summaries),
    };

    mockGetSelfPlayGameService.mockReturnValue(stubService);
    mockConnectDatabase.mockResolvedValue({
      game: {
        findFirst: jest.fn().mockResolvedValue(null),
      },
    });

    await selfPlayCliMain([
      '--mode',
      'import',
      '--db',
      '/tmp/selfplay.db',
      '--limit',
      '10',
      '--dry-run',
      '--tag',
      'exp:foo',
    ]);

    // In dry-run mode, we should not touch the database or call the importer.
    expect(mockConnectDatabase).not.toHaveBeenCalled();
    expect(mockImportSelfPlayGameAsGameRecord).not.toHaveBeenCalled();

    // But we should still have listed games from the requested DB.
    expect(mockGetSelfPlayGameService).toHaveBeenCalledTimes(1);
    expect(stubService.listGames).toHaveBeenCalledWith('/tmp/selfplay.db');
  });
});
