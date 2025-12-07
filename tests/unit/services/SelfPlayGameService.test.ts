import * as path from 'path';
import * as fs from 'fs';
import Database from 'better-sqlite3';
import { tmpdir } from 'os';

// Import types and helpers from the service module
// Note: Some helpers are not exported, so we test the public API
import {
  SelfPlayGameService,
  getSelfPlayGameService,
  SelfPlayGameSummary,
  SelfPlayGameDetail,
  SelfPlayPlayer,
  GameListOptions,
} from '../../../src/server/services/SelfPlayGameService';

describe('SelfPlayGameService', () => {
  let service: SelfPlayGameService;
  let testDbPath: string;

  beforeAll(() => {
    // Create a temporary test database
    testDbPath = path.join(tmpdir(), `ringrift-test-${Date.now()}.db`);
    const db = new Database(testDbPath);

    // Create the required tables matching the Python GameReplayDB schema
    db.exec(`
      CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        board_type TEXT NOT NULL,
        num_players INTEGER NOT NULL,
        winner INTEGER,
        total_moves INTEGER DEFAULT 0,
        total_turns INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        source TEXT,
        termination_reason TEXT,
        duration_ms INTEGER
      );

      CREATE TABLE IF NOT EXISTS game_initial_state (
        game_id TEXT PRIMARY KEY,
        initial_state_json TEXT NOT NULL,
        compressed INTEGER DEFAULT 0,
        FOREIGN KEY (game_id) REFERENCES games(game_id)
      );

      CREATE TABLE IF NOT EXISTS game_moves (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT NOT NULL,
        move_number INTEGER NOT NULL,
        turn_number INTEGER NOT NULL,
        player INTEGER NOT NULL,
        phase TEXT NOT NULL,
        move_type TEXT NOT NULL,
        move_json TEXT NOT NULL,
        think_time_ms INTEGER,
        engine_eval REAL,
        FOREIGN KEY (game_id) REFERENCES games(game_id)
      );

      CREATE TABLE IF NOT EXISTS game_players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT NOT NULL,
        player_number INTEGER NOT NULL,
        player_type TEXT NOT NULL,
        ai_type TEXT,
        ai_difficulty INTEGER,
        ai_profile_id TEXT,
        final_eliminated_rings INTEGER,
        final_territory_spaces INTEGER,
        FOREIGN KEY (game_id) REFERENCES games(game_id)
      );

      CREATE TABLE IF NOT EXISTS game_state_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT NOT NULL,
        move_number INTEGER NOT NULL,
        state_json TEXT NOT NULL,
        compressed INTEGER DEFAULT 0,
        FOREIGN KEY (game_id) REFERENCES games(game_id)
      );
    `);

    // Insert test data
    const now = new Date().toISOString();
    const completedAt = new Date(Date.now() + 60000).toISOString();

    // Game 1: Completed with winner
    db.prepare(`
      INSERT INTO games (game_id, board_type, num_players, winner, total_moves, total_turns, created_at, completed_at, source, termination_reason, duration_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run('game-001', 'square8', 2, 1, 50, 25, now, completedAt, 'selfplay', 'ring_elimination', 60000);

    // Game 2: Completed draw
    db.prepare(`
      INSERT INTO games (game_id, board_type, num_players, winner, total_moves, total_turns, created_at, completed_at, source, termination_reason, duration_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run('game-002', 'hexagonal', 3, null, 100, 50, now, completedAt, 'cmaes', 'timeout', 120000);

    // Game 3: In progress (no completed_at)
    db.prepare(`
      INSERT INTO games (game_id, board_type, num_players, winner, total_moves, total_turns, created_at, completed_at, source, termination_reason, duration_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run('game-003', 'square8', 2, null, 10, 5, now, null, 'selfplay', null, null);

    // Insert initial state for game-001
    db.prepare(`
      INSERT INTO game_initial_state (game_id, initial_state_json, compressed)
      VALUES (?, ?, ?)
    `).run('game-001', JSON.stringify({ boardType: 'square8', players: [{ playerNumber: 1 }, { playerNumber: 2 }] }), 0);

    // Insert moves for game-001
    db.prepare(`
      INSERT INTO game_moves (game_id, move_number, turn_number, player, phase, move_type, move_json, think_time_ms, engine_eval)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run('game-001', 1, 1, 1, 'placement', 'PLACE_RING', JSON.stringify({ type: 'PLACE_RING', position: 'a1' }), 100, 0.5);

    db.prepare(`
      INSERT INTO game_moves (game_id, move_number, turn_number, player, phase, move_type, move_json, think_time_ms, engine_eval)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run('game-001', 2, 1, 2, 'placement', 'PLACE_RING', JSON.stringify({ type: 'PLACE_RING', position: 'b2' }), 150, -0.2);

    // Insert players for game-001
    db.prepare(`
      INSERT INTO game_players (game_id, player_number, player_type, ai_type, ai_difficulty, ai_profile_id, final_eliminated_rings, final_territory_spaces)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run('game-001', 1, 'ai', 'mcts', 5, null, 3, 10);

    db.prepare(`
      INSERT INTO game_players (game_id, player_number, player_type, ai_type, ai_difficulty, ai_profile_id, final_eliminated_rings, final_territory_spaces)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run('game-001', 2, 'ai', 'mcts', 5, null, 5, 8);

    // Insert a snapshot for game-001
    db.prepare(`
      INSERT INTO game_state_snapshots (game_id, move_number, state_json, compressed)
      VALUES (?, ?, ?, ?)
    `).run('game-001', 25, JSON.stringify({ moveNumber: 25, phase: 'movement' }), 0);

    db.close();
  });

  afterAll(() => {
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  beforeEach(() => {
    service = new SelfPlayGameService();
  });

  afterEach(() => {
    service.closeAll();
  });

  describe('listGames', () => {
    it('lists all games from the database', () => {
      const games = service.listGames(testDbPath);

      expect(games).toHaveLength(3);
      expect(games.map((g) => g.gameId).sort()).toEqual(['game-001', 'game-002', 'game-003']);
    });

    it('filters by board type', () => {
      const games = service.listGames(testDbPath, { boardType: 'square8' });

      expect(games).toHaveLength(2);
      games.forEach((g) => expect(g.boardType).toBe('square8'));
    });

    it('filters by numPlayers', () => {
      const games = service.listGames(testDbPath, { numPlayers: 3 });

      expect(games).toHaveLength(1);
      expect(games[0].numPlayers).toBe(3);
    });

    it('filters by source', () => {
      const games = service.listGames(testDbPath, { source: 'cmaes' });

      expect(games).toHaveLength(1);
      expect(games[0].source).toBe('cmaes');
    });

    it('filters by hasWinner = true', () => {
      const games = service.listGames(testDbPath, { hasWinner: true });

      expect(games).toHaveLength(1);
      expect(games[0].winner).toBe(1);
    });

    it('filters by hasWinner = false', () => {
      const games = service.listGames(testDbPath, { hasWinner: false });

      expect(games).toHaveLength(2);
      games.forEach((g) => expect(g.winner).toBeNull());
    });

    it('respects limit option', () => {
      const games = service.listGames(testDbPath, { limit: 2 });

      expect(games).toHaveLength(2);
    });

    it('respects offset option', () => {
      const allGames = service.listGames(testDbPath);
      const offsetGames = service.listGames(testDbPath, { offset: 1, limit: 10 });

      expect(offsetGames).toHaveLength(allGames.length - 1);
    });

    it('combines multiple filters', () => {
      const games = service.listGames(testDbPath, {
        boardType: 'square8',
        hasWinner: true,
      });

      expect(games).toHaveLength(1);
      expect(games[0].gameId).toBe('game-001');
    });
  });

  describe('getGame', () => {
    it('returns full game details', () => {
      const game = service.getGame(testDbPath, 'game-001');

      expect(game).not.toBeNull();
      expect(game!.gameId).toBe('game-001');
      expect(game!.boardType).toBe('square8');
      expect(game!.numPlayers).toBe(2);
      expect(game!.winner).toBe(1);
      expect(game!.totalMoves).toBe(50);
      expect(game!.totalTurns).toBe(25);
      expect(game!.terminationReason).toBe('ring_elimination');
    });

    it('includes initial state', () => {
      const game = service.getGame(testDbPath, 'game-001');

      expect(game!.initialState).toBeDefined();
      expect((game!.initialState as any).boardType).toBe('square8');
    });

    it('includes parsed moves', () => {
      const game = service.getGame(testDbPath, 'game-001');

      expect(game!.moves).toHaveLength(2);
      expect(game!.moves[0].moveNumber).toBe(1);
      expect(game!.moves[0].player).toBe(1);
      expect(game!.moves[0].phase).toBe('placement');
      expect(game!.moves[0].moveType).toBe('PLACE_RING');
      expect((game!.moves[0].move as any).position).toBe('a1');
    });

    it('includes player data', () => {
      const game = service.getGame(testDbPath, 'game-001');

      expect(game!.players).toHaveLength(2);
      expect(game!.players[0].playerNumber).toBe(1);
      expect(game!.players[0].playerType).toBe('ai');
      expect(game!.players[0].aiType).toBe('mcts');
      expect(game!.players[0].aiDifficulty).toBe(5);
      expect(game!.players[0].finalEliminatedRings).toBe(3);
      expect(game!.players[0].finalTerritorySpaces).toBe(10);
    });

    it('returns null for non-existent game', () => {
      const game = service.getGame(testDbPath, 'non-existent');

      expect(game).toBeNull();
    });
  });

  describe('getStateAtMove', () => {
    it('returns snapshot when available', () => {
      const state = service.getStateAtMove(testDbPath, 'game-001', 25);

      expect(state).not.toBeNull();
      expect((state as any).moveNumber).toBe(25);
    });

    it('returns closest earlier snapshot', () => {
      // Request move 30, but only snapshot at 25 exists
      const state = service.getStateAtMove(testDbPath, 'game-001', 30);

      expect(state).not.toBeNull();
      expect((state as any).moveNumber).toBe(25);
    });

    it('returns initial state for move 0', () => {
      const state = service.getStateAtMove(testDbPath, 'game-001', 0);

      expect(state).not.toBeNull();
      expect((state as any).boardType).toBe('square8');
    });

    it('returns null for move without snapshot or initial state', () => {
      // game-002 has no initial state or snapshots
      const state = service.getStateAtMove(testDbPath, 'game-002', 50);

      expect(state).toBeNull();
    });
  });

  describe('getStats', () => {
    it('returns aggregate statistics', () => {
      const stats = service.getStats(testDbPath);

      expect(stats.totalGames).toBe(3);
      expect(stats.byBoardType['square8']).toBe(2);
      expect(stats.byBoardType['hexagonal']).toBe(1);
      expect(stats.byNumPlayers[2]).toBe(2);
      expect(stats.byNumPlayers[3]).toBe(1);
      expect(stats.byWinner['p1']).toBe(1);
      expect(stats.byWinner['draw']).toBe(2);
      expect(typeof stats.avgMoves).toBe('number');
    });
  });

  describe('closeAll', () => {
    it('closes all cached database connections', () => {
      // Access the database to cache a connection
      service.listGames(testDbPath);

      // Close all connections
      service.closeAll();

      // Accessing again should work (re-opens connection)
      const games = service.listGames(testDbPath);
      expect(games).toHaveLength(3);
    });
  });

  describe('error handling', () => {
    it('throws when database does not exist', () => {
      expect(() => {
        service.listGames('/non/existent/path.db');
      }).toThrow('Database not found');
    });
  });

  describe('getSelfPlayGameService singleton', () => {
    it('returns a singleton instance', () => {
      const instance1 = getSelfPlayGameService();
      const instance2 = getSelfPlayGameService();

      expect(instance1).toBe(instance2);
    });
  });
});

describe('SelfPlayGameService helper functions', () => {
  // Since mapSelfPlayTerminationToOutcome and buildFinalScoreFromSelfPlayPlayers
  // are not exported, we test their behavior indirectly through integration tests
  // or document expected behavior for future extraction

  describe('termination reason to outcome mapping (via behavior)', () => {
    // These document the expected mapping behavior
    // ring_elimination -> ring_elimination
    // territory/territory_control -> territory_control
    // last_player_standing -> last_player_standing
    // timeout/max_turns_reached/max_moves_reached -> timeout
    // resignation -> resignation
    // abandonment -> abandonment
    // draw/stalemate -> draw
    // unknown with winner -> last_player_standing
    // unknown without winner -> draw

    it('documents expected termination reason mappings', () => {
      const mappings = [
        { reason: 'ring_elimination', outcome: 'ring_elimination' },
        { reason: 'territory', outcome: 'territory_control' },
        { reason: 'territory_control', outcome: 'territory_control' },
        { reason: 'last_player_standing', outcome: 'last_player_standing' },
        { reason: 'timeout', outcome: 'timeout' },
        { reason: 'max_turns_reached', outcome: 'timeout' },
        { reason: 'max_moves_reached', outcome: 'timeout' },
        { reason: 'resignation', outcome: 'resignation' },
        { reason: 'abandonment', outcome: 'abandonment' },
        { reason: 'draw', outcome: 'draw' },
        { reason: 'stalemate', outcome: 'draw' },
      ];

      // This test documents expected behavior
      expect(mappings.length).toBeGreaterThan(0);
    });
  });

  describe('final score building (via behavior)', () => {
    it('documents expected FinalScore structure', () => {
      const players: SelfPlayPlayer[] = [
        {
          playerNumber: 1,
          playerType: 'ai',
          aiType: 'mcts',
          aiDifficulty: 5,
          aiProfileId: null,
          finalEliminatedRings: 3,
          finalTerritorySpaces: 10,
        },
        {
          playerNumber: 2,
          playerType: 'ai',
          aiType: 'mcts',
          aiDifficulty: 5,
          aiProfileId: null,
          finalEliminatedRings: 5,
          finalTerritorySpaces: 8,
        },
      ];

      // Document expected FinalScore structure
      // { ringsEliminated: { 1: 3, 2: 5 }, territorySpaces: { 1: 10, 2: 8 }, ringsRemaining: { 1: 0, 2: 0 } }
      expect(players[0].finalEliminatedRings).toBe(3);
      expect(players[1].finalEliminatedRings).toBe(5);
    });
  });
});
