/**
 * GameRecordRepository - CRUD operations for game records
 *
 * Provides database access for storing and retrieving completed game records,
 * supporting:
 * - Online game completion storage
 * - Self-play game recording (CMA-ES, soak tests)
 * - Training data export
 * - Replay system data access
 */

import { PrismaClient, Game, Move, BoardType } from '@prisma/client';
import { getDatabaseClient } from '../database/connection';
import {
  GameRecord,
  MoveRecord,
  GameRecordMetadata,
  FinalScore,
  GameOutcome,
  RecordSource,
  PlayerRecordInfo,
  gameRecordToJsonlLine,
} from '@shared/types/gameRecord';
import { GameState, MoveType } from '@shared/types/game';
import { logger } from '../utils/logger';

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface GameRecordFilter {
  boardType?: BoardType;
  numPlayers?: number;
  outcome?: GameOutcome;
  source?: RecordSource;
  isRated?: boolean;
  tags?: string[];
  fromDate?: Date;
  toDate?: Date;
  playerId?: string;
  limit?: number;
  offset?: number;
}

export interface GameRecordSummary {
  id: string;
  boardType: BoardType;
  numPlayers: number;
  winner: number | null;
  outcome: GameOutcome;
  totalMoves: number;
  totalDurationMs: number;
  startedAt: Date;
  endedAt: Date;
  source: RecordSource;
}

// Extended Game type for queries that access the new record fields
// (recordMetadata, finalScore, outcome) which may not be in the
// generated Prisma types until the client is fully regenerated.
type GameWithRecordFields = {
  recordMetadata?: unknown;
  finalScore?: unknown;
  outcome?: string | null;
};

// Type for Prisma query result with includes
interface GameWithRelations {
  id: string;
  boardType: BoardType;
  maxPlayers: number;
  isRated: boolean;
  rngSeed: number | null;
  status: string;
  createdAt: Date;
  startedAt: Date | null;
  endedAt: Date | null;
  finalState: unknown;
  finalScore: unknown;
  outcome: string | null;
  recordMetadata: unknown;
  moves: Array<Move & { player: { username: string } }>;
  player1: { username: string; rating: number } | null;
  player2: { username: string; rating: number } | null;
  player3: { username: string; rating: number } | null;
  player4: { username: string; rating: number } | null;
  winner: { username: string } | null;
}

// ────────────────────────────────────────────────────────────────────────────
// Repository
// ────────────────────────────────────────────────────────────────────────────

export class GameRecordRepository {
  private getDb(): PrismaClient {
    const db = getDatabaseClient();
    if (!db) {
      throw new Error('Database not connected');
    }
    return db;
  }

  /**
   * Save a completed game as a GameRecord.
   *
   * This updates the Game row with:
   * - finalState: Complete GameState snapshot
   * - finalScore: Per-player score breakdown
   * - outcome: How the game ended
   * - recordMetadata: Training pipeline metadata
   */
  async saveGameRecord(
    gameId: string,
    finalState: GameState,
    outcome: GameOutcome,
    finalScore: FinalScore,
    metadata: Partial<GameRecordMetadata> = {}
  ): Promise<void> {
    const db = this.getDb();

    const recordMetadata: GameRecordMetadata = {
      recordVersion: '1.0.0',
      createdAt: new Date(),
      source: metadata.source ?? 'online_game',
      tags: metadata.tags ?? [],
      ...(metadata.sourceId !== undefined && { sourceId: metadata.sourceId }),
      ...(metadata.generation !== undefined && { generation: metadata.generation }),
      ...(metadata.candidateId !== undefined && { candidateId: metadata.candidateId }),
    };

    // Use type assertion to work around Prisma client not yet having
    // the new record fields in its generated types. The actual database
    // schema has these fields after migration.
    await db.game.update({
      where: { id: gameId },
      data: {
        finalState: JSON.parse(JSON.stringify(finalState)),
        finalScore: JSON.parse(JSON.stringify(finalScore)),
        outcome,
        recordMetadata: JSON.parse(JSON.stringify(recordMetadata)),
        endedAt: new Date(),
      } as Parameters<typeof db.game.update>[0]['data'],
    });

    logger.info('Game record saved', { gameId, outcome, source: recordMetadata.source });
  }

  /**
   * Load a complete GameRecord by ID.
   *
   * Returns null if the game doesn't exist or hasn't been completed.
   */
  async getGameRecord(gameId: string): Promise<GameRecord | null> {
    const db = this.getDb();

    const game = await db.game.findUnique({
      where: { id: gameId },
      include: {
        moves: {
          orderBy: { moveNumber: 'asc' },
          include: { player: { select: { username: true } } },
        },
        player1: { select: { username: true, rating: true } },
        player2: { select: { username: true, rating: true } },
        player3: { select: { username: true, rating: true } },
        player4: { select: { username: true, rating: true } },
        winner: { select: { username: true } },
      },
    });

    // Cast to access outcome field (Prisma types may not be regenerated yet)
    const gameWithFields = game as unknown as GameWithRelations;
    if (!game || !game.finalState || !gameWithFields.outcome) {
      return null;
    }

    return this.gameToGameRecord(game as unknown as GameWithRelations);
  }

  /**
   * List game records with optional filtering.
   */
  async listGameRecords(filter: GameRecordFilter = {}): Promise<GameRecordSummary[]> {
    const db = this.getDb();

    // Build where clause dynamically
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const where: any = {
      finalState: { not: undefined },
      outcome: { not: null },
    };

    if (filter.boardType) where.boardType = filter.boardType;
    if (filter.numPlayers) where.maxPlayers = filter.numPlayers;
    if (filter.outcome) where.outcome = filter.outcome;
    if (filter.isRated !== undefined) where.isRated = filter.isRated;

    if (filter.fromDate || filter.toDate) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const dateFilter: any = {};
      if (filter.fromDate) dateFilter.gte = filter.fromDate;
      if (filter.toDate) dateFilter.lte = filter.toDate;
      where.endedAt = dateFilter;
    }

    if (filter.playerId) {
      where.OR = [
        { player1Id: filter.playerId },
        { player2Id: filter.playerId },
        { player3Id: filter.playerId },
        { player4Id: filter.playerId },
      ];
    }

    const games = await db.game.findMany({
      where,
      include: {
        _count: { select: { moves: true } },
      },
      orderBy: { endedAt: 'desc' },
      take: filter.limit ?? 50,
      skip: filter.offset ?? 0,
    });

    return games.map((game) => this.gameToSummary(game));
  }

  /**
   * Export game records as JSONL for training pipelines.
   *
   * Returns an async generator that yields one JSONL line per game.
   */
  async *exportAsJsonl(filter: GameRecordFilter = {}): AsyncGenerator<string> {
    const db = this.getDb();
    const batchSize = 100;
    let offset = 0;

    while (true) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const where: any = {
        finalState: { not: undefined },
        outcome: { not: null },
      };

      if (filter.boardType) where.boardType = filter.boardType;

      const games = await db.game.findMany({
        where,
        include: {
          moves: {
            orderBy: { moveNumber: 'asc' },
            include: { player: { select: { username: true } } },
          },
          player1: { select: { username: true, rating: true } },
          player2: { select: { username: true, rating: true } },
          player3: { select: { username: true, rating: true } },
          player4: { select: { username: true, rating: true } },
          winner: { select: { username: true } },
        },
        orderBy: { endedAt: 'desc' },
        take: batchSize,
        skip: offset,
      });

      if (games.length === 0) break;

      for (const game of games) {
        // Cast to access outcome field (Prisma types may not be regenerated yet)
        const gameWithFields = game as unknown as GameWithRecordFields;
        if (game.finalState && gameWithFields.outcome) {
          const record = this.gameToGameRecord(game as unknown as GameWithRelations);
          yield gameRecordToJsonlLine(record);
        }
      }

      offset += batchSize;
    }
  }

  /**
   * Count total game records matching filter.
   */
  async countGameRecords(filter: GameRecordFilter = {}): Promise<number> {
    const db = this.getDb();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const where: any = {
      finalState: { not: undefined },
      outcome: { not: null },
    };

    if (filter.boardType) where.boardType = filter.boardType;

    return db.game.count({ where });
  }

  /**
   * Delete old game records for data retention.
   */
  async deleteOldRecords(beforeDate: Date, source?: RecordSource): Promise<number> {
    const db = this.getDb();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const where: any = {
      endedAt: { lt: beforeDate },
      finalState: { not: undefined },
    };

    // Note: Filtering by source in JSON requires raw query or jsonPath
    // For now, delete all records before the date

    const result = await db.game.deleteMany({ where });
    logger.info('Deleted old game records', {
      deletedCount: result.count,
      beforeDate: beforeDate.toISOString(),
      source,
    });
    return result.count;
  }

  // ────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ────────────────────────────────────────────────────────────────────────

  private gameToGameRecord(game: GameWithRelations): GameRecord {
    const metadata = (game.recordMetadata ?? {
      recordVersion: '1.0.0',
      createdAt: game.endedAt ?? new Date(),
      source: 'online_game' as RecordSource,
      tags: [],
    }) as GameRecordMetadata;

    const players: PlayerRecordInfo[] = [];
    const playerRefs = [game.player1, game.player2, game.player3, game.player4];

    for (let i = 0; i < game.maxPlayers; i++) {
      const user = playerRefs[i];
      players.push({
        playerNumber: i + 1,
        username: user?.username ?? `AI Player ${i + 1}`,
        playerType: user ? 'human' : 'ai',
        ...(user && { ratingBefore: user.rating }),
      });
    }

    const moves: MoveRecord[] = game.moves.map((move) => ({
      moveNumber: move.moveNumber,
      player: players.findIndex((p) => p.username === move.player.username) + 1 || 1,
      type: move.moveType as MoveType,
      thinkTimeMs: 0, // Not tracked in current schema
      ...((move.moveData as Record<string, unknown>) ?? {}),
    }));

    const winnerIndex = game.winner
      ? players.findIndex((p) => p.username === game.winner?.username) + 1
      : undefined;

    const startedAt = game.startedAt ?? game.createdAt;
    const endedAt = game.endedAt ?? new Date();
    const durationMs = endedAt.getTime() - startedAt.getTime();

    const record: GameRecord = {
      id: game.id,
      boardType: game.boardType,
      numPlayers: game.maxPlayers,
      isRated: game.isRated,
      players,
      outcome: game.outcome as GameOutcome,
      finalScore: (game.finalScore ?? {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: {},
      }) as FinalScore,
      startedAt,
      endedAt,
      totalMoves: moves.length,
      totalDurationMs: durationMs,
      moves,
      metadata,
    };

    // Only add optional fields if they have values
    if (game.rngSeed !== null) {
      record.rngSeed = game.rngSeed;
    }
    if (winnerIndex !== undefined) {
      record.winner = winnerIndex;
    }

    return record;
  }

  private gameToSummary(game: Game & { _count: { moves: number } }): GameRecordSummary {
    const metadata = (game.recordMetadata ?? {}) as Partial<GameRecordMetadata>;
    const startedAt = game.startedAt ?? game.createdAt;
    const endedAt = game.endedAt ?? new Date();

    return {
      id: game.id,
      boardType: game.boardType,
      numPlayers: game.maxPlayers,
      winner: null, // Would need to resolve from winnerId
      outcome: (game.outcome ?? 'abandonment') as GameOutcome,
      totalMoves: game._count.moves,
      totalDurationMs: endedAt.getTime() - startedAt.getTime(),
      startedAt,
      endedAt,
      source: metadata.source ?? 'online_game',
    };
  }
}

// Singleton export
export const gameRecordRepository = new GameRecordRepository();
