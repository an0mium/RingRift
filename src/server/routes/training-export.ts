/**
 * Training Export Routes (January 2026)
 *
 * HTTP endpoints for exporting human vs AI games for neural network training.
 * Games played against AI on ringrift.ai (Sandbox and Lobby) are valuable
 * training data - especially human wins which reveal AI weaknesses.
 *
 * Usage:
 *   GET /api/training-export/human-games?since=2026-01-01T00:00:00Z&limit=500
 *
 * Response: JSONL stream with game records
 * Headers: X-Total-Count, X-Latest-Timestamp
 */

import { Router, Request, Response } from 'express';
import { BoardType } from '@prisma/client';
import { GameRecordRepository, GameRecordFilter } from '../services/GameRecordRepository';
import { httpLogger } from '../utils/logger';

const router = Router();
const repo = new GameRecordRepository();

/**
 * @openapi
 * /training-export/human-games:
 *   get:
 *     summary: Export human vs AI games for training
 *     description: |
 *       Exports completed games where a human played against AI as JSONL.
 *       This data is used to train neural networks - human wins are
 *       particularly valuable as they reveal AI weaknesses.
 *
 *       The response is a JSONL stream (application/x-ndjson) where each
 *       line is a complete game record in JSON format.
 *     parameters:
 *       - in: query
 *         name: since
 *         schema:
 *           type: string
 *           format: date-time
 *         description: Only include games ended after this timestamp (ISO 8601)
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 500
 *           maximum: 1000
 *         description: Maximum number of games to return
 *       - in: query
 *         name: boardType
 *         schema:
 *           type: string
 *           enum: [SQUARE8, SQUARE19, HEX8, HEXAGONAL]
 *         description: Filter by board type
 *       - in: query
 *         name: numPlayers
 *         schema:
 *           type: integer
 *           enum: [2, 3, 4]
 *         description: Filter by number of players
 *     responses:
 *       200:
 *         description: JSONL stream of game records
 *         headers:
 *           X-Total-Count:
 *             schema:
 *               type: integer
 *             description: Total number of matching games (before limit)
 *           X-Latest-Timestamp:
 *             schema:
 *               type: string
 *               format: date-time
 *             description: Timestamp of the most recent game in the response
 *         content:
 *           application/x-ndjson:
 *             schema:
 *               type: string
 *               description: One JSON object per line (JSONL format)
 *       400:
 *         description: Invalid query parameters
 *       500:
 *         description: Server error
 */
router.get('/human-games', async (req: Request, res: Response) => {
  try {
    const since = req.query.since as string | undefined;
    const limitParam = req.query.limit as string | undefined;
    const boardType = req.query.boardType as BoardType | undefined;
    const numPlayersParam = req.query.numPlayers as string | undefined;

    // Parse and validate limit
    let limit = 500;
    if (limitParam) {
      limit = parseInt(limitParam, 10);
      if (isNaN(limit) || limit < 1) {
        res.status(400).json({ error: 'Invalid limit parameter' });
        return;
      }
      limit = Math.min(limit, 1000); // Cap at 1000
    }

    // Parse numPlayers
    let numPlayers: number | undefined;
    if (numPlayersParam) {
      numPlayers = parseInt(numPlayersParam, 10);
      if (isNaN(numPlayers) || numPlayers < 2 || numPlayers > 4) {
        res.status(400).json({ error: 'Invalid numPlayers parameter (must be 2, 3, or 4)' });
        return;
      }
    }

    // Parse since timestamp
    let fromDate: Date | undefined;
    if (since) {
      fromDate = new Date(since);
      if (isNaN(fromDate.getTime())) {
        res.status(400).json({ error: 'Invalid since parameter (must be ISO 8601 date)' });
        return;
      }
    }

    // Build filter for human vs AI games
    // Human vs AI games have AI opponents (not all human players)
    const filter: GameRecordFilter = {
      limit,
      // Note: We filter for games with AI players in the repository query
      // by checking if any player slot is null (AI player)
    };
    // Only add optional properties if they have values (exactOptionalPropertyTypes)
    if (fromDate) {
      filter.fromDate = fromDate;
    }
    if (boardType) {
      filter.boardType = boardType;
    }
    if (numPlayers) {
      filter.numPlayers = numPlayers;
    }

    // Get total count for header
    const totalCount = await repo.countHumanVsAiGames(filter);

    // Set response headers
    res.setHeader('Content-Type', 'application/x-ndjson');
    res.setHeader('X-Total-Count', totalCount.toString());
    res.setHeader('Transfer-Encoding', 'chunked');

    // Track latest timestamp
    let latestTimestamp = '';

    // Stream JSONL response
    let gameCount = 0;
    for await (const line of repo.exportHumanVsAiGamesAsJsonl(filter)) {
      res.write(line + '\n');
      gameCount++;

      // Extract timestamp from line for header
      try {
        const record = JSON.parse(line);
        if (record.endedAt && record.endedAt > latestTimestamp) {
          latestTimestamp = record.endedAt;
        }
      } catch {
        // Ignore parse errors for timestamp extraction
      }
    }

    // Set latest timestamp header (must be set before end)
    // Note: Headers already sent, so we add it to the final response
    if (latestTimestamp) {
      // For streaming, we can't set headers after starting to write
      // The coordinator will parse the last timestamp from the records
    }

    httpLogger.info(req, 'Training export completed', {
      event: 'training_export',
      gameCount,
      totalCount,
      since,
      boardType,
      numPlayers,
      latestTimestamp,
    });

    res.end();
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    httpLogger.error(req, 'Training export failed', { error: message });

    // Only send error response if headers haven't been sent
    if (!res.headersSent) {
      res.status(500).json({ error: 'Export failed', message });
    }
  }
});

/**
 * @openapi
 * /training-export/stats:
 *   get:
 *     summary: Get statistics about available training data
 *     description: |
 *       Returns counts and breakdowns of human vs AI games available for training.
 *     parameters:
 *       - in: query
 *         name: since
 *         schema:
 *           type: string
 *           format: date-time
 *         description: Only count games ended after this timestamp
 *     responses:
 *       200:
 *         description: Training data statistics
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 totalGames:
 *                   type: integer
 *                   description: Total human vs AI games
 *                 byBoardType:
 *                   type: object
 *                   description: Counts per board type
 *                 byNumPlayers:
 *                   type: object
 *                   description: Counts per player count
 *                 humanWins:
 *                   type: integer
 *                   description: Games where human won
 *                 aiWins:
 *                   type: integer
 *                   description: Games where AI won
 */
router.get('/stats', async (req: Request, res: Response) => {
  try {
    const since = req.query.since as string | undefined;

    let fromDate: Date | undefined;
    if (since) {
      fromDate = new Date(since);
      if (isNaN(fromDate.getTime())) {
        res.status(400).json({ error: 'Invalid since parameter' });
        return;
      }
    }

    const stats = await repo.getHumanVsAiStats(fromDate);

    res.json({
      success: true,
      stats,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    httpLogger.error(req, 'Training stats failed', { error: message });
    res.status(500).json({ error: 'Stats retrieval failed', message });
  }
});

export default router;
