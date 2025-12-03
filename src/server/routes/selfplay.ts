/**
 * API routes for browsing and replaying recorded self-play games.
 *
 * These endpoints provide read-only access to games recorded during
 * CMA-ES training runs, self-play soaks, and other AI training activities.
 */

import { Router, Request, Response } from 'express';
import { getSelfPlayGameService } from '../services/SelfPlayGameService';
import { httpLogger } from '../utils/logger';

const router = Router();

/**
 * @openapi
 * /selfplay/databases:
 *   get:
 *     summary: List available game databases
 *     description: |
 *       Scans the project for SQLite databases containing recorded self-play games.
 *       Returns metadata about each database including game count.
 *     tags:
 *       - Self-Play
 *     responses:
 *       200:
 *         description: List of available databases
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 databases:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       path:
 *                         type: string
 *                       name:
 *                         type: string
 *                       gameCount:
 *                         type: integer
 *                       createdAt:
 *                         type: string
 *                         nullable: true
 */
router.get('/databases', (req: Request, res: Response) => {
  try {
    const service = getSelfPlayGameService();
    const databases = service.listDatabases(process.cwd());

    res.json({
      success: true,
      databases,
    });
  } catch (error) {
    httpLogger.error(req, 'Failed to list databases', { error });
    res.status(500).json({
      success: false,
      error: 'Failed to list databases',
    });
  }
});

/**
 * @openapi
 * /selfplay/games:
 *   get:
 *     summary: List games from a database
 *     description: |
 *       Returns a paginated list of games from the specified database.
 *       Supports filtering by board type, player count, and source.
 *     tags:
 *       - Self-Play
 *     parameters:
 *       - name: db
 *         in: query
 *         required: true
 *         description: Path to the database file
 *         schema:
 *           type: string
 *       - name: boardType
 *         in: query
 *         description: Filter by board type (square8, square19, hex)
 *         schema:
 *           type: string
 *       - name: numPlayers
 *         in: query
 *         description: Filter by number of players
 *         schema:
 *           type: integer
 *       - name: source
 *         in: query
 *         description: Filter by source (cmaes, selfplay, etc.)
 *         schema:
 *           type: string
 *       - name: hasWinner
 *         in: query
 *         description: Filter to only games with/without a winner
 *         schema:
 *           type: boolean
 *       - name: limit
 *         in: query
 *         description: Maximum number of games to return
 *         schema:
 *           type: integer
 *           default: 50
 *       - name: offset
 *         in: query
 *         description: Number of games to skip
 *         schema:
 *           type: integer
 *           default: 0
 *     responses:
 *       200:
 *         description: List of games
 *       400:
 *         description: Missing or invalid database path
 */
router.get('/games', (req: Request, res: Response) => {
  const dbPath = req.query.db as string | undefined;

  if (!dbPath) {
    res.status(400).json({
      success: false,
      error: 'Missing required parameter: db',
    });
    return;
  }

  try {
    const service = getSelfPlayGameService();

    const options = {
      boardType: req.query.boardType as string | undefined,
      numPlayers: req.query.numPlayers ? parseInt(req.query.numPlayers as string, 10) : undefined,
      source: req.query.source as string | undefined,
      hasWinner: req.query.hasWinner !== undefined ? req.query.hasWinner === 'true' : undefined,
      limit: req.query.limit ? parseInt(req.query.limit as string, 10) : 50,
      offset: req.query.offset ? parseInt(req.query.offset as string, 10) : 0,
    };

    const games = service.listGames(dbPath, options);

    res.json({
      success: true,
      games,
      pagination: {
        limit: options.limit,
        offset: options.offset,
        returned: games.length,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    httpLogger.error(req, 'Failed to list games', { error, dbPath });
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

/**
 * @openapi
 * /selfplay/games/{gameId}:
 *   get:
 *     summary: Get a single game with full details
 *     description: |
 *       Returns complete game data including initial state, all moves,
 *       and player metadata. Suitable for full replay.
 *     tags:
 *       - Self-Play
 *     parameters:
 *       - name: gameId
 *         in: path
 *         required: true
 *         description: The game ID
 *         schema:
 *           type: string
 *       - name: db
 *         in: query
 *         required: true
 *         description: Path to the database file
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Full game data
 *       404:
 *         description: Game not found
 */
router.get('/games/:gameId', (req: Request, res: Response) => {
  const { gameId } = req.params;
  const dbPath = req.query.db as string | undefined;

  if (!dbPath) {
    res.status(400).json({
      success: false,
      error: 'Missing required parameter: db',
    });
    return;
  }

  try {
    const service = getSelfPlayGameService();
    const game = service.getGame(dbPath, gameId);

    if (!game) {
      res.status(404).json({
        success: false,
        error: 'Game not found',
      });
      return;
    }

    res.json({
      success: true,
      game,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    httpLogger.error(req, 'Failed to get game', { error, gameId, dbPath });
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

/**
 * @openapi
 * /selfplay/games/{gameId}/state:
 *   get:
 *     summary: Get game state at a specific move
 *     description: |
 *       Returns the game state at a specific move number. Uses snapshots
 *       when available for faster retrieval.
 *     tags:
 *       - Self-Play
 *     parameters:
 *       - name: gameId
 *         in: path
 *         required: true
 *         description: The game ID
 *         schema:
 *           type: string
 *       - name: db
 *         in: query
 *         required: true
 *         description: Path to the database file
 *         schema:
 *           type: string
 *       - name: move
 *         in: query
 *         required: true
 *         description: Move number (0 for initial state)
 *         schema:
 *           type: integer
 *     responses:
 *       200:
 *         description: Game state at the specified move
 *       404:
 *         description: Game or state not found
 */
router.get('/games/:gameId/state', (req: Request, res: Response) => {
  const { gameId } = req.params;
  const dbPath = req.query.db as string | undefined;
  const moveStr = req.query.move as string | undefined;

  if (!dbPath) {
    res.status(400).json({
      success: false,
      error: 'Missing required parameter: db',
    });
    return;
  }

  if (moveStr === undefined) {
    res.status(400).json({
      success: false,
      error: 'Missing required parameter: move',
    });
    return;
  }

  const moveNumber = parseInt(moveStr, 10);
  if (isNaN(moveNumber) || moveNumber < 0) {
    res.status(400).json({
      success: false,
      error: 'Invalid move number',
    });
    return;
  }

  try {
    const service = getSelfPlayGameService();
    const state = service.getStateAtMove(dbPath, gameId, moveNumber);

    if (state === null) {
      res.status(404).json({
        success: false,
        error: 'State not found',
      });
      return;
    }

    res.json({
      success: true,
      moveNumber,
      state,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    httpLogger.error(req, 'Failed to get state', { error, gameId, dbPath, moveNumber });
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

/**
 * @openapi
 * /selfplay/stats:
 *   get:
 *     summary: Get aggregate statistics for a database
 *     description: |
 *       Returns statistics about the games in a database, including
 *       counts by board type, player count, and winner.
 *     tags:
 *       - Self-Play
 *     parameters:
 *       - name: db
 *         in: query
 *         required: true
 *         description: Path to the database file
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Database statistics
 */
router.get('/stats', (req: Request, res: Response) => {
  const dbPath = req.query.db as string | undefined;

  if (!dbPath) {
    res.status(400).json({
      success: false,
      error: 'Missing required parameter: db',
    });
    return;
  }

  try {
    const service = getSelfPlayGameService();
    const stats = service.getStats(dbPath);

    res.json({
      success: true,
      stats,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    httpLogger.error(req, 'Failed to get stats', { error, dbPath });
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

export default router;
