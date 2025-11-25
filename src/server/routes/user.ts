import { Router, Response } from 'express';
import { getDatabaseClient } from '../database/connection';
import { AuthenticatedRequest } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import { httpLogger } from '../utils/logger';
import {
  UpdateProfileSchema,
  GameListingQuerySchema,
  UserSearchQuerySchema,
  LeaderboardQuerySchema,
} from '../../shared/validation/schemas';

const router = Router();

/**
 * @openapi
 * /users/profile:
 *   get:
 *     summary: Get current user profile
 *     description: Returns the authenticated user's profile information including stats.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: User profile retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     user:
 *                       $ref: '#/components/schemas/User'
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/profile',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = req.user!.id;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        username: true,
        role: true,
        rating: true,
        gamesPlayed: true,
        gamesWon: true,
        createdAt: true,
        lastLoginAt: true,
        emailVerified: true,
        isActive: true,
      },
    });

    if (!user) {
      throw createError('User not found', 404, 'USER_NOT_FOUND');
    }

    res.json({
      success: true,
      data: { user },
    });
  })
);

/**
 * @openapi
 * /users/profile:
 *   put:
 *     summary: Update user profile
 *     description: |
 *       Updates the authenticated user's profile information.
 *       Only provided fields will be updated. All fields are optional.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/UpdateProfileRequest'
 *     responses:
 *       200:
 *         description: Profile updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     user:
 *                       $ref: '#/components/schemas/User'
 *                 message:
 *                   type: string
 *                   example: Profile updated successfully
 *       400:
 *         description: Invalid profile data
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: VALIDATION_INVALID_PROFILE_DATA
 *                 message: Invalid profile data
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       409:
 *         description: Username or email already taken
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               usernameExists:
 *                 summary: Username taken
 *                 value:
 *                   success: false
 *                   error:
 *                     code: RESOURCE_USERNAME_EXISTS
 *                     message: Username already taken
 *               emailExists:
 *                 summary: Email taken
 *                 value:
 *                   success: false
 *                   error:
 *                     code: RESOURCE_EMAIL_EXISTS
 *                     message: Email already registered
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.put(
  '/profile',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = req.user!.id;

    // Validate request body with Zod schema
    const parseResult = UpdateProfileSchema.safeParse(req.body);
    if (!parseResult.success) {
      const firstError = parseResult.error.issues[0];
      throw createError(firstError?.message || 'Invalid profile data', 400, 'INVALID_PROFILE_DATA');
    }
    const { username, email, preferences } = parseResult.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Check if username is already taken (if provided)
    if (username) {
      const existingUser = await prisma.user.findFirst({
        where: {
          username,
          NOT: { id: userId },
        },
      });

      if (existingUser) {
        throw createError('Username already taken', 409, 'USERNAME_EXISTS');
      }
    }

    // Check if email is already taken (if provided)
    if (email) {
      const existingUser = await prisma.user.findFirst({
        where: {
          email,
          NOT: { id: userId },
        },
      });

      if (existingUser) {
        throw createError('Email already registered', 409, 'EMAIL_EXISTS');
      }
    }

    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: {
        ...(username && { username }),
        ...(email && { email }),
        ...(preferences && { preferences: preferences as any }),
        updatedAt: new Date(),
      },
      select: {
        id: true,
        email: true,
        username: true,
        role: true,
        rating: true,
        gamesPlayed: true,
        gamesWon: true,
        createdAt: true,
        lastLoginAt: true,
        emailVerified: true,
        isActive: true,
      },
    });

    httpLogger.info(req, 'User profile updated', { userId });

    res.json({
      success: true,
      data: { user: updatedUser },
      message: 'Profile updated successfully',
    });
  })
);

/**
 * @openapi
 * /users/stats:
 *   get:
 *     summary: Get user statistics
 *     description: |
 *       Returns detailed statistics for the authenticated user including
 *       rating, win/loss record, recent games, and rating history.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Statistics retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     stats:
 *                       $ref: '#/components/schemas/UserStats'
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/stats',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = req.user!.id;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        rating: true,
        gamesPlayed: true,
        gamesWon: true,
      },
    });

    if (!user) {
      throw createError('User not found', 404, 'USER_NOT_FOUND');
    }

    // Get recent games
    const recentGames = await prisma.game.findMany({
      where: {
        OR: [
          { player1Id: userId },
          { player2Id: userId },
          { player3Id: userId },
          { player4Id: userId },
        ],
        // Cast to any so we don't depend on the exact Prisma GameStatus TS enum
        status: 'completed' as any,
      },
      orderBy: { endedAt: 'desc' },
      take: 10,
      select: {
        id: true,
        boardType: true,
        status: true,
        winnerId: true,
        endedAt: true,
        player1Id: true,
        player2Id: true,
        player3Id: true,
        player4Id: true,
      },
    });

    // Calculate win rate
    const winRate = user.gamesPlayed > 0 ? (user.gamesWon / user.gamesPlayed) * 100 : 0;

    // Get rating history (placeholder - would need a separate table in production)
    const ratingHistory = [{ date: new Date(), rating: user.rating }];

    const stats = {
      rating: user.rating,
      gamesPlayed: user.gamesPlayed,
      gamesWon: user.gamesWon,
      gamesLost: user.gamesPlayed - user.gamesWon,
      winRate: Math.round(winRate * 100) / 100,
      recentGames,
      ratingHistory,
    };

    res.json({
      success: true,
      data: { stats },
    });
  })
);

/**
 * @openapi
 * /users/games:
 *   get:
 *     summary: Get user's game history
 *     description: |
 *       Returns a paginated list of games the authenticated user has participated in.
 *       Can be filtered by game status.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: status
 *         schema:
 *           type: string
 *           enum: [waiting, active, completed, abandoned, paused]
 *         description: Filter by game status
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 20
 *         description: Number of results per page
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           minimum: 0
 *           default: 0
 *         description: Offset for pagination
 *     responses:
 *       200:
 *         description: Games retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     games:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/Game'
 *                     pagination:
 *                       $ref: '#/components/schemas/Pagination'
 *       400:
 *         description: Invalid query parameters
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: VALIDATION_INVALID_QUERY_PARAMS
 *                 message: Invalid query parameters
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/games',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = req.user!.id;

    // Validate query parameters with schema
    const queryResult = GameListingQuerySchema.safeParse(req.query);
    if (!queryResult.success) {
      throw createError('Invalid query parameters', 400, 'INVALID_QUERY_PARAMS');
    }
    const { status, limit, offset } = queryResult.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const whereClause: any = {
      OR: [
        { player1Id: userId },
        { player2Id: userId },
        { player3Id: userId },
        { player4Id: userId },
      ],
    };

    if (status) {
      whereClause.status = status;
    }

    const games = await prisma.game.findMany({
      where: whereClause,
      include: {
        player1: { select: { id: true, username: true, rating: true } },
        player2: { select: { id: true, username: true, rating: true } },
        player3: { select: { id: true, username: true, rating: true } },
        player4: { select: { id: true, username: true, rating: true } },
      },
      orderBy: { createdAt: 'desc' },
      take: limit,
      skip: offset,
    });

    const total = await prisma.game.count({ where: whereClause });

    res.json({
      success: true,
      data: {
        games,
        pagination: {
          total,
          limit,
          offset,
          hasMore: offset + limit < total,
        },
      },
    });
  })
);

/**
 * @openapi
 * /users/search:
 *   get:
 *     summary: Search users
 *     description: |
 *       Searches for users by username. Only returns active users.
 *       Results are sorted by rating (descending).
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: q
 *         required: true
 *         schema:
 *           type: string
 *           minLength: 1
 *           maxLength: 100
 *         description: Search query (matches username)
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 50
 *           default: 10
 *         description: Maximum results to return
 *     responses:
 *       200:
 *         description: Search results
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     users:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/UserPublic'
 *       400:
 *         description: Search query required or invalid
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               queryRequired:
 *                 summary: Query required
 *                 value:
 *                   success: false
 *                   error:
 *                     code: VALIDATION_SEARCH_QUERY_REQUIRED
 *                     message: Search query required
 *               invalidParams:
 *                 summary: Invalid parameters
 *                 value:
 *                   success: false
 *                   error:
 *                     code: VALIDATION_INVALID_QUERY_PARAMS
 *                     message: Invalid query parameters
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/search',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Validate query parameters with schema
    const queryResult = UserSearchQuerySchema.safeParse(req.query);
    if (!queryResult.success) {
      const firstError = queryResult.error.issues[0];
      // Preserve the specific error code for missing query
      if (firstError?.path[0] === 'q') {
        throw createError('Search query required', 400, 'SEARCH_QUERY_REQUIRED');
      }
      throw createError('Invalid query parameters', 400, 'INVALID_QUERY_PARAMS');
    }
    const { q, limit } = queryResult.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const users = await prisma.user.findMany({
      where: {
        username: {
          contains: q,
          mode: 'insensitive',
        },
        isActive: true,
      },
      select: {
        id: true,
        username: true,
        rating: true,
        gamesPlayed: true,
        gamesWon: true,
      },
      take: limit,
      orderBy: { rating: 'desc' },
    });

    res.json({
      success: true,
      data: { users },
    });
  })
);

function anonymizedEmail(user: { id: string; email: string }): string {
  return `deleted+${user.id}@example.invalid`;
}

function anonymizedUsername(user: { id: string; username: string }): string {
  return `DeletedPlayer_${user.id.slice(0, 8)}`;
}

/**
 * @openapi
 * /users/leaderboard:
 *   get:
 *     summary: Get leaderboard
 *     description: |
 *       Returns a paginated leaderboard of active users sorted by rating.
 *       Only includes users who have played at least one game.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 100
 *           default: 50
 *         description: Number of results per page
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           minimum: 0
 *           default: 0
 *         description: Offset for pagination
 *     responses:
 *       200:
 *         description: Leaderboard retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: object
 *                   properties:
 *                     users:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/LeaderboardEntry'
 *                     pagination:
 *                       $ref: '#/components/schemas/Pagination'
 *       400:
 *         description: Invalid query parameters
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: VALIDATION_INVALID_QUERY_PARAMS
 *                 message: Invalid query parameters
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.get(
  '/leaderboard',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Validate query parameters with schema
    const queryResult = LeaderboardQuerySchema.safeParse(req.query);
    if (!queryResult.success) {
      throw createError('Invalid query parameters', 400, 'INVALID_QUERY_PARAMS');
    }
    const { limit, offset } = queryResult.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const users = await prisma.user.findMany({
      where: {
        isActive: true,
        gamesPlayed: { gt: 0 },
      },
      select: {
        id: true,
        username: true,
        rating: true,
        gamesPlayed: true,
        gamesWon: true,
      },
      orderBy: { rating: 'desc' },
      take: limit,
      skip: offset,
    });

    const total = await prisma.user.count({
      where: {
        isActive: true,
        gamesPlayed: { gt: 0 },
      },
    });

    // Add rank to each user
    const usersWithRank = users.map((user: any, index: number) => ({
      ...user,
      rank: offset + index + 1,
      winRate:
        user.gamesPlayed > 0 ? Math.round((user.gamesWon / user.gamesPlayed) * 10000) / 100 : 0,
    }));

    res.json({
      success: true,
      data: {
        users: usersWithRank,
        pagination: {
          total,
          limit,
          offset,
          hasMore: offset + limit < total,
        },
      },
    });
  })
);

/**
 * @openapi
 * /users/me:
 *   delete:
 *     summary: Delete current user account
 *     description: |
 *       Soft-deletes the authenticated user's account. This action:
 *       - Deactivates the account (isActive = false)
 *       - Sets deletedAt timestamp
 *       - Invalidates all tokens (tokenVersion incremented)
 *       - Clears sensitive tokens (verification, reset)
 *       - Anonymizes email and username
 *       - Revokes all refresh tokens
 *
 *       The account cannot be recovered after deletion.
 *       Game history is preserved for other players.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Account deleted successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: Account deleted successfully
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       404:
 *         $ref: '#/components/responses/NotFound'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.delete(
  '/me',
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = req.user!.id;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    await prisma.$transaction(async (tx: any) => {
      const user = await tx.user.findUnique({
        where: { id: userId },
      });

      if (!user) {
        throw createError('User not found', 404, 'USER_NOT_FOUND');
      }

      if (user.deletedAt) {
        // Already soft-deleted; idempotent behaviour
        return;
      }

      await tx.user.update({
        where: { id: userId },
        data: {
          isActive: false,
          deletedAt: new Date(),
          tokenVersion: {
            increment: 1,
          },
          verificationToken: null,
          verificationTokenExpires: null,
          passwordResetToken: null,
          passwordResetExpires: null,
          email: anonymizedEmail(user),
          username: anonymizedUsername(user),
        },
      });

      // Best-effort cleanup of any persisted refresh tokens for this user.
      const refreshTokenModel = (tx as any).refreshToken;
      if (refreshTokenModel && typeof refreshTokenModel.deleteMany === 'function') {
        await refreshTokenModel.deleteMany({ where: { userId } });
      }
    });

    httpLogger.info(req, 'User account deleted (soft-delete)', {
      event: 'user_delete',
      userId,
    });

    res.status(200).json({
      success: true,
      message: 'Account deleted successfully',
    });
  })
);

export default router;
