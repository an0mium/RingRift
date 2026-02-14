import { Request, Response, NextFunction } from 'express';
import { ZodSchema } from 'zod';

/**
 * Middleware factory that validates `req.body` against a Zod schema.
 *
 * On success the parsed (and potentially transformed) data replaces
 * `req.body`, giving downstream handlers a typed, validated payload.
 *
 * On failure the ZodError is forwarded to the centralized error handler
 * which already knows how to map it to a 400 response with field-level
 * details.
 *
 * @example
 * ```ts
 * router.post('/register', validateBody(RegisterSchema), asyncHandler(async (req, res) => {
 *   // req.body is now the parsed RegisterSchema output
 * }));
 * ```
 */
export function validateBody<T>(schema: ZodSchema<T>) {
  return (req: Request, _res: Response, next: NextFunction): void => {
    try {
      req.body = schema.parse(req.body);
      next();
    } catch (err) {
      next(err);
    }
  };
}

/**
 * Middleware factory that validates `req.query` against a Zod schema.
 *
 * Parsed output is stored on `req.query` (coerced values, defaults, etc.).
 *
 * @example
 * ```ts
 * router.get('/games', validateQuery(GameListingQuerySchema), asyncHandler(async (req, res) => {
 *   const { status, limit, offset } = req.query as GameListingQueryInput;
 * }));
 * ```
 */
export function validateQuery<T>(schema: ZodSchema<T>) {
  return (req: Request, _res: Response, next: NextFunction): void => {
    try {
      // Cast back so Express typing doesn't complain; handlers should
      // cast to the expected inferred type from the schema.
      req.query = schema.parse(req.query) as typeof req.query;
      next();
    } catch (err) {
      next(err);
    }
  };
}

/**
 * Middleware factory that validates `req.params` against a Zod schema.
 *
 * @example
 * ```ts
 * router.get('/:gameId', validateParams(GameIdParamSchema), asyncHandler(async (req, res) => {
 *   const { gameId } = req.params;
 * }));
 * ```
 */
export function validateParams<T>(schema: ZodSchema<T>) {
  return (req: Request, _res: Response, next: NextFunction): void => {
    try {
      req.params = schema.parse(req.params) as typeof req.params;
      next();
    } catch (err) {
      next(err);
    }
  };
}
