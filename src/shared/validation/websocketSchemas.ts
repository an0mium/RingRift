import { z } from 'zod';
import { MoveSchema } from './schemas';

/**
 * Zod schemas for incoming WebSocket payloads.
 *
 * These schemas mirror the current contracts used by WebSocketServer and
 * GameSession. They are intentionally narrower than some internal domain
 * types so we can evolve backend representations without breaking clients.
 */

// --- Core game room events ---

export const JoinGamePayloadSchema = z.object({
  gameId: z.string().min(1),
});

export type JoinGamePayload = z.infer<typeof JoinGamePayloadSchema>;

export const LeaveGamePayloadSchema = z.object({
  gameId: z.string().min(1),
});

export type LeaveGamePayload = z.infer<typeof LeaveGamePayloadSchema>;

// --- Move submission events ---

export const PlayerMovePayloadSchema = z.object({
  gameId: z.string().min(1),
  move: MoveSchema,
});

export type PlayerMovePayload = z.infer<typeof PlayerMovePayloadSchema>;

export const PlayerMoveByIdPayloadSchema = z.object({
  gameId: z.string().min(1),
  moveId: z.string().min(1),
});

export type PlayerMoveByIdPayload = z.infer<typeof PlayerMoveByIdPayloadSchema>;

// --- Chat events ---

export const ChatMessagePayloadSchema = z.object({
  gameId: z.string().min(1),
  // The existing chat handler expects a simple { gameId, text } payload and
  // broadcasts the same text field back out. We intentionally avoid
  // UUID-only or HTTP ChatMessageSchema constraints here so tests and
  // legacy clients that use synthetic ids continue to work.
  text: z
    .string()
    .min(1, 'Message cannot be empty')
    .max(500, 'Message must be at most 500 characters')
    .trim(),
});

export type ChatMessagePayload = z.infer<typeof ChatMessagePayloadSchema>;

// --- Player choice system events ---

/**
 * PlayerChoiceResponse is structurally validated enough to protect the
 * interaction handler, but we deliberately keep selectedOption as
 * z.unknown() so that per-choice-type option shapes remain the single
 * source of truth on the engine side. WebSocketInteractionHandler performs
 * the semantic check that selectedOption matches one of choice.options.
 */
export const PlayerChoiceResponsePayloadSchema = z.object({
  choiceId: z.string().min(1),
  playerNumber: z.number().int().min(1),
  choiceType: z
    .enum([
      'line_order',
      'line_reward_option',
      'ring_elimination',
      'region_order',
      'capture_direction',
    ])
    .optional(),
  selectedOption: z.unknown(),
});

export type PlayerChoiceResponsePayload = z.infer<
  typeof PlayerChoiceResponsePayloadSchema
>;

// --- Event name &#8594; schema mapping used by WebSocketServer ---

export const WebSocketPayloadSchemas = {
  join_game: JoinGamePayloadSchema,
  leave_game: LeaveGamePayloadSchema,
  player_move: PlayerMovePayloadSchema,
  player_move_by_id: PlayerMoveByIdPayloadSchema,
  chat_message: ChatMessagePayloadSchema,
  player_choice_response: PlayerChoiceResponsePayloadSchema,
} as const;

export type WebSocketEventName = keyof typeof WebSocketPayloadSchemas;