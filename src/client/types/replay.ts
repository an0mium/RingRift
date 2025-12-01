/**
 * Types for the GameReplayDB API.
 *
 * These types match the REST API responses from the Python AI service
 * replay endpoints at `/api/replay/*`.
 *
 * See: docs/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md
 * See: ai-service/app/routes/replay.py
 */

import type { BoardType, GameState } from '../../shared/types/game';

// =============================================================================
// Player Metadata
// =============================================================================

export interface ReplayPlayerMetadata {
  playerNumber: number;
  playerType: 'ai' | 'human';
  aiType?: string;
  aiDifficulty?: number;
  finalEliminatedRings?: number;
  finalTerritorySpaces?: number;
  finalRingsInHand?: number;
}

// =============================================================================
// Game Metadata
// =============================================================================

export interface ReplayGameMetadata {
  gameId: string;
  boardType: BoardType | string;
  numPlayers: number;
  winner: number | null;
  terminationReason: string | null;
  totalMoves: number;
  totalTurns: number;
  createdAt: string;
  completedAt: string | null;
  durationMs: number | null;
  source: string | null;
  /** v2 fields */
  timeControlType?: string;
  initialTimeMs?: number;
  timeIncrementMs?: number;
  /** Player details (included when fetching single game) */
  players?: ReplayPlayerMetadata[];
}

// =============================================================================
// Game List Response
// =============================================================================

export interface ReplayGameListResponse {
  games: ReplayGameMetadata[];
  total: number;
  hasMore: boolean;
}

// =============================================================================
// Move Record
// =============================================================================

export interface ReplayMoveRecord {
  moveNumber: number;
  turnNumber: number;
  player: number;
  phase: string;
  moveType: string;
  move: Record<string, unknown>;
  timestamp: string | null;
  thinkTimeMs: number | null;
  /** v2 fields */
  timeRemainingMs?: number;
  engineEval?: number;
  engineEvalType?: string;
  engineDepth?: number;
  engineNodes?: number;
  enginePV?: string[];
  engineTimeMs?: number;
}

// =============================================================================
// Moves Response
// =============================================================================

export interface ReplayMovesResponse {
  moves: ReplayMoveRecord[];
  hasMore: boolean;
}

// =============================================================================
// State Response
// =============================================================================

export interface ReplayStateResponse {
  gameState: GameState;
  moveNumber: number;
  totalMoves: number;
  engineEval?: number;
  enginePV?: string[];
}

// =============================================================================
// Choice Record
// =============================================================================

export interface ReplayChoiceRecord {
  choiceType: string;
  player: number;
  options: Record<string, unknown>[];
  selected: Record<string, unknown>;
  reasoning?: string;
}

export interface ReplayChoicesResponse {
  choices: ReplayChoiceRecord[];
}

// =============================================================================
// Stats Response
// =============================================================================

export interface ReplayStatsResponse {
  totalGames: number;
  gamesByBoardType: Record<string, number>;
  gamesByStatus: Record<string, number>;
  gamesByTermination: Record<string, number>;
  totalMoves: number;
  schemaVersion: number;
}

// =============================================================================
// Store Game Request/Response
// =============================================================================

export interface StoreGameRequest {
  gameId?: string;
  initialState: GameState;
  finalState: GameState;
  moves: Record<string, unknown>[];
  choices?: Record<string, unknown>[];
  metadata?: Record<string, unknown>;
}

export interface StoreGameResponse {
  gameId: string;
  totalMoves: number;
  success: boolean;
}

// =============================================================================
// Query Parameters
// =============================================================================

export interface ReplayGameQueryParams {
  board_type?: string;
  num_players?: number;
  winner?: number;
  termination_reason?: string;
  source?: string;
  min_moves?: number;
  max_moves?: number;
  limit?: number;
  offset?: number;
}

// =============================================================================
// Playback State (for UI)
// =============================================================================

export type PlaybackSpeed = 0.5 | 1 | 2 | 4;

export interface ReplayPlaybackState {
  /** Currently loaded game ID */
  gameId: string | null;
  /** Game metadata */
  metadata: ReplayGameMetadata | null;
  /** Current move number (0 = initial state) */
  currentMoveNumber: number;
  /** Total moves in the game */
  totalMoves: number;
  /** Current game state at currentMoveNumber */
  currentState: GameState | null;
  /** Is auto-playing */
  isPlaying: boolean;
  /** Playback speed multiplier */
  playbackSpeed: PlaybackSpeed;
  /** Loading state */
  isLoading: boolean;
  /** Error message if any */
  error: string | null;
  /** Cached move records for current game */
  moves: ReplayMoveRecord[];
}
