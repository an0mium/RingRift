/**
 * ReplayService - HTTP client for the GameReplayDB REST API.
 *
 * This service communicates with the Python AI service's replay endpoints
 * to browse, query, and replay games stored in the SQLite database.
 *
 * Usage:
 *   const replayService = new ReplayService('http://localhost:8001');
 *   const games = await replayService.listGames({ board_type: 'square8' });
 *   const state = await replayService.getStateAtMove(gameId, 10);
 *
 * See: docs/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md
 */

import { readEnv } from '../../shared/utils/envFlags';
import type {
  ReplayGameListResponse,
  ReplayGameMetadata,
  ReplayGameQueryParams,
  ReplayMovesResponse,
  ReplayStateResponse,
  ReplayChoicesResponse,
  ReplayStatsResponse,
  StoreGameRequest,
  StoreGameResponse,
} from '../types/replay';

/**
 * Get the AI service URL from environment or use default.
 *
 * Priority:
 * 1. RINGRIFT_AI_SERVICE_URL (runtime env)
 * 2. Default localhost:8001 for development
 */
function getAIServiceUrl(): string {
  const envUrl = readEnv('RINGRIFT_AI_SERVICE_URL');
  if (envUrl && typeof envUrl === 'string') {
    return envUrl.replace(/\/$/, '');
  }

  // Default for local development
  return 'http://localhost:8001';
}

/**
 * ReplayService provides access to the GameReplayDB REST API.
 */
export class ReplayService {
  private baseUrl: string;

  constructor(aiServiceUrl?: string) {
    this.baseUrl = `${aiServiceUrl ?? getAIServiceUrl()}/api/replay`;
  }

  /**
   * List games with optional filters.
   */
  async listGames(filters: ReplayGameQueryParams = {}): Promise<ReplayGameListResponse> {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        params.set(key, String(value));
      }
    });

    const url = `${this.baseUrl}/games${params.toString() ? `?${params.toString()}` : ''}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to list games: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get detailed metadata for a specific game including player info.
   */
  async getGame(gameId: string): Promise<ReplayGameMetadata> {
    const response = await fetch(`${this.baseUrl}/games/${encodeURIComponent(gameId)}`);

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Game ${gameId} not found`);
      }
      throw new Error(`Failed to get game: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get reconstructed game state at a specific move.
   *
   * @param gameId - Game identifier
   * @param moveNumber - Move number (0 = initial state, N = after move N)
   */
  async getStateAtMove(gameId: string, moveNumber: number): Promise<ReplayStateResponse> {
    const params = new URLSearchParams({ move_number: String(moveNumber) });
    const response = await fetch(
      `${this.baseUrl}/games/${encodeURIComponent(gameId)}/state?${params.toString()}`
    );

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Game ${gameId} not found`);
      }
      throw new Error(`Failed to get state: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get moves for a game in a range.
   *
   * @param gameId - Game identifier
   * @param start - Start move number (inclusive, default 0)
   * @param end - End move number (exclusive, default all)
   * @param limit - Max moves to return (default 100)
   */
  async getMoves(
    gameId: string,
    start = 0,
    end?: number,
    limit = 100
  ): Promise<ReplayMovesResponse> {
    const params = new URLSearchParams({
      start: String(start),
      limit: String(limit),
    });
    if (end !== undefined) {
      params.set('end', String(end));
    }

    const response = await fetch(
      `${this.baseUrl}/games/${encodeURIComponent(gameId)}/moves?${params.toString()}`
    );

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Game ${gameId} not found`);
      }
      throw new Error(`Failed to get moves: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get player choices made at a specific move.
   */
  async getChoices(gameId: string, moveNumber: number): Promise<ReplayChoicesResponse> {
    const params = new URLSearchParams({ move_number: String(moveNumber) });
    const response = await fetch(
      `${this.baseUrl}/games/${encodeURIComponent(gameId)}/choices?${params.toString()}`
    );

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Game ${gameId} not found`);
      }
      throw new Error(`Failed to get choices: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get database statistics.
   */
  async getStats(): Promise<ReplayStatsResponse> {
    const response = await fetch(`${this.baseUrl}/stats`);

    if (!response.ok) {
      throw new Error(`Failed to get stats: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Store a game from the sandbox.
   *
   * Used by the sandbox UI to persist AI vs AI games to the database.
   */
  async storeGame(request: StoreGameRequest): Promise<StoreGameResponse> {
    const response = await fetch(`${this.baseUrl}/games`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Failed to store game: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Check if the replay service is available.
   */
  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/stats`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

/**
 * Singleton instance for convenience.
 * Use this when you don't need to customize the AI service URL.
 */
let defaultInstance: ReplayService | null = null;

export function getReplayService(): ReplayService {
  if (!defaultInstance) {
    defaultInstance = new ReplayService();
  }
  return defaultInstance;
}

/**
 * Reset the singleton instance (for testing).
 */
export function resetReplayService(): void {
  defaultInstance = null;
}
