/**
 * Client for communicating with the Python AI Service
 * Makes HTTP requests to the FastAPI microservice for AI move and choice selection.
 * Includes circuit breaker pattern for resilience.
 */

import axios, { AxiosInstance } from 'axios';
import { config } from '../config';
import {
  GameState,
  Move,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
} from '../../shared/types/game';
import { logger } from '../utils/logger';
import { aiMoveLatencyHistogram } from '../utils/rulesParityMetrics';

/**
 * Circuit breaker to prevent hammering a failing AI service
 */
class CircuitBreaker {
  private failureCount = 0;
  private lastFailureTime = 0;
  private isOpen = false;
  private readonly threshold = 5; // failures before opening
  private readonly timeout = 60000; // 1 minute cooldown

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.isOpen) {
      const now = Date.now();
      if (now - this.lastFailureTime > this.timeout) {
        this.reset(); // Try again after timeout
        logger.info('Circuit breaker transitioning from open to half-open');
      } else {
        throw new Error('Circuit breaker is open - AI service temporarily unavailable');
      }
    }

    try {
      const result = await fn();
      this.reset();
      return result;
    } catch (error) {
      this.recordFailure();
      throw error;
    }
  }

  private recordFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    if (this.failureCount >= this.threshold) {
      this.isOpen = true;
      logger.warn('Circuit breaker opened after repeated failures', {
        failureCount: this.failureCount,
        threshold: this.threshold,
      });
    }
  }

  private reset(): void {
    if (this.failureCount > 0 || this.isOpen) {
      logger.info('Circuit breaker reset', {
        previousFailures: this.failureCount,
        wasOpen: this.isOpen,
      });
    }
    this.failureCount = 0;
    this.isOpen = false;
  }

  getStatus(): { isOpen: boolean; failureCount: number } {
    return {
      isOpen: this.isOpen,
      failureCount: this.failureCount,
    };
  }
}

export interface AIConfig {
  difficulty: number;
  thinkTime?: number;
  randomness?: number;
}

export enum AIType {
  RANDOM = 'random',
  HEURISTIC = 'heuristic',
  MINIMAX = 'minimax',
  MCTS = 'mcts',
  DESCENT = 'descent',
}

export interface MoveRequest {
  game_state: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  seed?: number;
}

export interface MoveResponse {
  move: Move | null;
  evaluation: number;
  thinking_time_ms: number;
  ai_type: string;
  difficulty: number;
}

export interface EvaluationRequest {
  game_state: GameState;
  player_number: number;
}

export interface EvaluationResponse {
  score: number;
  breakdown: Record<string, number>;
}

export interface LineRewardChoiceRequestPayload {
  game_state?: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  options: LineRewardChoice['options'];
}

export interface LineRewardChoiceResponsePayload {
  selectedOption: LineRewardChoice['options'][number];
  aiType: string;
  difficulty: number;
}

export interface RingEliminationChoiceRequestPayload {
  // Optional for now so callers can omit GameState while we
  // progressively adopt full-game-state-aware heuristics on the
  // Python side.
  game_state?: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  options: RingEliminationChoice['options'];
}

export interface RingEliminationChoiceResponsePayload {
  selectedOption: RingEliminationChoice['options'][number];
  aiType: string;
  difficulty: number;
}

export interface RegionOrderChoiceRequestPayload {
  // Optional for now so callers can omit GameState while we
  // progressively adopt full-game-state-aware heuristics on the
  // Python side.
  game_state?: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
  options: RegionOrderChoice['options'];
}

export interface RegionOrderChoiceResponsePayload {
  selectedOption: RegionOrderChoice['options'][number];
  aiType: string;
  difficulty: number;
}

/**
 * Client for interacting with the Python AI microservice.
 * Includes circuit breaker for resilience and timeout handling.
 */
export class AIServiceClient {
  private client: AxiosInstance;
  private baseURL: string;
  private circuitBreaker: CircuitBreaker;

  constructor(baseURL?: string) {
    this.baseURL = baseURL || config.aiService.url;
    this.circuitBreaker = new CircuitBreaker();

    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000, // 30 second timeout for AI computation
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        // Enhanced error logging with categorization
        const errorType = this.categorizeError(error);
        // Attach categorized error type so downstream callers (e.g. AIEngine)
        // can emit structured fallback metrics without depending on axios internals.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (error as any).aiErrorType = errorType;

        logger.error('AI Service error:', {
          type: errorType,
          message: error.message,
          response: error.response?.data,
          status: error.response?.status,
          code: error.code,
        });
        throw error;
      }
    );
  }

  /**
   * Categorize error type for better diagnostics
   */
  private categorizeError(error: any): string {
    if (error.code === 'ECONNREFUSED') return 'connection_refused';
    if (error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') return 'timeout';
    if (error.response?.status === 500) return 'server_error';
    if (error.response?.status === 503) return 'service_unavailable';
    if (error.response?.status >= 400 && error.response?.status < 500) return 'client_error';
    return 'unknown';
  }

  /**
   * Get AI-selected move for current game state with circuit breaker protection.
   */
  async getAIMove(
    gameState: GameState,
    playerNumber: number,
    difficulty: number = 5,
    aiType?: AIType,
    seed?: number
  ): Promise<MoveResponse> {
    const startTime = performance.now();
    const difficultyLabel = String(difficulty ?? 'n/a');

    return this.circuitBreaker.execute(async () => {
      try {
        // Derive seed from gameState if not explicitly provided
        const effectiveSeed = seed ?? gameState.rngSeed;

        const request: MoveRequest = {
          game_state: gameState,
          player_number: playerNumber,
          difficulty,
          ...(aiType && { ai_type: aiType }),
          ...(effectiveSeed !== undefined && { seed: effectiveSeed }),
        };

        logger.info('Requesting AI move', {
          playerNumber,
          difficulty,
          aiType,
          phase: gameState.currentPhase,
        });

        const response = await this.client.post<MoveResponse>('/ai/move', request);
        const duration = performance.now() - startTime;

        // Record latency for successful Python-service-backed move selection.
        aiMoveLatencyHistogram.labels('python', difficultyLabel).observe(duration);

        logger.info('AI move received', {
          aiType: response.data.ai_type,
          thinkingTime: response.data.thinking_time_ms,
          evaluation: response.data.evaluation,
          latencyMs: Math.round(duration),
        });

        return response.data;
      } catch (error) {
        const duration = performance.now() - startTime;

        logger.error('Failed to get AI move', {
          error,
          latencyMs: Math.round(duration),
          playerNumber,
          difficulty,
        });

        throw new Error(
          `AI Service failed to generate move: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`
        );
      }
    });
  }

  /**
   * Get circuit breaker status for monitoring
   */
  getCircuitBreakerStatus(): { isOpen: boolean; failureCount: number } {
    return this.circuitBreaker.getStatus();
  }

  /**
   * Evaluate current position from a player's perspective.
   */
  async evaluatePosition(gameState: GameState, playerNumber: number): Promise<EvaluationResponse> {
    const startTime = performance.now();
    try {
      const request: EvaluationRequest = {
        game_state: gameState,
        player_number: playerNumber,
      };

      const response = await this.client.post<EvaluationResponse>('/ai/evaluate', request);
      const duration = performance.now() - startTime;

      logger.debug('Position evaluated', {
        playerNumber,
        score: response.data.score,
        breakdown: response.data.breakdown,
        latencyMs: Math.round(duration),
      });

      return response.data;
    } catch (error) {
      const duration = performance.now() - startTime;
      logger.error('Failed to evaluate position', {
        error,
        latencyMs: Math.round(duration),
      });
      throw new Error(
        `AI Service failed to evaluate position: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get AI-selected line reward option for a LineRewardChoice.
   *
   * For now this mirrors the TypeScript AIInteractionHandler heuristic by
   * preferring Option 2 when available, but delegates the decision to the
   * Python service to keep all AI behaviour behind a single façade.
   */
  async getLineRewardChoice(
    gameState: GameState | null,
    playerNumber: number,
    difficulty: number = 5,
    aiType: AIType | undefined,
    options: LineRewardChoice['options']
  ): Promise<LineRewardChoiceResponsePayload> {
    try {
      const request: LineRewardChoiceRequestPayload = {
        ...(gameState && { game_state: gameState }),
        player_number: playerNumber,
        difficulty,
        ...(aiType && { ai_type: aiType }),
        options,
      };

      logger.info('Requesting AI line_reward_option choice', {
        playerNumber,
        difficulty,
        aiType,
        options,
      });

      const response = await this.client.post<LineRewardChoiceResponsePayload>(
        '/ai/choice/line_reward_option',
        request
      );

      logger.info('AI line_reward_option choice received', {
        playerNumber,
        difficulty: response.data.difficulty,
        aiType: response.data.aiType,
        selectedOption: response.data.selectedOption,
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to get line_reward_option choice from AI service', {
        playerNumber,
        error,
      });
      throw new Error(
        `AI Service failed to choose line_reward_option: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get AI-selected ring elimination option for a RingEliminationChoice.
   */
  async getRingEliminationChoice(
    gameState: GameState | null,
    playerNumber: number,
    difficulty: number = 5,
    aiType: AIType | undefined,
    options: RingEliminationChoice['options']
  ): Promise<RingEliminationChoiceResponsePayload> {
    try {
      const request: RingEliminationChoiceRequestPayload = {
        ...(gameState && { game_state: gameState }),
        player_number: playerNumber,
        difficulty,
        ...(aiType && { ai_type: aiType }),
        options,
      };

      logger.info('Requesting AI ring_elimination choice', {
        playerNumber,
        difficulty,
        aiType,
        options,
      });

      const response = await this.client.post<RingEliminationChoiceResponsePayload>(
        '/ai/choice/ring_elimination',
        request
      );

      logger.info('AI ring_elimination choice received', {
        playerNumber,
        difficulty: response.data.difficulty,
        aiType: response.data.aiType,
        selectedOption: response.data.selectedOption,
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to get ring_elimination choice from AI service', {
        playerNumber,
        error,
      });
      throw new Error(
        `AI Service failed to choose ring_elimination: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get AI-selected region order option for a RegionOrderChoice.
   */
  async getRegionOrderChoice(
    gameState: GameState | null,
    playerNumber: number,
    difficulty: number = 5,
    aiType: AIType | undefined,
    options: RegionOrderChoice['options']
  ): Promise<RegionOrderChoiceResponsePayload> {
    try {
      const request: RegionOrderChoiceRequestPayload = {
        ...(gameState && { game_state: gameState }),
        player_number: playerNumber,
        difficulty,
        ...(aiType && { ai_type: aiType }),
        options,
      };

      logger.info('Requesting AI region_order choice', {
        playerNumber,
        difficulty,
        aiType,
        options,
      });

      const response = await this.client.post<RegionOrderChoiceResponsePayload>(
        '/ai/choice/region_order',
        request
      );

      logger.info('AI region_order choice received', {
        playerNumber,
        difficulty: response.data.difficulty,
        aiType: response.data.aiType,
        selectedOption: response.data.selectedOption,
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to get region_order choice from AI service', {
        playerNumber,
        error,
      });
      throw new Error(
        `AI Service failed to choose region_order: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Check if AI service is healthy.
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get('/health');
      return response.data.status === 'healthy';
    } catch (error) {
      logger.error('AI Service health check failed', { error });
      return false;
    }
  }

  /**
   * Clear AI service cache.
   */
  async clearCache(): Promise<void> {
    try {
      await this.client.delete('/ai/cache');
      logger.info('AI Service cache cleared');
    } catch (error) {
      logger.error('Failed to clear AI cache', { error });
      throw new Error(
        `AI Service failed to clear cache: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      );
    }
  }

  /**
   * Get service information.
   */
  async getServiceInfo(): Promise<any> {
    try {
      const response = await this.client.get('/');
      return response.data;
    } catch (error) {
      logger.error('Failed to get service info', { error });
      return null;
    }
  }
}

// Singleton instance
let aiServiceClient: AIServiceClient | null = null;

/**
 * Get the singleton AI Service client instance
 */
export function getAIServiceClient(): AIServiceClient {
  if (!aiServiceClient) {
    aiServiceClient = new AIServiceClient();
  }
  return aiServiceClient;
}

/**
 * Initialize AI Service client with custom URL
 */
export function initAIServiceClient(baseURL: string): AIServiceClient {
  aiServiceClient = new AIServiceClient(baseURL);
  return aiServiceClient;
}
