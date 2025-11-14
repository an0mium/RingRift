/**
 * Client for communicating with the Python AI Service
 * Makes HTTP requests to the FastAPI microservice for AI move selection
 */

import axios, { AxiosInstance } from 'axios';
import { GameState, Move } from '../../shared/types/game';
import { logger } from '../utils/logger';

export interface AIConfig {
  difficulty: number;
  thinkTime?: number;
  randomness?: number;
}

export enum AIType {
  RANDOM = 'random',
  HEURISTIC = 'heuristic',
  MINIMAX = 'minimax',
  MCTS = 'mcts'
}

export interface MoveRequest {
  game_state: GameState;
  player_number: number;
  difficulty: number;
  ai_type?: AIType;
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

/**
 * Client for interacting with the Python AI microservice
 */
export class AIServiceClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL?: string) {
    this.baseURL = baseURL || process.env.AI_SERVICE_URL || 'http://localhost:8001';
    
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000, // 30 second timeout for AI computation
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        logger.error('AI Service error:', {
          message: error.message,
          response: error.response?.data,
          status: error.response?.status
        });
        throw error;
      }
    );
  }

  /**
   * Get AI-selected move for current game state
   */
  async getAIMove(
    gameState: GameState,
    playerNumber: number,
    difficulty: number = 5,
    aiType?: AIType
  ): Promise<MoveResponse> {
    try {
      const request: MoveRequest = {
        game_state: gameState,
        player_number: playerNumber,
        difficulty,
        ...(aiType && { ai_type: aiType })
      };

      logger.info('Requesting AI move', {
        playerNumber,
        difficulty,
        aiType,
        phase: gameState.currentPhase
      });

      const response = await this.client.post<MoveResponse>('/ai/move', request);
      
      logger.info('AI move received', {
        aiType: response.data.ai_type,
        thinkingTime: response.data.thinking_time_ms,
        evaluation: response.data.evaluation
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to get AI move', { error });
      throw new Error(`AI Service failed to generate move: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Evaluate current position from a player's perspective
   */
  async evaluatePosition(
    gameState: GameState,
    playerNumber: number
  ): Promise<EvaluationResponse> {
    try {
      const request: EvaluationRequest = {
        game_state: gameState,
        player_number: playerNumber
      };

      const response = await this.client.post<EvaluationResponse>('/ai/evaluate', request);
      
      logger.debug('Position evaluated', {
        playerNumber,
        score: response.data.score,
        breakdown: response.data.breakdown
      });

      return response.data;
    } catch (error) {
      logger.error('Failed to evaluate position', { error });
      throw new Error(`AI Service failed to evaluate position: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Clear AI cache on the service
   */
  async clearCache(): Promise<void> {
    try {
      await this.client.delete('/ai/cache');
      logger.info('AI Service cache cleared');
    } catch (error) {
      logger.error('Failed to clear AI cache', { error });
      throw new Error(`AI Service failed to clear cache: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Check if AI service is healthy
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
   * Get service information
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
