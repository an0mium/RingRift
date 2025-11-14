/**
 * AI Engine - Manages AI Players and Move Selection
 * Delegates to Python AI microservice for move generation
 */

import { GameState, Move } from '../../../shared/types/game';
import { getAIServiceClient, AIType as ServiceAIType } from '../../services/AIServiceClient';
import { logger } from '../../utils/logger';

export enum AIType {
  RANDOM = 'random',
  HEURISTIC = 'heuristic',
  MINIMAX = 'minimax',
  MCTS = 'mcts'
}

export interface AIConfig {
  difficulty: number;
  thinkTime?: number;
  randomness?: number;
}

export const AI_DIFFICULTY_PRESETS: Record<number, Partial<AIConfig>> = {
  1: { randomness: 0.5, thinkTime: 500 },
  2: { randomness: 0.3, thinkTime: 700 },
  3: { randomness: 0.2, thinkTime: 1000 },
  4: { randomness: 0.1, thinkTime: 1200 },
  5: { randomness: 0.05, thinkTime: 1500 },
  6: { randomness: 0.02, thinkTime: 2000 },
  7: { randomness: 0.01, thinkTime: 2500 },
  8: { randomness: 0, thinkTime: 3000 },
  9: { randomness: 0, thinkTime: 4000 },
  10: { randomness: 0, thinkTime: 5000 },
};

export class AIEngine {
  private aiConfigs: Map<number, AIConfig> = new Map();
  private aiServiceClient = getAIServiceClient();

  /**
   * Create/configure an AI player
   * @param playerNumber - The player number for this AI
   * @param difficulty - Difficulty level (1-10)
   * @param type - AI type (optional, auto-selected based on difficulty if not provided)
   */
  createAI(
    playerNumber: number,
    difficulty: number = 5,
    type?: AIType
  ): void {
    // Validate difficulty
    if (difficulty < 1 || difficulty > 10) {
      throw new Error('AI difficulty must be between 1 and 10');
    }

    // Get configuration preset for this difficulty
    const config: AIConfig = {
      ...AI_DIFFICULTY_PRESETS[difficulty],
      difficulty,
    };

    // Store AI configuration
    this.aiConfigs.set(playerNumber, config);

    logger.info(`AI player configured`, {
      playerNumber,
      difficulty,
      type: type || this.selectAITypeForDifficulty(difficulty)
    });
  }

  /**
   * Get an AI config by player number
   */
  getAIConfig(playerNumber: number): AIConfig | undefined {
    return this.aiConfigs.get(playerNumber);
  }

  /**
   * Remove an AI player
   */
  removeAI(playerNumber: number): boolean {
    return this.aiConfigs.delete(playerNumber);
  }

  /**
   * Get move from AI player via Python microservice
   * @param playerNumber - The player number
   * @param gameState - Current game state
   * @returns The selected move or null if no valid moves
   */
  async getAIMove(playerNumber: number, gameState: GameState): Promise<Move | null> {
    const config = this.aiConfigs.get(playerNumber);
    
    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      // Call Python AI service
      const aiType = this.selectAITypeForDifficulty(config.difficulty);
      const response = await this.aiServiceClient.getAIMove(
        gameState,
        playerNumber,
        config.difficulty,
        aiType as unknown as ServiceAIType
      );

      logger.info('AI move generated', {
        playerNumber,
        moveType: response.move?.type,
        evaluation: response.evaluation,
        thinkingTime: response.thinking_time_ms,
        aiType: response.ai_type
      });

      return response.move;
    } catch (error) {
      logger.error('Failed to get AI move from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      throw error;
    }
  }

  /**
   * Evaluate a position from an AI's perspective via Python microservice
   */
  async evaluatePosition(playerNumber: number, gameState: GameState): Promise<number> {
    const config = this.aiConfigs.get(playerNumber);
    
    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const response = await this.aiServiceClient.evaluatePosition(
        gameState,
        playerNumber
      );

      logger.debug('Position evaluated', {
        playerNumber,
        score: response.score
      });

      return response.score;
    } catch (error) {
      logger.error('Failed to evaluate position from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      throw error;
    }
  }

  /**
   * Check if a player is controlled by AI
   */
  isAIPlayer(playerNumber: number): boolean {
    return this.aiConfigs.has(playerNumber);
  }

  /**
   * Get all AI player numbers
   */
  getAllAIPlayerNumbers(): number[] {
    return Array.from(this.aiConfigs.keys());
  }

  /**
   * Clear all AI players
   */
  clearAll(): void {
    this.aiConfigs.clear();
  }

  /**
   * Check AI service health
   */
  async checkServiceHealth(): Promise<boolean> {
    try {
      return await this.aiServiceClient.healthCheck();
    } catch (error) {
      logger.error('AI service health check failed', { error });
      return false;
    }
  }

  /**
   * Clear AI service cache
   */
  async clearServiceCache(): Promise<void> {
    try {
      await this.aiServiceClient.clearCache();
      logger.info('AI service cache cleared');
    } catch (error) {
      logger.error('Failed to clear AI service cache', { error });
      throw error;
    }
  }

  /**
   * Auto-select AI type based on difficulty level
   * @param difficulty - Difficulty level (1-10)
   * @returns Recommended AI type for this difficulty
   */
  private selectAITypeForDifficulty(difficulty: number): AIType {
    if (difficulty >= 1 && difficulty <= 2) {
      return AIType.RANDOM; // Levels 1-2: Random AI
    } else if (difficulty >= 3 && difficulty <= 5) {
      return AIType.HEURISTIC; // Levels 3-5: Heuristic AI
    } else if (difficulty >= 6 && difficulty <= 8) {
      return AIType.MINIMAX; // Levels 6-8: Minimax AI (falls back to Heuristic if not implemented)
    } else {
      return AIType.MCTS; // Levels 9-10: MCTS AI (falls back to Heuristic if not implemented)
    }
  }

  /**
   * Get AI description for difficulty level
   */
  static getAIDescription(difficulty: number): string {
    const descriptions: Record<number, string> = {
      1: 'Very Easy - Random moves with high error rate',
      2: 'Easy - Mostly random moves with some filtering',
      3: 'Medium-Easy - Basic strategy with occasional mistakes',
      4: 'Medium - Balanced play with tactical awareness',
      5: 'Medium-Hard - Strong tactical play',
      6: 'Hard - Advanced tactics and some planning',
      7: 'Very Hard - Deep planning and strong positional play',
      8: 'Expert - Excellent tactics and strategy',
      9: 'Master - Near-perfect play with deep calculation',
      10: 'Grandmaster - Optimal play across all phases',
    };

    return descriptions[difficulty] || 'Unknown difficulty level';
  }

  /**
   * Get recommended difficulty for player skill level
   */
  static getRecommendedDifficulty(skillLevel: 'beginner' | 'intermediate' | 'advanced' | 'expert'): number {
    const recommendations = {
      beginner: 2, // Easy
      intermediate: 4, // Medium
      advanced: 6, // Hard
      expert: 8, // Expert
    };

    return recommendations[skillLevel];
  }
}

/**
 * Singleton instance for global AI engine
 */
export const globalAIEngine = new AIEngine();
