/**
 * AI Engine - Manages AI Players and Move Selection
 * Delegates to Python AI microservice for move generation
 */

import {
  GameState,
  Move,
  AIProfile,
  AITacticType,
  AIControlMode,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  positionToString,
} from '../../../shared/types/game';
import { getAIServiceClient, AIType as ServiceAIType } from '../../services/AIServiceClient';
import { logger } from '../../utils/logger';
import { BoardManager } from '../BoardManager';
import { RuleEngine } from '../RuleEngine';
import {
  chooseLocalMoveFromCandidates as chooseSharedLocalMoveFromCandidates,
  LocalAIRng,
} from '../../../shared/engine/localAIMoveSelection';

export enum AIType {
  RANDOM = 'random',
  HEURISTIC = 'heuristic',
  MINIMAX = 'minimax',
  MCTS = 'mcts',
  DESCENT = 'descent',
}

export interface AIConfig {
  difficulty: number;
  thinkTime?: number;
  randomness?: number;
  /** Tactical engine chosen for this AI config. */
  aiType?: AIType;
  /** How this AI makes decisions about moves/choices. */
  mode?: AIControlMode;
}

/**
 * Lightweight per-player diagnostics for AI service usage. These counters are
 * incremented when the Python AI service fails or when the engine falls back
 * to local heuristics, and can be queried by tests and orchestration layers
 * to detect degraded AI quality modes.
 */
export interface AIDiagnostics {
  /** Number of times the AI service call failed for this player. */
  serviceFailureCount: number;
  /**
   * Number of times a local fallback move was generated because the AI
   * service was unavailable or failed. Note that this is counted at the
   * selection layer; downstream rules validation may still reject the move.
   */
  localFallbackCount: number;
}

/**
 * Canonical difficulty presets for the TypeScript backend. This table is kept
 * in lockstep with the Python service's difficulty ladder defined in
 * `ai-service/app/main.py` so that a given numeric difficulty corresponds to
 * the same underlying AI engine and coarse behaviour on both sides.
 *
 * Think times are intentionally modest (hundreds of milliseconds) to keep
 * tests and local development responsive; search-based engines (Minimax/MCTS)
 * interpret `thinkTime` as a search budget, while simpler engines treat it as
 * UX-oriented delay.
 */
export const AI_DIFFICULTY_PRESETS: Record<number, Partial<AIConfig> & { profileId: string }> = {
  1: {
    aiType: AIType.RANDOM,
    randomness: 0.5,
    thinkTime: 150,
    profileId: 'v1-random-1',
  },
  2: {
    aiType: AIType.RANDOM,
    randomness: 0.3,
    thinkTime: 200,
    profileId: 'v1-random-2',
  },
  3: {
    aiType: AIType.HEURISTIC,
    randomness: 0.2,
    thinkTime: 250,
    profileId: 'v1-heuristic-3',
  },
  4: {
    aiType: AIType.HEURISTIC,
    randomness: 0.1,
    thinkTime: 300,
    profileId: 'v1-heuristic-4',
  },
  5: {
    aiType: AIType.HEURISTIC,
    randomness: 0.05,
    thinkTime: 350,
    profileId: 'v1-heuristic-5',
  },
  6: {
    aiType: AIType.MINIMAX,
    randomness: 0.02,
    thinkTime: 400,
    profileId: 'v1-minimax-6',
  },
  7: {
    aiType: AIType.MINIMAX,
    randomness: 0.01,
    thinkTime: 450,
    profileId: 'v1-minimax-7',
  },
  8: {
    aiType: AIType.MINIMAX,
    randomness: 0.0,
    thinkTime: 500,
    profileId: 'v1-minimax-8',
  },
  9: {
    aiType: AIType.MCTS,
    randomness: 0.0,
    thinkTime: 600,
    profileId: 'v1-mcts-9',
  },
  10: {
    aiType: AIType.MCTS,
    randomness: 0.0,
    thinkTime: 700,
    profileId: 'v1-mcts-10',
  },
};

export class AIEngine {
  private aiConfigs: Map<number, AIConfig> = new Map();

  /**
   * Internal per-player diagnostics map keyed by playerNumber. This is kept
   * private to avoid accidental mutation; callers access a cloned snapshot
   * via getDiagnostics(...).
   */
  private diagnostics: Map<number, AIDiagnostics> = new Map();

  /**
   * Create/configure an AI player
   * @param playerNumber - The player number for this AI
   * @param difficulty - Difficulty level (1-10)
   * @param type - AI type (optional, auto-selected based on difficulty if not provided)
   */
  createAI(playerNumber: number, difficulty: number = 5, type?: AIType): void {
    // Backwards-compatible wrapper around createAIFromProfile.
    const profile: AIProfile = {
      difficulty,
      mode: 'service',
      ...(type && { aiType: this.mapAITypeToTactic(type) }),
    };

    this.createAIFromProfile(playerNumber, profile);
  }

  /**
   * Configure an AI player from a rich AIProfile. This is the
   * preferred entry point for new code paths.
   */
  createAIFromProfile(playerNumber: number, profile: AIProfile): void {
    const difficulty = profile.difficulty;

    // Validate difficulty
    if (difficulty < 1 || difficulty > 10) {
      throw new Error('AI difficulty must be between 1 and 10');
    }

    const basePreset = AI_DIFFICULTY_PRESETS[difficulty] ?? AI_DIFFICULTY_PRESETS[5];

    const aiType = profile.aiType
      ? this.mapAITacticToAIType(profile.aiType)
      : (basePreset.aiType ?? this.selectAITypeForDifficulty(difficulty));

    const config: AIConfig = {
      difficulty,
      aiType,
      mode: profile.mode ?? 'service',
    };

    if (typeof basePreset.randomness === 'number') {
      config.randomness = basePreset.randomness;
    }

    if (typeof basePreset.thinkTime === 'number') {
      config.thinkTime = basePreset.thinkTime;
    }

    this.aiConfigs.set(playerNumber, config);

    logger.info('AI player configured from profile', {
      playerNumber,
      difficulty,
      aiType,
      mode: config.mode,
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
   * Get move from AI player via Python microservice.
   *
   * @param playerNumber - The player number
   * @param gameState - Current game state
   * @param rng - Optional RNG hook used by local fallback paths. When
   *   provided, this is threaded through to getLocalAIMove so that test
   *   harnesses and parity tools can keep sandbox and backend AI on the
   *   same deterministic RNG stream.
   */
  async getAIMove(
    playerNumber: number,
    gameState: GameState,
    rng?: LocalAIRng
  ): Promise<Move | null> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      // Call Python AI service. Prefer any explicit aiType derived from
      // AIProfile, falling back to a difficulty-based default.
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const serviceAIType = this.mapInternalTypeToServiceType(aiType);
      const response = await getAIServiceClient().getAIMove(
        gameState,
        playerNumber,
        config.difficulty,
        serviceAIType
      );

      const normalizedMove = this.normalizeServiceMove(response.move, gameState, playerNumber);

      logger.info('AI move generated', {
        playerNumber,
        moveType: normalizedMove?.type,
        evaluation: response.evaluation,
        thinkingTime: response.thinking_time_ms,
        aiType: response.ai_type,
      });

      return normalizedMove;
    } catch (error) {
      logger.error('Failed to get AI move from service, falling back to local heuristic', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      // Record the service failure for diagnostics so that callers and tests
      // can detect degraded AI quality.
      const diag = this.getOrCreateDiagnostics(playerNumber);
      diag.serviceFailureCount += 1;

      // Fallback to local heuristic. When an explicit rng is provided,
      // thread it through so callers can keep local fallback decisions
      // on the same deterministic stream as any sandbox AI.
      const localMove = this.getLocalAIMove(playerNumber, gameState, rng);

      if (localMove) {
        diag.localFallbackCount += 1;
      }

      return localMove;
    }
  }

  /**
   * Generate a move using local heuristics when the AI service is unavailable.
   * Uses RuleEngine to find valid moves and selects one randomly. The
   * optional rng parameter allows test harnesses and parity tools to share
   * a deterministic RNG stream with other AI callers (e.g. sandbox AI).
   */
  private getLocalAIMove(
    playerNumber: number,
    gameState: GameState,
    rng: LocalAIRng = Math.random
  ): Move | null {
    try {
      const boardManager = new BoardManager(gameState.boardType);
      const ruleEngine = new RuleEngine(boardManager, gameState.boardType);

      let validMoves = ruleEngine.getValidMoves(gameState);

      // Enforce the canonical "must move placed stack" rule when available.
      // GameEngine/TurnEngine track the origin stack via mustMoveFromStackKey;
      // RuleEngine itself is stateless, so we need to apply this constraint
      // here before handing moves to the shared local-selection policy.
      if (
        gameState.currentPhase === 'movement' ||
        gameState.currentPhase === 'capture' ||
        gameState.currentPhase === 'chain_capture'
      ) {
        const mustMoveFromStackKey = gameState.mustMoveFromStackKey;

        if (mustMoveFromStackKey) {
          validMoves = validMoves.filter((m) => {
            const isMovementOrCaptureMove =
              m.type === 'move_stack' ||
              m.type === 'move_ring' ||
              m.type === 'build_stack' ||
              m.type === 'overtaking_capture' ||
              m.type === 'continue_capture_segment';

            if (!isMovementOrCaptureMove) {
              return true;
            }

            if (!m.from) {
              return false;
            }

            return positionToString(m.from) === mustMoveFromStackKey;
          });
        } else {
          // Backwards-compatible fallback for older fixtures/states that do
          // not populate mustMoveFromStackKey: infer the must-move origin
          // from the last place_ring move by the current player.
          const lastMove = gameState.moveHistory[gameState.moveHistory.length - 1];

          if (
            lastMove &&
            lastMove.type === 'place_ring' &&
            lastMove.player === gameState.currentPlayer &&
            lastMove.to
          ) {
            const placedKey = positionToString(lastMove.to);
            validMoves = validMoves.filter((m) => {
              const isMovementOrCaptureMove =
                m.type === 'move_stack' ||
                m.type === 'move_ring' ||
                m.type === 'overtaking_capture' ||
                m.type === 'build_stack' ||
                m.type === 'continue_capture_segment';

              if (!isMovementOrCaptureMove) {
                return true;
              }

              return m.from && positionToString(m.from) === placedKey;
            });
          }
        }
      }

      if (validMoves.length === 0) {
        return null;
      }

      // Delegate to the shared selection policy so that local fallback and
      // any external harnesses can share the same move preferences. When
      // an explicit rng is provided, it is threaded through so parity
      // harnesses can keep backend and sandbox AI on the same RNG stream.
      return this.chooseLocalMoveFromCandidates(playerNumber, gameState, validMoves, rng);
    } catch (error) {
      logger.error('Failed to generate local AI move', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      return null;
    }
  }

  /**
   * Public wrapper around the local heuristic move generator used by the
   * service fallback path. This allows orchestrators such as GameSession
   * to explicitly request a purely local move when the Python AI service
   * has failed or produced invalid moves, without re-entering the service
   * code path.
   */
  public getLocalFallbackMove(
    playerNumber: number,
    gameState: GameState,
    rng?: LocalAIRng
  ): Move | null {
    const move = this.getLocalAIMove(playerNumber, gameState, rng ?? Math.random);

    if (move) {
      const diag = this.getOrCreateDiagnostics(playerNumber);
      diag.localFallbackCount += 1;
    }

    return move;
  }

  /**
   * Shared local-selection policy used by the fallback path and test
   * harnesses. Given a set of already-legal moves (typically from
   * GameEngine.getValidMoves or RuleEngine.getValidMoves), prefer moves
   * that are more likely to make structural progress before falling back
   * to a pure random choice.
   */
  public chooseLocalMoveFromCandidates(
    playerNumber: number,
    gameState: GameState,
    candidates: Move[],
    rng: LocalAIRng = Math.random
  ): Move | null {
    const selectedMove = chooseSharedLocalMoveFromCandidates(
      playerNumber,
      gameState,
      candidates,
      rng
    );

    if (selectedMove) {
      logger.info('Local AI fallback move generated', {
        playerNumber,
        moveType: selectedMove.type,
      });
    }

    return selectedMove;
  }

  /**
   * Normalise a move returned from the Python AI service so that it
   * respects the backend placement semantics:
   * - Use the canonical 'place_ring' type for ring placements.
   * - On existing stacks, enforce exactly 1 ring per placement and set
   *   placedOnStack=true.
   * - On empty cells, allow small multi-ring placements by filling in
   *   placementCount when the service omits it, clamped by the
   *   player’s ringsInHand.
   *
   * This keeps the AI service relatively agnostic of RingRift’s
   * evolving placement rules while ensuring GameEngine/RuleEngine see
   * well-formed moves.
   */
  private normalizeServiceMove(
    move: Move | null,
    gameState: GameState,
    playerNumber: number
  ): Move | null {
    if (!move) {
      return null;
    }

    // Defensive: if board/players are missing (e.g. in unit tests that
    // mock GameState), return the move as-is.
    if (!gameState.board || !Array.isArray(gameState.players)) {
      return move;
    }

    const normalized: Move = { ...move };

    // Normalise any historical 'place' type to the canonical
    // 'place_ring'.
    if (normalized.type === ('place' as any)) {
      normalized.type = 'place_ring';
    }

    if (normalized.type !== 'place_ring') {
      return normalized;
    }

    const playerState = gameState.players.find((p) => p.playerNumber === playerNumber);
    const ringsInHand = playerState?.ringsInHand ?? 0;

    if (!normalized.to || ringsInHand <= 0) {
      // Let RuleEngine reject impossible placements; we only ensure the
      // metadata is consistent when a placement is otherwise plausible.
      return normalized;
    }

    const board = gameState.board;
    const posKey = positionToString(normalized.to);
    const stack = board.stacks.get(posKey as any);
    const isOccupied = !!stack && stack.rings.length > 0;

    if (isOccupied) {
      // Canonical rule: at most one ring per placement onto an existing
      // stack, and the placement is flagged as stacking.
      normalized.placedOnStack = true;
      normalized.placementCount = 1;
      return normalized;
    }

    // Empty cell: allow small multi-ring placements. If the service
    // already provided a placementCount, clamp it; otherwise fall back
    // to a deterministic default of 1 ring. This avoids introducing any
    // additional RNG at the AI–rules boundary while keeping older
    // service versions (that omit placementCount) working.
    const maxPerPlacement = ringsInHand;
    if (maxPerPlacement <= 0) {
      return normalized;
    }

    if (normalized.placementCount && normalized.placementCount > 0) {
      const clamped = Math.min(Math.max(normalized.placementCount, 1), maxPerPlacement);
      normalized.placementCount = clamped;
      normalized.placedOnStack = false;
      return normalized;
    }

    // If placementCount is missing for an otherwise valid empty-cell
    // placement, record this so we can add metrics and tighten the
    // contract in a future phase.
    logger.warn('AI service omitted placementCount for empty-cell place_ring; defaulting to 1', {
      playerNumber,
      position: normalized.to && positionToString(normalized.to),
      ringsInHand,
    });

    normalized.placementCount = 1;
    normalized.placedOnStack = false;

    return normalized;
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
      const response = await getAIServiceClient().evaluatePosition(gameState, playerNumber);

      logger.debug('Position evaluated', {
        playerNumber,
        score: response.score,
      });

      return response.score;
    } catch (error) {
      logger.error('Failed to evaluate position from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Ask the AI service to choose a line_reward_option for an AI-controlled
   * player. This is the service-backed analogue of the local heuristic in
   * AIInteractionHandler and keeps all remote AI behaviour behind this
   * façade.
   */
  async getLineRewardChoice(
    playerNumber: number,
    gameState: GameState | null,
    options: LineRewardChoice['options']
  ): Promise<LineRewardChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const serviceAIType = this.mapInternalTypeToServiceType(aiType);
      const response = await getAIServiceClient().getLineRewardChoice(
        gameState,
        playerNumber,
        config.difficulty,
        serviceAIType,
        options
      );

      logger.info('AI line_reward_option choice generated', {
        playerNumber,
        difficulty: response.difficulty,
        aiType: response.aiType,
        selectedOption: response.selectedOption,
      });

      return response.selectedOption;
    } catch (error) {
      logger.error('Failed to get line_reward_option choice from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Ask the AI service to choose a ring_elimination option for an
   * AI-controlled player. This mirrors the TypeScript
   * AIInteractionHandler heuristic (smallest capHeight, then
   * smallest totalHeight) but keeps the remote call behind this
   * façade so callers do not need to know about the Python
   * service directly.
   */
  async getRingEliminationChoice(
    playerNumber: number,
    gameState: GameState | null,
    options: RingEliminationChoice['options']
  ): Promise<RingEliminationChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const serviceAIType = this.mapInternalTypeToServiceType(aiType);
      const response = await getAIServiceClient().getRingEliminationChoice(
        gameState,
        playerNumber,
        config.difficulty,
        serviceAIType,
        options
      );

      logger.info('AI ring_elimination choice generated', {
        playerNumber,
        difficulty: response.difficulty,
        aiType: response.aiType,
        selectedOption: response.selectedOption,
      });

      return response.selectedOption;
    } catch (error) {
      logger.error('Failed to get ring_elimination choice from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Ask the AI service to choose a region_order option for an
   * AI-controlled player. This mirrors the TypeScript
   * AIInteractionHandler heuristic (largest region by size, with
   * additional context from GameState) but keeps the remote call
   * behind this façade so callers do not need to know about the
   * Python service directly.
   */
  async getRegionOrderChoice(
    playerNumber: number,
    gameState: GameState | null,
    options: RegionOrderChoice['options']
  ): Promise<RegionOrderChoice['options'][number]> {
    const config = this.aiConfigs.get(playerNumber);

    if (!config) {
      throw new Error(`No AI configuration found for player number ${playerNumber}`);
    }

    try {
      const aiType = config.aiType ?? this.selectAITypeForDifficulty(config.difficulty);
      const serviceAIType = this.mapInternalTypeToServiceType(aiType);
      const response = await getAIServiceClient().getRegionOrderChoice(
        gameState,
        playerNumber,
        config.difficulty,
        serviceAIType,
        options
      );

      logger.info('AI region_order choice generated', {
        playerNumber,
        difficulty: response.difficulty,
        aiType: response.aiType,
        selectedOption: response.selectedOption,
      });

      return response.selectedOption;
    } catch (error) {
      logger.error('Failed to get region_order choice from service', {
        playerNumber,
        error: error instanceof Error ? error.message : 'Unknown error',
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
    this.diagnostics.clear();
  }

  /**
   * Get a snapshot of diagnostics for a given AI-controlled player. The
   * returned object is a shallow clone so callers cannot mutate the
   * internal counters directly.
   */
  getDiagnostics(playerNumber: number): AIDiagnostics | undefined {
    const diag = this.diagnostics.get(playerNumber);
    return diag
      ? {
          serviceFailureCount: diag.serviceFailureCount,
          localFallbackCount: diag.localFallbackCount,
        }
      : undefined;
  }

  /**
   * Internal helper: ensure a diagnostics record exists for the given
   * player and return it.
   */
  private getOrCreateDiagnostics(playerNumber: number): AIDiagnostics {
    let diag = this.diagnostics.get(playerNumber);
    if (!diag) {
      diag = { serviceFailureCount: 0, localFallbackCount: 0 };
      this.diagnostics.set(playerNumber, diag);
    }
    return diag;
  }

  /**
   * Check AI service health
   */
  async checkServiceHealth(): Promise<boolean> {
    try {
      return await getAIServiceClient().healthCheck();
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
      await getAIServiceClient().clearCache();
      logger.info('AI service cache cleared');
    } catch (error) {
      logger.error('Failed to clear AI service cache', { error });
      throw error;
    }
  }

  /**
   * Auto-select AI type based on difficulty level.
   *
   * This is a thin wrapper over the canonical AI_DIFFICULTY_PRESETS table so
   * that both Python and TypeScript resolve difficulty→engine in the same way.
   */
  private selectAITypeForDifficulty(difficulty: number): AIType {
    const clamped = difficulty < 1 ? 1 : difficulty > 10 ? 10 : difficulty;
    const preset = AI_DIFFICULTY_PRESETS[clamped];
    return preset.aiType ?? AIType.HEURISTIC;
  }

  /** Map shared AITacticType values onto the internal AIType enum. */
  private mapAITacticToAIType(tactic: AITacticType): AIType {
    switch (tactic) {
      case 'random':
        return AIType.RANDOM;
      case 'heuristic':
        return AIType.HEURISTIC;
      case 'minimax':
        return AIType.MINIMAX;
      case 'mcts':
        return AIType.MCTS;
      case 'descent':
        return AIType.DESCENT;
      default: {
        // Exhaustive check so that adding a new AITacticType forces this
        // mapping to be updated.
        const exhaustiveCheck: never = tactic;
        throw new Error(`Unhandled AITacticType in mapAITacticToAIType: ${exhaustiveCheck}`);
      }
    }
  }

  /** Map internal AIType to the shared AITacticType union. */
  private mapAITypeToTactic(type: AIType): AITacticType {
    switch (type) {
      case AIType.RANDOM:
        return 'random';
      case AIType.HEURISTIC:
        return 'heuristic';
      case AIType.MINIMAX:
        return 'minimax';
      case AIType.MCTS:
        return 'mcts';
      case AIType.DESCENT:
        return 'descent';
      default: {
        // Exhaustive check so that adding a new AIType forces this mapping
        // (and downstream service wiring) to be updated.
        const exhaustiveCheck: never = type;
        throw new Error(`Unhandled AIType in mapAITypeToTactic: ${exhaustiveCheck}`);
      }
    }
  }

  /**
   * Map the internal AIType enum used by the server onto the AIType enum
   * understood by the Python AI service. This indirection keeps the
   * wire-level contract stable even if the server or service introduce
   * additional implementation-specific variants in future.
   */
  private mapInternalTypeToServiceType(type: AIType): ServiceAIType {
    switch (type) {
      case AIType.RANDOM:
        return ServiceAIType.RANDOM;
      case AIType.HEURISTIC:
        return ServiceAIType.HEURISTIC;
      case AIType.MINIMAX:
        return ServiceAIType.MINIMAX;
      case AIType.MCTS:
        return ServiceAIType.MCTS;
      case AIType.DESCENT:
        return ServiceAIType.DESCENT;
      default: {
        const exhaustiveCheck: never = type;
        throw new Error(`Unhandled AIType in mapInternalTypeToServiceType: ${exhaustiveCheck}`);
      }
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
  static getRecommendedDifficulty(
    skillLevel: 'beginner' | 'intermediate' | 'advanced' | 'expert'
  ): number {
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
