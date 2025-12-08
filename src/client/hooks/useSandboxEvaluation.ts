/**
 * useSandboxEvaluation - AI evaluation management for sandbox mode
 *
 * This hook extracts AI evaluation functionality from SandboxGameHost:
 * - Requesting AI position evaluation
 * - Auto-evaluation when developer tools are enabled
 * - Tracking evaluation history for visualization
 * - Managing evaluation loading state
 *
 * @module hooks/useSandboxEvaluation
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import type { GameState } from '../../shared/types/game';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import type { ClientSandboxEngine } from '../services/ClientSandboxEngine';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Evaluation data from AI service.
 */
export type EvaluationData = PositionEvaluationPayload['data'];

/**
 * Options for the evaluation hook.
 */
export interface SandboxEvaluationOptions {
  /** Sandbox engine instance */
  engine: ClientSandboxEngine | null;
  /** Current game state */
  gameState: GameState | null;
  /** Whether developer tools are enabled */
  developerToolsEnabled: boolean;
  /** Whether in replay mode (skip auto-evaluation) */
  isInReplayMode?: boolean;
  /** Whether viewing history (skip auto-evaluation) */
  isViewingHistory?: boolean;
  /** API endpoint for evaluation */
  evaluationEndpoint?: string;
}

/**
 * Return type for useSandboxEvaluation.
 */
export interface SandboxEvaluationState {
  /** Evaluation history for visualization */
  evaluationHistory: EvaluationData[];
  /** Current evaluation error message */
  evaluationError: string | null;
  /** Whether evaluation is in progress */
  isEvaluating: boolean;
  /** Request a new evaluation */
  requestEvaluation: () => Promise<void>;
  /** Clear evaluation history */
  clearHistory: () => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for managing AI evaluation in sandbox mode.
 */
export function useSandboxEvaluation(
  options: SandboxEvaluationOptions
): SandboxEvaluationState {
  const {
    engine,
    gameState,
    developerToolsEnabled,
    isInReplayMode = false,
    isViewingHistory = false,
    evaluationEndpoint = '/api/evaluate',
  } = options;

  // State
  const [evaluationHistory, setEvaluationHistory] = useState<EvaluationData[]>([]);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  // Ref to track last evaluated move count for auto-evaluation
  const lastEvaluatedMoveRef = useRef<number>(-1);

  // Request evaluation
  const requestEvaluation = useCallback(async () => {
    if (!engine || !gameState) {
      setEvaluationError('No game state available');
      return;
    }

    if (isEvaluating) {
      return; // Already running
    }

    setIsEvaluating(true);
    setEvaluationError(null);

    try {
      const response = await fetch(evaluationEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          gameState: gameState,
          playerNumber: gameState.currentPlayer,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Evaluation failed: ${response.status}`);
      }

      const data = await response.json();

      // Add to history
      const evalData: EvaluationData = {
        gameId: gameState.gameId,
        moveNumber: gameState.moveHistory?.length ?? 0,
        playerNumber: gameState.currentPlayer,
        evaluation: data.evaluation ?? data.score ?? 0,
        breakdown: data.breakdown ?? {},
        timestamp: Date.now(),
      };

      setEvaluationHistory((prev) => [...prev, evalData]);
      lastEvaluatedMoveRef.current = gameState.moveHistory?.length ?? 0;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Evaluation failed';
      console.error('[useSandboxEvaluation] Evaluation failed:', err);
      setEvaluationError(message);
    } finally {
      setIsEvaluating(false);
    }
  }, [engine, gameState, isEvaluating, evaluationEndpoint]);

  // Clear history
  const clearHistory = useCallback(() => {
    setEvaluationHistory([]);
    setEvaluationError(null);
    lastEvaluatedMoveRef.current = -1;
  }, []);

  // Reset on engine change
  useEffect(() => {
    clearHistory();
  }, [engine, clearHistory]);

  // Auto-evaluation when developer tools are enabled
  useEffect(() => {
    if (!developerToolsEnabled || !engine || !gameState) {
      return;
    }

    // Skip if in replay or history viewing mode
    if (isInReplayMode || isViewingHistory) {
      return;
    }

    // Skip if game is not active
    if (gameState.gameStatus !== 'active') {
      return;
    }

    // Skip if already evaluating
    if (isEvaluating) {
      return;
    }

    // Check if we need to evaluate (new move since last evaluation)
    const currentMoveCount = gameState.moveHistory?.length ?? 0;
    if (currentMoveCount <= lastEvaluatedMoveRef.current) {
      return;
    }

    // Debounce evaluation requests
    const timeoutId = setTimeout(() => {
      requestEvaluation();
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [
    developerToolsEnabled,
    engine,
    gameState,
    isInReplayMode,
    isViewingHistory,
    isEvaluating,
    requestEvaluation,
  ]);

  return {
    evaluationHistory,
    evaluationError,
    isEvaluating,
    requestEvaluation,
    clearHistory,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format evaluation score for display.
 */
export function formatEvaluationScore(score: number): string {
  if (score > 0) {
    return `+${score.toFixed(2)}`;
  }
  return score.toFixed(2);
}

/**
 * Get evaluation trend from history.
 */
export function getEvaluationTrend(
  history: EvaluationData[],
  playerNumber: number
): 'improving' | 'declining' | 'stable' | 'unknown' {
  if (history.length < 2) {
    return 'unknown';
  }

  // Get last few evaluations for the player
  const playerEvals = history
    .filter((e) => e.playerNumber === playerNumber)
    .slice(-3);

  if (playerEvals.length < 2) {
    return 'unknown';
  }

  const first = playerEvals[0].evaluation;
  const last = playerEvals[playerEvals.length - 1].evaluation;
  const diff = last - first;

  if (Math.abs(diff) < 0.1) {
    return 'stable';
  }

  return diff > 0 ? 'improving' : 'declining';
}

/**
 * Get key features from evaluation breakdown.
 */
export function getKeyFeatures(
  breakdown: Record<string, number>,
  limit: number = 5
): Array<{ name: string; value: number }> {
  return Object.entries(breakdown)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, limit);
}
