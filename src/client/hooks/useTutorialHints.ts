/**
 * Tutorial Hints Hook
 *
 * Manages contextual hints for the "Learn the Basics" tutorial mode.
 * Triggers hints when entering new game phases for the first time.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { GameState, GamePhase } from '../../shared/types/game';
import type { TeachingTopic } from '../components/TeachingOverlay';

export interface TutorialHint {
  /** Game phase that triggered this hint */
  phase: GamePhase;
  /** Title shown in the hint banner */
  title: string;
  /** Main message explaining what to do */
  message: string;
  /** Teaching topic to open when "Learn More" is clicked */
  topic: TeachingTopic;
  /** Icon for the hint */
  icon: string;
}

/** Maps game phases to their corresponding teaching topics */
const PHASE_TO_TOPIC: Partial<Record<GamePhase, TeachingTopic>> = {
  ring_placement: 'ring_placement',
  movement: 'stack_movement',
  capture: 'capturing',
  chain_capture: 'chain_capture',
  line_processing: 'line_bonus',
  territory_processing: 'territory',
  forced_elimination: 'forced_elimination',
};

/** Hint content for each phase */
const PHASE_HINTS: Partial<Record<GamePhase, Omit<TutorialHint, 'phase' | 'topic'>>> = {
  ring_placement: {
    title: 'Place Your Rings',
    message: 'Click empty cells to place rings. Adjacent rings form stronger positions!',
    icon: '🎯',
  },
  movement: {
    title: 'Move Your Stacks',
    message:
      'Click a stack, then a destination. You must move at least as far as the stack is tall.',
    icon: '↗️',
  },
  capture: {
    title: 'Capture!',
    message:
      'Jump over an adjacent enemy stack to capture it. Your stack must be at least as tall.',
    icon: '⚔️',
  },
  chain_capture: {
    title: 'Keep Capturing!',
    message: 'You started a capture chain - you must continue until no captures remain!',
    icon: '⇉',
  },
  line_processing: {
    title: 'Line Formed!',
    message: 'Your markers formed a line! Choose how to collapse it into territory.',
    icon: '━',
  },
  territory_processing: {
    title: 'Claim Territory',
    message: 'A region is surrounded. Pay the cost from an outside stack to claim it.',
    icon: '🏰',
  },
  forced_elimination: {
    title: 'No Moves Available',
    message: 'You have no legal moves. Select a stack to eliminate rings from.',
    icon: '💎',
  },
};

export interface UseTutorialHintsOptions {
  /** Current game state */
  gameState: GameState | null;
  /** Whether tutorial mode is active */
  isLearnMode: boolean;
  /** Phases that have already been seen */
  seenPhases: string[];
  /** Whether hints are enabled by user preference */
  hintsEnabled: boolean;
}

export interface UseTutorialHintsResult {
  /** Current hint to display, or null if none */
  currentHint: TutorialHint | null;
  /** Dismiss the current hint */
  dismissHint: () => void;
  /** Get the teaching topic for the current hint (for "Learn More") */
  getTeachingTopic: () => TeachingTopic | null;
}

/**
 * Hook that manages tutorial hints based on game phase changes.
 *
 * Shows hints when entering a game phase for the first time in learn mode.
 * Hints can be dismissed and won't reappear for the same phase.
 */
export function useTutorialHints({
  gameState,
  isLearnMode,
  seenPhases,
  hintsEnabled,
}: UseTutorialHintsOptions): UseTutorialHintsResult {
  const [currentHint, setCurrentHint] = useState<TutorialHint | null>(null);
  const [dismissedPhase, setDismissedPhase] = useState<GamePhase | null>(null);
  const lastPhaseRef = useRef<GamePhase | null>(null);

  // Track phase changes and trigger hints
  useEffect(() => {
    if (!isLearnMode || !hintsEnabled || !gameState) {
      setCurrentHint(null);
      return;
    }

    const currentPhase = gameState.currentPhase;

    // Skip if same phase as before or already dismissed this phase
    if (currentPhase === lastPhaseRef.current || currentPhase === dismissedPhase) {
      return;
    }

    // Skip game_over phase
    if (currentPhase === 'game_over') {
      setCurrentHint(null);
      return;
    }

    lastPhaseRef.current = currentPhase;

    // Check if this phase has already been seen
    if (seenPhases.includes(currentPhase)) {
      setCurrentHint(null);
      return;
    }

    // Get hint content for this phase
    const hintContent = PHASE_HINTS[currentPhase];
    const topic = PHASE_TO_TOPIC[currentPhase];

    if (!hintContent || !topic) {
      setCurrentHint(null);
      return;
    }

    // Show the hint
    setCurrentHint({
      phase: currentPhase,
      topic,
      ...hintContent,
    });
  }, [gameState, isLearnMode, seenPhases, hintsEnabled, dismissedPhase]);

  // Clear dismissed phase when phase changes
  useEffect(() => {
    if (gameState && gameState.currentPhase !== dismissedPhase) {
      setDismissedPhase(null);
    }
  }, [gameState, dismissedPhase]);

  const dismissHint = useCallback(() => {
    if (currentHint) {
      setDismissedPhase(currentHint.phase);
    }
    setCurrentHint(null);
  }, [currentHint]);

  const getTeachingTopic = useCallback(() => {
    return currentHint?.topic ?? null;
  }, [currentHint]);

  return {
    currentHint,
    dismissHint,
    getTeachingTopic,
  };
}
