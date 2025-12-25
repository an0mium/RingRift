/**
 * @fileoverview useBackendDecisionUI Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for backend decision UI state.
 * It manages pending choice state and countdown timer wiring, not rules logic.
 *
 * Canonical SSoT:
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - Choice handling: `src/shared/decisions/PlayerChoice.ts`
 *
 * This adapter:
 * - Encapsulates pending choice state from usePendingChoice
 * - Provides countdown timer state for ChoiceDialog component
 * - Exposes respond callback for submitting choice selections
 * - Provides rich decision-phase view for HUD and ChoiceDialog
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/BACKEND_GAME_HOST_DECOMPOSITION_PLAN.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { usePendingChoice, type PendingChoiceView } from './useGameActions';
import type { PlayerChoice } from '../../shared/types/game';

/**
 * State returned by the useBackendDecisionUI hook.
 */
export interface BackendDecisionUIState {
  /** Currently pending PlayerChoice (if any) */
  pendingChoice: PlayerChoice | null;
  /** Deadline timestamp for the choice (if any) */
  choiceDeadline: number | null;
  /** Remaining time in milliseconds (if any) */
  choiceTimeRemainingMs: number | null;
  /**
   * Respond to a pending choice with the selected option.
   * The choice argument is ignored for safety - the hook uses the current pending choice.
   */
  respondToChoice: <T>(choice: PlayerChoice, selectedOption: T) => void;
  /**
   * Rich decision-phase view derived from choiceViewModels, used to provide
   * consistent copy/timeout semantics to both HUD and ChoiceDialog.
   */
  pendingChoiceView: PendingChoiceView | null;
}

/**
 * Custom hook for managing backend decision UI state.
 *
 * Encapsulates pending choice state and countdown timer wiring for the
 * ChoiceDialog component. Provides:
 * - Current pending choice and its deadline
 * - Time remaining for the choice
 * - Respond callback for submitting selections
 * - Rich view model for consistent HUD/dialog rendering
 *
 * Extracted from BackendGameHost to reduce component complexity.
 *
 * @returns Object with pending choice state and respond action
 */
export function useBackendDecisionUI(): BackendDecisionUIState {
  const { choice, deadline, respond, timeRemaining, view } = usePendingChoice();

  return {
    pendingChoice: choice,
    choiceDeadline: deadline,
    choiceTimeRemainingMs: timeRemaining,
    // The underlying hook already knows which choice is pending; we ignore the
    // explicit choice argument and delegate to respond() for safety.
    respondToChoice: (_choice, selectedOption) => {
      respond(selectedOption as PlayerChoice['options'][number]);
    },
    pendingChoiceView: view,
  };
}

export default useBackendDecisionUI;
