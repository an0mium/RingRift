import type { GameState, GamePhase } from '../types/game';
import type { PerTurnState, TurnLogicDelegates } from './turnLogic';
import { advanceTurnAndPhase } from './turnLogic';

/**
 * Shared, host-agnostic helpers that sit on top of {@link advanceTurnAndPhase}
 * to model higher-level turn lifecycle concepts:
 *
 * - Advancing from the end-of-movement boundary (after all automatic
 *   post-move work has completed) to the next interactive state.
 * - Starting an interactive turn for the current player, including
 *   forced-elimination / skipping when appropriate.
 *
 * These helpers are intentionally minimal and purely functional so they
 * can be reused by the backend GameEngine, the client sandbox engine,
 * and any future reference engines without introducing new stateful
 * abstractions. Hosts remain responsible for owning the underlying
 * GameState instance and any additional per-turn / per-host metadata.
 */

export interface TurnLifecycleContext {
  /** Current game state; host owns the actual storage. */
  state: GameState;
  /** Per-turn flags shared with turnLogic (hasPlacedThisTurn, mustMoveFromStackKey). */
  perTurn: PerTurnState;
}

/** Optional hooks to integrate host-specific behaviour. */
export interface TurnLifecycleHooks {
  /**
   * Called whenever a new interactive turn (ring_placement / movement /
   * capture / chain_capture) begins for the active player, after all
   * forced-elimination / skipping has been resolved. Hosts typically use
   * this to run LPS tracking and last-player-standing checks.
   */
  onStartInteractiveTurn?: (state: GameState, perTurn: PerTurnState) => void;

  /**
   * Called after a forced-elimination step ends but the game remains
   * active. The {@code eliminatedPlayer} is the player who just paid the
   * elimination cost. This is useful for logging and metrics.
   */
  onForcedElimination?: (state: GameState, eliminatedPlayer: number) => void;
}

export interface TurnLifecycleDeps {
  delegates: TurnLogicDelegates;
  hooks?: TurnLifecycleHooks;
}

function isInteractivePhase(phase: GamePhase): boolean {
  return (
    phase === 'ring_placement' ||
    phase === 'movement' ||
    phase === 'capture' ||
    phase === 'chain_capture'
  );
}

/**
 * Advance from the end-of-movement boundary to the next state chosen by
 * the shared turn logic.
 *
 * Contract:
 * - Call this after all automatic post-move consequences (lines,
 *   territory, victory checks) for the *current* player have been
 *   resolved.
 * - The caller is expected to normalise {@link GameState.currentPhase}
 *   to 'territory_processing' before invoking this helper, signalling
 *   that all automatic bookkeeping for the turn has completed.
 */
export function advanceFromMovementBoundary(
  ctx: TurnLifecycleContext,
  deps: TurnLifecycleDeps
): TurnLifecycleContext {
  const { state, perTurn } = ctx;

  const { nextState, nextTurn } = advanceTurnAndPhase(state, perTurn, deps.delegates);

  const result: TurnLifecycleContext = {
    state: nextState,
    perTurn: nextTurn,
  };

  if (nextState.gameStatus === 'active' && isInteractivePhase(nextState.currentPhase)) {
    deps.hooks?.onStartInteractiveTurn?.(nextState, nextTurn);
  }

  return result;
}

/**
 * Start an interactive turn for the current player:
 *
 * - Resets per-turn flags.
 * - Applies forced-elimination when the player controls stacks but has
 *   no legal placements / movements / captures.
 * - Skips players who are completely out of material.
 * - Chooses an initial interactive phase for the first player in seat
 *   order who still has material.
 *
 * This is a higher-level convenience around {@link advanceTurnAndPhase}
 * for hosts that want to centralise start-of-turn behaviour in a shared
 * place rather than duplicating the “who can act?” loop.
 */
export function startInteractiveTurnForCurrentPlayer(
  ctx: TurnLifecycleContext,
  deps: TurnLifecycleDeps
): TurnLifecycleContext {
  let { state, perTurn } = ctx;

  // Reset per-turn flags at the start of any player's turn.
  perTurn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

  // We may need to iterate when successive forced eliminations or
  // material-less players require skipping. Bound by 2x players as a
  // conservative safety net.
  const maxIterations = state.players.length * 2;

  for (let i = 0; i < maxIterations; i++) {
    if (state.gameStatus !== 'active') {
      return { state, perTurn };
    }

    const beforePlayer = state.currentPlayer;
    const beforePhase = state.currentPhase;

    const { nextState, nextTurn } = advanceTurnAndPhase(state, perTurn, deps.delegates);

    // Detect whether a forced-elimination step occurred by checking for
    // a change in player/phase while the game remains active.
    const forcedEliminationOccurred =
      nextState.gameStatus === 'active' &&
      (nextState.currentPlayer !== beforePlayer || nextState.currentPhase !== beforePhase) &&
      beforePhase === 'territory_processing';

    if (forcedEliminationOccurred && deps.hooks?.onForcedElimination) {
      deps.hooks.onForcedElimination(nextState, beforePlayer);
    }

    state = nextState;
    perTurn = nextTurn;

    if (state.gameStatus !== 'active') {
      return { state, perTurn };
    }

    if (isInteractivePhase(state.currentPhase)) {
      deps.hooks?.onStartInteractiveTurn?.(state, perTurn);
      return { state, perTurn };
    }

    // Otherwise we landed in a non-interactive phase (e.g. another
    // round of territory_processing); loop and let the core sequencer
    // continue advancing until we either reach an interactive phase or
    // the game ends.
  }

  return { state, perTurn };
}
