import type { GameState, Position } from '../types/game';
import { flagEnabled, debugLog } from '../utils/envFlags';

/**
 * Shared, host-agnostic helpers for turn/phase progression.
 *
 * This module centralises the minimal state machine that decides:
 *
 * - How phases advance within a player's turn
 *   (ring_placement → movement/capture → chain_capture → line_processing → territory_processing → forced_elimination).
 * - When the active player changes and which phase the next player starts in.
 * - When forced elimination is triggered for a player who controls stacks but has
 *   no legal placement / movement / capture actions (7th phase per RR-CANON-R070).
 *
 * The concrete rules for:
 * - Which placements / movements / captures are legal,
 * - How forced elimination is applied (which stack, how many rings), and
 * - How victory is detected after forced elimination
 *
 * are all delegated to the host via {@link TurnLogicDelegates}. This keeps the
 * phase/turn sequencing identical across:
 *
 * - The backend server engine (GameEngine + TurnEngine),
 * - The client sandbox engine, and
 * - The shared reference engine used in parity tests.
 *
 * The semantics are derived from the backend TurnEngine and the rules
 * documented in ringrift_complete_rules.md §4 and §15.2, and
 * RULES_ENGINE_ARCHITECTURE.md (turn/phase orchestration).
 */
export interface PerTurnState {
  /**
   * True once the active player has performed a place_ring this turn.
   * Hosts may use this to enforce must-move semantics but the core
   * sequencer itself only propagates the flag.
   */
  hasPlacedThisTurn: boolean;
  /**
   * Optional board key (positionToString) of the stack that must move
   * this turn after a placement. Semantics are host-defined; the shared
   * sequencer simply preserves the value when rotating turns (it is
   * always reset for the next player).
   */
  mustMoveFromStackKey?: string | undefined;
}

/**
 * Delegates required by the shared turn/phase sequencer.
 *
 * Hosts provide small, testable adapters that answer questions about
 * available actions and apply forced elimination when needed.
 */
export interface TurnLogicDelegates {
  /**
   * Return all stacks controlled by the specified player. Only
   * position and stackHeight are required by the sequencer; hosts are
   * free to attach richer objects.
   */
  getPlayerStacks(
    state: GameState,
    player: number
  ): Array<{
    position: Position;
    stackHeight: number;
  }>;

  /**
   * True when the player has at least one legal ring placement in the
   * current state (respecting any host-specific rules such as
   * no-dead-placement and per-player caps).
   */
  hasAnyPlacement(state: GameState, player: number): boolean;

  /**
   * True when the player has at least one non-capturing movement
   * available in the current state, respecting any per-turn
   * constraints encoded in the provided {@link PerTurnState}.
   */
  hasAnyMovement(state: GameState, player: number, turn: PerTurnState): boolean;

  /**
   * True when the player has at least one overtaking capture available
   * in the current state, respecting any per-turn constraints encoded
   * in the provided {@link PerTurnState}.
   */
  hasAnyCapture(state: GameState, player: number, turn: PerTurnState): boolean;

  /**
   * Apply forced elimination for the specified player when they control
   * at least one stack but have no legal placement/movement/capture
   * actions (compact rules §4.4 / FAQ Q24).
   *
   * Implementations may mutate and/or clone the provided state but must
   * return the updated instance that should continue through the state
   * machine. They are also responsible for running any victory checks
   * and updating gameStatus / winner as appropriate.
   */
  applyForcedElimination(state: GameState, player: number): GameState;

  /**
   * Compute the next player's number in turn order after {@code current}.
   *
   * This keeps the sequencer agnostic to specific seat ordering
   * conventions (e.g. gaps in playerNumber or arbitrary seating maps).
   */
  getNextPlayerNumber(state: GameState, current: number): number;

  /**
   * True when the player has at least one ring anywhere on the board or in hand.
   * This includes:
   * - Rings in controlled stacks (top ring is player's colour)
   * - Rings buried inside stacks controlled by other players
   * - Rings in hand (not yet placed)
   *
   * A player with no rings anywhere is permanently eliminated and should
   * be skipped in turn rotation. A player who has rings (even if only
   * buried) may still be recovery-eligible and should receive turns.
   */
  playerHasAnyRings(state: GameState, player: number): boolean;
}

export interface TurnAdvanceResult {
  /** New GameState after applying the phase/turn transition. */
  nextState: GameState;
  /** Updated per-turn context for the (new) active player. */
  nextTurn: PerTurnState;
}

/**
 * Advance {@link GameState.currentPhase} and {@link GameState.currentPlayer}
 * according to the canonical RingRift turn/phase rules.
 *
 * Usage contract:
 *
 * - Call this helper *after* applying a canonical Move and any automatic
 *   geometric consequences (line detection, territory detection, etc.).
 * - The helper does **not** perform geometry, move validation, or victory
 *   evaluation; those remain responsibilities of the host engine.
 * - When a forced-elimination step is required, the helper invokes
 *   {@link TurnLogicDelegates.applyForcedElimination} and then, if the game
 *   is still active, resumes the state machine from the updated state.
 *
 * The implementation deliberately avoids mutating the input {@code state};
 * instead it returns a shallow-cloned {@code nextState} with updated
 * phase/player fields. Hosts that maintain a single mutable GameState
 * instance (e.g. the backend GameEngine) can selectively copy those
 * top-level fields back onto their internal object.
 */
export function advanceTurnAndPhase(
  state: GameState,
  turn: PerTurnState,
  delegates: TurnLogicDelegates
): TurnAdvanceResult {
  // If the game is no longer active, leave phase/player unchanged.
  if (state.gameStatus !== 'active') {
    return {
      nextState: state,
      nextTurn: turn,
    };
  }

  const cloneState = (patch: Partial<GameState>): GameState => ({
    ...state,
    ...patch,
  });

  // Default: carry the existing per-turn context through unchanged.
  let nextTurn: PerTurnState = { ...turn };
  let nextState: GameState = state;

  switch (state.currentPhase) {
    case 'ring_placement': {
      // Per RR-CANON-R075: All phases must be visited with explicit moves.
      // Always enter movement phase - if no moves exist, the player must
      // emit no_movement_action which then advances to line_processing.
      // No silent phase skipping is permitted.
      nextState = cloneState({ currentPhase: 'movement' });
      break;
    }

    case 'movement':
    case 'capture':
    case 'chain_capture': {
      // After movement/capture (including the final segment of a chain),
      // bookkeeping proceeds through line_processing.
      nextState = cloneState({ currentPhase: 'line_processing' });
      break;
    }

    case 'line_processing': {
      // After processing lines, proceed to territory processing.
      nextState = cloneState({ currentPhase: 'territory_processing' });
      break;
    }

    case 'territory_processing': {
      // Turn boundary: hand control to the next player in seat order,
      // then determine whether they must immediately pay a forced
      // elimination cost or can start a normal turn.
      const initialNextPlayer = delegates.getNextPlayerNumber(state, state.currentPlayer);

      let workingState: GameState = cloneState({ currentPlayer: initialNextPlayer });
      nextTurn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      const stacksForCurrent = delegates.getPlayerStacks(workingState, workingState.currentPlayer);
      const hasStacksForCurrent = stacksForCurrent.length > 0;

      const hasAnyActionForCurrent =
        delegates.hasAnyPlacement(workingState, workingState.currentPlayer) ||
        delegates.hasAnyMovement(workingState, workingState.currentPlayer, nextTurn) ||
        delegates.hasAnyCapture(workingState, workingState.currentPlayer, nextTurn);

      if (hasStacksForCurrent && !hasAnyActionForCurrent) {
        // Forced elimination when blocked with material but no legal
        // actions (compact rules §4.4 / FAQ Q24).
        workingState = delegates.applyForcedElimination(workingState, workingState.currentPlayer);

        if (workingState.gameStatus !== 'active') {
          // Host has ended the game; propagate terminal state without
          // selecting a new interactive phase.
          nextState = workingState;
          nextTurn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };
          break;
        }

        // Game continues: the same player remains active and now begins
        // an interactive turn in the movement phase.
        workingState = { ...workingState, currentPhase: 'movement' };
        nextState = workingState;
        nextTurn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };
        break;
      }

      // Normal turn progression. Only skip players who are permanently
      // eliminated (no rings anywhere - not in stacks, not buried, not in hand).
      const maxSkips = workingState.players.length;
      let skips = 0;

      while (skips < maxSkips) {
        const currentPlayerNumber = workingState.currentPlayer;
        const currentPlayer = workingState.players.find(
          (p) => p.playerNumber === currentPlayerNumber
        );

        if (!currentPlayer) {
          break;
        }

        const stacks = delegates.getPlayerStacks(workingState, currentPlayerNumber);
        const hasStacks = stacks.length > 0;

        // Check if player is permanently eliminated (no rings anywhere).
        // Players who have rings somewhere (even if only buried) get turns
        // because they may be recovery-eligible (RR-CANON-R201).
        const hasAnyRings = delegates.playerHasAnyRings(workingState, currentPlayerNumber);

        // Debug logging for parity investigation
        debugLog(
          flagEnabled('RINGRIFT_TRACE_DEBUG'),
          '[turnLogic.advanceTurnAndPhase.while] player=',
          currentPlayerNumber,
          'hasStacks=',
          hasStacks,
          'stackCount=',
          stacks.length,
          'ringsInHand=',
          currentPlayer.ringsInHand,
          'hasAnyRings=',
          hasAnyRings,
          'willSkip=',
          !hasAnyRings
        );

        if (!hasAnyRings) {
          // Player is permanently eliminated (no rings anywhere on board
          // or in hand). Skip them in turn rotation. This is different from
          // "no turn-material" - a player with buried rings but no stacks
          // and no rings in hand is NOT skipped because they may be
          // recovery-eligible.
          const nextPlayerNumber = delegates.getNextPlayerNumber(workingState, currentPlayerNumber);
          // Debug: log skip for parity debugging
          debugLog(
            flagEnabled('RINGRIFT_TRACE_DEBUG'),
            '[turnLogic.advanceTurnAndPhase] SKIP permanently eliminated player',
            currentPlayerNumber,
            'hasStacks=',
            hasStacks,
            'ringsInHand=',
            currentPlayer.ringsInHand,
            'hasAnyRings=',
            hasAnyRings,
            '-> next=',
            nextPlayerNumber
          );
          workingState = { ...workingState, currentPlayer: nextPlayerNumber };
          skips += 1;
          continue;
        }

        // All players always start in ring_placement phase. Players with
        // ringsInHand == 0 will emit no_placement_action and transition to
        // movement. This ensures consistent phase traversal per RR-CANON-R075
        // ("All phases must be visited with explicit moves").
        workingState = { ...workingState, currentPhase: 'ring_placement' };
        break;
      }

      nextState = workingState;
      nextTurn = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      debugLog(
        flagEnabled('RINGRIFT_TRACE_DEBUG'),
        '[turnLogic.advanceTurnAndPhase.territory_processing.after]',
        {
          nextPlayer: nextState.currentPlayer,
          nextPhase: nextState.currentPhase,
        }
      );

      break;
    }

    default: {
      // For any unrecognised phase (future extensions), leave the state
      // unchanged so hosts can layer additional semantics without
      // surprising the shared sequencer.
      nextState = state;
      nextTurn = turn;
      break;
    }
  }

  return { nextState, nextTurn };
}
