/**
 * Shared swap sides (pie rule) helpers for RingRift engine.
 *
 * The pie rule allows Player 2 to swap colours/seats with Player 1 immediately
 * after Player 1's first turn, as a balancing mechanism for 2-player games.
 *
 * These functions are pure and can be used by both server-side GameEngine
 * and client-side ClientSandboxEngine to determine swap eligibility.
 *
 * @module swapSidesHelpers
 *
 * Rule Reference: RR-CANON R180-R184 (Pie Rule / Swap Sides)
 */

import type { GameState, GamePhase } from '../types/game';

/**
 * Phases where swap_sides can be offered.
 *
 * Per RR-CANON R180-R184, the pie rule is only available at the start of
 * Player 2's first turn, which always begins in ring_placement.
 */
const SWAP_ELIGIBLE_PHASES: GamePhase[] = ['ring_placement'];

/**
 * Determine whether the current game state should offer a swap_sides
 * meta-move (pie rule) for Player 2.
 *
 * The swap_sides move is available exactly once, for Player 2, at the start
 * of their first interactive turn after Player 1 has completed a full turn.
 *
 * Gate conditions:
 * 1. rulesOptions.swapRuleEnabled is true
 * 2. Game is active
 * 3. Exactly 2 players
 * 4. Current player is Player 2
 * 5. Current phase is ring_placement
 * 6. At least one move from Player 1 exists
 * 7. No moves from Player 2 exist (other than potential swap_sides)
 * 8. No swap_sides move has been applied yet
 *
 * @param state - Current game state to evaluate
 * @returns true if swap_sides should be offered to Player 2
 *
 * @example
 * ```typescript
 * if (shouldOfferSwapSides(gameState)) {
 *   // Add swap_sides to valid moves for Player 2
 *   validMoves.push(createSwapSidesMove(2));
 * }
 * ```
 */
export function shouldOfferSwapSides(state: GameState): boolean {
  // 1. Config gating: swap_sides must be explicitly enabled
  if (!state.rulesOptions?.swapRuleEnabled) {
    return false;
  }

  // 2. Game must be active
  if (state.gameStatus !== 'active') {
    return false;
  }

  // 3. Only available in 2-player games
  if (state.players.length !== 2) {
    return false;
  }

  // 4. Only Player 2 can invoke swap_sides
  if (state.currentPlayer !== 2) {
    return false;
  }

  // 5. Only available in ring_placement
  if (!SWAP_ELIGIBLE_PHASES.includes(state.currentPhase)) {
    return false;
  }

  // 6-8. Check move history for P1 move, no P2 move, no prior swap
  const moveHistory = state.moveHistory;

  if (moveHistory.length === 0) {
    return false;
  }

  const hasSwapMove = moveHistory.some((m) => m.type === 'swap_sides');
  if (hasSwapMove) {
    return false;
  }

  const hasP1Move = moveHistory.some((m) => m.player === 1);
  const hasP2Move = moveHistory.some((m) => m.player === 2 && m.type !== 'swap_sides');

  return hasP1Move && !hasP2Move;
}

/**
 * Validate that a swap_sides move can be applied to the given state.
 *
 * This performs the same checks as shouldOfferSwapSides plus validates
 * the player number in the move matches the expected player (2).
 *
 * @param state - Current game state
 * @param playerNumber - Player number attempting the swap
 * @returns Object with valid flag and optional error message
 */
export function validateSwapSidesMove(
  state: GameState,
  playerNumber: number
): { valid: true } | { valid: false; reason: string } {
  if (!state.rulesOptions?.swapRuleEnabled) {
    return { valid: false, reason: 'swap_sides is disabled for this game' };
  }

  if (state.gameStatus !== 'active') {
    return { valid: false, reason: 'swap_sides is only available in active games' };
  }

  if (state.players.length !== 2) {
    return { valid: false, reason: 'swap_sides is only defined for 2-player games' };
  }

  if (state.currentPlayer !== playerNumber) {
    return { valid: false, reason: 'Only the active player may request swap_sides' };
  }

  if (playerNumber !== 2) {
    return { valid: false, reason: 'Only Player 2 may request swap_sides' };
  }

  if (!SWAP_ELIGIBLE_PHASES.includes(state.currentPhase)) {
    return {
      valid: false,
      reason: 'swap_sides is only available at the start of ring placement',
    };
  }

  const moveHistory = state.moveHistory;
  const hasP1Move = moveHistory.some((m) => m.player === 1);
  const hasP2Move = moveHistory.some((m) => m.player === 2 && m.type !== 'swap_sides');
  const hasSwapMove = moveHistory.some((m) => m.type === 'swap_sides');

  if (!hasP1Move || hasP2Move || hasSwapMove) {
    return {
      valid: false,
      reason: "swap_sides is only available immediately after Player 1's first turn",
    };
  }

  return { valid: true };
}

/**
 * Apply the swap_sides identity swap to player objects.
 *
 * This swaps the identity fields (id, username, type, rating, AI config)
 * between the players in seats 1 and 2 while keeping the seat numbers stable.
 * Board geometry and on-seat statistics (ringsInHand, eliminatedRings, etc.)
 * remain with the seats.
 *
 * @param players - Current player array (will be copied, not mutated)
 * @returns New player array with identities swapped
 */
export function applySwapSidesIdentitySwap<
  T extends {
    playerNumber: number;
    id: string;
    username: string;
    type: 'human' | 'ai';
    rating?: number;
    aiDifficulty?: number;
    aiProfile?: unknown;
  },
>(players: T[]): T[] {
  const p1 = players.find((p) => p.playerNumber === 1);
  const p2 = players.find((p) => p.playerNumber === 2);

  if (!p1 || !p2) {
    // Safety: if seats 1/2 don't exist, return unchanged
    return players;
  }

  return players.map((p) => {
    if (p.playerNumber === 1) {
      return {
        ...p,
        id: p2.id,
        username: p2.username,
        type: p2.type,
        rating: p2.rating,
        aiDifficulty: p2.aiDifficulty,
        aiProfile: p2.aiProfile,
      };
    }
    if (p.playerNumber === 2) {
      return {
        ...p,
        id: p1.id,
        username: p1.username,
        type: p1.type,
        rating: p1.rating,
        aiDifficulty: p1.aiDifficulty,
        aiProfile: p1.aiProfile,
      };
    }
    return p;
  }) as T[];
}
