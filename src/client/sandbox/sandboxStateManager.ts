/**
 * @fileoverview Sandbox State Manager - Centralized State Access and Mutation
 *
 * This module provides a centralized interface for accessing and modifying
 * game state in the ClientSandboxEngine. It handles:
 * - GameState getter/setter with defensive cloning
 * - Victory result and game end explanation access
 * - Per-turn state flag management (hasPlacedThisTurn, mustMoveFromStackKey, etc.)
 * - LPS tracking state access
 * - State serialization/deserialization helpers
 *
 * The module uses a hooks pattern to avoid circular dependencies and
 * maintain testability. State ownership remains with ClientSandboxEngine
 * while this module provides the access interface.
 *
 * @see CLIENT_SANDBOX_ENGINE_DECOMPOSITION_PLAN.md - Phase 1 extraction
 */

import type { BoardState, GameResult, GameState } from '../../shared/types/game';
import type { GameEndExplanation } from '../../shared/engine/gameEndExplanation';
import type { LpsTrackingState } from '../../shared/engine';
import type { SerializedGameState } from '../../shared/engine/contracts/serialization';
import { serializeGameState } from '../../shared/engine/contracts/serialization';

/**
 * Per-turn state flags that track the current turn's progression.
 * These are reset at the start of each player's turn.
 */
export interface PerTurnState {
  /** Whether the current player has placed a ring this turn */
  hasPlacedThisTurn: boolean;
  /** Position key where the player must move from after placing */
  mustMoveFromStackKey: string | undefined;
  /** Currently selected stack key for movement */
  selectedStackKey: string | undefined;
  /** Number of rings placed this turn (max 3) */
  ringsPlacedThisTurn: number;
  /** Position key where rings were placed this turn (all must go to same position) */
  placementPositionThisTurn: string | undefined;
  /** Whether there's a pending line reward elimination */
  pendingLineRewardElimination: boolean;
  /** Whether there's a pending territory self-elimination */
  pendingTerritorySelfElimination: boolean;
}

/**
 * Create initial per-turn state with default values.
 */
export function createInitialPerTurnState(): PerTurnState {
  return {
    hasPlacedThisTurn: false,
    mustMoveFromStackKey: undefined,
    selectedStackKey: undefined,
    ringsPlacedThisTurn: 0,
    placementPositionThisTurn: undefined,
    pendingLineRewardElimination: false,
    pendingTerritorySelfElimination: false,
  };
}

/**
 * Reset per-turn state for a new turn.
 */
export function resetPerTurnState(): PerTurnState {
  return createInitialPerTurnState();
}

/**
 * Deep-clone a BoardState for defensive copies.
 *
 * @param board - The BoardState to clone
 * @returns A deep copy with cloned Map/array fields
 */
export function cloneBoardState(board: BoardState): BoardState {
  return {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };
}

/**
 * Create a defensive snapshot of a GameState.
 *
 * Unlike simple spread, this deep-clones the board's Map/array fields
 * so that parity/debug tooling (and any callers holding onto past
 * snapshots) see stable pre/post views rather than aliases that are
 * mutated by subsequent moves.
 *
 * @param state - The GameState to snapshot
 * @returns A defensive copy with cloned Map/array fields
 */
export function createGameStateSnapshot(state: GameState): GameState {
  const clonedBoard = cloneBoardState(state.board);

  return {
    ...state,
    board: clonedBoard,
    moveHistory: [...state.moveHistory],
    history: [...state.history],
    players: state.players.map((p) => ({ ...p })),
    spectators: [...state.spectators],
  };
}

/**
 * Get a serialized snapshot of the current game state.
 * Used for saving custom scenarios and exporting game state.
 *
 * @param state - The GameState to serialize
 * @returns SerializedGameState suitable for JSON storage
 */
export function getSerializedState(state: GameState): SerializedGameState {
  return serializeGameState(state);
}

/**
 * Extract LPS tracking state for UI display.
 *
 * @param lpsState - The internal LPS tracking state
 * @returns A subset of LPS state suitable for UI consumption
 */
export function getLpsTrackingStateForUI(lpsState: LpsTrackingState): {
  roundIndex: number;
  consecutiveExclusiveRounds: number;
  consecutiveExclusivePlayer: number | null;
  exclusivePlayerForCompletedRound: number | null;
} {
  return {
    roundIndex: lpsState.roundIndex,
    consecutiveExclusiveRounds: lpsState.consecutiveExclusiveRounds,
    consecutiveExclusivePlayer: lpsState.consecutiveExclusivePlayer,
    exclusivePlayerForCompletedRound: lpsState.exclusivePlayerForCompletedRound,
  };
}

/**
 * Validate that a GameState is in a consistent state.
 * Used for debugging and testing.
 *
 * @param state - The GameState to validate
 * @returns Array of error messages, empty if valid
 */
export function validateGameStateConsistency(state: GameState): string[] {
  const errors: string[] = [];
  const board = state.board;

  // Invariant 1: no stacks may exist on collapsed territory.
  for (const key of board.stacks.keys()) {
    if (board.collapsedSpaces.has(key)) {
      errors.push(`stack present on collapsed space at ${key}`);
    }
  }

  // Invariant 2: a cell may not host both a stack and a marker.
  for (const key of board.stacks.keys()) {
    if (board.markers.has(key)) {
      errors.push(`stack and marker coexist at ${key}`);
    }
  }

  // Invariant 3: a cell may not host both a marker and collapsed territory.
  for (const key of board.markers.keys()) {
    if (board.collapsedSpaces.has(key)) {
      errors.push(`marker present on collapsed space at ${key}`);
    }
  }

  return errors;
}

/**
 * Interface for sandbox engine state accessors.
 * Provides a unified API for accessing all sandbox state.
 */
export interface SandboxStateAccessors {
  /** Get a defensive copy of the current game state */
  getGameState(): GameState;

  /** Get the victory result (null if game not ended) */
  getVictoryResult(): GameResult | null;

  /** Get the game end explanation (null if game not ended) */
  getGameEndExplanation(): GameEndExplanation | null;

  /** Get the current per-turn state flags */
  getPerTurnState(): PerTurnState;

  /** Get the LPS tracking state for UI */
  getLpsTrackingState(): {
    roundIndex: number;
    consecutiveExclusiveRounds: number;
    consecutiveExclusivePlayer: number | null;
    exclusivePlayerForCompletedRound: number | null;
  };

  /** Get a serialized snapshot of the current game state */
  getSerializedState(): SerializedGameState;
}

/**
 * Interface for sandbox engine state mutators.
 * Provides a unified API for modifying sandbox state.
 */
export interface SandboxStateMutators {
  /** Update the game state */
  updateGameState(state: GameState): void;

  /** Set the victory result */
  setVictoryResult(result: GameResult | null): void;

  /** Set the game end explanation */
  setGameEndExplanation(explanation: GameEndExplanation | null): void;

  /** Update per-turn state flags */
  updatePerTurnState(partial: Partial<PerTurnState>): void;

  /** Reset per-turn state for a new turn */
  resetPerTurnState(): void;
}

/**
 * Combined interface for accessing and modifying sandbox state.
 */
export interface SandboxStateManager extends SandboxStateAccessors, SandboxStateMutators {}

/**
 * Check if the current phase is an interactive turn phase (placement, movement, capture).
 *
 * @param phase - The current game phase
 * @returns Whether this is an interactive phase
 */
export function isInteractivePhase(phase: GameState['currentPhase']): boolean {
  return (
    phase === 'ring_placement' ||
    phase === 'movement' ||
    phase === 'capture' ||
    phase === 'chain_capture'
  );
}

/**
 * Check if the game is in an active (playable) state.
 *
 * @param state - The GameState to check
 * @returns Whether the game is active and can accept moves
 */
export function isGameActive(state: GameState): boolean {
  return state.gameStatus === 'active';
}

/**
 * Get the current player number from the game state.
 *
 * @param state - The GameState to check
 * @returns The current player's number (1-based)
 */
export function getCurrentPlayer(state: GameState): number {
  return state.currentPlayer;
}

/**
 * Check if a specific player is human.
 *
 * @param state - The GameState to check
 * @param playerNumber - The player number to check (1-based)
 * @returns Whether the player is human
 */
export function isHumanPlayer(state: GameState, playerNumber: number): boolean {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  return player?.type === 'human';
}

/**
 * Check if a specific player is AI.
 *
 * @param state - The GameState to check
 * @param playerNumber - The player number to check (1-based)
 * @returns Whether the player is AI
 */
export function isAIPlayer(state: GameState, playerNumber: number): boolean {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  return player?.type === 'ai';
}

/**
 * Get the next player number in turn order.
 *
 * @param state - The GameState
 * @param current - Current player number (1-based)
 * @returns Next player number (1-based)
 */
export function getNextPlayerNumber(state: GameState, current: number): number {
  const players = state.players;
  const idx = players.findIndex((p) => p.playerNumber === current);
  const nextIdx = (idx + 1) % players.length;
  return players[nextIdx].playerNumber;
}
