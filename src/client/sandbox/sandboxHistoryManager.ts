/**
 * @fileoverview Sandbox History Manager - History Entry and Snapshot Management
 *
 * This module manages game history entries and state snapshots for the
 * ClientSandboxEngine. It handles:
 * - Creating and appending history entries for moves
 * - Managing state snapshots for history playback
 * - Reconstructing snapshots from move history for loaded fixtures
 * - Providing state access at specific move indices
 *
 * The module uses a hooks pattern to avoid circular dependencies and
 * maintain testability. All state is owned by ClientSandboxEngine and
 * accessed via the provided hooks interface.
 *
 * @see CLIENT_SANDBOX_ENGINE_DECOMPOSITION_PLAN.md - Phase 1 extraction
 */

import type { BoardState, GameState, Move } from '../../shared/types/game';
import { createHistoryEntry } from '../../shared/engine';
import { createInitialGameState } from '../../shared/engine/initialState';
import { applyMoveForReplay } from '../../shared/engine/orchestration/turnOrchestrator';
import { BOARD_CONFIGS } from '../../shared/engine';

/**
 * Interface for accessing and modifying game state from the sandbox engine.
 * This allows the history manager to operate without direct state ownership.
 */
export interface HistoryManagerHooks {
  /** Get the current GameState snapshot */
  getGameState(): GameState;

  /** Update the GameState (used for history entry recording) */
  updateGameState(state: GameState): void;

  /** Get the array of state snapshots after each move */
  getStateSnapshots(): GameState[];

  /** Set the state snapshots array */
  setStateSnapshots(snapshots: GameState[]): void;

  /** Get the initial state snapshot (before any moves) */
  getInitialStateSnapshot(): GameState | null;

  /** Set the initial state snapshot */
  setInitialStateSnapshot(snapshot: GameState | null): void;
}

/**
 * Options for appendHistoryEntry
 */
export interface HistoryEntryOptions {
  /** When true, skip updating moveHistory (use when orchestrator already added it) */
  skipMoveHistory?: boolean;
}

/**
 * Deep-clone a GameState for safe history viewing.
 * Similar to getGameState() but operates on any state object.
 *
 * @param state - The GameState to clone
 * @returns A deep copy with cloned Map/array fields
 */
export function cloneGameState(state: GameState): GameState {
  const board = state.board;

  const clonedBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };

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
 * Append a structured history entry for a canonical move applied to the
 * sandbox game state. This mirrors the backend GameEngine
 * appendHistoryEntry but runs entirely client-side.
 *
 * @param hooks - Interface for accessing/updating state
 * @param before - GameState before the move was applied
 * @param action - The Move that was applied
 * @param opts - Optional settings for history recording
 */
export function appendHistoryEntry(
  hooks: HistoryManagerHooks,
  before: GameState,
  action: Move,
  opts?: HistoryEntryOptions
): void {
  const after = hooks.getGameState();

  // Use the shared helper to create a consistent history entry.
  // normalizeMoveNumber ensures sandbox history uses a contiguous 1..N
  // sequence regardless of how callers populated Move.moveNumber.
  const entry = createHistoryEntry(before, after, action, {
    normalizeMoveNumber: true,
  });

  // Capture state snapshots for history playback.
  // For the first move, also capture the initial state (before any moves).
  const currentHistory = after.history;
  if (currentHistory.length === 0) {
    hooks.setInitialStateSnapshot(cloneGameState(before));
  }

  // Capture the "after" state for this move.
  const snapshots = hooks.getStateSnapshots();
  hooks.setStateSnapshots([...snapshots, cloneGameState(after)]);

  const newState: GameState = {
    ...after,
    // Keep moveHistory in sync with canonical actions so board
    // animation hooks (useAutoMoveAnimation) can observe new moves
    // in both backend and sandbox hosts.
    // When skipMoveHistory is true, the orchestrator adapter has already
    // added the move to moveHistory, so we don't add it again.
    moveHistory: opts?.skipMoveHistory ? after.moveHistory : [...after.moveHistory, action],
    history: [...after.history, entry],
  };

  hooks.updateGameState(newState);
}

/**
 * Record history playback snapshots for an action that was already appended
 * to `moveHistory`/`history` by the SandboxOrchestratorAdapter.
 *
 * This avoids double-recording GameHistoryEntry rows (which inflates
 * `historyLength` and produces odd/even moveNumber gaps in exported fixtures),
 * while still keeping `_stateSnapshots` aligned for HistoryPlayback UX.
 *
 * @param hooks - Interface for accessing/updating state
 * @param before - GameState before the move was applied
 */
export function recordHistorySnapshotsOnly(hooks: HistoryManagerHooks, before: GameState): void {
  const afterSnapshot = cloneGameState(hooks.getGameState());

  const snapshots = hooks.getStateSnapshots();
  if (snapshots.length === 0) {
    hooks.setInitialStateSnapshot(cloneGameState(before));
  }

  hooks.setStateSnapshots([...snapshots, afterSnapshot]);
}

/**
 * Get the game state at a specific move index for history playback.
 *
 * @param hooks - Interface for accessing state
 * @param moveIndex - The move index to retrieve state for:
 *   - 0 = initial state (before any moves)
 *   - N = state after move N (1-indexed moves)
 *   - Total moves = current/final state
 * @returns The GameState at that point, or null if unavailable
 *
 * This method extracts state from the history entries which contain
 * before/after snapshots for each move. For fixtures loaded with
 * pre-existing move history, historical states are only available
 * if the history entries contain the snapshots.
 */
export function getStateAtMoveIndex(
  hooks: HistoryManagerHooks,
  moveIndex: number
): GameState | null {
  const currentState = hooks.getGameState();
  const history = currentState.history;
  const totalMoves = history.length;

  // If at or beyond total moves, return current state
  if (moveIndex >= totalMoves) {
    return cloneGameState(currentState);
  }

  // If negative, invalid
  if (moveIndex < 0) {
    return null;
  }

  // For index 0 (initial state), return the initial state snapshot
  if (moveIndex === 0) {
    const initialSnapshot = hooks.getInitialStateSnapshot();
    if (initialSnapshot) {
      return cloneGameState(initialSnapshot);
    }
    // No initial snapshot - can't determine initial state for pre-loaded fixtures
    return null;
  }

  // For index N (1..totalMoves-1), return snapshot N-1's state
  // _stateSnapshots[0] = state after move 1
  // _stateSnapshots[N-1] = state after move N
  // So for moveIndex = N, we need _stateSnapshots[N-1]
  const snapshotIndex = moveIndex - 1;
  const snapshots = hooks.getStateSnapshots();
  if (snapshotIndex >= 0 && snapshotIndex < snapshots.length) {
    return cloneGameState(snapshots[snapshotIndex]);
  }

  // Snapshot not available (e.g., for pre-loaded fixtures without snapshots)
  return null;
}

/**
 * Rebuild state snapshots from a game's move history.
 *
 * When loading a fixture/saved state that has move history, this method
 * reconstructs the intermediate game states by replaying all moves from
 * a fresh initial state. This enables history playback for loaded games.
 *
 * @param hooks - Interface for accessing/updating state
 * @param finalState - The final game state (after all moves)
 */
export function rebuildSnapshotsFromMoveHistory(
  hooks: HistoryManagerHooks,
  finalState: GameState
): void {
  const moveHistory = finalState.moveHistory;

  // No moves to replay - nothing to do
  if (!moveHistory || moveHistory.length === 0) {
    return;
  }

  try {
    // Create a fresh initial state matching the fixture's configuration.
    // We use the final state's metadata (boardType, players, timeControl, etc.)
    // but with an empty board and no moves.
    const initialState = createInitialGameState(
      finalState.id,
      finalState.boardType,
      // Reset player state to initial values
      finalState.players.map((p) => ({
        ...p,
        ringsInHand: BOARD_CONFIGS[finalState.boardType].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      })),
      finalState.timeControl,
      finalState.isRated,
      finalState.rngSeed,
      finalState.rulesOptions
    );

    // Mark the game as active (createInitialGameState sets it to 'waiting')
    let currentState: GameState = {
      ...initialState,
      gameStatus: 'active',
    };

    // Store the initial state snapshot (state before any moves)
    hooks.setInitialStateSnapshot(cloneGameState(currentState));

    // Replay each move and capture snapshots
    const newSnapshots: GameState[] = [];
    for (const move of moveHistory) {
      try {
        const result = applyMoveForReplay(currentState, move);
        currentState = result.nextState;

        // Add the move to the replayed state's history
        currentState = {
          ...currentState,
          moveHistory: [...currentState.moveHistory, move],
        };

        // Capture snapshot after this move
        newSnapshots.push(cloneGameState(currentState));
      } catch (err) {
        // If a move fails to apply, stop reconstruction but keep what we have
        if (process.env.NODE_ENV === 'development') {
          console.warn(
            '[sandboxHistoryManager] Failed to apply move during snapshot reconstruction:',
            move,
            err
          );
        }
        break;
      }
    }

    hooks.setStateSnapshots(newSnapshots);

    if (process.env.NODE_ENV === 'development') {
      // eslint-disable-next-line no-console
      console.log(
        `[sandboxHistoryManager] Rebuilt ${newSnapshots.length} snapshots from ${moveHistory.length} moves`
      );
    }
  } catch (err) {
    // If initial state creation fails, leave snapshots empty
    if (process.env.NODE_ENV === 'development') {
      console.warn('[sandboxHistoryManager] Failed to rebuild snapshots from move history:', err);
    }
  }
}

/**
 * Clear all history snapshots.
 * Used when resetting the engine state (e.g., when loading a new scenario).
 *
 * @param hooks - Interface for accessing/updating state
 */
export function clearHistorySnapshots(hooks: HistoryManagerHooks): void {
  hooks.setStateSnapshots([]);
  hooks.setInitialStateSnapshot(null);
}

/**
 * Get the total number of moves in the history.
 *
 * @param hooks - Interface for accessing state
 * @returns The number of moves recorded
 */
export function getHistoryLength(hooks: HistoryManagerHooks): number {
  return hooks.getGameState().history.length;
}
