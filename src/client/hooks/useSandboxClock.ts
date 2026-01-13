/**
 * @fileoverview useSandboxClock Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for sandbox UI state.
 * It manages game clock state, not rules logic.
 *
 * Canonical SSoT:
 * - Turn logic: `src/shared/engine/orchestration/turnOrchestrator.ts`
 *
 * This adapter:
 * - Initializes player times when clock is enabled
 * - Applies time increment when turns change
 * - Decrements the active player's clock each second
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import type { GameState } from '../../shared/types/game';

/**
 * Time control configuration for sandbox clock.
 */
export interface SandboxTimeControl {
  /** Initial time per player in milliseconds */
  initialTimeMs: number;
  /** Time increment added after each move in milliseconds */
  incrementMs: number;
}

/**
 * Default time control: 15 minutes + 10 second increment.
 */
export const DEFAULT_SANDBOX_TIME_CONTROL: SandboxTimeControl = {
  initialTimeMs: 15 * 60 * 1000, // 15 minutes
  incrementMs: 10 * 1000, // 10 seconds
};

/**
 * Return type for useSandboxClock hook.
 */
export interface UseSandboxClockReturn {
  /** Whether the clock is enabled for this game */
  clockEnabled: boolean;
  /** Enable or disable the clock */
  setClockEnabled: (enabled: boolean) => void;
  /** Current time control settings */
  timeControl: SandboxTimeControl;
  /** Set time control settings */
  setTimeControl: React.Dispatch<React.SetStateAction<SandboxTimeControl>>;
  /** Per-player time remaining in milliseconds (playerNumber → ms) */
  playerTimes: Record<number, number>;
  /** Reset all player times (e.g., when starting a new game) */
  resetPlayerTimes: () => void;
}

/**
 * Custom hook for managing sandbox game clock state and timing.
 *
 * Handles:
 * - Initializing player times when clock is enabled
 * - Applying time increment when turns change
 * - Decrementing the active player's clock each second
 *
 * @param gameState - The current game state (or null if no game active)
 * @returns Clock state and control functions
 */
export function useSandboxClock(gameState: GameState | null): UseSandboxClockReturn {
  // Clock enabled toggle - enabled by default so sandbox games show a timer
  const [clockEnabled, setClockEnabled] = useState(true);

  // Time control settings
  const [timeControl, setTimeControl] = useState<SandboxTimeControl>(DEFAULT_SANDBOX_TIME_CONTROL);

  // Per-player time remaining (playerNumber → ms remaining)
  const [playerTimes, setPlayerTimes] = useState<Record<number, number>>({});

  // Track when the current player's turn started (for clock decrement)
  const turnStartTimeRef = useRef<number | null>(null);

  // Track the previous current player to detect turn changes
  const prevCurrentPlayerRef = useRef<number | null>(null);

  // Reset player times (e.g., when toggling clock or starting new game)
  const resetPlayerTimes = useCallback(() => {
    setPlayerTimes({});
    turnStartTimeRef.current = null;
    prevCurrentPlayerRef.current = null;
  }, []);

  // Initialize player times when clock is enabled and game is active
  useEffect(() => {
    if (!clockEnabled || !gameState) {
      return;
    }

    // Initialize times for all players if not already set
    const needsInit = gameState.players.some((p) => playerTimes[p.playerNumber] === undefined);

    if (needsInit) {
      const initialTimes: Record<number, number> = {};
      gameState.players.forEach((p) => {
        initialTimes[p.playerNumber] = timeControl.initialTimeMs;
      });
      setPlayerTimes(initialTimes);
      turnStartTimeRef.current = Date.now();
      prevCurrentPlayerRef.current = gameState.currentPlayer;
    }
  }, [clockEnabled, gameState, timeControl.initialTimeMs, playerTimes]);

  // Handle turn changes: apply increment to the player who just moved
  useEffect(() => {
    if (!clockEnabled || !gameState) {
      return;
    }

    const currentPlayer = gameState.currentPlayer;
    const prevPlayer = prevCurrentPlayerRef.current;

    // Detect turn change
    if (prevPlayer !== null && prevPlayer !== currentPlayer && gameState.gameStatus === 'active') {
      // Apply increment to the player who just finished their turn
      setPlayerTimes((prev) => {
        const prevTime = prev[prevPlayer] ?? timeControl.initialTimeMs;
        // Deduct time spent on this turn before adding increment
        const elapsed = turnStartTimeRef.current ? Date.now() - turnStartTimeRef.current : 0;
        const timeAfterMove = Math.max(0, prevTime - elapsed);
        const timeWithIncrement = timeAfterMove + timeControl.incrementMs;
        return { ...prev, [prevPlayer]: timeWithIncrement };
      });

      // Reset turn start time for the new player
      turnStartTimeRef.current = Date.now();
    }

    prevCurrentPlayerRef.current = currentPlayer;
  }, [clockEnabled, gameState, timeControl.incrementMs, timeControl.initialTimeMs]);

  // Track base time at start of current turn (before elapsed deduction)
  const turnBaseTimeRef = useRef<number | null>(null);

  // Store base time when turn starts
  useEffect(() => {
    if (!clockEnabled || !gameState) {
      return;
    }

    const currentPlayer = gameState.currentPlayer;
    // When turn changes or game starts, capture the base time for this turn
    if (turnBaseTimeRef.current === null && playerTimes[currentPlayer] !== undefined) {
      turnBaseTimeRef.current = playerTimes[currentPlayer];
    }
  }, [clockEnabled, gameState, playerTimes]);

  // Reset base time ref when turn changes
  useEffect(() => {
    if (!clockEnabled || !gameState) {
      return;
    }

    const currentPlayer = gameState.currentPlayer;
    const prevPlayer = prevCurrentPlayerRef.current;

    if (prevPlayer !== null && prevPlayer !== currentPlayer) {
      // Turn changed - reset base time ref for next turn
      turnBaseTimeRef.current = null;
    }
  }, [clockEnabled, gameState]);

  // Force re-render every second to update display (without modifying playerTimes)
  const [, setTick] = useState(0);

  useEffect(() => {
    if (!clockEnabled || !gameState || gameState.gameStatus !== 'active') {
      return;
    }

    const interval = setInterval(() => {
      setTick((t) => t + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [clockEnabled, gameState?.gameStatus]);

  // Compute display times on-the-fly (don't mutate playerTimes during turn)
  const displayPlayerTimes = { ...playerTimes };
  if (
    clockEnabled &&
    gameState &&
    gameState.gameStatus === 'active' &&
    turnStartTimeRef.current !== null
  ) {
    const currentPlayer = gameState.currentPlayer;
    const baseTime =
      turnBaseTimeRef.current ?? playerTimes[currentPlayer] ?? timeControl.initialTimeMs;
    const elapsed = Date.now() - turnStartTimeRef.current;
    displayPlayerTimes[currentPlayer] = Math.max(0, baseTime - elapsed);
  }

  return {
    clockEnabled,
    setClockEnabled,
    timeControl,
    setTimeControl,
    playerTimes: displayPlayerTimes,
    resetPlayerTimes,
  };
}

export default useSandboxClock;
