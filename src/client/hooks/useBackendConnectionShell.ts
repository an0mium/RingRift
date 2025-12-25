/**
 * @fileoverview useBackendConnectionShell Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for backend game connection lifecycle.
 * It manages WebSocket connection state, not rules logic.
 *
 * Canonical SSoT:
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - Connection: `src/client/hooks/useGameConnection.ts`
 *
 * This adapter:
 * - Wraps useGameConnection and owns connect/disconnect lifecycle
 * - Manages early spectator detection before gameState is available
 * - Tracks connection status, errors, and heartbeat
 * - Provides reconnect functionality
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/BACKEND_GAME_HOST_DECOMPOSITION_PLAN.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useEffect } from 'react';
import { useGameConnection } from './useGameConnection';
import type { ConnectionStatus } from '../contexts/GameContext';

/**
 * State returned by the useBackendConnectionShell hook.
 */
export interface BackendConnectionShellState {
  /** The route game ID that was passed to the hook */
  routeGameId: string;
  /** The actual game ID from the connection (may differ from route) */
  gameId: string | null;
  /** Current connection status */
  connectionStatus: ConnectionStatus;
  /** Whether a connection attempt is in progress */
  isConnecting: boolean;
  /** Any error message from the connection */
  error: string | null;
  /** Timestamp of the last successful heartbeat */
  lastHeartbeatAt: number | null;
  /** Function to trigger a reconnection attempt */
  reconnect: () => void;
}

/**
 * Custom hook for managing backend game connection lifecycle.
 *
 * Wraps useGameConnection and owns the connect/disconnect lifecycle for a
 * specific :gameId route. Handles:
 * - Auto-connect when routeGameId changes
 * - Auto-disconnect on unmount or routeGameId change
 * - Reconnection functionality
 *
 * Extracted from BackendGameHost to reduce component complexity.
 *
 * @param routeGameId - The game ID from the route (e.g. /game/:gameId)
 * @returns Object with connection state and reconnect action
 */
export function useBackendConnectionShell(routeGameId: string): BackendConnectionShellState {
  const { gameId, status, isConnecting, error, lastHeartbeatAt, connectToGame, disconnect } =
    useGameConnection();

  useEffect(() => {
    if (!routeGameId) {
      disconnect();
      return;
    }

    void connectToGame(routeGameId);

    return () => {
      disconnect();
    };
  }, [routeGameId, connectToGame, disconnect]);

  return {
    routeGameId,
    gameId,
    connectionStatus: status,
    isConnecting,
    error,
    lastHeartbeatAt,
    reconnect: () => {
      if (!routeGameId) {
        return;
      }
      void connectToGame(routeGameId);
    },
  };
}

export default useBackendConnectionShell;
