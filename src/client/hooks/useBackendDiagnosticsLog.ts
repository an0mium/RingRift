/**
 * @fileoverview useBackendDiagnosticsLog Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for backend game diagnostics.
 * It manages event logging state for debugging, not rules logic.
 *
 * Canonical SSoT:
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - Weird state reasons: `src/shared/engine/weirdStateReasons.ts`
 *
 * This adapter:
 * - Produces a rolling log of phase/player/choice transitions
 * - Tracks connection-status events
 * - Logs decision auto-resolution events
 * - Logs weird-state (ANM, FE, stalemate, LPS) diagnostics
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/BACKEND_GAME_HOST_DECOMPOSITION_PLAN.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useEffect, useRef } from 'react';
import type { GameState, GameResult, PlayerChoice } from '../../shared/types/game';
import type {
  DecisionAutoResolvedMeta,
  DecisionPhaseTimeoutWarningPayload,
} from '../../shared/types/websocket';
import type { ConnectionStatus } from '../contexts/GameContext';
import { getWeirdStateBanner } from '../utils/gameStateWeirdness';

/**
 * State returned by the useBackendDiagnosticsLog hook.
 */
export interface BackendDiagnosticsState {
  /** Rolling event log entries (newest first) */
  eventLog: string[];
  /** Whether to show system events in addition to move events */
  showSystemEventsInLog: boolean;
  /** Toggle system events visibility */
  setShowSystemEventsInLog: React.Dispatch<React.SetStateAction<boolean>>;
}

/**
 * Context for tracking forced elimination sequences.
 */
interface ForcedElimContext {
  active: boolean;
  startTotal: number;
  playerNumber: number | null;
}

/**
 * Describe a decision auto-resolved event in human-readable format.
 *
 * @param meta - The auto-resolved decision metadata
 * @returns Human-readable description
 */
export function describeDecisionAutoResolved(meta: DecisionAutoResolvedMeta): string {
  const playerLabel = `P${meta.actingPlayerNumber}`;
  const reasonLabel = meta.reason === 'timeout' ? 'timeout' : meta.reason.replace(/_/g, ' ');

  const choiceKindLabel = (() => {
    switch (meta.choiceKind) {
      case 'line_order':
        return 'line order';
      case 'line_reward':
        return 'line reward';
      case 'ring_elimination':
        return 'ring elimination';
      case 'territory_region_order':
        return 'territory region order';
      case 'capture_direction':
        return 'capture direction';
      default:
        return meta.choiceKind.replace(/_/g, ' ');
    }
  })();

  const movePart = meta.resolvedMoveId ? ` (moveId: ${meta.resolvedMoveId})` : '';

  return `Decision auto-resolved for ${playerLabel}: ${choiceKindLabel} (reason: ${reasonLabel})${movePart}`;
}

/**
 * Custom hook for managing backend game diagnostics logging.
 *
 * Produces a rolling log of:
 * - Phase/player/choice transitions
 * - Connection status changes
 * - Decision auto-resolution events
 * - Decision timeout warnings
 * - Weird-state (ANM, forced elimination, structural stalemate, LPS) events
 *
 * Extracted from BackendGameHost to reduce component complexity.
 *
 * @param gameState - Current game state from backend
 * @param pendingChoice - Current pending choice (if any)
 * @param connectionStatus - WebSocket connection status
 * @param decisionAutoResolved - Auto-resolved decision metadata (if any)
 * @param decisionPhaseTimeoutWarning - Timeout warning payload (if any)
 * @param victoryState - Victory result (if game ended)
 * @returns Object with event log and system events toggle
 */
export function useBackendDiagnosticsLog(
  gameState: GameState | null,
  pendingChoice: PlayerChoice | null,
  connectionStatus: ConnectionStatus,
  decisionAutoResolved: DecisionAutoResolvedMeta | null,
  decisionPhaseTimeoutWarning: DecisionPhaseTimeoutWarningPayload | null,
  victoryState: GameResult | null
): BackendDiagnosticsState {
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [showSystemEventsInLog, setShowSystemEventsInLog] = useState(true);

  const lastPhaseRef = useRef<string | null>(null);
  const lastCurrentPlayerRef = useRef<number | null>(null);
  const lastChoiceIdRef = useRef<string | null>(null);
  const lastAutoResolvedKeyRef = useRef<string | null>(null);
  const lastConnectionStatusRef = useRef<ConnectionStatus | null>(null);
  const lastTimeoutWarningKeyRef = useRef<string | null>(null);
  const lastWeirdStateTypeRef = useRef<string | null>(null);
  const forcedElimContextRef = useRef<ForcedElimContext | null>(null);

  // Phase / current player / choice transitions
  useEffect(() => {
    if (!gameState) {
      lastPhaseRef.current = null;
      lastCurrentPlayerRef.current = null;
      return;
    }

    const events: string[] = [];

    if (gameState.currentPhase !== lastPhaseRef.current) {
      if (lastPhaseRef.current !== null) {
        events.push(`Phase changed: ${lastPhaseRef.current} → ${gameState.currentPhase}`);
      } else {
        events.push(`Phase: ${gameState.currentPhase}`);
      }
      lastPhaseRef.current = gameState.currentPhase;
    }

    if (gameState.currentPlayer !== lastCurrentPlayerRef.current) {
      events.push(`Current player: P${gameState.currentPlayer}`);
      lastCurrentPlayerRef.current = gameState.currentPlayer;
    }

    if (pendingChoice && pendingChoice.id !== lastChoiceIdRef.current) {
      events.push(`Choice requested: ${pendingChoice.type} for P${pendingChoice.playerNumber}`);
      lastChoiceIdRef.current = pendingChoice.id;
    } else if (!pendingChoice && lastChoiceIdRef.current) {
      events.push('Choice resolved');
      lastChoiceIdRef.current = null;
    }

    if (events.length > 0) {
      setEventLog((prev) => {
        const next = [...events, ...prev];
        return next.slice(0, 50);
      });
    }
  }, [gameState, pendingChoice]);

  // Auto-resolved decision events
  useEffect(() => {
    if (!decisionAutoResolved) {
      return;
    }

    const { actingPlayerNumber, choiceKind, reason, resolvedMoveId } = decisionAutoResolved;
    const key = resolvedMoveId ?? `${actingPlayerNumber}:${choiceKind}:${reason}`;

    if (lastAutoResolvedKeyRef.current === key) {
      return;
    }

    lastAutoResolvedKeyRef.current = key;

    const label = describeDecisionAutoResolved(decisionAutoResolved);

    setEventLog((prev) => [label, ...prev].slice(0, 50));
  }, [decisionAutoResolved]);

  // Decision-phase timeout warning events
  useEffect(() => {
    if (!decisionPhaseTimeoutWarning) {
      return;
    }

    const { gameId, playerNumber, phase, remainingMs, choiceId } = decisionPhaseTimeoutWarning.data;

    const key = `${gameId}:${playerNumber}:${phase}:${choiceId ?? ''}:${remainingMs}`;
    if (lastTimeoutWarningKeyRef.current === key) {
      return;
    }
    lastTimeoutWarningKeyRef.current = key;

    const seconds = Math.max(1, Math.round(remainingMs / 1000));
    const label = `Decision timeout warning: P${playerNumber} in ${phase} (~${seconds}s remaining)`;

    setEventLog((prev) => [label, ...prev].slice(0, 50));
  }, [decisionPhaseTimeoutWarning]);

  // Connection status changes
  useEffect(() => {
    if (!connectionStatus || lastConnectionStatusRef.current === connectionStatus) {
      lastConnectionStatusRef.current = connectionStatus;
      return;
    }

    const label =
      connectionStatus === 'connected'
        ? 'Connection restored'
        : connectionStatus === 'reconnecting'
          ? 'Connection interrupted – reconnecting'
          : connectionStatus === 'connecting'
            ? 'Connecting to server…'
            : 'Disconnected from server';

    setEventLog((prev) => [label, ...prev].slice(0, 50));
    lastConnectionStatusRef.current = connectionStatus;
  }, [connectionStatus]);

  // Weird-state (ANM / forced elimination / structural stalemate / LPS) diagnostics.
  useEffect(() => {
    if (!gameState) {
      lastWeirdStateTypeRef.current = null;
      forcedElimContextRef.current = null;
      return;
    }

    const weird = getWeirdStateBanner(gameState, { victoryState });
    const prevType = lastWeirdStateTypeRef.current;
    const nextType = weird.type;

    const events: string[] = [];

    // Detect entry into ANM states (movement / line / territory).
    if (
      nextType === 'active-no-moves-movement' ||
      nextType === 'active-no-moves-line' ||
      nextType === 'active-no-moves-territory'
    ) {
      if (prevType !== nextType) {
        const phaseLabel =
          nextType === 'active-no-moves-movement'
            ? 'movement'
            : nextType === 'active-no-moves-line'
              ? 'line processing'
              : 'territory processing';
        const playerNumber = (weird as Extract<typeof weird, { playerNumber: number }>)
          .playerNumber;
        events.push(
          `Active–No–Moves detected for P${playerNumber} during ${phaseLabel}; the engine will apply forced resolution according to the rulebook.`
        );
      }
    }

    // Detect start and completion of forced elimination sequences.
    if (nextType === 'forced-elimination') {
      if (prevType !== 'forced-elimination') {
        const playerNumber = (weird as Extract<typeof weird, { playerNumber: number }>)
          .playerNumber;
        const startTotal =
          (gameState as GameState & { totalRingsEliminated?: number }).totalRingsEliminated ?? 0;
        forcedElimContextRef.current = {
          active: true,
          startTotal,
          playerNumber,
        };
        events.push(`Forced elimination sequence started for P${playerNumber}.`);
      }
    } else if (prevType === 'forced-elimination') {
      const ctx = forcedElimContextRef.current;
      const endTotal =
        (gameState as GameState & { totalRingsEliminated?: number }).totalRingsEliminated ?? 0;
      if (ctx) {
        const delta = Math.max(0, endTotal - ctx.startTotal);
        const playerLabel = ctx.playerNumber ? `P${ctx.playerNumber}` : 'the active player';
        const detail =
          delta > 0
            ? `${delta} ring${delta === 1 ? '' : 's'} were eliminated during the forced elimination sequence.`
            : 'No additional rings were eliminated during the forced elimination sequence.';
        events.push(`Forced elimination sequence completed for ${playerLabel}. ${detail}`);
      } else {
        events.push('Forced elimination sequence completed.');
      }
      forcedElimContextRef.current = null;
    }

    // Detect structural stalemate / plateau end conditions.
    if (nextType === 'structural-stalemate' && prevType !== 'structural-stalemate') {
      events.push(
        'Structural stalemate: no legal placements, movements, captures, or forced eliminations remain. The game ended by plateau auto-resolution.'
      );
    }

    // Detect last-player-standing terminal condition.
    if (nextType === 'last-player-standing' && prevType !== 'last-player-standing') {
      const winner = (weird as Extract<typeof weird, { winner?: number }>).winner;
      const label = typeof winner === 'number' ? `P${winner}` : 'A player';
      events.push(
        `Last Player Standing: ${label} won after three complete rounds where only they had real moves available.`
      );
    }

    if (events.length > 0) {
      setEventLog((prev) => {
        const merged = [...events, ...prev];
        return merged.slice(0, 50);
      });
    }

    lastWeirdStateTypeRef.current = nextType;
  }, [gameState, victoryState]);

  return {
    eventLog,
    showSystemEventsInLog,
    setShowSystemEventsInLog,
  };
}

export default useBackendDiagnosticsLog;
