/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MobileGameHUD Component
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * A compact, mobile-optimized HUD that surfaces essential game information
 * without overwhelming small screens. Designed to work alongside the touch
 * controls panel and responsive board.
 *
 * Key differences from full GameHUD:
 * - Single-row phase indicator instead of full-width card
 * - Collapsed player summary (expandable on tap)
 * - Timer and turn info always visible in a sticky header bar
 * - Victory conditions moved to a help button
 */

import React, { useState } from 'react';
import type { HUDViewModel, PlayerViewModel } from '../adapters/gameViewModels';
import type { TimeControl } from '../../shared/types/game';
import { getCountdownSeverity } from '../utils/countdown';
import { Tooltip } from './ui/Tooltip';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

export interface MobileGameHUDProps {
  viewModel: HUDViewModel;
  timeControl?: TimeControl;
  isLocalSandboxOnly?: boolean;
  onShowBoardControls?: () => void;
  /** Handler for tapping a player card (e.g. to show details) */
  onPlayerTap?: (player: PlayerViewModel) => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Components
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compact turn/phase indicator bar for mobile
 */
function MobilePhaseBar({
  phase,
  turnNumber,
  isMyTurn,
}: {
  phase: HUDViewModel['phase'];
  turnNumber: number;
  isMyTurn: boolean;
}) {
  return (
    <div
      className={`flex items-center justify-between px-3 py-2 rounded-lg ${phase.colorClass}`}
      data-testid="mobile-phase-bar"
    >
      <div className="flex items-center gap-2 min-w-0">
        {phase.icon && <span className="text-lg">{phase.icon}</span>}
        <span className="font-semibold text-sm text-white truncate">{phase.label}</span>
      </div>
      <div className="flex items-center gap-2 text-xs text-white/90">
        <span className="font-mono">Turn {turnNumber}</span>
        {isMyTurn && (
          <span className="px-1.5 py-0.5 rounded-full bg-white/20 text-[10px] font-semibold uppercase">
            Your turn
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Compact player summary row for mobile
 */
function MobilePlayerRow({ player, isExpanded }: { player: PlayerViewModel; isExpanded: boolean }) {
  const { ringStats, territorySpaces, aiInfo } = player;

  return (
    <div
      className={`flex items-center justify-between gap-2 px-2 py-1.5 rounded-lg transition-colors ${
        player.isCurrentPlayer ? 'bg-blue-900/40 border border-blue-500/50' : 'bg-slate-800/50'
      } ${player.isUserPlayer ? 'ring-1 ring-green-400/50' : ''}`}
    >
      {/* Player identity */}
      <div className="flex items-center gap-2 min-w-0 flex-1">
        <div className={`w-3 h-3 rounded-full shrink-0 ${player.colorClass}`} />
        <span className="text-xs font-medium text-slate-100 truncate">
          {player.isUserPlayer ? 'You' : player.username}
        </span>
        {aiInfo.isAI && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-slate-700 text-slate-300">AI</span>
        )}
        {player.isCurrentPlayer && (
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
        )}
      </div>

      {/* Quick stats */}
      <div className="flex items-center gap-3 text-[10px] text-slate-300">
        <span className="flex items-center gap-1">
          <span className="text-red-300">🔴</span>
          <span className="font-mono">
            {ringStats.eliminated}/{ringStats.total}
          </span>
        </span>
        <span className="flex items-center gap-1">
          <span className="text-emerald-300">🏰</span>
          <span className="font-mono">{territorySpaces}</span>
        </span>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] text-slate-300 pt-2 border-t border-slate-700">
          <div className="text-center">
            <div className="font-bold text-slate-100">{ringStats.inHand}</div>
            <div>In Hand</div>
          </div>
          <div className="text-center">
            <div className="font-bold text-slate-100">{ringStats.onBoard}</div>
            <div>On Board</div>
          </div>
          <div className="text-center">
            <div className="font-bold text-red-400">{ringStats.eliminated}</div>
            <div>Captured</div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Decision timer pill for mobile
 */
function MobileDecisionTimer({
  timeRemainingMs,
  isServerCapped,
}: {
  timeRemainingMs: number;
  isServerCapped?: boolean;
}) {
  const totalSeconds = Math.max(0, Math.floor(timeRemainingMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const timeLabel = `${minutes}:${seconds.toString().padStart(2, '0')}`;

  const severity = getCountdownSeverity(timeRemainingMs);

  const bgClass =
    severity === 'critical'
      ? 'bg-red-900/80 border-red-400/60'
      : severity === 'warning'
        ? 'bg-amber-900/80 border-amber-400/60'
        : 'bg-slate-800/80 border-slate-600';

  const textClass =
    severity === 'critical'
      ? 'text-red-100'
      : severity === 'warning'
        ? 'text-amber-100'
        : 'text-slate-100';

  return (
    <div
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-full border text-xs ${bgClass} ${
        severity === 'critical' ? 'animate-pulse' : ''
      }`}
      data-testid="mobile-decision-timer"
      data-severity={severity ?? undefined}
    >
      <span className="text-[10px]">⏱</span>
      <span className={`font-mono ${textClass}`}>{timeLabel}</span>
      {isServerCapped && <span className="text-[9px] text-amber-200">*</span>}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════════

/**
 * MobileGameHUD - Compact HUD optimized for mobile viewports
 *
 * Usage:
 * ```tsx
 * const isMobile = useIsMobile();
 * {isMobile ? (
 *   <MobileGameHUD viewModel={hudViewModel} timeControl={timeControl} />
 * ) : (
 *   <GameHUD viewModel={hudViewModel} timeControl={timeControl} />
 * )}
 * ```
 */
export function MobileGameHUD({
  viewModel,
  timeControl: _timeControl,
  isLocalSandboxOnly = false,
  onShowBoardControls,
}: MobileGameHUDProps) {
  const [expandedPlayerId, setExpandedPlayerId] = useState<string | null>(null);

  const { phase, players, turnNumber, instruction, connectionStatus, isSpectator, decisionPhase } =
    viewModel;

  const isMyTurn = players.some((p) => p.isUserPlayer && p.isCurrentPlayer);

  const connectionColor =
    connectionStatus === 'connected'
      ? 'text-emerald-400'
      : connectionStatus === 'reconnecting'
        ? 'text-amber-400'
        : 'text-rose-400';

  const togglePlayerExpand = (playerId: string) => {
    setExpandedPlayerId((prev) => (prev === playerId ? null : playerId));
  };

  return (
    <div className="space-y-2" data-testid="mobile-game-hud">
      {/* Local sandbox banner (compact) */}
      {isLocalSandboxOnly && (
        <div className="px-2 py-1 rounded bg-slate-900/70 border border-slate-600 text-[10px] text-slate-300">
          <span className="font-semibold">Local sandbox</span> – not logged in
        </div>
      )}

      {/* Spectator badge */}
      {isSpectator && (
        <div className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-purple-900/40 border border-purple-500/40 text-xs text-purple-100">
          <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
            <path
              fillRule="evenodd"
              d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
              clipRule="evenodd"
            />
          </svg>
          <span>Spectating</span>
        </div>
      )}

      {/* Phase + turn bar */}
      <MobilePhaseBar phase={phase} turnNumber={turnNumber} isMyTurn={isMyTurn} />

      {/* Decision timer (when active) */}
      {decisionPhase &&
        decisionPhase.isActive &&
        decisionPhase.showCountdown &&
        decisionPhase.timeRemainingMs !== null && (
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-slate-300 truncate flex-1">
              {decisionPhase.label}
            </span>
            <MobileDecisionTimer
              timeRemainingMs={decisionPhase.timeRemainingMs}
              isServerCapped={decisionPhase.isServerCapped}
            />
          </div>
        )}

      {/* Instruction banner (compact) */}
      {instruction && (
        <div className="px-2 py-1.5 rounded bg-slate-700/50 border border-slate-600 text-xs text-slate-200 text-center">
          {instruction}
        </div>
      )}

      {/* Player list (compact) */}
      <div className="space-y-1">
        {players.map((player) => (
          <button
            key={player.id}
            className="w-full text-left"
            onClick={() => togglePlayerExpand(player.id)}
            aria-expanded={expandedPlayerId === player.id}
          >
            <MobilePlayerRow player={player} isExpanded={expandedPlayerId === player.id} />
          </button>
        ))}
      </div>

      {/* Footer bar: connection + help */}
      <div className="flex items-center justify-between text-[10px] text-slate-400">
        <span className={connectionColor}>● {connectionStatus}</span>
        <div className="flex items-center gap-2">
          {onShowBoardControls && (
            <button
              onClick={onShowBoardControls}
              className="px-2 py-1 rounded border border-slate-600 bg-slate-900/70 hover:border-slate-400 transition-colors"
              aria-label="Board controls"
              data-testid="mobile-controls-button"
            >
              <Tooltip content="Keyboard shortcuts and controls">
                <span>? Help</span>
              </Tooltip>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default MobileGameHUD;
