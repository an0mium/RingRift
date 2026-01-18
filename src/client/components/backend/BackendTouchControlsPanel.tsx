import React from 'react';
import type { Position } from '../../../shared/types/game';

export interface BackendTouchControlsPanelProps {
  /** Currently selected position */
  selectedPosition?: Position | undefined;
  /** Details about the selected stack (if any) */
  selectedStackDetails?: {
    height: number;
    cap: number;
    controllingPlayer: number;
  } | null;
  /** Valid move target positions */
  validTargets: Position[];
  /** Whether capture direction selection is pending */
  isCaptureDirectionPending: boolean;
  /** Current phase label */
  phaseLabel: string;
  /** Optional phase hint text */
  phaseHint?: string;
  /** Whether player is spectating */
  isSpectator: boolean;
  /** Whether it's the player's turn */
  isMyTurn: boolean;

  // Skip action flags
  canSkipCapture?: boolean;
  canSkipTerritoryProcessing?: boolean;
  canSkipRecovery?: boolean;

  // Skip action handlers
  onSkipCapture?: () => void;
  onSkipTerritoryProcessing?: () => void;
  onSkipRecovery?: () => void;

  // Selection handlers
  onClearSelection: () => void;

  // Visual aid toggles
  showMovementGrid: boolean;
  onToggleMovementGrid: (next: boolean) => void;
  showValidTargets: boolean;
  onToggleValidTargets: (next: boolean) => void;
  showLineOverlays?: boolean;
  onToggleLineOverlays?: (next: boolean) => void;
  showTerritoryOverlays?: boolean;
  onToggleTerritoryOverlays?: (next: boolean) => void;
}

/**
 * BackendTouchControlsPanel - Touch-optimized controls for lobby (server-backed) games.
 *
 * Provides mobile-friendly controls for:
 * - Viewing current selection state
 * - Skip actions (capture, territory, recovery)
 * - Visual overlay toggles
 *
 * January 2026: Ported from SandboxTouchControlsPanel with backend-specific adaptations.
 */
export const BackendTouchControlsPanel: React.FC<BackendTouchControlsPanelProps> = ({
  selectedPosition,
  selectedStackDetails,
  validTargets,
  isCaptureDirectionPending,
  phaseLabel,
  phaseHint,
  isSpectator,
  isMyTurn,
  canSkipCapture,
  canSkipTerritoryProcessing,
  canSkipRecovery,
  onSkipCapture,
  onSkipTerritoryProcessing,
  onSkipRecovery,
  onClearSelection,
  showMovementGrid,
  onToggleMovementGrid,
  showValidTargets,
  onToggleValidTargets,
  showLineOverlays,
  onToggleLineOverlays,
  showTerritoryOverlays,
  onToggleTerritoryOverlays,
}) => {
  const hasSelection = !!selectedPosition;
  const hasTargets = validTargets.length > 0;

  const selectionLabel = selectedPosition
    ? `(${selectedPosition.x}, ${selectedPosition.y}${
        typeof selectedPosition.z === 'number' ? `, ${selectedPosition.z}` : ''
      })`
    : 'None';

  const stackSummary = selectedStackDetails
    ? `H${selectedStackDetails.height} - C${selectedStackDetails.cap} - P${selectedStackDetails.controllingPlayer}`
    : null;

  // Disable controls for spectators or when not player's turn
  const controlsDisabled = isSpectator || !isMyTurn;

  return (
    <div
      className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3 text-xs text-slate-100"
      data-testid="backend-touch-controls"
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold">Touch Controls</h2>
          <p className="text-[11px] text-slate-400">
            {isSpectator
              ? 'Spectating - controls disabled'
              : isMyTurn
                ? 'Tap cells to make moves'
                : 'Waiting for opponent...'}
          </p>
          {phaseHint && <p className="mt-0.5 text-[10px] text-amber-300">{phaseHint}</p>}
        </div>
        <span className="px-2 py-0.5 rounded-full bg-slate-800/80 border border-slate-600 text-[10px] uppercase tracking-wide text-slate-300">
          Phase: {phaseLabel}
        </span>
      </div>

      {/* Selection summary */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-[11px]">Selection</span>
          <span className="text-[11px] text-slate-400">Targets: {validTargets.length}</span>
        </div>

        {hasSelection ? (
          <div className="space-y-1">
            <div className="font-mono text-sm text-white">{selectionLabel}</div>
            {stackSummary ? (
              <div className="text-[11px] text-slate-300">{stackSummary}</div>
            ) : (
              <div className="text-[11px] text-slate-300">Empty cell selected</div>
            )}
            {hasTargets && !isCaptureDirectionPending && (
              <p className="text-[11px] text-slate-400">
                Tap a highlighted destination on the board to complete your move.
              </p>
            )}
            {isCaptureDirectionPending && (
              <p className="text-[11px] text-amber-300">
                Capture available - tap a highlighted landing cell to choose direction.
              </p>
            )}
          </div>
        ) : (
          <p className="text-[11px] text-slate-300">
            {isMyTurn
              ? 'Tap any stack or empty cell to begin.'
              : 'Wait for your turn to make a move.'}
          </p>
        )}
      </div>

      {/* Skip actions - only show when available and it's player's turn */}
      {isMyTurn && !isSpectator && (
        <>
          {canSkipCapture && onSkipCapture && (
            <div className="space-y-1">
              <button
                type="button"
                onClick={onSkipCapture}
                className="w-full px-4 py-2.5 min-h-[44px] rounded-lg border border-amber-400 text-[11px] font-semibold text-amber-100 bg-amber-900/40 hover:border-amber-200 hover:bg-amber-800/70 active:scale-[0.98] transition touch-manipulation"
                data-testid="backend-skip-capture-button"
              >
                Skip Capture
              </button>
              <p className="text-[10px] text-amber-200/80">
                Decline optional capture and continue to line/territory processing.
              </p>
            </div>
          )}

          {canSkipTerritoryProcessing && onSkipTerritoryProcessing && (
            <div className="space-y-1">
              <button
                type="button"
                onClick={onSkipTerritoryProcessing}
                className="w-full px-4 py-2.5 min-h-[44px] rounded-lg border border-amber-400 text-[11px] font-semibold text-amber-100 bg-amber-900/40 hover:border-amber-200 hover:bg-amber-800/70 active:scale-[0.98] transition touch-manipulation"
                data-testid="backend-skip-territory-button"
              >
                Skip Territory Processing
              </button>
              <p className="text-[10px] text-amber-200/80">
                Leave remaining disconnected regions unprocessed for this turn.
              </p>
            </div>
          )}

          {canSkipRecovery && onSkipRecovery && (
            <div className="space-y-1">
              <button
                type="button"
                onClick={onSkipRecovery}
                className="w-full px-4 py-2.5 min-h-[44px] rounded-lg border border-amber-400 text-[11px] font-semibold text-amber-100 bg-amber-900/40 hover:border-amber-200 hover:bg-amber-800/70 active:scale-[0.98] transition touch-manipulation"
                data-testid="backend-skip-recovery-button"
              >
                Skip Recovery
              </button>
              <p className="text-[10px] text-amber-200/80">
                Decline recovery this turn and continue.
              </p>
            </div>
          )}
        </>
      )}

      {/* Clear selection button */}
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onClearSelection}
          disabled={!hasSelection || controlsDisabled}
          className={`px-4 py-2.5 min-h-[44px] rounded-lg border text-[11px] font-semibold active:scale-[0.98] transition touch-manipulation ${
            !hasSelection || controlsDisabled
              ? 'border-slate-700 text-slate-500 cursor-not-allowed opacity-60'
              : 'border-slate-500 text-slate-100 hover:border-emerald-400 hover:text-emerald-200'
          }`}
        >
          Clear Selection
        </button>
      </div>

      {/* Visual aids */}
      <div className="border-t border-slate-700 pt-3 space-y-2">
        <span className="font-semibold text-[11px]">Visual Aids</span>
        <div className="flex flex-col gap-1 text-[11px]">
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-slate-600 bg-slate-900 text-emerald-500 focus:ring-emerald-500"
              checked={showValidTargets}
              onChange={(e) => onToggleValidTargets(e.target.checked)}
            />
            <span className="text-slate-200">Show valid targets</span>
          </label>
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-slate-600 bg-slate-900 text-sky-500 focus:ring-sky-500"
              checked={showMovementGrid}
              onChange={(e) => onToggleMovementGrid(e.target.checked)}
            />
            <span className="text-slate-200">Show movement grid</span>
          </label>
        </div>
      </div>

      {/* Debug overlays - optional */}
      {onToggleLineOverlays !== undefined && onToggleTerritoryOverlays !== undefined && (
        <div className="border-t border-slate-700 pt-3 space-y-2">
          <span className="font-semibold text-[11px]">Debug Overlays</span>
          <div className="flex flex-col gap-1 text-[11px]">
            <label className="inline-flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                className="rounded border-slate-600 bg-slate-900 text-amber-500 focus:ring-amber-500"
                checked={showLineOverlays ?? false}
                onChange={(e) => onToggleLineOverlays(e.target.checked)}
              />
              <span className="text-slate-200">Show detected lines</span>
            </label>
            <label className="inline-flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                className="rounded border-slate-600 bg-slate-900 text-fuchsia-500 focus:ring-fuchsia-500"
                checked={showTerritoryOverlays ?? false}
                onChange={(e) => onToggleTerritoryOverlays(e.target.checked)}
              />
              <span className="text-slate-200">Show territory regions</span>
            </label>
          </div>
        </div>
      )}
    </div>
  );
};

export default BackendTouchControlsPanel;
