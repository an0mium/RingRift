import React, { useEffect } from 'react';
import { Position, positionToString, RingStack } from '../../shared/types/game';

export interface CellInfoToastProps {
  /** Position of the cell being inspected */
  position: Position;
  /** Stack at this position, if any */
  stack?: RingStack | null;
  /** Whether this cell is a collapsed (territory) space */
  isCollapsed?: boolean;
  /** Owner of collapsed space if applicable */
  collapsedOwner?: number;
  /** Whether this cell is a valid move target */
  isValidTarget?: boolean;
  /** Callback to dismiss the toast */
  onDismiss: () => void;
  /** Auto-dismiss timeout in ms (default: 3000) */
  autoDismissMs?: number;
}

/**
 * Touch-friendly cell info toast displayed at the bottom of the screen.
 * Shows stack details, territory status, and move validity for a cell.
 * Appears on long-press of board cells on touch devices.
 */
export const CellInfoToast: React.FC<CellInfoToastProps> = ({
  position,
  stack,
  isCollapsed = false,
  collapsedOwner,
  isValidTarget = false,
  onDismiss,
  autoDismissMs = 3000,
}) => {
  // Auto-dismiss after timeout
  useEffect(() => {
    const timer = setTimeout(() => {
      onDismiss();
    }, autoDismissMs);

    return () => clearTimeout(timer);
  }, [onDismiss, autoDismissMs]);

  const posKey = positionToString(position);
  const hasZ = typeof position.z === 'number';
  const coordLabel = hasZ
    ? `(${position.x}, ${position.y}, ${position.z})`
    : `(${position.x}, ${position.y})`;

  return (
    <div
      className="touch-info-badge safe-area-bottom"
      role="status"
      aria-live="polite"
      onClick={onDismiss}
      onKeyDown={(e) => {
        if (e.key === 'Escape' || e.key === 'Enter') {
          onDismiss();
        }
      }}
      tabIndex={0}
    >
      <div className="bg-slate-900/95 border border-slate-600 rounded-xl px-4 py-3 shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          {/* Position indicator */}
          <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-slate-800 border border-slate-500 flex items-center justify-center">
            <span className="text-xs font-mono text-slate-200">{posKey.slice(0, 3)}</span>
          </div>

          {/* Cell details */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-slate-100">Cell {coordLabel}</span>
              {isValidTarget && (
                <span className="px-1.5 py-0.5 text-[10px] font-medium bg-emerald-900/80 text-emerald-200 rounded-full border border-emerald-500/50">
                  Valid target
                </span>
              )}
            </div>

            {stack ? (
              <div className="mt-1 space-y-0.5">
                <div className="text-xs text-slate-300">
                  <span className="font-medium">Stack:</span> Height {stack.stackHeight}, Cap{' '}
                  {stack.capHeight}
                </div>
                <div className="text-xs text-slate-400">
                  Controlled by Player {stack.controllingPlayer}
                </div>
                {stack.rings && stack.rings.length > 0 && (
                  <div className="flex items-center gap-1 mt-1">
                    <span className="text-[10px] text-slate-500">Rings:</span>
                    <div className="flex gap-0.5">
                      {stack.rings.slice(0, 6).map((player, idx) => (
                        <div
                          key={idx}
                          className={`w-3 h-3 rounded-full border ${getPlayerColorClasses(player)}`}
                          title={`Player ${player}`}
                        />
                      ))}
                      {stack.rings.length > 6 && (
                        <span className="text-[10px] text-slate-500">
                          +{stack.rings.length - 6}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : isCollapsed ? (
              <div className="mt-1 text-xs text-slate-400">
                Territory space
                {collapsedOwner && ` (Player ${collapsedOwner})`}
              </div>
            ) : (
              <div className="mt-1 text-xs text-slate-400">Empty cell</div>
            )}
          </div>

          {/* Dismiss button */}
          <button
            type="button"
            onClick={onDismiss}
            className="flex-shrink-0 p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition touch-manipulation"
            aria-label="Dismiss cell info"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Tap hint */}
        <div className="mt-2 text-center text-[10px] text-slate-500">Tap anywhere to dismiss</div>
      </div>
    </div>
  );
};

function getPlayerColorClasses(player: number): string {
  switch (player) {
    case 1:
      return 'bg-emerald-400 border-emerald-200';
    case 2:
      return 'bg-sky-600 border-sky-300';
    case 3:
      return 'bg-amber-400 border-amber-200';
    case 4:
      return 'bg-fuchsia-400 border-fuchsia-200';
    default:
      return 'bg-slate-400 border-slate-200';
  }
}

export default CellInfoToast;
