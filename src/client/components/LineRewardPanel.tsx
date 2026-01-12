/**
 * @fileoverview LineRewardPanel - Floating panel for overlength line choice
 *
 * RR-FIX-2026-01-12: Provides a non-modal floating panel that displays
 * overlength line reward options. The panel shows a legend of segment
 * colors and instructs the user to click on the highlighted segments
 * on the board to make their selection.
 *
 * This component is rendered when a `line_reward_option` choice with
 * segments is pending, enabling graphical board-based selection.
 */

import React from 'react';
import type { LineRewardChoice } from '../../shared/types/game';

export interface LineRewardPanelProps {
  /** The pending line reward choice with segment data */
  choice: LineRewardChoice;
  /** Callback when user selects an option (from button click, not board click) */
  onSelect: (optionId: string) => void;
}

/**
 * Floating panel for overlength line segment selection.
 *
 * Displays a compact legend showing:
 * - Amber: Collapse All (costs 1 ring)
 * - Cyan: Minimum Collapse options (free)
 *
 * User can either:
 * 1. Click directly on highlighted segments on the board
 * 2. Click the buttons in this panel
 */
export function LineRewardPanel({ choice, onSelect }: LineRewardPanelProps) {
  const hasSegments = choice.segments && choice.segments.length > 0;

  // Find collapse-all and minimum collapse segments
  const collapseAllSegment = choice.segments?.find((s) => s.isCollapseAll);
  const minCollapseSegments = choice.segments?.filter((s) => !s.isCollapseAll) ?? [];

  return (
    <div
      className="absolute top-4 left-1/2 -translate-x-1/2 z-50 pointer-events-auto"
      role="region"
      aria-label="Overlength line choice"
    >
      <div className="bg-slate-900/95 backdrop-blur-sm rounded-xl border border-slate-700 shadow-2xl p-4 min-w-[280px] max-w-[360px]">
        {/* Header */}
        <div className="mb-3">
          <h3 className="text-sm font-semibold text-slate-100">Overlength Line</h3>
          <p className="text-xs text-slate-400 mt-1">
            {hasSegments
              ? 'Click a highlighted segment on the board, or use buttons below'
              : 'Choose how to resolve this overlength line'}
          </p>
        </div>

        {/* Segment legend and buttons */}
        <div className="space-y-2">
          {/* Collapse All option */}
          {collapseAllSegment && (
            <button
              type="button"
              onClick={() => onSelect(collapseAllSegment.optionId)}
              className="w-full flex items-center gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-left hover:bg-amber-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400 transition-colors"
            >
              <span className="w-4 h-4 rounded-sm bg-amber-400 flex-shrink-0 ring-2 ring-amber-400/50 animate-pulse" />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-amber-100">Collapse All</div>
                <div className="text-xs text-amber-200/70 truncate">
                  {collapseAllSegment.positions.length} markers (costs 1 ring)
                </div>
              </div>
            </button>
          )}

          {/* Minimum Collapse options */}
          {minCollapseSegments.length > 0 && (
            <div className="space-y-1.5">
              {minCollapseSegments.length === 1 ? (
                <button
                  type="button"
                  onClick={() => onSelect(minCollapseSegments[0].optionId)}
                  className="w-full flex items-center gap-3 rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-left hover:bg-cyan-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 transition-colors"
                >
                  <span className="w-4 h-4 rounded-sm bg-cyan-400 flex-shrink-0 ring-2 ring-cyan-400/50 animate-pulse" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-cyan-100">Minimum Collapse</div>
                    <div className="text-xs text-cyan-200/70 truncate">
                      {minCollapseSegments[0].positions.length} markers (free)
                    </div>
                  </div>
                </button>
              ) : (
                <>
                  <div className="text-xs text-slate-400 px-1">
                    Minimum Collapse Options (click on board or button):
                  </div>
                  {minCollapseSegments.map((segment, index) => {
                    // Use different colors for multiple segments
                    const colorClasses = [
                      'border-cyan-500/40 bg-cyan-500/10 hover:bg-cyan-500/20 focus-visible:ring-cyan-400',
                      'border-emerald-500/40 bg-emerald-500/10 hover:bg-emerald-500/20 focus-visible:ring-emerald-400',
                      'border-violet-500/40 bg-violet-500/10 hover:bg-violet-500/20 focus-visible:ring-violet-400',
                      'border-sky-500/40 bg-sky-500/10 hover:bg-sky-500/20 focus-visible:ring-sky-400',
                    ];
                    const dotColors = [
                      'bg-cyan-400',
                      'bg-emerald-400',
                      'bg-violet-400',
                      'bg-sky-400',
                    ];
                    const textColors = [
                      'text-cyan-100',
                      'text-cyan-200/70',
                      'text-emerald-100',
                      'text-emerald-200/70',
                      'text-violet-100',
                      'text-violet-200/70',
                      'text-sky-100',
                      'text-sky-200/70',
                    ];

                    const colorIndex = index % colorClasses.length;

                    return (
                      <button
                        key={segment.optionId}
                        type="button"
                        onClick={() => onSelect(segment.optionId)}
                        className={`w-full flex items-center gap-3 rounded-lg border px-3 py-2 text-left focus:outline-none focus-visible:ring-2 transition-colors ${colorClasses[colorIndex]}`}
                      >
                        <span
                          className={`w-4 h-4 rounded-sm flex-shrink-0 ring-2 ring-opacity-50 animate-pulse ${dotColors[colorIndex]}`}
                        />
                        <div className="flex-1 min-w-0">
                          <div
                            className={`text-sm font-medium ${textColors[colorIndex * 2] || 'text-slate-100'}`}
                          >
                            Segment {index + 1}
                          </div>
                          <div
                            className={`text-xs truncate ${textColors[colorIndex * 2 + 1] || 'text-slate-300'}`}
                          >
                            {segment.positions.length} markers (free)
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </>
              )}
            </div>
          )}

          {/* Fallback for choices without segment data */}
          {!hasSegments && (
            <div className="grid gap-2">
              <button
                type="button"
                onClick={() => onSelect('option_2_min_collapse_no_elimination')}
                className="flex items-center gap-3 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2.5 text-left hover:bg-emerald-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 transition-colors"
              >
                <div className="flex-1">
                  <div className="text-sm font-medium text-emerald-100">
                    Minimum Collapse (Free)
                  </div>
                  <div className="text-xs text-emerald-200/70">
                    Collapse minimum markers, no ring cost
                  </div>
                </div>
              </button>
              <button
                type="button"
                onClick={() => onSelect('option_1_collapse_all_and_eliminate')}
                className="flex items-center gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2.5 text-left hover:bg-amber-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400 transition-colors"
              >
                <div className="flex-1">
                  <div className="text-sm font-medium text-amber-100">Collapse All (Cost)</div>
                  <div className="text-xs text-amber-200/70">
                    Collapse all markers, eliminate 1 ring
                  </div>
                </div>
              </button>
            </div>
          )}
        </div>

        {/* Tip */}
        <div className="mt-3 pt-2 border-t border-slate-700">
          <p className="text-xs text-slate-500">Tip: Amber = costs ring, Cyan/others = free</p>
        </div>
      </div>
    </div>
  );
}

export default LineRewardPanel;
