/**
 * Tutorial Hint Banner
 *
 * A non-intrusive banner that displays contextual hints during gameplay
 * in "Learn the Basics" tutorial mode.
 */

import React from 'react';
import type { TutorialHint } from '../../hooks/useTutorialHints';

export interface TutorialHintBannerProps {
  /** The hint to display */
  hint: TutorialHint;
  /** Called when user clicks "Got it" to dismiss */
  onDismiss: () => void;
  /** Called when user clicks "Learn More" to see full teaching content */
  onLearnMore: () => void;
  /** Called when user clicks "Don't show hints" to disable future hints */
  onDisableHints: () => void;
}

/**
 * Displays a contextual tutorial hint banner above the game board.
 *
 * Features:
 * - Shows phase-specific guidance with icon and message
 * - "Learn More" opens the TeachingOverlay for deeper explanation
 * - "Got it" dismisses and marks the phase as seen
 * - "Don't show" disables all future hints
 */
export function TutorialHintBanner({
  hint,
  onDismiss,
  onLearnMore,
  onDisableHints,
}: TutorialHintBannerProps) {
  return (
    <div className="animate-in slide-in-from-top duration-300 mx-2 sm:mx-4 mb-3">
      <div className="rounded-xl border border-emerald-500/30 bg-slate-800/90 backdrop-blur-sm shadow-lg overflow-hidden">
        <div className="p-3 sm:p-4">
          <div className="flex items-start gap-3">
            {/* Icon */}
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center text-xl">
              {hint.icon}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs uppercase tracking-wide text-emerald-400 font-medium">
                  Tutorial
                </span>
                <h3 className="text-sm sm:text-base font-semibold text-white">{hint.title}</h3>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">{hint.message}</p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-700/50">
            <button
              onClick={onDisableHints}
              className="text-xs text-slate-500 hover:text-slate-400 transition"
            >
              Don't show hints
            </button>

            <div className="flex items-center gap-2">
              <button
                onClick={onLearnMore}
                className="px-3 py-1.5 rounded-lg border border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/10 hover:border-emerald-400 transition text-sm font-medium"
              >
                Learn More
              </button>
              <button
                onClick={onDismiss}
                className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition text-sm font-medium"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
