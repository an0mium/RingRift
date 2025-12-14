/**
 * PlaybackControls - Transport controls for game replay.
 *
 * Provides step forward/backward, play/pause, speed selector, and scrubber.
 */

import React, { useCallback } from 'react';
import type { PlaybackSpeed } from '../../types/replay';

export interface PlaybackControlsProps {
  currentMove: number;
  totalMoves: number;
  isPlaying: boolean;
  playbackSpeed: PlaybackSpeed;
  isLoading?: boolean;
  canStepForward: boolean;
  canStepBackward: boolean;
  onStepForward: () => void;
  onStepBackward: () => void;
  onJumpToStart: () => void;
  onJumpToEnd: () => void;
  onJumpToMove: (move: number) => void;
  onTogglePlay: () => void;
  onSetSpeed: (speed: PlaybackSpeed) => void;
  className?: string;
}

const SPEED_OPTIONS: PlaybackSpeed[] = [0.5, 1, 2, 4];

export function PlaybackControls({
  currentMove,
  totalMoves,
  isPlaying,
  playbackSpeed,
  isLoading = false,
  canStepForward,
  canStepBackward,
  onStepForward,
  onStepBackward,
  onJumpToStart,
  onJumpToEnd,
  onJumpToMove,
  onTogglePlay,
  onSetSpeed,
  className = '',
}: PlaybackControlsProps) {
  const progress = totalMoves > 0 ? (currentMove / totalMoves) * 100 : 0;

  const handleScrubberChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = parseInt(e.target.value, 10);
      onJumpToMove(value);
    },
    [onJumpToMove]
  );

  const handleScrubberClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percent = x / rect.width;
      const move = Math.round(percent * totalMoves);
      onJumpToMove(Math.max(0, Math.min(totalMoves, move)));
    },
    [totalMoves, onJumpToMove]
  );

  const buttonClass =
    'px-3 py-2 min-h-[44px] rounded border border-slate-600 text-xs ' +
    'disabled:opacity-40 disabled:cursor-not-allowed ' +
    'hover:border-slate-400 hover:bg-slate-700/50 transition ' +
    'touch-manipulation';

  return (
    <div className={`space-y-2 ${className}`}>
      {/* Transport buttons */}
      <div className="flex items-center justify-center gap-1">
        <button
          type="button"
          onClick={onJumpToStart}
          disabled={!canStepBackward || isLoading}
          className={buttonClass}
          aria-label="Jump to start"
          title="Jump to start"
        >
          ⏮
        </button>
        <button
          type="button"
          onClick={onStepBackward}
          disabled={!canStepBackward || isLoading}
          className={buttonClass}
          aria-label="Step backward"
          title="Step backward (←)"
        >
          ◀
        </button>
        <button
          type="button"
          onClick={onTogglePlay}
          disabled={!canStepForward && !isPlaying}
          className={`${buttonClass} px-3 ${isPlaying ? 'bg-emerald-900/40 border-emerald-500/50' : ''}`}
          aria-label={isPlaying ? 'Pause' : 'Play'}
          title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button
          type="button"
          onClick={onStepForward}
          disabled={!canStepForward || isLoading}
          className={buttonClass}
          aria-label="Step forward"
          title="Step forward (→)"
        >
          ▶
        </button>
        <button
          type="button"
          onClick={onJumpToEnd}
          disabled={!canStepForward || isLoading}
          className={buttonClass}
          aria-label="Jump to end"
          title="Jump to end"
        >
          ⏭
        </button>
      </div>

      {/* Speed selector */}
      <div className="flex items-center justify-center gap-2 text-[10px]">
        <span className="text-slate-400">Speed:</span>
        {SPEED_OPTIONS.map((speed) => (
          <button
            key={speed}
            type="button"
            onClick={() => onSetSpeed(speed)}
            className={`px-1.5 py-0.5 rounded border transition ${
              playbackSpeed === speed
                ? 'border-emerald-500/50 bg-emerald-900/40 text-emerald-200'
                : 'border-slate-600 text-slate-300 hover:border-slate-400'
            }`}
          >
            {speed}x
          </button>
        ))}
      </div>

      {/* Scrubber / progress bar */}
      <div className="space-y-1">
        <div
          className="relative h-2 bg-slate-700 rounded-full cursor-pointer overflow-hidden"
          onClick={handleScrubberClick}
          role="slider"
          aria-label="Playback position"
          aria-valuemin={0}
          aria-valuemax={totalMoves}
          aria-valuenow={currentMove}
        >
          <div
            className="absolute left-0 top-0 h-full bg-emerald-500 rounded-full transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
          {/* Thumb indicator */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow border border-slate-400"
            style={{ left: `calc(${progress}% - 6px)` }}
          />
        </div>

        {/* Hidden range input for keyboard accessibility */}
        <input
          type="range"
          min={0}
          max={totalMoves}
          value={currentMove}
          onChange={handleScrubberChange}
          className="sr-only"
          aria-label="Playback position"
        />

        {/* Move counter */}
        <div className="flex items-center justify-between text-[10px] text-slate-400">
          <span>Move {currentMove}</span>
          <span>of {totalMoves}</span>
        </div>
      </div>
    </div>
  );
}
