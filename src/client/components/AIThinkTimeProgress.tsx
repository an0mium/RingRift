import React, { useEffect, useState } from 'react';

/**
 * Estimate AI think time based on difficulty level.
 * Higher difficulty = longer expected think time.
 */
function getExpectedThinkTimeMs(difficulty: number): number {
  // Rough estimates based on AI type at each difficulty:
  // 1-2 (Random/Heuristic): 500-800ms
  // 3-4 (Minimax): 1000-1500ms
  // 5-8 (MCTS): 2000-4000ms
  // 9-10 (Descent): 4000-6000ms
  if (difficulty <= 2) return 600;
  if (difficulty <= 4) return 1200;
  if (difficulty <= 6) return 2500;
  if (difficulty <= 8) return 3500;
  return 5000;
}

interface AIThinkTimeProgressProps {
  /** Whether an AI is currently thinking */
  isAiThinking: boolean;
  /** When the AI started thinking (timestamp) */
  thinkingStartedAt: number | null;
  /** AI difficulty level (1-10) */
  aiDifficulty: number;
  /** Optional: AI player name */
  aiPlayerName?: string;
}

/**
 * Progress bar showing AI think time elapsed vs expected.
 * Shrinks from full width as time elapses.
 */
export function AIThinkTimeProgress({
  isAiThinking,
  thinkingStartedAt,
  aiDifficulty,
  aiPlayerName = 'AI',
}: AIThinkTimeProgressProps) {
  const [elapsed, setElapsed] = useState(0);

  const expectedMs = getExpectedThinkTimeMs(aiDifficulty);

  // Update elapsed time while AI is thinking
  useEffect(() => {
    if (!isAiThinking || !thinkingStartedAt) {
      setElapsed(0);
      return;
    }

    // Initialize elapsed immediately
    setElapsed(Date.now() - thinkingStartedAt);

    const interval = setInterval(() => {
      setElapsed(Date.now() - thinkingStartedAt);
    }, 50);

    return () => clearInterval(interval);
  }, [isAiThinking, thinkingStartedAt]);

  // Don't render if AI is not thinking
  if (!isAiThinking || !thinkingStartedAt) {
    return null;
  }

  // Calculate progress (0 to 1, where 1 = expected time reached)
  const progress = Math.min(elapsed / expectedMs, 1);
  // Remaining percentage (shrinking bar)
  const remainingPercent = Math.max(0, 100 - progress * 100);

  // Color shifts from blue to amber as time elapses
  const barColorClass =
    progress < 0.5 ? 'bg-blue-500' : progress < 0.8 ? 'bg-amber-500' : 'bg-orange-500';

  // Pulse animation when close to expected time
  const pulseClass = progress > 0.8 ? 'animate-pulse' : '';

  return (
    <div
      className="px-3 py-2 rounded-lg border border-slate-700 bg-slate-900/60"
      role="status"
      aria-live="polite"
      aria-label={`${aiPlayerName} is thinking`}
      data-testid="ai-think-time-progress"
    >
      <div className="flex items-center justify-between text-[11px] text-slate-300 mb-1.5">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
          <span className="font-medium text-slate-100">{aiPlayerName} is thinking...</span>
        </span>
        <span className="text-slate-400 font-mono text-[10px]">{(elapsed / 1000).toFixed(1)}s</span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColorClass} ${pulseClass} transition-all duration-75`}
          style={{ width: `${remainingPercent}%` }}
        />
      </div>
      <div className="mt-1 text-[10px] text-slate-500 text-right">
        Expected: ~{(expectedMs / 1000).toFixed(1)}s
      </div>
    </div>
  );
}
