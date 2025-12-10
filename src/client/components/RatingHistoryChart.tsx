/**
 * RatingHistoryChart Component
 *
 * Displays a visual representation of a user's rating history over time.
 * Uses a simple SVG-based line chart to show rating progression.
 */

import React, { useMemo } from 'react';
import clsx from 'clsx';

interface RatingHistoryEntry {
  date: string;
  rating: number;
  change: number;
  gameId: string | null;
}

interface RatingHistoryChartProps {
  history: RatingHistoryEntry[];
  currentRating: number;
  className?: string;
}

const CHART_HEIGHT = 120;
const CHART_WIDTH = 400;
const PADDING = { top: 10, right: 10, bottom: 20, left: 40 };

export function RatingHistoryChart({ history, currentRating, className }: RatingHistoryChartProps) {
  const chartData = useMemo(() => {
    if (history.length === 0) {
      return null;
    }

    // Include current rating as the last point
    const dataPoints = [
      ...history.map((h) => ({
        rating: h.rating,
        date: new Date(h.date),
        change: h.change,
      })),
    ].sort((a, b) => a.date.getTime() - b.date.getTime());

    if (dataPoints.length === 0) {
      return null;
    }

    const ratings = dataPoints.map((d) => d.rating);
    const minRating = Math.min(...ratings) - 50;
    const maxRating = Math.max(...ratings) + 50;
    const ratingRange = maxRating - minRating || 100;

    const innerWidth = CHART_WIDTH - PADDING.left - PADDING.right;
    const innerHeight = CHART_HEIGHT - PADDING.top - PADDING.bottom;

    const points = dataPoints.map((d, i) => ({
      x: PADDING.left + (i / Math.max(dataPoints.length - 1, 1)) * innerWidth,
      y: PADDING.top + innerHeight - ((d.rating - minRating) / ratingRange) * innerHeight,
      rating: d.rating,
      change: d.change,
      date: d.date,
    }));

    // Create path for the line
    const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

    // Create gradient area path
    const areaPath = [
      `M ${points[0].x} ${PADDING.top + innerHeight}`,
      ...points.map((p) => `L ${p.x} ${p.y}`),
      `L ${points[points.length - 1].x} ${PADDING.top + innerHeight}`,
      'Z',
    ].join(' ');

    return {
      points,
      linePath,
      areaPath,
      minRating,
      maxRating,
      innerHeight,
    };
  }, [history]);

  if (!chartData || history.length < 2) {
    return (
      <div
        className={clsx(
          'flex items-center justify-center h-32 bg-slate-900/50 rounded-lg border border-slate-700 text-slate-500 text-sm',
          className
        )}
      >
        {history.length === 0
          ? 'No rating history yet'
          : 'Play more games to see your rating trend'}
      </div>
    );
  }

  const { points, linePath, areaPath, minRating, maxRating, innerHeight } = chartData;
  const latestChange = points[points.length - 1]?.change ?? 0;

  return (
    <div className={clsx('bg-slate-900/50 rounded-lg border border-slate-700 p-4', className)}>
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-medium text-slate-300">Rating History</h3>
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-white">{currentRating}</span>
          {latestChange !== 0 && (
            <span
              className={clsx(
                'text-xs font-medium px-1.5 py-0.5 rounded',
                latestChange > 0
                  ? 'text-emerald-400 bg-emerald-900/30'
                  : 'text-red-400 bg-red-900/30'
              )}
            >
              {latestChange > 0 ? '+' : ''}
              {latestChange}
            </span>
          )}
        </div>
      </div>

      <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} className="w-full h-auto">
        {/* Gradient definition */}
        <defs>
          <linearGradient id="ratingGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgb(16, 185, 129)" stopOpacity="0.3" />
            <stop offset="100%" stopColor="rgb(16, 185, 129)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Y-axis labels */}
        <text
          x={PADDING.left - 5}
          y={PADDING.top + 5}
          textAnchor="end"
          className="fill-slate-500 text-[10px]"
        >
          {Math.round(maxRating)}
        </text>
        <text
          x={PADDING.left - 5}
          y={PADDING.top + innerHeight}
          textAnchor="end"
          className="fill-slate-500 text-[10px]"
        >
          {Math.round(minRating)}
        </text>

        {/* Gradient area */}
        <path d={areaPath} fill="url(#ratingGradient)" />

        {/* Line */}
        <path
          d={linePath}
          fill="none"
          stroke="rgb(16, 185, 129)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Points */}
        {points.map((point, i) => (
          <circle
            key={i}
            cx={point.x}
            cy={point.y}
            r="3"
            fill="rgb(16, 185, 129)"
            stroke="rgb(15, 23, 42)"
            strokeWidth="2"
          />
        ))}

        {/* X-axis labels (first and last dates) */}
        {points.length > 0 && (
          <>
            <text
              x={points[0].x}
              y={CHART_HEIGHT - 5}
              textAnchor="start"
              className="fill-slate-500 text-[9px]"
            >
              {points[0].date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
            </text>
            <text
              x={points[points.length - 1].x}
              y={CHART_HEIGHT - 5}
              textAnchor="end"
              className="fill-slate-500 text-[9px]"
            >
              {points[points.length - 1].date.toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric',
              })}
            </text>
          </>
        )}
      </svg>

      {/* Legend */}
      <div className="flex justify-between items-center mt-2 text-xs text-slate-500">
        <span>{history.length} games</span>
        <span>Last 30 days</span>
      </div>
    </div>
  );
}

export default RatingHistoryChart;
