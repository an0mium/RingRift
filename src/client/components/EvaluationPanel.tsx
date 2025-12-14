import React from 'react';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import type { Player } from '../../shared/types/game';
import { getPlayerColors } from '../adapters/gameViewModels';
import { useAccessibility } from '../contexts/AccessibilityContext';

export interface EvaluationPanelProps {
  evaluationHistory: PositionEvaluationPayload['data'][];
  players: Player[];
  className?: string;
}

export function EvaluationPanel({
  evaluationHistory,
  players,
  className = '',
}: EvaluationPanelProps) {
  const { colorVisionMode } = useAccessibility();
  const latest =
    evaluationHistory && evaluationHistory.length > 0
      ? evaluationHistory[evaluationHistory.length - 1]
      : null;

  // Build a simple per-player sparkline over the evaluation history.
  const sparkline = React.useMemo(() => {
    if (!evaluationHistory || evaluationHistory.length === 0) {
      return null;
    }

    // Collect all player numbers that appear in any snapshot.
    const playerNumbers = new Set<number>();
    for (const snapshot of evaluationHistory) {
      for (const key of Object.keys(snapshot.perPlayer)) {
        const pn = Number.parseInt(key, 10);
        if (Number.isFinite(pn)) {
          playerNumbers.add(pn);
        }
      }
    }

    if (playerNumbers.size === 0) {
      return null;
    }

    // Determine vertical scale from all totalEval values.
    let maxAbs = 0;
    for (const snapshot of evaluationHistory) {
      for (const pn of playerNumbers) {
        const ev = snapshot.perPlayer[pn];
        if (!ev || typeof ev.totalEval !== 'number') continue;
        const abs = Math.abs(ev.totalEval);
        if (abs > maxAbs) {
          maxAbs = abs;
        }
      }
    }

    if (maxAbs === 0) {
      maxAbs = 1;
    }

    const width = 220;
    const height = 60;
    const verticalMargin = 4;
    const midY = height / 2;
    const count = evaluationHistory.length;
    const step = count > 1 ? width / (count - 1) : 0;

    const playerByNumber = new Map<number, Player>();
    for (const p of players) {
      playerByNumber.set(p.playerNumber, p);
    }

    const series = Array.from(playerNumbers)
      .sort((a, b) => a - b)
      .map((playerNumber) => {
        const colors = getPlayerColors(playerNumber, colorVisionMode);

        const points: string[] = [];
        evaluationHistory.forEach((snapshot, index) => {
          const ev = snapshot.perPlayer[playerNumber];
          if (!ev || typeof ev.totalEval !== 'number') {
            return;
          }
          const value = ev.totalEval;
          const normalized = value / maxAbs;
          const x = count > 1 ? index * step : width / 2;
          const y = midY - normalized * (midY - verticalMargin);
          points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
        });

        if (points.length === 0) {
          return null;
        }

        const player = playerByNumber.get(playerNumber);
        const label = player?.username || `P${playerNumber}`;

        return {
          playerNumber,
          label,
          stroke: colors.hex,
          points: points.join(' '),
        };
      })
      .filter(Boolean) as Array<{
      playerNumber: number;
      label: string;
      stroke: string;
      points: string;
    }>;

    if (series.length === 0) {
      return null;
    }

    return { width, height, series };
  }, [evaluationHistory, players, colorVisionMode]);

  return (
    <div
      className={`border border-slate-700 rounded bg-slate-900/70 p-3 ${className}`}
      data-testid="evaluation-panel"
    >
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-slate-100">AI Evaluation</h2>
        {latest ? (
          <span className="text-[11px] text-slate-400">
            Move {latest.moveNumber} • {latest.engineProfile}
          </span>
        ) : (
          <span className="text-[11px] text-slate-500">Waiting for analysis…</span>
        )}
      </div>

      {latest ? (
        <>
          <div className="space-y-1 mb-2">
            {(() => {
              const playerByNumber = new Map<number, Player>();
              for (const p of players) {
                playerByNumber.set(p.playerNumber, p);
              }

              const playerNumbers = Object.keys(latest.perPlayer)
                .map((k) => Number.parseInt(k, 10))
                .filter((n) => Number.isFinite(n))
                .sort((a, b) => a - b);

              const bestPlayer =
                playerNumbers.reduce<number | null>((best, pn) => {
                  const current = latest.perPlayer[pn]?.totalEval ?? 0;
                  if (best === null) return pn;
                  const bestVal = latest.perPlayer[best]?.totalEval ?? 0;
                  return current > bestVal ? pn : best;
                }, null) ?? null;

              return playerNumbers.map((playerNumber) => {
                const ev = latest.perPlayer[playerNumber];
                if (!ev) return null;

                const player = playerByNumber.get(playerNumber);
                const name = player?.username || `P${playerNumber}`;
                const colors = getPlayerColors(playerNumber, colorVisionMode);

                const total = ev.totalEval ?? 0;
                const territory = ev.territoryEval ?? 0;
                const rings = ev.ringEval ?? 0;
                const isBest = bestPlayer === playerNumber;

                const sign = total > 0 ? '+' : total < 0 ? '' : '';
                const totalClass =
                  total > 0 ? 'text-emerald-300' : total < 0 ? 'text-rose-300' : 'text-slate-200';

                return (
                  <div key={playerNumber} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span
                        className={`inline-block w-2 h-2 rounded-full ${colors.ring}`}
                        aria-hidden="true"
                      />
                      <span className="font-medium text-slate-200">
                        {name}
                        {isBest && (
                          <span className="text-[10px] text-emerald-400 ml-1">(ahead)</span>
                        )}
                      </span>
                    </div>
                    <div className={`font-mono ${totalClass}`}>
                      {sign}
                      {total.toFixed(1)}{' '}
                      <span className="text-[10px] text-slate-400">
                        (T {territory >= 0 ? '+' : ''}
                        {territory.toFixed(1)}, R {rings >= 0 ? '+' : ''}
                        {rings.toFixed(1)})
                      </span>
                    </div>
                  </div>
                );
              });
            })()}
          </div>

          <div className="text-[11px] text-slate-500">
            Positive values favour the listed player; totals are relative margins combining
            territory and eliminated-ring advantage.
          </div>

          {sparkline && (
            <div className="mt-3">
              <svg
                width={sparkline.width}
                height={sparkline.height}
                viewBox={`0 0 ${sparkline.width} ${sparkline.height}`}
                className="w-full h-16 text-slate-500"
                aria-hidden="true"
                data-testid="evaluation-sparkline"
              >
                <rect
                  x="0"
                  y="0"
                  width={sparkline.width}
                  height={sparkline.height}
                  className="fill-slate-900/0"
                />
                {/* Zero line */}
                <line
                  x1={0}
                  x2={sparkline.width}
                  y1={sparkline.height / 2}
                  y2={sparkline.height / 2}
                  className="stroke-slate-700"
                  strokeWidth={0.5}
                  strokeDasharray="2 2"
                />
                {sparkline.series.map((s) => (
                  <polyline
                    key={s.playerNumber}
                    fill="none"
                    stroke={s.stroke}
                    strokeWidth={1.5}
                    points={s.points}
                    className="opacity-80"
                  />
                ))}
              </svg>
              <div className="mt-1 flex flex-wrap gap-2 text-[10px] text-slate-400">
                {sparkline.series.map((s) => (
                  <span key={s.playerNumber} className="inline-flex items-center gap-1">
                    <span
                      className="inline-block w-2 h-2 rounded-full"
                      style={{ backgroundColor: s.stroke }}
                    />
                    <span>{s.label}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="text-[11px] text-slate-500">
          No evaluation data has been received yet. When analysis is enabled, this panel will show
          per-player advantage for each move.
        </div>
      )}
    </div>
  );
}
