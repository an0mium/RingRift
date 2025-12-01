import React from 'react';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import type { Player } from '../../shared/types/game';
import { PLAYER_COLORS } from '../adapters/gameViewModels';

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
  const latest =
    evaluationHistory && evaluationHistory.length > 0
      ? evaluationHistory[evaluationHistory.length - 1]
      : null;

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
                const colors = PLAYER_COLORS[playerNumber as keyof typeof PLAYER_COLORS] ?? {
                  ring: 'bg-slate-300',
                  hex: '#64748b',
                };

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
