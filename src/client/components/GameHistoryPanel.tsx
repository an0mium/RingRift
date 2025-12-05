import React, { useState, useEffect } from 'react';
import {
  gameApi,
  GameHistoryResponse,
  GameHistoryMove,
  GameDetailsResponse,
} from '../services/api';
import { Badge } from './ui/Badge';
import { formatVictoryReason } from '../adapters/gameViewModels';
import { BoardView } from './BoardView';
import { MoveHistory } from './MoveHistory';
import { HistoryPlaybackPanel } from './HistoryPlaybackPanel';
import { reconstructStateAtMove } from '../../shared/engine/replayHelpers';
import type {
  GameRecord,
  MoveRecord,
  GameOutcome,
  FinalScore,
  PlayerRecordInfo,
} from '../../shared/types/gameRecord';
import type {
  GameState,
  Move,
  Position,
  BoardType,
  LineInfo,
  Territory,
} from '../../shared/types/game';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

export interface GameHistoryPanelProps {
  /** Game ID to fetch history for */
  gameId: string;
  /** Whether the panel is initially collapsed */
  defaultCollapsed?: boolean;
  /** Optional class name for additional styling */
  className?: string;
  /** Called when an error occurs while fetching history */
  onError?: (error: Error) => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format a move type for display
 */
function formatMoveType(moveType: string): string {
  const typeMap: Record<string, string> = {
    place_ring: 'Place Ring',
    skip_placement: 'Skip Placement',
    move_ring: 'Move Ring',
    move_stack: 'Move Stack',
    build_stack: 'Build Stack',
    overtaking_capture: 'Capture',
    continue_capture_segment: 'Continue Capture',
    process_line: 'Process Line',
    choose_line_reward: 'Line Reward',
    process_territory_region: 'Territory',
    eliminate_rings_from_stack: 'Eliminate Rings',
    line_formation: 'Line Formation',
    territory_claim: 'Territory Claim',
  };
  return typeMap[moveType] || moveType.replace(/_/g, ' ');
}

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

/**
 * Extract position description from move data
 */
function getPositionDescription(moveData: Record<string, unknown>): string {
  const parts: string[] = [];

  if (moveData.from && typeof moveData.from === 'object') {
    const from = moveData.from as { x: number; y: number; z?: number };
    parts.push(`from (${from.x},${from.y}${from.z !== undefined ? `,${from.z}` : ''})`);
  }

  if (moveData.to && typeof moveData.to === 'object') {
    const to = moveData.to as { x: number; y: number; z?: number };
    parts.push(`to (${to.x},${to.y}${to.z !== undefined ? `,${to.z}` : ''})`);
  }

  return parts.join(' → ');
}

/**
 * Narrow unknown to Position
 */
function isPosition(value: unknown): value is Position {
  if (!value || typeof value !== 'object') return false;
  const obj = value as Record<string, unknown>;
  return typeof obj.x === 'number' && typeof obj.y === 'number';
}

/**
 * Normalise raw boardType string from the backend into a shared BoardType.
 * Falls back to 'square8' for unknown values.
 */
function toBoardType(raw: string): BoardType {
  if (raw === 'square8' || raw === 'square19' || raw === 'hexagonal') {
    return raw;
  }
  return 'square8';
}

/**
 * Map a GameHistoryResponse.result into a GameOutcome for GameRecord.
 * Defaults to 'abandonment' for unknown or missing reasons so that
 * downstream tooling always sees a valid outcome.
 */
function toGameOutcome(result: GameHistoryResponse['result'] | undefined): GameOutcome {
  if (!result) {
    return 'abandonment';
  }

  const reason = result.reason as string;
  const allowed: GameOutcome[] = [
    'ring_elimination',
    'territory_control',
    'last_player_standing',
    'timeout',
    'resignation',
    'draw',
    'abandonment',
  ];

  if ((allowed as string[]).includes(reason)) {
    return reason as GameOutcome;
  }

  return 'abandonment';
}

/**
 * Build a minimal but valid FinalScore structure with zeroed fields for all
 * players. Backend history does not currently expose full score breakdown,
 * but replayHelpers only need structural correctness here.
 */
function createEmptyFinalScore(numPlayers: number): FinalScore {
  const ringsEliminated: FinalScore['ringsEliminated'] = {};
  const territorySpaces: FinalScore['territorySpaces'] = {};
  const ringsRemaining: FinalScore['ringsRemaining'] = {};

  for (let p = 1; p <= numPlayers; p += 1) {
    ringsEliminated[p] = 0;
    territorySpaces[p] = 0;
    ringsRemaining[p] = 0;
  }

  return { ringsEliminated, territorySpaces, ringsRemaining };
}

/**
 * Adapt a backend GameHistoryResponse + GameDetailsResponse pair into:
 * - A minimal GameRecord suitable for replayHelpers.reconstructStateAtMove.
 * - A parallel Move[] array for feeding MoveHistory.
 *
 * This keeps the mapping logic in one place and avoids leaking GameRecord
 * details throughout the UI.
 */
function adaptHistoryToGameRecord(
  history: GameHistoryResponse,
  details: GameDetailsResponse
): { record: GameRecord; movesForDisplay: Move[] } {
  const boardType = toBoardType(details.boardType);
  const historyMoves = history.moves ?? [];

  // Map backend player IDs to seat indices for robust playerNumber mapping.
  const playerIdToSeat = new Map<string, number>();
  details.players.forEach((p, index) => {
    if (p && p.id) {
      playerIdToSeat.set(p.id, index + 1);
    }
  });

  // Infer number of players from details first, then from history payload.
  const distinctSeatsFromHistory = new Set<number>();
  historyMoves.forEach((entry) => {
    const raw = entry.moveData ?? {};
    const seat =
      typeof (raw as any).player === 'number' ? ((raw as any).player as number) : undefined;
    if (seat) distinctSeatsFromHistory.add(seat);
  });

  const numPlayers =
    details.players.length > 0 ? details.players.length : distinctSeatsFromHistory.size || 2;

  // Build PlayerRecordInfo array from game details, stubbing unknown seats.
  const players: PlayerRecordInfo[] = [];
  for (let i = 0; i < numPlayers; i += 1) {
    const seatIndex = i;
    const p = details.players[seatIndex];
    players.push({
      playerNumber: i + 1,
      username: p?.username ?? `Player ${i + 1}`,
      playerType: 'human',
      ...(typeof p?.rating === 'number' ? { ratingBefore: p.rating } : {}),
    });
  }

  // Helper to choose a player seat for a history entry.
  const inferPlayerSeat = (entry: GameHistoryMove): number => {
    const raw = entry.moveData ?? {};
    const seatFromMove =
      typeof (raw as any).player === 'number' ? ((raw as any).player as number) : undefined;
    if (seatFromMove && seatFromMove >= 1 && seatFromMove <= numPlayers) {
      return seatFromMove;
    }
    const seatFromId = playerIdToSeat.get(entry.playerId);
    if (seatFromId && seatFromId >= 1 && seatFromId <= numPlayers) {
      return seatFromId;
    }
    return 1;
  };

  const moveRecords: MoveRecord[] = [];
  const movesForDisplay: Move[] = [];

  historyMoves.forEach((entry) => {
    const raw = entry.moveData ?? {};
    const type = ((raw as any).type ?? entry.moveType) as MoveRecord['type'];

    const from = isPosition((raw as any).from) ? ((raw as any).from as Position) : undefined;
    const to = isPosition((raw as any).to) ? ((raw as any).to as Position) : undefined;
    const captureTarget = isPosition((raw as any).captureTarget)
      ? ((raw as any).captureTarget as Position)
      : undefined;

    const placementCount =
      typeof (raw as any).placementCount === 'number'
        ? ((raw as any).placementCount as number)
        : undefined;
    const placedOnStack =
      typeof (raw as any).placedOnStack === 'boolean'
        ? ((raw as any).placedOnStack as boolean)
        : undefined;

    const formedLines =
      Array.isArray((raw as any).formedLines) && (raw as any).formedLines.length > 0
        ? ((raw as any).formedLines as LineInfo[])
        : undefined;
    const collapsedMarkers =
      Array.isArray((raw as any).collapsedMarkers) && (raw as any).collapsedMarkers.length > 0
        ? ((raw as any).collapsedMarkers as Position[])
        : undefined;
    const disconnectedRegions =
      Array.isArray((raw as any).disconnectedRegions) && (raw as any).disconnectedRegions.length > 0
        ? ((raw as any).disconnectedRegions as Territory[])
        : undefined;
    const eliminatedRings =
      Array.isArray((raw as any).eliminatedRings) && (raw as any).eliminatedRings.length > 0
        ? ((raw as any).eliminatedRings as { player: number; count: number }[])
        : undefined;

    const thinkTimeCandidate = (raw as any).thinkTimeMs ?? (raw as any).thinkTime;
    const thinkTimeMs =
      typeof thinkTimeCandidate === 'number' && Number.isFinite(thinkTimeCandidate)
        ? (thinkTimeCandidate as number)
        : 0;

    const player = inferPlayerSeat(entry);

    const recordMove: MoveRecord = {
      moveNumber: entry.moveNumber,
      player,
      type,
      thinkTimeMs,
      ...(from ? { from } : {}),
      ...(to ? { to } : {}),
      ...(captureTarget ? { captureTarget } : {}),
      ...(placementCount !== undefined ? { placementCount } : {}),
      ...(placedOnStack !== undefined ? { placedOnStack } : {}),
      ...(formedLines ? { formedLines } : {}),
      ...(collapsedMarkers ? { collapsedMarkers } : {}),
      ...(disconnectedRegions ? { disconnectedRegions } : {}),
      ...(eliminatedRings ? { eliminatedRings } : {}),
    };

    moveRecords.push(recordMove);

    const uiMove: Move = {
      id: `history-${history.gameId}-${entry.moveNumber}`,
      type,
      player,
      ...(from ? { from } : {}),
      to: to ?? from ?? { x: 0, y: 0 },
      ...(captureTarget ? { captureTarget } : {}),
      ...(formedLines ? { formedLines } : {}),
      ...(collapsedMarkers ? { collapsedMarkers } : {}),
      ...(disconnectedRegions ? { disconnectedRegions } : {}),
      ...(eliminatedRings ? { eliminatedRings } : {}),
      timestamp: new Date(entry.timestamp),
      thinkTime: thinkTimeMs,
      moveNumber: entry.moveNumber,
    };

    movesForDisplay.push(uiMove);
  });

  const firstHistoryTs = historyMoves[0]?.timestamp;
  const lastHistoryTs = historyMoves[historyMoves.length - 1]?.timestamp ?? firstHistoryTs;

  const startedAt = details.startedAt ?? firstHistoryTs ?? new Date().toISOString();
  const endedAt = details.endedAt ?? lastHistoryTs ?? startedAt;

  const totalDurationMs = Math.max(0, new Date(endedAt).getTime() - new Date(startedAt).getTime());

  const outcome = toGameOutcome(history.result);
  const finalScore = createEmptyFinalScore(numPlayers);

  const record: GameRecord = {
    id: history.gameId,
    boardType,
    numPlayers,
    isRated: details.isRated,
    players,
    winner: typeof history.result?.winner === 'number' ? history.result.winner : undefined,
    outcome,
    finalScore,
    startedAt,
    endedAt,
    totalMoves: history.totalMoves,
    totalDurationMs,
    moves: moveRecords,
    metadata: {
      recordVersion: '1.0.0-client-replay',
      createdAt: endedAt,
      source: 'online_game',
      sourceId: history.gameId,
      tags: ['client_replay', 'backend_history'],
    },
  };

  return { record, movesForDisplay };
}

// ═══════════════════════════════════════════════════════════════════════════
// Sub-Components
// ═══════════════════════════════════════════════════════════════════════════

interface MoveItemProps {
  move: GameHistoryMove;
}

function MoveItem({ move }: MoveItemProps) {
  const [expanded, setExpanded] = useState(false);

  const positionDesc = getPositionDescription(move.moveData);
  const hasDetails =
    Object.keys(move.moveData).length > 0 &&
    Object.keys(move.moveData).some((k) => !['id', 'type', 'player', 'from', 'to'].includes(k));

  const isAutoResolved = !!move.autoResolved;
  let autoResolvedLabel: string | null = null;

  if (isAutoResolved && move.autoResolved) {
    const reason = move.autoResolved.reason;

    let reasonDisplay: string;
    if (reason === 'timeout') reasonDisplay = 'timeout';
    else if (reason === 'disconnected') reasonDisplay = 'disconnect';
    else if (reason === 'fallback') reasonDisplay = 'fallback move';
    else reasonDisplay = reason;

    autoResolvedLabel = `Auto-resolved (${reasonDisplay})`;
  }

  return (
    <div className="border-b border-slate-700/50 last:border-b-0">
      <div
        className={`px-3 py-2 flex items-center gap-2 ${hasDetails ? 'cursor-pointer hover:bg-slate-800/50' : ''}`}
        onClick={() => hasDetails && setExpanded(!expanded)}
        role={hasDetails ? 'button' : undefined}
        tabIndex={hasDetails ? 0 : undefined}
        onKeyDown={(e) => {
          if (hasDetails && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            setExpanded(!expanded);
          }
        }}
      >
        {/* Move Number */}
        <span className="text-xs font-mono text-slate-500 w-8">#{move.moveNumber}</span>

        {/* Player */}
        <span className="text-xs font-semibold text-blue-400 w-16 truncate" title={move.playerName}>
          {move.playerName}
        </span>

        {/* Move Type + Auto-resolve badge (if present) */}
        <div className="text-xs text-slate-300 flex-1 flex items-center gap-2 min-w-0">
          <span className="truncate">{formatMoveType(move.moveType)}</span>
          {autoResolvedLabel && (
            <Badge
              variant="warning"
              className="shrink-0"
              data-testid="auto-resolved-badge"
              aria-label={autoResolvedLabel}
            >
              {autoResolvedLabel}
            </Badge>
          )}
        </div>

        {/* Position (if available) */}
        {positionDesc && (
          <span className="text-xs text-slate-400 hidden sm:inline">{positionDesc}</span>
        )}

        {/* Timestamp */}
        <span className="text-[10px] text-slate-500">{formatTimestamp(move.timestamp)}</span>

        {/* Expand indicator */}
        {hasDetails && <span className="text-slate-500 text-xs">{expanded ? '▼' : '▶'}</span>}
      </div>

      {/* Expanded details */}
      {expanded && hasDetails && (
        <div className="px-3 py-2 bg-slate-800/30 text-xs">
          <pre className="text-slate-400 overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(move.moveData, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════════

/**
 * GameHistoryPanel Component
 *
 * A collapsible panel that displays the complete move history for a game.
 * Fetches data from the API and displays moves in a scrollable list with
 * expandable details for each move.
 *
 * @example
 * ```tsx
 * <GameHistoryPanel
 *   gameId="abc123"
 *   defaultCollapsed={false}
 *   onError={(err) => console.error('Failed to load history:', err)}
 * />
 * ```
 */
export function GameHistoryPanel({
  gameId,
  defaultCollapsed = false,
  className = '',
  onError,
}: GameHistoryPanelProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<GameHistoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadVersion, setReloadVersion] = useState(0);

  // Backend replay state (single-game, backend-history based).
  const [replayOpen, setReplayOpen] = useState(false);
  const [replayLoading, setReplayLoading] = useState(false);
  const [replayRecord, setReplayRecord] = useState<GameRecord | null>(null);
  const [replayMoves, setReplayMoves] = useState<Move[]>([]);
  const [replayGameState, setReplayGameState] = useState<GameState | null>(null);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0);
  const [isViewingHistory, setIsViewingHistory] = useState(false);

  // Fetch history when panel is expanded or gameId changes
  useEffect(() => {
    if (collapsed || !gameId) return;

    let cancelled = false;

    async function fetchHistory() {
      setLoading(true);
      setError(null);

      try {
        const data = await gameApi.getGameHistory(gameId);
        if (!cancelled) {
          setHistory(data);
        }
      } catch (err) {
        if (!cancelled) {
          const errorMessage = err instanceof Error ? err.message : 'Failed to load history';
          setError(errorMessage);
          onError?.(err instanceof Error ? err : new Error(errorMessage));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchHistory();

    return () => {
      cancelled = true;
    };
  }, [gameId, collapsed, onError, reloadVersion]);

  // Recompute reconstructed GameState when replay record or index changes.
  useEffect(() => {
    if (!replayRecord || !replayOpen) {
      setReplayGameState(null);
      return;
    }

    const effectiveIndex = isViewingHistory ? currentMoveIndex : replayRecord.moves.length;

    try {
      const next = reconstructStateAtMove(replayRecord, effectiveIndex);
      setReplayGameState(next);
      setReplayError(null);
    } catch (err) {
      // Log for devs while surfacing a compact message in the UI.

      console.error('Failed to reconstruct replay state from backend history', err);
      setReplayGameState(null);
      setReplayError('Failed to reconstruct game state for replay.');
    }
  }, [replayRecord, replayOpen, currentMoveIndex, isViewingHistory]);

  const handleToggleReplay = async () => {
    if (!history || history.moves.length === 0) {
      return;
    }

    // Simple toggle when already initialized
    if (replayOpen) {
      setReplayOpen(false);
      return;
    }

    // If we already have a record, just reopen.
    if (replayRecord) {
      setReplayOpen(true);
      return;
    }

    setReplayLoading(true);
    setReplayError(null);

    try {
      const details = await gameApi.getGameDetails(gameId);
      const { record, movesForDisplay } = adaptHistoryToGameRecord(history, details);
      setReplayRecord(record);
      setReplayMoves(movesForDisplay);
      setCurrentMoveIndex(record.moves.length);
      setIsViewingHistory(false);
      setReplayOpen(true);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load replay metadata for this game';
      setReplayError(message);
      onError?.(err instanceof Error ? err : new Error(message));
    } finally {
      setReplayLoading(false);
    }
  };

  const handleMoveIndexChange = (index: number) => {
    if (!replayRecord) return;
    const clamped = Math.max(0, Math.min(index, replayRecord.moves.length));
    setCurrentMoveIndex(clamped);
  };

  const handleEnterHistoryView = () => {
    if (!replayRecord) return;
    setIsViewingHistory(true);
  };

  const handleExitHistoryView = () => {
    if (!replayRecord) return;
    setIsViewingHistory(false);
    setCurrentMoveIndex(replayRecord.moves.length);
  };

  const handleMoveClick = (index: number) => {
    if (!replayRecord) return;
    setIsViewingHistory(true);
    setCurrentMoveIndex(index + 1);
  };

  const hasReplaySnapshots = !!replayRecord && !!replayGameState;

  const activeMoveIndex =
    isViewingHistory && currentMoveIndex > 0 && replayRecord
      ? currentMoveIndex - 1
      : replayRecord
        ? replayRecord.moves.length - 1
        : undefined;

  return (
    <div
      className={`border border-slate-700 rounded-lg bg-slate-900/70 overflow-hidden ${className}`}
      data-testid="game-history-panel"
    >
      {/* Header */}
      <button
        className="w-full px-4 py-3 flex items-center justify-between bg-slate-800/50 hover:bg-slate-800/70 transition-colors"
        onClick={() => setCollapsed(!collapsed)}
        aria-expanded={!collapsed}
        aria-controls="game-history-content"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-200">Move History</span>
          {history && <span className="text-xs text-slate-400">({history.totalMoves} moves)</span>}
        </div>
        <span className="text-slate-400">{collapsed ? '▶' : '▼'}</span>
      </button>

      {/* Content */}
      {!collapsed && (
        <div
          id="game-history-content"
          className="max-h-80 overflow-y-auto"
          role="region"
          aria-label="Move history"
        >
          {/* Terminal result summary, when available */}
          {history?.result && !loading && !error && (
            <div className="px-4 py-2 border-b border-slate-700/50 bg-slate-900/60 text-xs text-slate-200 flex items-center justify-between">
              <span className="font-semibold">
                Result: {formatVictoryReason(history.result.reason)}
              </span>
              {history.result.winner !== undefined && history.result.winner !== null && (
                <span className="text-slate-400">Winner: P{history.result.winner}</span>
              )}
            </div>
          )}

          {/* Backend replay entry point for finished games */}
          {history && history.result && history.moves.length > 0 && !loading && !error && (
            <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-900/50 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-slate-200">Replay this game</span>
                <button
                  type="button"
                  className="px-2 py-1 text-[11px] rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={handleToggleReplay}
                  disabled={replayLoading}
                  data-testid="open-replay-button"
                >
                  {replayOpen ? 'Hide replay' : 'Replay'}
                </button>
              </div>

              {replayOpen && (
                <div className="mt-2 space-y-3" data-testid="backend-replay-panel">
                  {replayLoading && (
                    <div className="text-[11px] text-slate-400">Preparing replay…</div>
                  )}

                  {replayError && !replayLoading && (
                    <div className="text-[11px] text-red-400">
                      Replay unavailable: {replayError}
                    </div>
                  )}

                  {!replayLoading && !replayError && replayRecord && (
                    <>
                      <HistoryPlaybackPanel
                        totalMoves={replayRecord.moves.length}
                        currentMoveIndex={currentMoveIndex}
                        isViewingHistory={isViewingHistory}
                        onMoveIndexChange={handleMoveIndexChange}
                        onExitHistoryView={handleExitHistoryView}
                        onEnterHistoryView={handleEnterHistoryView}
                        visible={true}
                        hasSnapshots={hasReplaySnapshots}
                      />

                      {replayGameState && (
                        <div className="flex flex-col md:flex-row gap-3">
                          <div className="flex-1 min-w-0 border-t border-slate-800 pt-3 md:border-t-0 md:border-r md:pr-3">
                            <BoardView
                              boardType={replayGameState.boardType}
                              board={replayGameState.board}
                              showCoordinateLabels={replayGameState.boardType === 'square8'}
                              showMovementGrid={false}
                              showLineOverlays={false}
                              showTerritoryRegionOverlays={false}
                            />
                          </div>
                          <div className="w-full md:w-56">
                            <MoveHistory
                              moves={replayMoves}
                              boardType={replayRecord.boardType}
                              currentMoveIndex={activeMoveIndex}
                              onMoveClick={handleMoveClick}
                              maxHeight="max-h-48"
                            />
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="px-4 py-8 text-center">
              <div className="animate-spin w-6 h-6 border-2 border-slate-600 border-t-blue-500 rounded-full mx-auto mb-2"></div>
              <span className="text-xs text-slate-400">Loading history...</span>
            </div>
          )}

          {/* Error state */}
          {error && !loading && (
            <div className="px-4 py-6 text-center">
              <div className="text-red-400 text-sm mb-2">⚠ {error}</div>
              <button
                className="text-xs text-blue-400 hover:text-blue-300 underline"
                onClick={() => {
                  // Clear previous history and error, then trigger a refetch.
                  setHistory(null);
                  setError(null);
                  setReloadVersion((v) => v + 1);
                }}
              >
                Retry
              </button>
            </div>
          )}

          {/* Empty state */}
          {!loading && !error && history && history.moves.length === 0 && (
            <div className="px-4 py-6 text-center text-slate-400 text-sm">
              No moves recorded yet.
            </div>
          )}

          {/* Move list */}
          {!loading && !error && history && history.moves.length > 0 && (
            <div className="divide-y divide-slate-700/30">
              {history.moves.map((move) => (
                <MoveItem key={`${move.moveNumber}-${move.playerId}`} move={move} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default GameHistoryPanel;
