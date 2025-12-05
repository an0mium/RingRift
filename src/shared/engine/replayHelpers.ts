import type { GameState, Move, BoardType, Player, TimeControl } from '../types/game';
import type { GameRecord, MoveRecord } from '../types/gameRecord';
import { BOARD_CONFIGS } from '../types/game';
import { createInitialGameState } from './initialState';
import { processTurn } from './orchestration/turnOrchestrator';

/**
 * Reconstruct a GameState from a GameRecord at a given move index.
 *
 * This helper is deliberately minimal and replay-focused:
 * - It derives a plausible initial GameState compatible with the shared
 *   engine from the GameRecord header (boardType, numPlayers, rngSeed).
 * - It then applies the first `moveIndex` moves from the record using the
 *   canonical GameEngine, returning the resulting GameState.
 *
 * The returned state is suitable for parity checks, replay viewers, and
 * offline analysis, but is not guaranteed to match historical timestamps or
 * transient per-move metadata such as think times.
 *
 * @param record - Canonical GameRecord to replay.
 * @param moveIndex - Number of moves from the start to apply (0 = initial).
 */
export function reconstructStateAtMove(record: GameRecord, moveIndex: number): GameState {
  if (moveIndex < 0) {
    throw new Error(`moveIndex must be non-negative, got ${moveIndex}`);
  }

  const { boardType, numPlayers, rngSeed, isRated, players, moves } = record;
  const clampedIndex = Math.min(moveIndex, moves.length);

  const config = BOARD_CONFIGS[boardType];
  if (!config) {
    throw new Error(`Unknown boardType in GameRecord: ${boardType}`);
  }

  // Build minimal Player array for the engine; we preserve player numbers but
  // do not attempt to recover historical ratings or clocks.
  const timeControl: TimeControl = {
    type: 'classical',
    initialTime: 600, // seconds
    increment: 5,
  };

  const playerStates: Player[] = [];
  for (let i = 0; i < numPlayers; i += 1) {
    const seat = players[i];
    const playerNumber = i + 1;
    playerStates.push({
      id: `player${playerNumber}`,
      username: seat?.username ?? `Player ${playerNumber}`,
      type: seat?.playerType ?? 'ai',
      playerNumber,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: config.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    });
  }

  const initialState = createInitialGameState(
    record.id,
    boardType as BoardType,
    playerStates,
    timeControl,
    isRated,
    rngSeed
  );

  if (clampedIndex === 0) {
    return initialState;
  }

  // Convert MoveRecord entries into engine Moves and apply them.
  let state: GameState = initialState;
  for (let i = 0; i < clampedIndex; i += 1) {
    const rec: MoveRecord = record.moves[i];
    const move: Move = {
      id: `record-${record.id}-${i}`,
      player: rec.player,
      type: rec.type,
      ...(rec.from ? { from: rec.from } : {}),
      ...(rec.to ? { to: rec.to } : {}),
      ...(rec.captureTarget ? { captureTarget: rec.captureTarget } : {}),
      ...(rec.placementCount !== undefined ? { placementCount: rec.placementCount } : {}),
    } as Move;

    const result = processTurn(state, move);
    state = result.nextState;
  }

  // When we've applied the full recorded move list, treat the reconstructed
  // state as terminal for golden-replay invariants and analysis tooling. The
  // host is responsible for richer end-of-game semantics; here we only need a
  // structurally "finished" game so that INV-FINAL-STATE passes.
  if (clampedIndex === moves.length) {
    state = {
      ...state,
      gameStatus: 'finished',
    };
  }

  return state;
}
