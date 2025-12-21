import type { GameState } from '../../types/game';
import type { GameRecord } from '../../types/gameRecord';
import { reconstructStateAtMoveWithOptions } from '../replayHelpers';
import { normalizeLegacyMoveType } from './legacyMoveTypes';

/**
 * Legacy replay compatibility helper.
 *
 * @deprecated Prefer strict canonical replay via `reconstructStateAtMove`.
 * This exists only for legacy logs/parity fixtures recorded before strict
 * RR-CANON-R075 enforcement. Migrate or quarantine non-canonical records
 * rather than relying on this path long-term.
 */
export function reconstructStateAtMoveLegacy(record: GameRecord, moveIndex: number): GameState {
  const normalizedRecord: GameRecord = {
    ...record,
    moves: record.moves.map((move) => ({
      ...move,
      type: normalizeLegacyMoveType(move.type),
    })),
  };

  return reconstructStateAtMoveWithOptions(normalizedRecord, moveIndex, {
    replayCompatibility: true,
    skipAutoLineProcessing: true,
    skipSingleTerritoryAutoProcess: true,
  });
}
