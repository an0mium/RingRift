import { BOARD_CONFIGS, BoardType, RulesOptions } from '../types/game';

/**
 * Compute the effective required line length for collapse / rewards for a
 * given board + player-count combination.
 *
 * Canonical semantics (RR-CANON-R120):
 * - square8 2-player: line length = 4
 * - square8 3-4 player: line length = 3
 * - hex8 2-player: line length = 4
 * - hex8 3-4 player: line length = 3
 * - square19 and hexagonal: line length = 4 (all player counts)
 * - rulesOptions is reserved for future per-game overrides (for example,
 *   per-ruleset lineLength tweaks) and is currently unused.
 */
export function getEffectiveLineLengthThreshold(
  boardType: BoardType,
  numPlayers: number,
  _rulesOptions?: RulesOptions
): number {
  // Per RR-CANON-R120: square8 and hex8 2-player games require line length 4,
  // while 3-4 player games require line length 3.
  if (boardType === 'square8' || boardType === 'hex8') {
    return numPlayers === 2 ? 4 : 3;
  }

  // For all other configurations, use the base line_length from BOARD_CONFIGS:
  // - square19: 4
  // - hexagonal: 4
  return BOARD_CONFIGS[boardType].lineLength;
}

/**
 * Compute the effective rings-per-player supply cap for a given board.
 *
 * Canonical default is `BOARD_CONFIGS[boardType].ringsPerPlayer`.
 * For controlled experiments / ablations, callers may override via
 * `rulesOptions.ringsPerPlayer`.
 */
export function getEffectiveRingsPerPlayer(
  boardType: BoardType,
  rulesOptions?: RulesOptions
): number {
  return rulesOptions?.ringsPerPlayer ?? BOARD_CONFIGS[boardType].ringsPerPlayer;
}
