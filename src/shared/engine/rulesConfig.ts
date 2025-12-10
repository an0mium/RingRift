import { BOARD_CONFIGS, BoardType, RulesOptions } from '../types/game';

/**
 * Compute the effective required line length for collapse / rewards for a
 * given board + player-count combination.
 *
 * Canonical semantics (RR-CANON-R120):
 * - square8 2-player: line length = 4
 * - square8 3-4 player: line length = 3
 * - square19 and hexagonal: line length = 4 (all player counts)
 * - rulesOptions is reserved for future per-game overrides (for example,
 *   per-ruleset lineLength tweaks) and is currently unused.
 */
export function getEffectiveLineLengthThreshold(
  boardType: BoardType,
  numPlayers: number,
  _rulesOptions?: RulesOptions
): number {
  // Per RR-CANON-R120: square8 2-player games require line length 4,
  // while 3-4 player games require line length 3.
  if (boardType === 'square8' && numPlayers === 2) {
    return 4;
  }

  // For all other configurations, use the base line_length from BOARD_CONFIGS:
  // - square8 3-4p: 3
  // - square19: 4
  // - hexagonal: 4
  return BOARD_CONFIGS[boardType].lineLength;
}
