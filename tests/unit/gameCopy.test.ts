import { getGameOverBannerText } from '../../src/client/utils/gameCopy';
import type { GameResult } from '../../src/shared/types/game';

describe('getGameOverBannerText', () => {
  const cases: Array<[GameResult['reason'], string]> = [
    ['ring_elimination', 'Victory! All opponent rings eliminated.'],
    ['territory_control', 'Victory! Territory dominance achieved.'],
    ['last_player_standing', 'Victory! Last player standing wins.'],
    ['timeout', 'Victory! Opponent ran out of time.'],
    ['resignation', 'Victory! Opponent resigned.'],
    ['abandonment', 'Game abandoned.'],
    ['draw', 'Draw — evenly matched!'],
  ];

  it.each(cases)('returns expected copy for reason %s', (reason, expected) => {
    expect(getGameOverBannerText(reason)).toBe(expected);
  });

  it('falls back to generic copy for unknown reason', () => {
    expect(getGameOverBannerText('unknown' as GameResult['reason'])).toBe('Game over.');
  });
});
