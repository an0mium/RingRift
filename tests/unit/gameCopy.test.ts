import { getGameOverBannerText } from '../../src/client/utils/gameCopy';
import type { GameResult } from '../../src/shared/types/game';

describe('getGameOverBannerText', () => {
  const cases: Array<[GameResult['reason'], string]> = [
    ['ring_elimination', 'Game over – victory by ring elimination.'],
    ['territory_control', 'Game over – victory by territory control.'],
    ['last_player_standing', 'Game over – victory by last player standing.'],
    ['timeout', 'Game over – victory on time.'],
    ['resignation', 'Game over – victory by resignation.'],
    ['abandonment', 'Game over – game abandoned.'],
    ['draw', 'Game over – draw.'],
  ];

  it.each(cases)('returns expected copy for reason %s', (reason, expected) => {
    expect(getGameOverBannerText(reason)).toBe(expected);
  });

  it('falls back to generic copy for unknown reason', () => {
    expect(getGameOverBannerText('unknown' as GameResult['reason'])).toBe('Game over.');
  });
});
