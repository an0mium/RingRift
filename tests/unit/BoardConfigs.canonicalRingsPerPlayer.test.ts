import { BOARD_CONFIGS } from '../../src/shared/types/game';

describe('BOARD_CONFIGS canonical ringsPerPlayer', () => {
  it('keeps square19/hex ringsPerPlayer stable', () => {
    expect(BOARD_CONFIGS.square19.ringsPerPlayer).toBe(72);
    expect(BOARD_CONFIGS.hexagonal.ringsPerPlayer).toBe(96);
  });
});
