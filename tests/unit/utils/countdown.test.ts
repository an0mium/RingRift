import { getCountdownSeverity, msToDisplaySeconds } from '../../../src/client/utils/countdown';

describe('countdown utilities', () => {
  describe('getCountdownSeverity', () => {
    it('returns normal above 10s', () => {
      expect(getCountdownSeverity(15_000)).toBe('normal');
    });

    it('returns warning between 3s and 10s', () => {
      expect(getCountdownSeverity(5_000)).toBe('warning');
    });

    it('returns critical at or below 3s (including negative)', () => {
      expect(getCountdownSeverity(3_000)).toBe('critical');
      expect(getCountdownSeverity(-1)).toBe('critical');
    });

    it('returns null for non-numeric input', () => {
      expect(getCountdownSeverity(null)).toBeNull();
      expect(getCountdownSeverity(undefined)).toBeNull();
      // @ts-expect-error intentional bad type
      expect(getCountdownSeverity('1000')).toBeNull();
    });
  });

  describe('msToDisplaySeconds', () => {
    it('ceil rounds positive milliseconds to display seconds', () => {
      expect(msToDisplaySeconds(1)).toBe(1);
      expect(msToDisplaySeconds(999)).toBe(1);
      expect(msToDisplaySeconds(1_001)).toBe(2);
    });

    it('clamps negative to zero', () => {
      expect(msToDisplaySeconds(-5)).toBe(0);
    });

    it('returns null for non-finite inputs', () => {
      expect(msToDisplaySeconds(null)).toBeNull();
      expect(msToDisplaySeconds(undefined)).toBeNull();
      expect(msToDisplaySeconds(Number.NaN)).toBeNull();
      expect(msToDisplaySeconds(Number.POSITIVE_INFINITY)).toBeNull();
    });
  });
});
