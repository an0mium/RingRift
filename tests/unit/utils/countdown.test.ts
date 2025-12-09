import { getCountdownSeverity, msToDisplaySeconds } from '../../../src/client/utils/countdown';

describe('countdown utilities', () => {
  describe('getCountdownSeverity', () => {
    it('returns null for non-numeric or nullish values', () => {
      expect(getCountdownSeverity(null)).toBeNull();
      expect(getCountdownSeverity(undefined)).toBeNull();
      // @ts-expect-error intentional NaN test
      expect(getCountdownSeverity(NaN)).toBeNull();
    });

    it('classifies above 10s as normal', () => {
      expect(getCountdownSeverity(10_001)).toBe('normal');
      expect(getCountdownSeverity(25_000)).toBe('normal');
    });

    it('classifies between 3s and 10s as warning', () => {
      expect(getCountdownSeverity(10_000)).toBe('warning');
      expect(getCountdownSeverity(5_000)).toBe('warning');
      expect(getCountdownSeverity(3_001)).toBe('warning');
    });

    it('classifies 3s and below (including negative) as critical', () => {
      expect(getCountdownSeverity(3_000)).toBe('critical');
      expect(getCountdownSeverity(1_000)).toBe('critical');
      expect(getCountdownSeverity(0)).toBe('critical');
      expect(getCountdownSeverity(-1_000)).toBe('critical');
    });
  });

  describe('msToDisplaySeconds', () => {
    it('returns null for non-finite or nullish values', () => {
      expect(msToDisplaySeconds(null)).toBeNull();
      expect(msToDisplaySeconds(undefined)).toBeNull();
      expect(msToDisplaySeconds(Infinity)).toBeNull();
      expect(msToDisplaySeconds(-Infinity)).toBeNull();
      // @ts-expect-error intentional NaN test
      expect(msToDisplaySeconds(NaN)).toBeNull();
    });

    it('clamps negative values to zero seconds', () => {
      expect(msToDisplaySeconds(-500)).toBe(0);
      expect(msToDisplaySeconds(0)).toBe(0);
    });

    it('uses ceiling to avoid showing zero when time remains', () => {
      expect(msToDisplaySeconds(1)).toBe(1);
      expect(msToDisplaySeconds(999)).toBe(1);
      expect(msToDisplaySeconds(1_001)).toBe(2);
    });

    it('handles multi-second values', () => {
      expect(msToDisplaySeconds(5_000)).toBe(5);
      expect(msToDisplaySeconds(12_345)).toBe(13);
    });
  });
});
