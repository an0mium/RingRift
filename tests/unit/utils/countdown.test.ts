import { getCountdownSeverity, msToDisplaySeconds } from '../../../src/client/utils/countdown';

describe('getCountdownSeverity', () => {
  it('returns null for non-numeric or null input', () => {
    expect(getCountdownSeverity(null)).toBeNull();
    expect(getCountdownSeverity(undefined)).toBeNull();
    expect(getCountdownSeverity(NaN as unknown as number)).toBeNull();
  });

  it('returns "normal" when time remaining is greater than 10 seconds (ms-based)', () => {
    expect(getCountdownSeverity(11_000)).toBe('normal');
    expect(getCountdownSeverity(20_000)).toBe('normal');
  });

  it('returns "warning" when time remaining is between 3 and 10 seconds in ms (exclusive lower bound, inclusive upper bound)', () => {
    expect(getCountdownSeverity(10_000)).toBe('warning');
    expect(getCountdownSeverity(3_001)).toBe('warning');
  });

  it('returns "critical" when time remaining is at or below 3 seconds in ms, including zero and negative values', () => {
    expect(getCountdownSeverity(3_000)).toBe('critical');
    expect(getCountdownSeverity(0)).toBe('critical');
    expect(getCountdownSeverity(-1)).toBe('critical');
  });
});

describe('msToDisplaySeconds', () => {
  it('returns null for nullish or non-finite input', () => {
    expect(msToDisplaySeconds(null)).toBeNull();
    expect(msToDisplaySeconds(undefined)).toBeNull();
    expect(msToDisplaySeconds(NaN as unknown as number)).toBeNull();
    expect(msToDisplaySeconds(Infinity as unknown as number)).toBeNull();
    expect(msToDisplaySeconds(-Infinity as unknown as number)).toBeNull();
  });

  it('converts positive milliseconds to whole display seconds using ceil, never showing 0 when time remains', () => {
    expect(msToDisplaySeconds(1)).toBe(1);
    expect(msToDisplaySeconds(999)).toBe(1);
    expect(msToDisplaySeconds(1_000)).toBe(1);
    expect(msToDisplaySeconds(1_001)).toBe(2);
    expect(msToDisplaySeconds(1_500)).toBe(2);
    expect(msToDisplaySeconds(2_000)).toBe(2);
  });

  it('clamps zero and negative milliseconds to 0 display seconds', () => {
    expect(msToDisplaySeconds(0)).toBe(0);
    expect(msToDisplaySeconds(-1)).toBe(0);
    expect(msToDisplaySeconds(-1_000)).toBe(0);
  });
});

describe('msToDisplaySeconds + getCountdownSeverity integration', () => {
  it.each([
    { timeRemainingMs: 15_000, expectedSeconds: 15, expectedSeverity: 'normal' as const },
    { timeRemainingMs: 9_000, expectedSeconds: 9, expectedSeverity: 'warning' as const },
    // 2_500ms is still > 0ms so we display 3s while severity is already critical.
    { timeRemainingMs: 2_500, expectedSeconds: 3, expectedSeverity: 'critical' as const },
    { timeRemainingMs: 0, expectedSeconds: 0, expectedSeverity: 'critical' as const },
    { timeRemainingMs: -500, expectedSeconds: 0, expectedSeverity: 'critical' as const },
  ])(
    'keeps ms-based severity and seconds-based display consistent for $timeRemainingMs ms',
    ({ timeRemainingMs, expectedSeconds, expectedSeverity }) => {
      expect(msToDisplaySeconds(timeRemainingMs)).toBe(expectedSeconds);
      expect(getCountdownSeverity(timeRemainingMs)).toBe(expectedSeverity);
    }
  );
});
