/**
 * Unit tests for useCountdown hook
 *
 * Tests the countdown timer functionality including:
 * - Basic countdown behavior
 * - Pause/resume functionality
 * - Server time reconciliation
 * - Expiration callbacks
 * - Time formatting utilities
 */

import { renderHook, act } from '@testing-library/react';
import {
  useCountdown,
  useDecisionTimer,
  formatTime,
  formatTimeWithMs,
  formatTimeAdaptive,
} from '../../src/client/hooks/useCountdown';

// Mock timers for deterministic testing
jest.useFakeTimers();

describe('useCountdown', () => {
  afterEach(() => {
    jest.clearAllTimers();
  });

  describe('basic countdown behavior', () => {
    it('initializes with correct remaining time', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 10000,
        })
      );

      expect(result.current.remainingMs).toBe(10000);
      expect(result.current.remainingSeconds).toBe(10);
      expect(result.current.isExpired).toBe(false);
      expect(result.current.isActive).toBe(true);
    });

    it('counts down by intervalMs on each tick', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 5000,
          intervalMs: 1000,
        })
      );

      expect(result.current.remainingMs).toBe(5000);

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      expect(result.current.remainingMs).toBe(4000);

      act(() => {
        jest.advanceTimersByTime(2000);
      });

      expect(result.current.remainingMs).toBe(2000);
    });

    it('stops at minMs and sets isExpired', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 2000,
          intervalMs: 1000,
          minMs: 0,
        })
      );

      act(() => {
        jest.advanceTimersByTime(3000);
      });

      expect(result.current.remainingMs).toBe(0);
      expect(result.current.isExpired).toBe(true);
      expect(result.current.isActive).toBe(false);
    });

    it('respects custom minMs', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 5000,
          intervalMs: 1000,
          minMs: 2000,
        })
      );

      act(() => {
        jest.advanceTimersByTime(10000);
      });

      expect(result.current.remainingMs).toBe(2000);
      expect(result.current.isExpired).toBe(true);
    });
  });

  describe('callbacks', () => {
    it('calls onTick with remaining time on each tick', () => {
      const onTick = jest.fn();

      renderHook(() =>
        useCountdown({
          initialMs: 3000,
          intervalMs: 1000,
          onTick,
        })
      );

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      expect(onTick).toHaveBeenCalledWith(2000);

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      expect(onTick).toHaveBeenCalledWith(1000);
    });

    it('calls onExpire when countdown reaches minMs', () => {
      const onExpire = jest.fn();

      renderHook(() =>
        useCountdown({
          initialMs: 2000,
          intervalMs: 1000,
          onExpire,
        })
      );

      expect(onExpire).not.toHaveBeenCalled();

      act(() => {
        jest.advanceTimersByTime(2000);
      });

      // onExpire is called via setTimeout(0), so advance timers again
      act(() => {
        jest.advanceTimersByTime(0);
      });

      expect(onExpire).toHaveBeenCalledTimes(1);
    });

    it('only calls onExpire once', () => {
      const onExpire = jest.fn();

      renderHook(() =>
        useCountdown({
          initialMs: 2000,
          intervalMs: 1000,
          onExpire,
        })
      );

      // Advance past expiration
      act(() => {
        jest.advanceTimersByTime(2000);
      });

      // Run any pending setTimeout(0) callbacks
      act(() => {
        jest.runAllTimers();
      });

      // Continue advancing to ensure onExpire isn't called again
      act(() => {
        jest.advanceTimersByTime(3000);
      });

      expect(onExpire).toHaveBeenCalledTimes(1);
    });
  });

  describe('pause/resume', () => {
    it('pauses countdown when pause is called', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 5000,
          intervalMs: 1000,
        })
      );

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      expect(result.current.remainingMs).toBe(4000);

      act(() => {
        result.current.pause();
      });

      act(() => {
        jest.advanceTimersByTime(2000);
      });

      // Should still be 4000 because paused
      expect(result.current.remainingMs).toBe(4000);
      expect(result.current.isActive).toBe(false);
    });

    it('resumes countdown when resume is called', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 5000,
          intervalMs: 1000,
        })
      );

      act(() => {
        result.current.pause();
      });

      act(() => {
        jest.advanceTimersByTime(2000);
      });

      expect(result.current.remainingMs).toBe(5000);

      act(() => {
        result.current.resume();
      });

      act(() => {
        jest.advanceTimersByTime(2000);
      });

      expect(result.current.remainingMs).toBe(3000);
    });
  });

  describe('reset', () => {
    it('resets to initial value', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 5000,
          intervalMs: 1000,
        })
      );

      act(() => {
        jest.advanceTimersByTime(3000);
      });

      expect(result.current.remainingMs).toBe(2000);

      act(() => {
        result.current.reset();
      });

      expect(result.current.remainingMs).toBe(5000);
      expect(result.current.isExpired).toBe(false);
    });

    it('resets to new value when provided', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 5000,
          intervalMs: 1000,
        })
      );

      act(() => {
        result.current.reset(10000);
      });

      expect(result.current.remainingMs).toBe(10000);
    });

    it('resumes countdown after reset even if previously paused', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 5000,
          intervalMs: 1000,
        })
      );

      act(() => {
        result.current.pause();
        result.current.reset();
      });

      expect(result.current.isActive).toBe(true);

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      expect(result.current.remainingMs).toBe(4000);
    });
  });

  describe('server reconciliation', () => {
    it('reconciles with server time when provided', () => {
      const { result, rerender } = renderHook(
        ({ serverRemainingMs }) =>
          useCountdown({
            initialMs: 10000,
            intervalMs: 1000,
            serverRemainingMs,
          }),
        { initialProps: { serverRemainingMs: null as number | null } }
      );

      expect(result.current.remainingMs).toBe(10000);

      // Server says only 5 seconds left
      rerender({ serverRemainingMs: 5000 });

      expect(result.current.remainingMs).toBe(5000);
    });

    it('takes minimum of server and client time', () => {
      const { result, rerender } = renderHook(
        ({ serverRemainingMs }) =>
          useCountdown({
            initialMs: 10000,
            intervalMs: 1000,
            serverRemainingMs,
          }),
        { initialProps: { serverRemainingMs: null as number | null } }
      );

      act(() => {
        jest.advanceTimersByTime(3000);
      });

      // Client is at 7000ms
      expect(result.current.remainingMs).toBe(7000);

      // Server says 8000ms (more than client) - should keep client value
      rerender({ serverRemainingMs: 8000 });
      expect(result.current.remainingMs).toBe(7000);

      // Server says 5000ms (less than client) - should use server value
      rerender({ serverRemainingMs: 5000 });
      expect(result.current.remainingMs).toBe(5000);
    });
  });

  describe('isActive control', () => {
    it('does not tick when isActive is false', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 5000,
          intervalMs: 1000,
          isActive: false,
        })
      );

      act(() => {
        jest.advanceTimersByTime(3000);
      });

      expect(result.current.remainingMs).toBe(5000);
      expect(result.current.isActive).toBe(false);
    });

    it('starts ticking when isActive becomes true', () => {
      const { result, rerender } = renderHook(
        ({ isActive }) =>
          useCountdown({
            initialMs: 5000,
            intervalMs: 1000,
            isActive,
          }),
        { initialProps: { isActive: false } }
      );

      act(() => {
        jest.advanceTimersByTime(2000);
      });

      expect(result.current.remainingMs).toBe(5000);

      rerender({ isActive: true });

      act(() => {
        jest.advanceTimersByTime(2000);
      });

      expect(result.current.remainingMs).toBe(3000);
    });
  });

  describe('formattedTime', () => {
    it('formats time correctly', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 125000, // 2:05
        })
      );

      expect(result.current.formattedTime).toBe('2:05');
    });

    it('shows just seconds when under a minute', () => {
      const { result } = renderHook(() =>
        useCountdown({
          initialMs: 45000, // 45 seconds
        })
      );

      expect(result.current.formattedTime).toBe('45');
    });
  });
});

describe('formatTime', () => {
  it('formats milliseconds to MM:SS for times over 60 seconds', () => {
    expect(formatTime(125000)).toBe('2:05');
    expect(formatTime(3600000)).toBe('60:00');
    expect(formatTime(65000)).toBe('1:05');
  });

  it('formats to just seconds for times under 60 seconds', () => {
    expect(formatTime(45000)).toBe('45');
    expect(formatTime(5000)).toBe('5');
    expect(formatTime(0)).toBe('0');
  });

  it('handles negative values by showing 0', () => {
    expect(formatTime(-1000)).toBe('0');
  });
});

describe('formatTimeWithMs', () => {
  it('formats with milliseconds for times over 60 seconds', () => {
    expect(formatTimeWithMs(125500)).toBe('2:05.500');
  });

  it('formats without minutes for times under 60 seconds', () => {
    expect(formatTimeWithMs(45123)).toBe('45.123');
    expect(formatTimeWithMs(5000)).toBe('5.000');
  });
});

describe('formatTimeAdaptive', () => {
  it('shows tenths for times under 10 seconds', () => {
    expect(formatTimeAdaptive(9500)).toBe('9.5');
    expect(formatTimeAdaptive(5000)).toBe('5.0');
    expect(formatTimeAdaptive(1234)).toBe('1.2');
  });

  it('shows whole seconds for 10-60 seconds', () => {
    expect(formatTimeAdaptive(45000)).toBe('45');
    expect(formatTimeAdaptive(10000)).toBe('10');
  });

  it('shows MM:SS for times over 60 seconds', () => {
    expect(formatTimeAdaptive(125000)).toBe('2:05');
    expect(formatTimeAdaptive(61000)).toBe('1:01');
  });
});

describe('useDecisionTimer', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2025-01-01T00:00:00Z'));
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it('calculates remaining time from deadline', () => {
    const deadline = Date.now() + 10000; // 10 seconds from now

    const { result } = renderHook(() =>
      useDecisionTimer({
        deadlineMs: deadline,
      })
    );

    expect(result.current.remainingMs).toBe(10000);
    expect(result.current.isExpired).toBe(false);
  });

  it('sets isWarning when within warning threshold', () => {
    const deadline = Date.now() + 5000; // 5 seconds from now

    const { result } = renderHook(() =>
      useDecisionTimer({
        deadlineMs: deadline,
        warningThresholdMs: 10000, // Warn when under 10 seconds
      })
    );

    expect(result.current.isWarning).toBe(true);
  });

  it('calls onWarning when warning threshold is crossed', () => {
    const onWarning = jest.fn();
    const deadline = Date.now() + 15000; // 15 seconds from now

    renderHook(() =>
      useDecisionTimer({
        deadlineMs: deadline,
        warningThresholdMs: 10000,
        onWarning,
      })
    );

    expect(onWarning).not.toHaveBeenCalled();

    // Advance to within warning threshold
    act(() => {
      jest.advanceTimersByTime(6000); // Now at 9 seconds remaining
    });

    expect(onWarning).toHaveBeenCalledTimes(1);
  });

  it('handles null deadline', () => {
    const { result } = renderHook(() =>
      useDecisionTimer({
        deadlineMs: null,
      })
    );

    expect(result.current.remainingMs).toBe(0);
    expect(result.current.isActive).toBe(false);
  });
});
