import { renderHook, act } from '@testing-library/react';
import { useFirstTimePlayer } from '../../../src/client/hooks/useFirstTimePlayer';

const STORAGE_KEY = 'ringrift_onboarding';

describe('useFirstTimePlayer', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();
    jest.clearAllMocks();
  });

  describe('initial state', () => {
    it('returns default state for new users', () => {
      const { result } = renderHook(() => useFirstTimePlayer());

      expect(result.current.state.hasSeenWelcome).toBe(false);
      expect(result.current.state.hasCompletedFirstGame).toBe(false);
      expect(result.current.state.hasSeenControlsHelp).toBe(false);
      expect(result.current.state.gamesPlayed).toBe(0);
      expect(typeof result.current.state.firstVisit).toBe('number');
    });

    it('isFirstTimePlayer is true for new users', () => {
      const { result } = renderHook(() => useFirstTimePlayer());
      expect(result.current.isFirstTimePlayer).toBe(true);
    });

    it('shouldShowWelcome is true for new users', () => {
      const { result } = renderHook(() => useFirstTimePlayer());
      expect(result.current.shouldShowWelcome).toBe(true);
    });
  });

  describe('localStorage persistence', () => {
    it('saves state to localStorage on changes', () => {
      const { result } = renderHook(() => useFirstTimePlayer());

      act(() => {
        result.current.markWelcomeSeen();
      });

      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      expect(stored.hasSeenWelcome).toBe(true);
    });

    it('loads existing state from localStorage', () => {
      // Set up existing state
      const existingState = {
        hasSeenWelcome: true,
        hasCompletedFirstGame: true,
        hasSeenControlsHelp: true,
        gamesPlayed: 5,
        firstVisit: 1700000000000,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(existingState));

      const { result } = renderHook(() => useFirstTimePlayer());

      expect(result.current.state.hasSeenWelcome).toBe(true);
      expect(result.current.state.hasCompletedFirstGame).toBe(true);
      expect(result.current.state.hasSeenControlsHelp).toBe(true);
      expect(result.current.state.gamesPlayed).toBe(5);
      expect(result.current.state.firstVisit).toBe(1700000000000);
      expect(result.current.isFirstTimePlayer).toBe(false);
      expect(result.current.shouldShowWelcome).toBe(false);
    });

    it('handles partial state in localStorage', () => {
      // Only some fields set
      const partialState = {
        hasSeenWelcome: true,
        gamesPlayed: 2,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(partialState));

      const { result } = renderHook(() => useFirstTimePlayer());

      expect(result.current.state.hasSeenWelcome).toBe(true);
      expect(result.current.state.gamesPlayed).toBe(2);
      // Defaults should fill in missing fields
      expect(result.current.state.hasCompletedFirstGame).toBe(false);
      expect(result.current.state.hasSeenControlsHelp).toBe(false);
    });

    it('handles invalid JSON in localStorage gracefully', () => {
      localStorage.setItem(STORAGE_KEY, 'not valid json');

      const { result } = renderHook(() => useFirstTimePlayer());

      // Should fall back to defaults
      expect(result.current.state.hasSeenWelcome).toBe(false);
      expect(result.current.state.hasCompletedFirstGame).toBe(false);
      expect(result.current.state.gamesPlayed).toBe(0);
    });
  });

  describe('markWelcomeSeen', () => {
    it('sets hasSeenWelcome to true', () => {
      const { result } = renderHook(() => useFirstTimePlayer());

      expect(result.current.state.hasSeenWelcome).toBe(false);
      expect(result.current.shouldShowWelcome).toBe(true);

      act(() => {
        result.current.markWelcomeSeen();
      });

      expect(result.current.state.hasSeenWelcome).toBe(true);
      expect(result.current.shouldShowWelcome).toBe(false);
    });

    it('preserves other state when marking welcome seen', () => {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ gamesPlayed: 3, firstVisit: 1700000000000 })
      );

      const { result } = renderHook(() => useFirstTimePlayer());

      act(() => {
        result.current.markWelcomeSeen();
      });

      expect(result.current.state.gamesPlayed).toBe(3);
      expect(result.current.state.firstVisit).toBe(1700000000000);
    });
  });

  describe('markGameCompleted', () => {
    it('sets hasCompletedFirstGame and increments gamesPlayed', () => {
      const { result } = renderHook(() => useFirstTimePlayer());

      expect(result.current.state.hasCompletedFirstGame).toBe(false);
      expect(result.current.state.gamesPlayed).toBe(0);
      expect(result.current.isFirstTimePlayer).toBe(true);

      act(() => {
        result.current.markGameCompleted();
      });

      expect(result.current.state.hasCompletedFirstGame).toBe(true);
      expect(result.current.state.gamesPlayed).toBe(1);
      expect(result.current.isFirstTimePlayer).toBe(false);
    });

    it('increments gamesPlayed on subsequent completions', () => {
      const { result } = renderHook(() => useFirstTimePlayer());

      act(() => {
        result.current.markGameCompleted();
      });
      expect(result.current.state.gamesPlayed).toBe(1);

      act(() => {
        result.current.markGameCompleted();
      });
      expect(result.current.state.gamesPlayed).toBe(2);

      act(() => {
        result.current.markGameCompleted();
      });
      expect(result.current.state.gamesPlayed).toBe(3);
    });
  });

  describe('markControlsHelpSeen', () => {
    it('sets hasSeenControlsHelp to true', () => {
      const { result } = renderHook(() => useFirstTimePlayer());

      expect(result.current.state.hasSeenControlsHelp).toBe(false);

      act(() => {
        result.current.markControlsHelpSeen();
      });

      expect(result.current.state.hasSeenControlsHelp).toBe(true);
    });
  });

  describe('resetOnboarding', () => {
    it('resets all state to defaults', () => {
      // Set up a used state
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          hasSeenWelcome: true,
          hasCompletedFirstGame: true,
          hasSeenControlsHelp: true,
          gamesPlayed: 10,
          firstVisit: 1600000000000,
        })
      );

      const { result } = renderHook(() => useFirstTimePlayer());

      // Verify loaded state
      expect(result.current.state.hasSeenWelcome).toBe(true);
      expect(result.current.state.gamesPlayed).toBe(10);

      act(() => {
        result.current.resetOnboarding();
      });

      expect(result.current.state.hasSeenWelcome).toBe(false);
      expect(result.current.state.hasCompletedFirstGame).toBe(false);
      expect(result.current.state.hasSeenControlsHelp).toBe(false);
      expect(result.current.state.gamesPlayed).toBe(0);
      // firstVisit should be reset to current time
      expect(result.current.state.firstVisit).toBeGreaterThan(1600000000000);
    });

    it('clears localStorage before syncing new default state', () => {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ hasSeenWelcome: true, gamesPlayed: 10 }));

      const { result } = renderHook(() => useFirstTimePlayer());

      act(() => {
        result.current.resetOnboarding();
      });

      // After reset, the new default state is synced to localStorage by the effect
      // so storage won't be null, but it will have the reset values
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      expect(stored.hasSeenWelcome).toBe(false);
      expect(stored.gamesPlayed).toBe(0);
      expect(stored.hasCompletedFirstGame).toBe(false);
    });

    it('returns to first-time player state after reset', () => {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          hasSeenWelcome: true,
          hasCompletedFirstGame: true,
        })
      );

      const { result } = renderHook(() => useFirstTimePlayer());

      expect(result.current.isFirstTimePlayer).toBe(false);
      expect(result.current.shouldShowWelcome).toBe(false);

      act(() => {
        result.current.resetOnboarding();
      });

      expect(result.current.isFirstTimePlayer).toBe(true);
      expect(result.current.shouldShowWelcome).toBe(true);
    });
  });

  describe('derived flags', () => {
    it('isFirstTimePlayer is based on hasCompletedFirstGame', () => {
      const { result } = renderHook(() => useFirstTimePlayer());

      // New user - first time player
      expect(result.current.isFirstTimePlayer).toBe(true);

      // See welcome but don't complete game - still first time
      act(() => {
        result.current.markWelcomeSeen();
      });
      expect(result.current.isFirstTimePlayer).toBe(true);

      // Complete game - no longer first time
      act(() => {
        result.current.markGameCompleted();
      });
      expect(result.current.isFirstTimePlayer).toBe(false);
    });

    it('shouldShowWelcome is based on hasSeenWelcome', () => {
      const { result } = renderHook(() => useFirstTimePlayer());

      // New user - should show
      expect(result.current.shouldShowWelcome).toBe(true);

      // Mark as seen - should not show
      act(() => {
        result.current.markWelcomeSeen();
      });
      expect(result.current.shouldShowWelcome).toBe(false);
    });
  });

  describe('callback stability', () => {
    it('callbacks have stable references', () => {
      const { result, rerender } = renderHook(() => useFirstTimePlayer());

      const markWelcomeSeen1 = result.current.markWelcomeSeen;
      const markGameCompleted1 = result.current.markGameCompleted;
      const markControlsHelpSeen1 = result.current.markControlsHelpSeen;
      const resetOnboarding1 = result.current.resetOnboarding;

      rerender();

      expect(result.current.markWelcomeSeen).toBe(markWelcomeSeen1);
      expect(result.current.markGameCompleted).toBe(markGameCompleted1);
      expect(result.current.markControlsHelpSeen).toBe(markControlsHelpSeen1);
      expect(result.current.resetOnboarding).toBe(resetOnboarding1);
    });
  });
});
