/**
 * @fileoverview Tests for ScreenReaderAnnouncer component and hooks.
 *
 * Tests cover:
 * - Basic announcement functionality (simple message mode)
 * - Priority queue mode (category-based announcements)
 * - Politeness levels (polite vs assertive)
 * - Debouncing behavior
 * - Game announcement helpers
 * - State announcement hooks
 */

import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import { renderHook } from '@testing-library/react';
import {
  ScreenReaderAnnouncer,
  useScreenReaderAnnouncement,
  useGameAnnouncements,
  useGameStateAnnouncements,
  GameAnnouncements,
  mergeAnnouncementQueues,
} from '../../../src/client/components/ScreenReaderAnnouncer';
import type {
  QueuedAnnouncement,
  AnnouncementCategory,
} from '../../../src/client/components/ScreenReaderAnnouncer';

// Silence console.error during tests for cleaner output
const originalError = console.error;
beforeAll(() => {
  console.error = jest.fn();
});
afterAll(() => {
  console.error = originalError;
});

describe('ScreenReaderAnnouncer', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Simple message mode', () => {
    it('renders visually hidden live regions', () => {
      render(<ScreenReaderAnnouncer message="" />);

      expect(screen.getByTestId('sr-announcer-polite-1')).toBeInTheDocument();
      expect(screen.getByTestId('sr-announcer-polite-2')).toBeInTheDocument();
      expect(screen.getByTestId('sr-announcer-assertive-1')).toBeInTheDocument();
      expect(screen.getByTestId('sr-announcer-assertive-2')).toBeInTheDocument();
    });

    it('announces a message in the polite region by default', () => {
      const { rerender } = render(<ScreenReaderAnnouncer message="" />);

      rerender(<ScreenReaderAnnouncer message="Hello screen reader" />);

      // One of the polite regions should contain the message
      const polite1 = screen.getByTestId('sr-announcer-polite-1');
      const polite2 = screen.getByTestId('sr-announcer-polite-2');

      expect(
        polite1.textContent === 'Hello screen reader' ||
          polite2.textContent === 'Hello screen reader'
      ).toBe(true);
    });

    it('announces a message in the assertive region when politeness is assertive', () => {
      const { rerender } = render(<ScreenReaderAnnouncer message="" politeness="assertive" />);

      rerender(<ScreenReaderAnnouncer message="Urgent message" politeness="assertive" />);

      const assertive1 = screen.getByTestId('sr-announcer-assertive-1');
      const assertive2 = screen.getByTestId('sr-announcer-assertive-2');

      expect(
        assertive1.textContent === 'Urgent message' || assertive2.textContent === 'Urgent message'
      ).toBe(true);
    });

    it('clears message after timeout to allow re-announcement', async () => {
      const { rerender } = render(<ScreenReaderAnnouncer message="" />);

      rerender(<ScreenReaderAnnouncer message="First message" />);

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      const polite1 = screen.getByTestId('sr-announcer-polite-1');
      const polite2 = screen.getByTestId('sr-announcer-polite-2');

      // After timeout, message should be cleared
      expect(polite1.textContent).toBe('');
      expect(polite2.textContent).toBe('');
    });

    it('alternates between regions for consecutive announcements', () => {
      const { rerender } = render(<ScreenReaderAnnouncer message="" />);

      // First announcement
      rerender(<ScreenReaderAnnouncer message="Message 1" />);
      const polite1After1 = screen.getByTestId('sr-announcer-polite-1').textContent;

      // Clear and advance
      act(() => {
        jest.advanceTimersByTime(1100);
      });

      // Second announcement
      rerender(<ScreenReaderAnnouncer message="Message 2" />);
      const polite1After2 = screen.getByTestId('sr-announcer-polite-1').textContent;
      const polite2After2 = screen.getByTestId('sr-announcer-polite-2').textContent;

      // If first was in polite-1, second should be in polite-2 (or vice versa)
      if (polite1After1 === 'Message 1') {
        expect(polite2After2).toBe('Message 2');
      } else {
        expect(polite1After2).toBe('Message 2');
      }
    });
  });

  describe('Queue mode', () => {
    it('processes announcements from the queue', () => {
      const mockOnSpoken = jest.fn();
      const queue: QueuedAnnouncement[] = [
        {
          id: 'ann-1',
          message: 'First in queue',
          category: 'info',
          priority: 'low',
          politeness: 'polite',
          timestamp: Date.now(),
        },
      ];

      render(<ScreenReaderAnnouncer queue={queue} onAnnouncementSpoken={mockOnSpoken} />);

      const polite1 = screen.getByTestId('sr-announcer-polite-1');
      const polite2 = screen.getByTestId('sr-announcer-polite-2');

      expect(
        polite1.textContent === 'First in queue' || polite2.textContent === 'First in queue'
      ).toBe(true);
    });

    it('calls onAnnouncementSpoken after speaking', () => {
      const mockOnSpoken = jest.fn();
      const queue: QueuedAnnouncement[] = [
        {
          id: 'ann-1',
          message: 'Test message',
          category: 'info',
          priority: 'low',
          politeness: 'polite',
          timestamp: Date.now(),
        },
      ];

      render(<ScreenReaderAnnouncer queue={queue} onAnnouncementSpoken={mockOnSpoken} />);

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      expect(mockOnSpoken).toHaveBeenCalledWith('ann-1');
    });

    it('prioritizes high priority announcements', () => {
      const mockOnSpoken = jest.fn();
      const now = Date.now();
      const queue: QueuedAnnouncement[] = [
        {
          id: 'low-1',
          message: 'Low priority',
          category: 'info',
          priority: 'low',
          politeness: 'polite',
          timestamp: now,
        },
        {
          id: 'high-1',
          message: 'High priority',
          category: 'victory',
          priority: 'high',
          politeness: 'assertive',
          timestamp: now + 100, // Added later but higher priority
        },
      ];

      render(<ScreenReaderAnnouncer queue={queue} onAnnouncementSpoken={mockOnSpoken} />);

      // High priority should be announced first
      const assertive1 = screen.getByTestId('sr-announcer-assertive-1');
      const assertive2 = screen.getByTestId('sr-announcer-assertive-2');

      expect(
        assertive1.textContent === 'High priority' || assertive2.textContent === 'High priority'
      ).toBe(true);
    });

    it('uses assertive region for high priority assertive announcements', () => {
      const queue: QueuedAnnouncement[] = [
        {
          id: 'urgent-1',
          message: 'Victory!',
          category: 'victory',
          priority: 'high',
          politeness: 'assertive',
          timestamp: Date.now(),
        },
      ];

      render(<ScreenReaderAnnouncer queue={queue} />);

      const assertive1 = screen.getByTestId('sr-announcer-assertive-1');
      const assertive2 = screen.getByTestId('sr-announcer-assertive-2');

      expect(assertive1.textContent === 'Victory!' || assertive2.textContent === 'Victory!').toBe(
        true
      );
    });
  });
});

describe('useScreenReaderAnnouncement', () => {
  it('returns message and announce function', () => {
    const { result } = renderHook(() => useScreenReaderAnnouncement());

    expect(result.current.message).toBe('');
    expect(typeof result.current.announce).toBe('function');
  });

  it('updates message when announce is called', () => {
    const { result } = renderHook(() => useScreenReaderAnnouncement());

    act(() => {
      result.current.announce('Test announcement');
    });

    expect(result.current.message).toBe('Test announcement');
  });
});

describe('useGameAnnouncements', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('returns queue and management functions', () => {
    const { result } = renderHook(() => useGameAnnouncements());

    expect(result.current.queue).toEqual([]);
    expect(typeof result.current.announce).toBe('function');
    expect(typeof result.current.removeAnnouncement).toBe('function');
    expect(typeof result.current.clearQueue).toBe('function');
  });

  it('adds announcement to queue with correct properties', () => {
    const { result } = renderHook(() => useGameAnnouncements());

    act(() => {
      result.current.announce("It's your turn!", 'your_turn');
    });

    expect(result.current.queue.length).toBe(1);
    expect(result.current.queue[0].message).toBe("It's your turn!");
    expect(result.current.queue[0].category).toBe('your_turn');
    expect(result.current.queue[0].priority).toBe('high');
    expect(result.current.queue[0].politeness).toBe('polite');
  });

  it('removes announcement from queue', () => {
    const { result } = renderHook(() => useGameAnnouncements());

    act(() => {
      result.current.announce('Test', 'info');
    });

    const id = result.current.queue[0].id;

    act(() => {
      result.current.removeAnnouncement(id);
    });

    expect(result.current.queue.length).toBe(0);
  });

  it('clears all announcements', () => {
    const { result } = renderHook(() => useGameAnnouncements());

    act(() => {
      result.current.announce('Test 1', 'info');
      result.current.announce('Test 2', 'move');
      result.current.announce('Test 3', 'capture');
    });

    expect(result.current.queue.length).toBe(3);

    act(() => {
      result.current.clearQueue();
    });

    expect(result.current.queue.length).toBe(0);
  });

  it('debounces announcements of the same category', () => {
    const { result } = renderHook(() => useGameAnnouncements());

    act(() => {
      result.current.announce('Move 1', 'move');
      result.current.announce('Move 2', 'move'); // Should be debounced
      result.current.announce('Move 3', 'move'); // Should be debounced
    });

    expect(result.current.queue.length).toBe(1);
    expect(result.current.queue[0].message).toBe('Move 1');
  });

  it('allows different categories without debouncing', () => {
    const { result } = renderHook(() => useGameAnnouncements());

    act(() => {
      result.current.announce('Move', 'move');
      result.current.announce('Capture', 'capture');
      result.current.announce('Turn', 'turn_change');
    });

    expect(result.current.queue.length).toBe(3);
  });

  it('allows same category after debounce period', () => {
    const { result } = renderHook(() => useGameAnnouncements());

    act(() => {
      result.current.announce('Move 1', 'move');
    });

    expect(result.current.queue.length).toBe(1);

    act(() => {
      // Move category has 200ms debounce
      jest.advanceTimersByTime(300);
    });

    act(() => {
      result.current.announce('Move 2', 'move');
    });

    expect(result.current.queue.length).toBe(2);
  });
});

describe('GameAnnouncements helpers', () => {
  describe('turnChange', () => {
    it('returns your turn message when isYourTurn is true', () => {
      expect(GameAnnouncements.turnChange('Player 1', true)).toBe("It's your turn!");
    });

    it('returns other player turn message when isYourTurn is false', () => {
      expect(GameAnnouncements.turnChange('Player 2', false)).toBe("It's Player 2's turn.");
    });
  });

  describe('phaseTransition', () => {
    it('returns phase name only when no description', () => {
      expect(GameAnnouncements.phaseTransition('Movement Phase')).toBe('Movement Phase');
    });

    it('includes description when provided', () => {
      expect(GameAnnouncements.phaseTransition('Ring Placement', 'Place a ring on the board')).toBe(
        'Ring Placement. Place a ring on the board'
      );
    });
  });

  describe('move', () => {
    it('returns your move message', () => {
      expect(GameAnnouncements.move('Me', 'a1', 'b2', true)).toBe('You moved from a1 to b2.');
    });

    it('returns other player move message', () => {
      expect(GameAnnouncements.move('Player 2', 'c3', 'd4', false)).toBe(
        'Player 2 moved from c3 to d4.'
      );
    });
  });

  describe('placement', () => {
    it('returns simple placement message', () => {
      expect(GameAnnouncements.placement('Me', 'e5', 1, true)).toBe('You placed a ring at e5.');
    });

    it('includes stack height when greater than 1', () => {
      expect(GameAnnouncements.placement('Player 2', 'f6', 3, false)).toBe(
        'Player 2 placed a ring at f6. Stack height is now 3.'
      );
    });
  });

  describe('capture', () => {
    it('returns your capture message with rings gained', () => {
      expect(GameAnnouncements.capture('Me', 'a1', 'c3', 2, true)).toBe(
        'You captured at a1, landing at c3. Gained 2 rings.'
      );
    });

    it('handles singular ring', () => {
      expect(GameAnnouncements.capture('Player', 'a1', 'c3', 1, false)).toBe(
        'Player captured at a1, landing at c3. Gained 1 ring.'
      );
    });

    it('omits rings info when zero', () => {
      expect(GameAnnouncements.capture('Me', 'a1', 'c3', 0, true)).toBe(
        'You captured at a1, landing at c3.'
      );
    });
  });

  describe('chainCapture', () => {
    it('lists positions in path', () => {
      expect(GameAnnouncements.chainCapture(['a1', 'c3', 'e5'])).toBe(
        'Chain capture in progress. Path: a1 to c3 to e5.'
      );
    });
  });

  describe('lineFormed', () => {
    it('returns your line message', () => {
      expect(GameAnnouncements.lineFormed('Me', 5, true)).toBe(
        'You formed a line of 5! Choose your reward.'
      );
    });

    it('returns other player line message', () => {
      expect(GameAnnouncements.lineFormed('Player 2', 4, false)).toBe(
        'Player 2 formed a line of 4.'
      );
    });
  });

  describe('territoryClaimed', () => {
    it('returns your territory message with plural', () => {
      expect(GameAnnouncements.territoryClaimed('Me', 3, 10, true)).toBe(
        'You claimed 3 territory spaces. Total: 10.'
      );
    });

    it('returns singular for 1 space', () => {
      expect(GameAnnouncements.territoryClaimed('Player', 1, 5, false)).toBe(
        'Player claimed 1 territory space. Total: 5.'
      );
    });
  });

  describe('victory', () => {
    it('returns your victory by elimination', () => {
      expect(GameAnnouncements.victory('Me', 'elimination', true)).toBe(
        'Victory! You won the game by ring elimination!'
      );
    });

    it('returns other player territory victory', () => {
      expect(GameAnnouncements.victory('Player 2', 'territory', false)).toBe(
        'Game over. Player 2 won by territory control.'
      );
    });

    it('returns last player standing victory', () => {
      expect(GameAnnouncements.victory('Player 1', 'last_player_standing', false)).toBe(
        'Game over. Player 1 won as last player standing.'
      );
    });
  });

  describe('playerEliminated', () => {
    it('returns your elimination message', () => {
      expect(GameAnnouncements.playerEliminated('Me', true)).toBe(
        "You've been eliminated from the game."
      );
    });

    it('returns other player elimination', () => {
      expect(GameAnnouncements.playerEliminated('Player 3', false)).toBe(
        'Player 3 has been eliminated.'
      );
    });
  });

  describe('timerWarning', () => {
    it('returns urgent warning for 10 seconds or less', () => {
      expect(GameAnnouncements.timerWarning(5)).toBe('Warning! 5 seconds remaining!');
    });

    it('returns normal warning for more than 10 seconds', () => {
      expect(GameAnnouncements.timerWarning(30)).toBe('30 seconds remaining.');
    });
  });

  describe('cellSelected', () => {
    it('returns simple selection message', () => {
      expect(GameAnnouncements.cellSelected('a1')).toBe('Selected a1.');
    });

    it('includes stack info when provided', () => {
      expect(GameAnnouncements.cellSelected('b2', 'Height 3, cap 2')).toBe(
        'Selected b2. Height 3, cap 2'
      );
    });
  });

  describe('validMoves', () => {
    it('returns no valid moves message', () => {
      expect(GameAnnouncements.validMoves(0)).toBe('No valid moves available.');
    });

    it('returns singular move message', () => {
      expect(GameAnnouncements.validMoves(1)).toBe('1 valid move available.');
    });

    it('returns plural moves message', () => {
      expect(GameAnnouncements.validMoves(5)).toBe('5 valid moves available.');
    });
  });

  describe('decisionRequired', () => {
    it('returns decision message with options count', () => {
      expect(GameAnnouncements.decisionRequired('Line reward', 3)).toBe(
        'Decision required: Line reward. 3 options available.'
      );
    });

    it('handles singular option', () => {
      expect(GameAnnouncements.decisionRequired('Territory region', 1)).toBe(
        'Decision required: Territory region. 1 option available.'
      );
    });
  });

  describe('ringStats', () => {
    it('returns formatted ring statistics', () => {
      expect(GameAnnouncements.ringStats('Player 1', 5, 10, 3)).toBe(
        'Player 1: 5 rings in hand, 10 on board, 3 eliminated.'
      );
    });
  });
});

describe('useGameStateAnnouncements', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('announces turn change when isYourTurn changes', () => {
    const mockAnnounce = jest.fn();

    const { rerender } = renderHook(
      ({ isYourTurn }) =>
        useGameStateAnnouncements({
          currentPlayerName: 'Player 1',
          isYourTurn,
          announce: mockAnnounce,
        }),
      { initialProps: { isYourTurn: false } }
    );

    // First render with isYourTurn=false
    expect(mockAnnounce).toHaveBeenCalledWith("It's Player 1's turn.", 'turn_change');

    mockAnnounce.mockClear();

    // Change to your turn
    rerender({ isYourTurn: true });

    expect(mockAnnounce).toHaveBeenCalledWith("It's your turn!", 'your_turn');
  });

  it('announces phase transitions', () => {
    const mockAnnounce = jest.fn();

    const { rerender } = renderHook(
      ({ phase, previousPhase }: { phase: string; previousPhase: string | undefined }) =>
        useGameStateAnnouncements({
          phase,
          previousPhase,
          phaseDescription: 'Move your pieces',
          announce: mockAnnounce,
        }),
      { initialProps: { phase: 'ring_placement', previousPhase: undefined as string | undefined } }
    );

    // No announcement on initial render (no previousPhase)
    expect(mockAnnounce).not.toHaveBeenCalled();

    // Change phase
    rerender({ phase: 'movement', previousPhase: 'ring_placement' });

    expect(mockAnnounce).toHaveBeenCalledWith('movement. Move your pieces', 'phase_transition');
  });

  it('announces timer warnings at thresholds', () => {
    const mockAnnounce = jest.fn();

    const { rerender } = renderHook(
      ({ timeRemaining }) =>
        useGameStateAnnouncements({
          timeRemaining,
          announce: mockAnnounce,
        }),
      { initialProps: { timeRemaining: 120000 } } // 2 minutes
    );

    // No warning at 2 minutes
    expect(mockAnnounce).not.toHaveBeenCalled();

    // Cross 60 second threshold
    rerender({ timeRemaining: 59000 });
    expect(mockAnnounce).toHaveBeenCalledWith('59 seconds remaining.', 'timer_warning');

    mockAnnounce.mockClear();

    // Cross 30 second threshold
    rerender({ timeRemaining: 29000 });
    expect(mockAnnounce).toHaveBeenCalledWith('29 seconds remaining.', 'timer_warning');

    mockAnnounce.mockClear();

    // Cross 10 second threshold (urgent)
    rerender({ timeRemaining: 9000 });
    expect(mockAnnounce).toHaveBeenCalledWith('Warning! 9 seconds remaining!', 'timer_warning');
  });

  it('announces game over', () => {
    const mockAnnounce = jest.fn();

    const { rerender } = renderHook(
      ({ isGameOver }) =>
        useGameStateAnnouncements({
          isGameOver,
          winnerName: 'Player 1',
          victoryCondition: 'elimination',
          isWinner: true,
          announce: mockAnnounce,
        }),
      { initialProps: { isGameOver: false } }
    );

    expect(mockAnnounce).not.toHaveBeenCalled();

    rerender({ isGameOver: true });

    expect(mockAnnounce).toHaveBeenCalledWith(
      'Victory! You won the game by ring elimination!',
      'victory'
    );
  });

  it('announces defeat when not the winner', () => {
    const mockAnnounce = jest.fn();

    const { rerender } = renderHook(
      ({ isGameOver }) =>
        useGameStateAnnouncements({
          isGameOver,
          winnerName: 'Player 2',
          victoryCondition: 'territory',
          isWinner: false,
          announce: mockAnnounce,
        }),
      { initialProps: { isGameOver: false } }
    );

    rerender({ isGameOver: true });

    expect(mockAnnounce).toHaveBeenCalledWith(
      'Game over. Player 2 won by territory control.',
      'defeat'
    );
  });
});

describe('mergeAnnouncementQueues', () => {
  it('merges multiple queues and sorts by priority', () => {
    const now = Date.now();

    const queue1: QueuedAnnouncement[] = [
      {
        id: 'low-1',
        message: 'Low',
        category: 'info',
        priority: 'low',
        politeness: 'polite',
        timestamp: now,
      },
    ];

    const queue2: QueuedAnnouncement[] = [
      {
        id: 'high-1',
        message: 'High',
        category: 'victory',
        priority: 'high',
        politeness: 'assertive',
        timestamp: now,
      },
      {
        id: 'medium-1',
        message: 'Medium',
        category: 'capture',
        priority: 'medium',
        politeness: 'polite',
        timestamp: now,
      },
    ];

    const merged = mergeAnnouncementQueues(queue1, queue2);

    expect(merged.length).toBe(3);
    expect(merged[0].priority).toBe('high');
    expect(merged[1].priority).toBe('medium');
    expect(merged[2].priority).toBe('low');
  });

  it('sorts by timestamp when priority is equal', () => {
    const now = Date.now();

    const queue: QueuedAnnouncement[] = [
      {
        id: 'later',
        message: 'Later',
        category: 'info',
        priority: 'low',
        politeness: 'polite',
        timestamp: now + 100,
      },
      {
        id: 'earlier',
        message: 'Earlier',
        category: 'move',
        priority: 'low',
        politeness: 'polite',
        timestamp: now,
      },
    ];

    const merged = mergeAnnouncementQueues(queue);

    expect(merged[0].id).toBe('earlier');
    expect(merged[1].id).toBe('later');
  });

  it('handles empty queues', () => {
    expect(mergeAnnouncementQueues()).toEqual([]);
    expect(mergeAnnouncementQueues([], [])).toEqual([]);
  });
});
