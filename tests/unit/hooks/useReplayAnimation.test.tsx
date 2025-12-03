/**
 * Tests for useReplayAnimation hook.
 *
 * This hook handles move animations during game replay playback.
 */

import React, { useState, useCallback } from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import {
  useReplayAnimation,
  UseReplayAnimationOptions,
} from '../../../src/client/hooks/useReplayAnimation';
import type { ReplayMoveRecord } from '../../../src/client/types/replay';
import type { MoveAnimationData } from '../../../src/client/components/BoardView';

// Helper to create mock move records
function createMoveRecord(
  moveNumber: number,
  moveType: string,
  from?: { x: number; y: number; z?: number },
  to?: { x: number; y: number; z?: number },
  player = 1
): ReplayMoveRecord {
  return {
    moveNumber,
    turnNumber: moveNumber,
    player,
    phase: 'movement',
    moveType,
    move: {
      ...(from ? { from } : {}),
      ...(to ? { to } : {}),
    },
    timestamp: null,
    thinkTimeMs: null,
  };
}

// Test harness component
interface HarnessProps {
  initialMoveNumber?: number;
  initialMoves?: ReplayMoveRecord[];
  initialIsPlaying?: boolean;
  initialEnabled?: boolean;
}

function TestHarness({
  initialMoveNumber = 0,
  initialMoves = [],
  initialIsPlaying = false,
  initialEnabled = true,
}: HarnessProps) {
  const [moveNumber, setMoveNumber] = useState(initialMoveNumber);
  const [moves, setMoves] = useState(initialMoves);
  const [isPlaying, setIsPlaying] = useState(initialIsPlaying);
  const [enabled, setEnabled] = useState(initialEnabled);

  const { pendingAnimation, clearAnimation } = useReplayAnimation({
    currentMoveNumber: moveNumber,
    moves,
    isPlaying,
    enabled,
  });

  // Expose control functions for tests
  const stepForward = useCallback(() => {
    setMoveNumber((n) => n + 1);
  }, []);

  const stepBackward = useCallback(() => {
    setMoveNumber((n) => Math.max(0, n - 1));
  }, []);

  const jumpTo = useCallback((n: number) => {
    setMoveNumber(n);
  }, []);

  const togglePlay = useCallback(() => {
    setIsPlaying((p) => !p);
  }, []);

  const toggleEnabled = useCallback(() => {
    setEnabled((e) => !e);
  }, []);

  const updateMoves = useCallback((newMoves: ReplayMoveRecord[]) => {
    setMoves(newMoves);
  }, []);

  return (
    <div>
      <div data-testid="move-number">{moveNumber}</div>
      <div data-testid="is-playing">{isPlaying ? 'true' : 'false'}</div>
      <div data-testid="enabled">{enabled ? 'true' : 'false'}</div>
      <div data-testid="pending-animation">
        {pendingAnimation ? JSON.stringify(pendingAnimation) : 'null'}
      </div>
      <div data-testid="animation-type">{pendingAnimation?.type ?? 'none'}</div>
      <div data-testid="animation-to">
        {pendingAnimation?.to ? `${pendingAnimation.to.x},${pendingAnimation.to.y}` : 'none'}
      </div>
      <button data-testid="step-forward" onClick={stepForward}>
        Step Forward
      </button>
      <button data-testid="step-backward" onClick={stepBackward}>
        Step Backward
      </button>
      <button data-testid="jump-to-5" onClick={() => jumpTo(5)}>
        Jump to 5
      </button>
      <button data-testid="toggle-play" onClick={togglePlay}>
        Toggle Play
      </button>
      <button data-testid="toggle-enabled" onClick={toggleEnabled}>
        Toggle Enabled
      </button>
      <button data-testid="clear-animation" onClick={clearAnimation}>
        Clear Animation
      </button>
      <button
        data-testid="update-moves"
        onClick={() =>
          updateMoves([createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 })])
        }
      >
        Update Moves
      </button>
    </div>
  );
}

describe('useReplayAnimation', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('initial state', () => {
    it('should return null pendingAnimation initially', () => {
      render(<TestHarness />);
      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });

    it('should start with no animation type', () => {
      render(<TestHarness />);
      expect(screen.getByTestId('animation-type').textContent).toBe('none');
    });
  });

  describe('stepping forward', () => {
    const moves = [
      createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 }),
      createMoveRecord(2, 'move_stack', { x: 2, y: 2 }, { x: 3, y: 3 }),
      createMoveRecord(3, 'overtaking_capture', { x: 4, y: 4 }, { x: 5, y: 5 }),
    ];

    it('should trigger animation when stepping forward by 1', async () => {
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      expect(screen.getByTestId('animation-type').textContent).toBe('move');
      expect(screen.getByTestId('animation-to').textContent).toBe('1,1');
    });

    it('should update animation for each forward step', async () => {
      render(<TestHarness initialMoves={moves} />);

      // Step to move 1
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      expect(screen.getByTestId('animation-to').textContent).toBe('1,1');

      // Step to move 2
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      expect(screen.getByTestId('animation-to').textContent).toBe('3,3');

      // Step to move 3 (capture)
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      expect(screen.getByTestId('animation-type').textContent).toBe('capture');
      expect(screen.getByTestId('animation-to').textContent).toBe('5,5');
    });
  });

  describe('stepping backward', () => {
    const moves = [
      createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 }),
      createMoveRecord(2, 'move_stack', { x: 2, y: 2 }, { x: 3, y: 3 }),
    ];

    it('should clear animation when stepping backward', async () => {
      render(<TestHarness initialMoveNumber={2} initialMoves={moves} />);

      // First step forward to trigger an animation
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      // The animation might be set, but let's step backward now
      await act(async () => {
        screen.getByTestId('step-backward').click();
      });

      // Animation should be cleared on backward step
      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });
  });

  describe('jumping to move', () => {
    const moves = Array.from({ length: 10 }, (_, i) =>
      createMoveRecord(i + 1, 'move_stack', { x: i, y: i }, { x: i + 1, y: i + 1 })
    );

    it('should clear animation when jumping more than 1 move', async () => {
      render(<TestHarness initialMoves={moves} />);

      // First trigger an animation
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      expect(screen.getByTestId('animation-type').textContent).toBe('move');

      // Jump to move 5 (more than 1 step)
      await act(async () => {
        screen.getByTestId('jump-to-5').click();
      });

      // Animation should be cleared
      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });
  });

  describe('animation types', () => {
    it('should return "place" for place_ring moves', async () => {
      const moves = [createMoveRecord(1, 'place_ring', undefined, { x: 1, y: 1 })];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      expect(screen.getByTestId('animation-type').textContent).toBe('place');
    });

    it('should return "capture" for overtaking_capture moves', async () => {
      const moves = [createMoveRecord(1, 'overtaking_capture', { x: 0, y: 0 }, { x: 1, y: 1 })];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      expect(screen.getByTestId('animation-type').textContent).toBe('capture');
    });

    it('should return "capture" for chain_capture moves', async () => {
      const moves = [createMoveRecord(1, 'chain_capture', { x: 0, y: 0 }, { x: 1, y: 1 })];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      expect(screen.getByTestId('animation-type').textContent).toBe('capture');
    });

    it('should return "chain_capture" for continue_capture_segment moves', async () => {
      const moves = [
        createMoveRecord(1, 'continue_capture_segment', { x: 0, y: 0 }, { x: 1, y: 1 }),
      ];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      expect(screen.getByTestId('animation-type').textContent).toBe('chain_capture');
    });

    it('should return "move" for unknown move types', async () => {
      const moves = [createMoveRecord(1, 'unknown_type', { x: 0, y: 0 }, { x: 1, y: 1 })];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      expect(screen.getByTestId('animation-type').textContent).toBe('move');
    });
  });

  describe('position extraction', () => {
    it('should handle moves with only "to" position', async () => {
      const moves = [createMoveRecord(1, 'place_ring', undefined, { x: 5, y: 5 })];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      expect(screen.getByTestId('animation-to').textContent).toBe('5,5');
    });

    it('should skip moves without destination', async () => {
      const moves = [
        {
          moveNumber: 1,
          turnNumber: 1,
          player: 1,
          phase: 'movement',
          moveType: 'skip_placement',
          move: {},
          timestamp: null,
          thinkTimeMs: null,
        } as ReplayMoveRecord,
      ];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      // No animation should be triggered for skip moves
      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });

    it('should handle hex positions with z coordinate', async () => {
      const moves = [
        createMoveRecord(1, 'move_stack', { x: 0, y: 0, z: 0 }, { x: 1, y: -1, z: 0 }),
      ];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      const animation = JSON.parse(screen.getByTestId('pending-animation').textContent!);
      expect(animation.to).toEqual({ x: 1, y: -1, z: 0 });
      expect(animation.from).toEqual({ x: 0, y: 0, z: 0 });
    });
  });

  describe('enabled flag', () => {
    it('should not trigger animations when disabled', async () => {
      const moves = [createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 })];
      render(<TestHarness initialMoves={moves} initialEnabled={false} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });

    it('should track move changes even when disabled', async () => {
      const moves = [
        createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 }),
        createMoveRecord(2, 'move_stack', { x: 2, y: 2 }, { x: 3, y: 3 }),
      ];
      render(<TestHarness initialMoves={moves} initialEnabled={false} />);

      // Step forward while disabled
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      expect(screen.getByTestId('move-number').textContent).toBe('1');
      expect(screen.getByTestId('pending-animation').textContent).toBe('null');

      // Re-enable and step forward
      await act(async () => {
        screen.getByTestId('toggle-enabled').click();
      });
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      // Should animate the new step
      expect(screen.getByTestId('move-number').textContent).toBe('2');
      expect(screen.getByTestId('animation-to').textContent).toBe('3,3');
    });
  });

  describe('clearAnimation', () => {
    it('should clear the pending animation when called', async () => {
      const moves = [createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 })];
      render(<TestHarness initialMoves={moves} />);

      // Trigger animation
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      expect(screen.getByTestId('animation-type').textContent).toBe('move');

      // Clear it
      await act(async () => {
        screen.getByTestId('clear-animation').click();
      });
      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });
  });

  describe('auto-clear during playback', () => {
    it('should auto-clear animation after delay when playing', async () => {
      const moves = [createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 })];
      render(<TestHarness initialMoves={moves} initialIsPlaying={true} />);

      // Trigger animation
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      expect(screen.getByTestId('animation-type').textContent).toBe('move');

      // Advance timers to trigger auto-clear
      await act(async () => {
        jest.advanceTimersByTime(450);
      });

      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });

    it('should not auto-clear when not playing', async () => {
      const moves = [createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 })];
      render(<TestHarness initialMoves={moves} initialIsPlaying={false} />);

      // Trigger animation
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      expect(screen.getByTestId('animation-type').textContent).toBe('move');

      // Advance timers
      await act(async () => {
        jest.advanceTimersByTime(450);
      });

      // Animation should still be present
      expect(screen.getByTestId('animation-type').textContent).toBe('move');
    });
  });

  describe('player information', () => {
    it('should include player number in animation data', async () => {
      const moves = [createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 }, 2)];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      const animation = JSON.parse(screen.getByTestId('pending-animation').textContent!);
      expect(animation.playerNumber).toBe(2);
    });
  });

  describe('edge cases', () => {
    it('should handle empty moves array', async () => {
      render(<TestHarness initialMoves={[]} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });

    it('should handle move number 0', async () => {
      const moves = [createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 })];
      render(<TestHarness initialMoveNumber={0} initialMoves={moves} />);

      expect(screen.getByTestId('move-number').textContent).toBe('0');
      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });

    it('should generate unique animation IDs', async () => {
      const moves = [
        createMoveRecord(1, 'move_stack', { x: 0, y: 0 }, { x: 1, y: 1 }),
        createMoveRecord(2, 'move_stack', { x: 1, y: 1 }, { x: 2, y: 2 }),
      ];
      render(<TestHarness initialMoves={moves} />);

      // First step
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      const animation1 = JSON.parse(screen.getByTestId('pending-animation').textContent!);

      // Second step
      await act(async () => {
        screen.getByTestId('step-forward').click();
      });
      const animation2 = JSON.parse(screen.getByTestId('pending-animation').textContent!);

      expect(animation1.id).not.toBe(animation2.id);
      expect(animation1.id).toContain('replay-anim-');
      expect(animation2.id).toContain('replay-anim-');
    });

    it('should handle malformed move data gracefully', async () => {
      const moves = [
        {
          moveNumber: 1,
          turnNumber: 1,
          player: 1,
          phase: 'movement',
          moveType: 'move_stack',
          move: { from: 'invalid', to: { x: 'bad', y: 'data' } },
          timestamp: null,
          thinkTimeMs: null,
        } as unknown as ReplayMoveRecord,
      ];
      render(<TestHarness initialMoves={moves} />);

      await act(async () => {
        screen.getByTestId('step-forward').click();
      });

      // Should not crash, and should not create animation with invalid data
      expect(screen.getByTestId('pending-animation').textContent).toBe('null');
    });
  });
});
