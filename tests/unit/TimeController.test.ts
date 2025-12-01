/**
 * Timeout Acceleration E2E Tests
 * ============================================================================
 *
 * This test suite demonstrates the usage of TimeController for testing
 * timeout scenarios that would normally take 30+ seconds in real time.
 *
 * The TimeController provides:
 * - Jest fake timer installation/uninstallation
 * - Programmatic time advancement
 * - Ability to test 30-second timeouts in milliseconds
 *
 * RUN COMMAND: npx jest tests/e2e/timeout.acceleration.test.ts
 *
 * NOTE: These tests use Jest's fake timers and don't require a running
 * backend. They test the TimeController utility itself and demonstrate
 * patterns for timeout testing.
 */

import {
  TimeController,
  createTimeController,
  createFastTimeoutController,
  createHybridTimeController,
  withTimeControl,
  waitForConditionWithTimeAdvance,
} from '../helpers/TimeController';

describe('TimeController API Verification', () => {
  let timeController: TimeController;

  afterEach(() => {
    // Ensure timers are restored after each test
    if (timeController?.isInstalled()) {
      timeController.uninstall();
    }
  });

  describe('Installation and Uninstallation', () => {
    it('creates a TimeController with default options', () => {
      timeController = createTimeController();
      expect(timeController).toBeTruthy();
      expect(timeController).toBeInstanceOf(TimeController);
      expect(timeController.isInstalled()).toBe(false);
      expect(timeController.getState()).toBe('idle');
    });

    it('installs fake timers successfully', () => {
      timeController = createTimeController();
      timeController.install();

      expect(timeController.isInstalled()).toBe(true);
      expect(timeController.getState()).toBe('installed');
    });

    it('uninstalls fake timers successfully', () => {
      timeController = createTimeController();
      timeController.install();
      timeController.uninstall();

      expect(timeController.isInstalled()).toBe(false);
      expect(timeController.getState()).toBe('idle');
    });

    it('throws when installing twice', () => {
      timeController = createTimeController();
      timeController.install();

      expect(() => timeController.install()).toThrow('TimeController is already installed');
    });

    it('uninstall is idempotent', () => {
      timeController = createTimeController();
      // Should not throw even when not installed
      expect(() => timeController.uninstall()).not.toThrow();
      expect(() => timeController.uninstall()).not.toThrow();
    });
  });

  describe('Time Advancement', () => {
    beforeEach(() => {
      timeController = createTimeController();
      timeController.install();
    });

    it('advances time by specified milliseconds', async () => {
      const startTime = timeController.getCurrentTime();

      await timeController.advanceTime(5000);

      expect(timeController.getCurrentTime()).toBe(startTime + 5000);
    });

    it('advances time by 30 seconds (decision timeout duration)', async () => {
      const startTime = timeController.getCurrentTime();

      // This would take 30 real seconds without time acceleration
      await timeController.advanceTime(30_000);

      expect(timeController.getCurrentTime()).toBe(startTime + 30_000);
    });

    it('advances time using convenience methods', async () => {
      const startTime = timeController.getCurrentTime();

      await timeController.advanceTimeBySeconds(15);

      expect(timeController.getCurrentTime()).toBe(startTime + 15_000);
    });

    it('advances time by minutes', async () => {
      const startTime = timeController.getCurrentTime();

      await timeController.advanceTimeByMinutes(2);

      expect(timeController.getCurrentTime()).toBe(startTime + 120_000);
    });

    it('fires timers when time advances', async () => {
      let timerFired = false;
      setTimeout(() => {
        timerFired = true;
      }, 5000);

      expect(timerFired).toBe(false);

      await timeController.advanceTime(5000);

      expect(timerFired).toBe(true);
    });

    it('fires multiple timers in correct order', async () => {
      const order: number[] = [];

      setTimeout(() => order.push(1), 1000);
      setTimeout(() => order.push(2), 2000);
      setTimeout(() => order.push(3), 3000);

      await timeController.advanceTime(3500);

      expect(order).toEqual([1, 2, 3]);
    });

    it('does not fire timers that have not elapsed', async () => {
      let timerFired = false;
      setTimeout(() => {
        timerFired = true;
      }, 10000);

      await timeController.advanceTime(5000);

      expect(timerFired).toBe(false);
    });

    it('throws when advancing time without installation', async () => {
      const controller = createTimeController();
      // Not installed

      await expect(controller.advanceTime(1000)).rejects.toThrow(
        'TimeController must be installed'
      );
    });
  });

  describe('Time Setting', () => {
    it('sets virtual time to a specific timestamp', () => {
      timeController = createTimeController();
      timeController.install();

      const targetTime = new Date('2024-06-15T12:00:00Z').getTime();
      timeController.setTime(targetTime);

      expect(timeController.getCurrentTime()).toBe(targetTime);
      expect(Date.now()).toBe(targetTime);
    });

    it('advanceTimeTo advances to specific timestamp', async () => {
      timeController = createTimeController({ initialTime: 1000 });
      timeController.install();

      await timeController.advanceTimeTo(5000);

      expect(timeController.getCurrentTime()).toBe(5000);
    });

    it('advanceTimeTo throws when going backwards', async () => {
      timeController = createTimeController({ initialTime: 10000 });
      timeController.install();

      await expect(timeController.advanceTimeTo(5000)).rejects.toThrow('would go backwards');
    });
  });

  describe('Timer Management', () => {
    beforeEach(() => {
      timeController = createTimeController();
      timeController.install();
    });

    it('reports pending timer count', () => {
      expect(timeController.getPendingTimerCount()).toBe(0);

      setTimeout(() => {}, 1000);
      setTimeout(() => {}, 2000);
      setInterval(() => {}, 5000);

      expect(timeController.getPendingTimerCount()).toBe(3);
    });

    it('clears all pending timers', async () => {
      let fired = false;
      setTimeout(() => {
        fired = true;
      }, 1000);

      expect(timeController.getPendingTimerCount()).toBe(1);

      timeController.clearAllTimers();

      expect(timeController.getPendingTimerCount()).toBe(0);

      // Timer should not fire even after advancing
      await timeController.advanceTime(5000);
      expect(fired).toBe(false);
    });

    it('runs all pending timers immediately', async () => {
      const order: number[] = [];

      setTimeout(() => order.push(1), 1000);
      setTimeout(() => order.push(2), 50000);
      setTimeout(() => order.push(3), 100000);

      await timeController.runAllTimers();

      expect(order).toEqual([1, 2, 3]);
    });

    it('runs only pending timers (not newly scheduled)', async () => {
      const order: number[] = [];

      setTimeout(() => {
        order.push(1);
        setTimeout(() => order.push(2), 1000); // Newly scheduled
      }, 1000);

      await timeController.runOnlyPendingTimers();

      expect(order).toEqual([1]);
    });
  });

  describe('Pause and Resume', () => {
    it('pauses time advancement', () => {
      timeController = createTimeController();
      timeController.install();

      timeController.pause();

      expect(timeController.isPaused()).toBe(true);
      expect(timeController.getState()).toBe('paused');
    });

    it('resumes time advancement', () => {
      timeController = createTimeController();
      timeController.install();
      timeController.pause();

      timeController.resume();

      expect(timeController.isPaused()).toBe(false);
      expect(timeController.getState()).toBe('installed');
    });
  });

  describe('Configuration', () => {
    it('sets and gets acceleration factor', () => {
      timeController = createTimeController({ accelerationFactor: 100 });

      expect(timeController.getAcceleration()).toBe(100);

      timeController.setAcceleration(500);
      expect(timeController.getAcceleration()).toBe(500);
    });

    it('throws for invalid acceleration factor', () => {
      timeController = createTimeController();

      expect(() => timeController.setAcceleration(0)).toThrow(
        'Acceleration factor must be positive'
      );
      expect(() => timeController.setAcceleration(-1)).toThrow(
        'Acceleration factor must be positive'
      );
    });

    it('resets controller to initial state', () => {
      timeController = createTimeController({ initialTime: 5000 });
      timeController.install();
      timeController.setTime(99999);

      timeController.reset();

      expect(timeController.isInstalled()).toBe(false);
      expect(timeController.getState()).toBe('idle');
      // After reset, the internal virtualTime is reset to initialTime
      // Reinstall to verify
      timeController.install();
      expect(timeController.getCurrentTime()).toBe(5000);
      timeController.uninstall();
    });
  });
});

describe('Factory Functions', () => {
  afterEach(() => {
    jest.useRealTimers();
  });

  it('createFastTimeoutController creates optimized controller', () => {
    const controller = createFastTimeoutController();

    expect(controller).toBeInstanceOf(TimeController);
    expect(controller.getAcceleration()).toBe(1000);
  });

  it('createHybridTimeController creates auto-advance controller', () => {
    const controller = createHybridTimeController();

    expect(controller).toBeInstanceOf(TimeController);
    expect(controller.getAcceleration()).toBe(100);
  });

  it('withTimeControl auto-installs and uninstalls', async () => {
    let wasInstalled = false;

    await withTimeControl(async (controller) => {
      wasInstalled = controller.isInstalled();
      await controller.advanceTime(1000);
    });

    expect(wasInstalled).toBe(true);
    // After withTimeControl, real timers should be restored
    // We can verify by checking Date.now works normally
    const beforeDate = Date.now();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const afterDate = Date.now();
    expect(afterDate).toBeGreaterThanOrEqual(beforeDate);
  });
});

describe('Decision Timeout Scenarios', () => {
  let timeController: TimeController;

  beforeEach(() => {
    timeController = createTimeController();
    timeController.install();
  });

  afterEach(() => {
    timeController.uninstall();
  });

  it('simulates 30-second decision timeout expiry', async () => {
    // Simulate a decision phase with 30-second timeout
    const DECISION_TIMEOUT_MS = 30_000;
    let isTimedOut = false;
    let warningEmitted = false;

    // Set up timeout handlers
    setTimeout(() => {
      warningEmitted = true;
    }, DECISION_TIMEOUT_MS - 5000); // Warning at 25s

    setTimeout(() => {
      isTimedOut = true;
    }, DECISION_TIMEOUT_MS);

    // Verify initial state
    expect(warningEmitted).toBe(false);
    expect(isTimedOut).toBe(false);

    // Advance past warning threshold
    await timeController.advanceTime(26_000);
    expect(warningEmitted).toBe(true);
    expect(isTimedOut).toBe(false);

    // Advance past timeout
    await timeController.advanceTime(5_000); // Total: 31s
    expect(isTimedOut).toBe(true);
  });

  it('simulates reconnection window expiry (30s)', async () => {
    // Simulate reconnection window
    const RECONNECT_WINDOW_MS = 30_000;
    let connectionExpired = false;

    setTimeout(() => {
      connectionExpired = true;
    }, RECONNECT_WINDOW_MS);

    // Advance time in chunks to simulate checking connection status
    for (let i = 0; i < 6; i++) {
      expect(connectionExpired).toBe(false);
      await timeController.advanceTime(4_000);
    }

    // After 24 seconds, still not expired
    expect(connectionExpired).toBe(false);

    // Advance past the window
    await timeController.advanceTime(10_000); // Total: 34s
    expect(connectionExpired).toBe(true);
  });

  it('simulates move timeout with auto-pass', async () => {
    const MOVE_TIMEOUT_MS = 30_000;
    let autoPassApplied = false;
    let moveTimeoutWarnings = 0;

    // Set up move timeout handlers
    setInterval(() => {
      moveTimeoutWarnings++;
    }, 10_000); // Warn every 10s

    setTimeout(() => {
      autoPassApplied = true;
    }, MOVE_TIMEOUT_MS);

    // Advance time to trigger warnings
    await timeController.advanceTime(25_000);
    expect(moveTimeoutWarnings).toBe(2); // At 10s and 20s
    expect(autoPassApplied).toBe(false);

    // Advance past timeout
    await timeController.advanceTime(10_000); // Total: 35s
    expect(moveTimeoutWarnings).toBe(3); // At 30s too
    expect(autoPassApplied).toBe(true);
  });

  it('advances 30s of virtual time rapidly', async () => {
    // This test verifies that 30 seconds of virtual time can be advanced
    // without actually waiting 30 seconds. While Jest's fake timers still
    // need to process microtasks (taking some real time), it should be
    // dramatically faster than real time.
    const startTime = timeController.getCurrentTime();

    // This simulates 30 seconds of virtual time
    await timeController.advanceTime(30_000);

    const endTime = timeController.getCurrentTime();

    // Verify 30 seconds of virtual time passed
    expect(endTime - startTime).toBe(30_000);
  });
});

describe('Condition Waiting Utility', () => {
  it('waitForConditionWithTimeAdvance resolves when condition is met', async () => {
    await withTimeControl(async (controller) => {
      let phase = 'waiting';

      setTimeout(() => {
        phase = 'timeout_processing';
      }, 30_000);

      await waitForConditionWithTimeAdvance(controller, () => phase === 'timeout_processing', {
        maxTime: 35_000,
        stepSize: 5_000,
      });

      expect(phase).toBe('timeout_processing');
    });
  });

  it('waitForConditionWithTimeAdvance times out with error', async () => {
    await withTimeControl(async (controller) => {
      const neverTrue = () => false;

      await expect(
        waitForConditionWithTimeAdvance(controller, neverTrue, {
          maxTime: 10_000,
          stepSize: 2_000,
          description: 'never happening condition',
        })
      ).rejects.toThrow('Timeout waiting for never happening condition');
    });
  });

  it('waitForConditionWithTimeAdvance supports async conditions', async () => {
    await withTimeControl(async (controller) => {
      let counter = 0;

      setInterval(() => {
        counter++;
      }, 1000);

      const asyncCondition = async () => {
        return counter >= 5;
      };

      await waitForConditionWithTimeAdvance(controller, asyncCondition, {
        maxTime: 10_000,
        stepSize: 1_000,
      });

      expect(counter).toBeGreaterThanOrEqual(5);
    });
  });
});

describe('Integration Patterns', () => {
  it('demonstrates game session timeout testing pattern', async () => {
    /**
     * This test demonstrates how TimeController would be used
     * to test GameSession decision phase timeouts.
     *
     * In a real test, you would:
     * 1. Create a GameSession with a pending decision
     * 2. Use TimeController to advance time past the timeout
     * 3. Verify the decision was auto-resolved
     */

    await withTimeControl(async (controller) => {
      // Simulate game session state
      interface MockGameState {
        phase: 'placement' | 'movement' | 'line_processing' | 'timeout_processing';
        decisionPending: boolean;
        autoResolved: boolean;
      }

      const gameState: MockGameState = {
        phase: 'line_processing',
        decisionPending: true,
        autoResolved: false,
      };

      // Simulate decision timeout handler (like in GameSession.ts)
      const DECISION_TIMEOUT_MS = 30_000;
      setTimeout(() => {
        if (gameState.decisionPending) {
          gameState.autoResolved = true;
          gameState.decisionPending = false;
          gameState.phase = 'timeout_processing';
        }
      }, DECISION_TIMEOUT_MS);

      // Test: Advance time and verify auto-resolution
      expect(gameState.phase).toBe('line_processing');
      expect(gameState.decisionPending).toBe(true);
      expect(gameState.autoResolved).toBe(false);

      await controller.advanceTime(DECISION_TIMEOUT_MS + 1000);

      expect(gameState.phase).toBe('timeout_processing');
      expect(gameState.decisionPending).toBe(false);
      expect(gameState.autoResolved).toBe(true);
    });
  });

  it('demonstrates player reconnection timeout testing pattern', async () => {
    /**
     * This test demonstrates how TimeController would be used
     * to test player reconnection window expiry.
     */

    await withTimeControl(async (controller) => {
      // Simulate player connection state
      interface MockPlayerState {
        connected: boolean;
        disconnectedAt: number | null;
        reconnectionExpired: boolean;
      }

      const playerState: MockPlayerState = {
        connected: true,
        disconnectedAt: null,
        reconnectionExpired: false,
      };

      // Simulate disconnect
      playerState.connected = false;
      playerState.disconnectedAt = controller.getCurrentTime();

      // Simulate reconnection window timer
      const RECONNECT_WINDOW_MS = 30_000;
      setTimeout(() => {
        if (!playerState.connected) {
          playerState.reconnectionExpired = true;
        }
      }, RECONNECT_WINDOW_MS);

      // Test: Player stays disconnected
      await controller.advanceTime(20_000);
      expect(playerState.reconnectionExpired).toBe(false);

      // Test: Window expires
      await controller.advanceTime(15_000);
      expect(playerState.reconnectionExpired).toBe(true);

      // Verify duration tracking
      const disconnectDuration = controller.getCurrentTime() - (playerState.disconnectedAt ?? 0);
      expect(disconnectDuration).toBe(35_000);
    });
  });
});
