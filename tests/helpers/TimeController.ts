/**
 * TimeController - Time Acceleration Utility for E2E Timeout Tests
 * ============================================================================
 *
 * A test utility class for controlling time in tests, enabling rapid testing
 * of timeout scenarios that would otherwise take 30+ seconds in real time.
 *
 * This implementation uses Jest's built-in fake timers, which provide:
 * - Well-tested timer mocking (setTimeout, setInterval, Date.now, etc.)
 * - Proper handling of async/await with timer advancement
 * - Correct timer ordering and firing
 *
 * Features:
 * - Install/uninstall time mocking
 * - Advance time programmatically
 * - Set acceleration factor for real-time tests
 * - Pause/resume time flow
 *
 * @example
 * ```typescript
 * describe('Timeout tests', () => {
 *   let timeController: TimeController;
 *
 *   beforeEach(() => {
 *     timeController = createTimeController();
 *     timeController.install();
 *   });
 *
 *   afterEach(() => {
 *     timeController.uninstall();
 *   });
 *
 *   it('should timeout decision after 30 seconds', async () => {
 *     // Setup game with pending decision
 *     await setupGame();
 *
 *     // Advance time by 30 seconds (instant in test time)
 *     await timeController.advanceTime(30000);
 *
 *     // Verify timeout occurred
 *     expect(game.state.phase).toBe('timeout_processing');
 *   });
 * });
 * ```
 */

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration options for TimeController.
 */
export interface TimeControllerOptions {
  /**
   * Acceleration factor for real-time mode.
   * 100 means time passes 100x faster (100ms real = 10s virtual).
   * Only used with `useRealTime: true` mode.
   * @default 1000
   */
  accelerationFactor?: number;

  /**
   * Whether to start with time paused.
   * @default false
   */
  startPaused?: boolean;

  /**
   * Initial virtual time (as Unix timestamp in ms).
   * @default Date.now() at install time
   */
  initialTime?: number;

  /**
   * Whether to use Jest's modern fake timers.
   * @default true
   */
  useModernTimers?: boolean;

  /**
   * Whether to auto-advance timers by a small delta on each real tick.
   * Useful for testing code that mixes real async with timers.
   * @default false
   */
  shouldAdvanceTime?: boolean;

  /**
   * Delta in ms for auto-advancement when shouldAdvanceTime is true.
   * @default 20
   */
  advanceTimeDelta?: number;
}

/**
 * State of the TimeController.
 */
export type TimeControllerState = 'idle' | 'installed' | 'paused';

// ============================================================================
// TimeController Class
// ============================================================================

/**
 * Controls time for testing timeout scenarios.
 *
 * Uses Jest's built-in fake timers for robust, well-tested timer mocking.
 */
export class TimeController {
  private state: TimeControllerState = 'idle';
  private virtualTime: number;
  private accelerationFactor: number;
  private readonly initialTime: number;
  private readonly useModernTimers: boolean;
  private readonly shouldAdvanceTime: boolean;
  private readonly advanceTimeDelta: number;

  // Store references to real timing functions (in case needed for debugging)
  private realDateNow: typeof Date.now | null = null;
  private realSetTimeout: typeof setTimeout | null = null;
  private realSetInterval: typeof setInterval | null = null;
  private realClearTimeout: typeof clearTimeout | null = null;
  private realClearInterval: typeof clearInterval | null = null;

  /**
   * Creates a new TimeController.
   *
   * @param options - Configuration options
   */
  constructor(options: TimeControllerOptions = {}) {
    this.initialTime = options.initialTime ?? Date.now();
    this.virtualTime = this.initialTime;
    this.accelerationFactor = options.accelerationFactor ?? 1000;
    this.useModernTimers = options.useModernTimers ?? true;
    this.shouldAdvanceTime = options.shouldAdvanceTime ?? false;
    this.advanceTimeDelta = options.advanceTimeDelta ?? 20;

    if (options.startPaused) {
      this.state = 'paused';
    }
  }

  // ==========================================================================
  // Core Control Methods
  // ==========================================================================

  /**
   * Install fake timers, replacing global timing functions.
   *
   * After calling this, setTimeout, setInterval, Date.now, and related
   * functions will use virtual time that you control.
   *
   * @throws Error if already installed
   */
  install(): void {
    if (this.state === 'installed') {
      throw new Error('TimeController is already installed. Call uninstall() first.');
    }

    // Store references to real functions for potential debugging
    this.realDateNow = Date.now;
    this.realSetTimeout = global.setTimeout;
    this.realSetInterval = global.setInterval;
    this.realClearTimeout = global.clearTimeout;
    this.realClearInterval = global.clearInterval;

    // Configure Jest fake timers
    if (this.shouldAdvanceTime) {
      // Auto-advance mode: useful for code mixing real async with timers
      jest.useFakeTimers({
        now: this.virtualTime,
        advanceTimers: this.advanceTimeDelta,
      });
    } else {
      // Standard mode: full control over time
      jest.useFakeTimers({
        now: this.virtualTime,
      });
    }

    this.state = 'installed';
  }

  /**
   * Uninstall fake timers, restoring original global timing functions.
   *
   * Always call this in afterEach/afterAll to clean up.
   */
  uninstall(): void {
    if (this.state === 'idle') {
      return; // Nothing to uninstall
    }

    jest.useRealTimers();

    // Clear stored references
    this.realDateNow = null;
    this.realSetTimeout = null;
    this.realSetInterval = null;
    this.realClearTimeout = null;
    this.realClearInterval = null;

    this.state = 'idle';
  }

  /**
   * Check if fake timers are currently installed.
   */
  isInstalled(): boolean {
    return this.state === 'installed' || this.state === 'paused';
  }

  // ==========================================================================
  // Time Manipulation Methods
  // ==========================================================================

  /**
   * Advance virtual time by the specified duration and run any pending timers.
   *
   * This is the primary method for testing timeouts. For example, to test
   * a 30-second timeout, call `await advanceTime(30000)`.
   *
   * @param ms - Duration to advance in milliseconds
   * @returns Promise that resolves after timers have been processed
   * @throws Error if not installed
   *
   * @example
   * ```typescript
   * // Test a 30-second decision timeout
   * await timeController.advanceTime(30_000);
   *
   * // Test with smaller increments
   * await timeController.advanceTime(15_000); // Half way
   * await timeController.advanceTime(15_000); // Rest of the way
   * ```
   */
  async advanceTime(ms: number): Promise<void> {
    this.ensureInstalled('advanceTime');

    // Update our tracked virtual time
    this.virtualTime += ms;

    // Use Jest's async timer advancement for proper Promise handling
    await jest.advanceTimersByTimeAsync(ms);
  }

  /**
   * Advance virtual time to a specific point and run pending timers.
   *
   * Useful when you need to advance to an absolute timestamp.
   *
   * @param timestamp - Target Unix timestamp in milliseconds
   * @throws Error if timestamp is in the past
   */
  async advanceTimeTo(timestamp: number): Promise<void> {
    this.ensureInstalled('advanceTimeTo');

    const delta = timestamp - this.virtualTime;
    if (delta < 0) {
      throw new Error(
        `Cannot advance to ${timestamp}: current time is ${this.virtualTime} (would go backwards)`
      );
    }

    await this.advanceTime(delta);
  }

  /**
   * Set the current virtual time without running timers.
   *
   * Use this to set an initial time before starting a test scenario.
   * Note: This does NOT fire any timers - use advanceTime for that.
   *
   * @param timestamp - Unix timestamp in milliseconds
   */
  setTime(timestamp: number): void {
    this.ensureInstalled('setTime');

    this.virtualTime = timestamp;
    jest.setSystemTime(timestamp);
  }

  /**
   * Get the current virtual time.
   *
   * @returns Current virtual time as Unix timestamp in milliseconds
   */
  getCurrentTime(): number {
    if (this.state === 'idle') {
      return Date.now();
    }
    return this.virtualTime;
  }

  /**
   * Run all pending timers immediately (without advancing time).
   *
   * Useful for flushing any scheduled callbacks. Use with caution
   * as this can cause infinite loops if timers re-schedule themselves.
   */
  async runAllTimers(): Promise<void> {
    this.ensureInstalled('runAllTimers');
    await jest.runAllTimersAsync();
  }

  /**
   * Run only currently pending timers (not newly scheduled ones).
   *
   * Safer than runAllTimers when dealing with recurring timers.
   */
  async runOnlyPendingTimers(): Promise<void> {
    this.ensureInstalled('runOnlyPendingTimers');
    await jest.runOnlyPendingTimersAsync();
  }

  /**
   * Clear all pending timers without running them.
   *
   * Useful for cleanup or when you want to cancel all scheduled work.
   */
  clearAllTimers(): void {
    this.ensureInstalled('clearAllTimers');
    jest.clearAllTimers();
  }

  /**
   * Get the number of pending timers.
   *
   * Useful for assertions or debugging.
   */
  getPendingTimerCount(): number {
    this.ensureInstalled('getPendingTimerCount');
    return jest.getTimerCount();
  }

  // ==========================================================================
  // Configuration Methods
  // ==========================================================================

  /**
   * Set the acceleration factor for real-time hybrid mode.
   *
   * This is used when running tests that mix real async operations
   * with timer-based logic. The acceleration factor determines how
   * much faster virtual time advances compared to real time.
   *
   * @param factor - Acceleration multiplier (e.g., 100 = 100x faster)
   */
  setAcceleration(factor: number): void {
    if (factor <= 0) {
      throw new Error('Acceleration factor must be positive');
    }
    this.accelerationFactor = factor;
  }

  /**
   * Get the current acceleration factor.
   */
  getAcceleration(): number {
    return this.accelerationFactor;
  }

  /**
   * Pause time advancement (for debugging or step-by-step testing).
   *
   * When paused, advanceTime calls will still update virtual time
   * but no new auto-advancement will occur in shouldAdvanceTime mode.
   */
  pause(): void {
    if (this.state === 'installed') {
      this.state = 'paused';
    }
  }

  /**
   * Resume time advancement after a pause.
   */
  resume(): void {
    if (this.state === 'paused') {
      this.state = 'installed';
    }
  }

  /**
   * Check if time is currently paused.
   */
  isPaused(): boolean {
    return this.state === 'paused';
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Reset the controller to initial state.
   *
   * Uninstalls fake timers and resets virtual time.
   */
  reset(): void {
    this.uninstall();
    this.virtualTime = this.initialTime;
  }

  /**
   * Get the current state of the TimeController.
   */
  getState(): TimeControllerState {
    return this.state;
  }

  /**
   * Convenience method: advance time in seconds.
   *
   * @param seconds - Number of seconds to advance
   */
  async advanceTimeBySeconds(seconds: number): Promise<void> {
    await this.advanceTime(seconds * 1000);
  }

  /**
   * Convenience method: advance time in minutes.
   *
   * @param minutes - Number of minutes to advance
   */
  async advanceTimeByMinutes(minutes: number): Promise<void> {
    await this.advanceTime(minutes * 60 * 1000);
  }

  // ==========================================================================
  // Private Helpers
  // ==========================================================================

  /**
   * Ensure the controller is installed before performing an operation.
   */
  private ensureInstalled(operation: string): void {
    if (this.state === 'idle') {
      throw new Error(
        `TimeController must be installed before calling ${operation}(). ` + 'Call install() first.'
      );
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new TimeController with default options.
 *
 * @param options - Optional configuration
 * @returns A new TimeController instance
 *
 * @example
 * ```typescript
 * // Basic usage
 * const timeController = createTimeController();
 *
 * // With custom options
 * const timeController = createTimeController({
 *   accelerationFactor: 100,
 *   initialTime: new Date('2024-01-01').getTime(),
 * });
 * ```
 */
export function createTimeController(options?: TimeControllerOptions): TimeController {
  return new TimeController(options);
}

/**
 * Create a TimeController configured for fast timeout testing.
 *
 * This preset is optimized for testing 30-second decision timeouts
 * and similar long-duration scenarios in minimal real time.
 *
 * @param initialTime - Optional initial timestamp
 * @returns A configured TimeController
 */
export function createFastTimeoutController(initialTime?: number): TimeController {
  return new TimeController({
    accelerationFactor: 1000,
    initialTime,
    shouldAdvanceTime: false,
  });
}

/**
 * Create a TimeController for hybrid async/timer testing.
 *
 * This preset auto-advances time slowly, useful when testing code
 * that mixes real async operations with timer-based logic.
 *
 * @param options - Override options
 * @returns A configured TimeController
 */
export function createHybridTimeController(
  options: Partial<TimeControllerOptions> = {}
): TimeController {
  return new TimeController({
    accelerationFactor: 100,
    shouldAdvanceTime: true,
    advanceTimeDelta: 10,
    ...options,
  });
}

// ============================================================================
// Test Helper Functions
// ============================================================================

/**
 * Run a test with controlled time, automatically setting up and tearing down.
 *
 * @param testFn - The test function to run
 * @param options - TimeController options
 *
 * @example
 * ```typescript
 * it('should timeout after 30s', async () => {
 *   await withTimeControl(async (tc) => {
 *     // Set up test
 *     const game = createGame();
 *
 *     // Advance time
 *     await tc.advanceTime(30_000);
 *
 *     // Assert
 *     expect(game.isTimedOut).toBe(true);
 *   });
 * });
 * ```
 */
export async function withTimeControl<T>(
  testFn: (controller: TimeController) => Promise<T>,
  options?: TimeControllerOptions
): Promise<T> {
  const controller = createTimeController(options);
  controller.install();

  try {
    return await testFn(controller);
  } finally {
    controller.uninstall();
  }
}

/**
 * Wait for a condition to become true, advancing time in steps.
 *
 * Useful for testing state machines or async processes that
 * depend on timers.
 *
 * @param controller - The TimeController instance
 * @param condition - Function that returns true when condition is met
 * @param options - Configuration for the wait
 * @returns Promise that resolves when condition is met or times out
 *
 * @example
 * ```typescript
 * await waitForConditionWithTimeAdvance(
 *   timeController,
 *   () => game.state.phase === 'timeout_processing',
 *   { maxTime: 35_000, stepSize: 5_000 }
 * );
 * ```
 */
export async function waitForConditionWithTimeAdvance(
  controller: TimeController,
  condition: () => boolean | Promise<boolean>,
  options: {
    maxTime?: number;
    stepSize?: number;
    description?: string;
  } = {}
): Promise<void> {
  const { maxTime = 60_000, stepSize = 1_000, description = 'condition' } = options;

  let elapsed = 0;

  while (elapsed < maxTime) {
    const result = await condition();
    if (result) {
      return;
    }

    await controller.advanceTime(stepSize);
    elapsed += stepSize;
  }

  throw new Error(`Timeout waiting for ${description} after ${elapsed}ms of virtual time`);
}
