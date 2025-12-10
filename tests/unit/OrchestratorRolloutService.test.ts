/**
 * Unit tests for OrchestratorRolloutService
 *
 * Tests the rollout decision logic for the orchestrator adapter, including:
 * - Kill switch precedence
 * - Denylist/allowlist targeting
 * - Circuit breaker behavior
 * - Default enabled mode (Phase 3 - percentage rollout removed)
 */

import {
  OrchestratorRolloutService,
  EngineSelection,
  RolloutDecision,
} from '../../src/server/services/OrchestratorRolloutService';
import { logger } from '../../src/server/utils/logger';

// Mock the logger
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

// Create a mock config object that we can modify per test
// Note: shadowModeEnabled has been removed as FSM is now canonical
const mockOrchestratorConfig = {
  adapterEnabled: true,
  rolloutPercentage: 100,
  allowlistUsers: [] as string[],
  denylistUsers: [] as string[],
  circuitBreaker: {
    enabled: true,
    errorThresholdPercent: 5,
    errorWindowSeconds: 300,
  },
  latencyThresholdMs: 500,
};

// Mock the config module
// NOTE: shadowModeEnabled has been removed as FSM is now canonical
jest.mock('../../src/server/config', () => ({
  config: {
    featureFlags: {
      orchestrator: {
        get adapterEnabled() {
          return mockOrchestratorConfig.adapterEnabled;
        },
        get rolloutPercentage() {
          return mockOrchestratorConfig.rolloutPercentage;
        },
        get allowlistUsers() {
          return mockOrchestratorConfig.allowlistUsers;
        },
        get denylistUsers() {
          return mockOrchestratorConfig.denylistUsers;
        },
        get circuitBreaker() {
          return mockOrchestratorConfig.circuitBreaker;
        },
        get latencyThresholdMs() {
          return mockOrchestratorConfig.latencyThresholdMs;
        },
      },
    },
  },
}));

describe('OrchestratorRolloutService', () => {
  let service: OrchestratorRolloutService;

  // Helper to reset config to defaults
  // NOTE: shadowModeEnabled removed as FSM is now canonical
  const resetConfig = () => {
    mockOrchestratorConfig.adapterEnabled = true;
    mockOrchestratorConfig.rolloutPercentage = 100;
    mockOrchestratorConfig.allowlistUsers = [];
    mockOrchestratorConfig.denylistUsers = [];
    mockOrchestratorConfig.circuitBreaker = {
      enabled: true,
      errorThresholdPercent: 5,
      errorWindowSeconds: 300,
    };
  };

  beforeEach(() => {
    jest.clearAllMocks();
    resetConfig();
    service = new OrchestratorRolloutService();
  });

  describe('Decision Priority', () => {
    describe('1. Kill Switch', () => {
      it('should return LEGACY with reason kill_switch when adapter is disabled', () => {
        mockOrchestratorConfig.adapterEnabled = false;
        mockOrchestratorConfig.rolloutPercentage = 100;

        const decision = service.selectEngine('session-123', 'user-456');

        expect(decision.engine).toBe(EngineSelection.LEGACY);
        expect(decision.reason).toBe('kill_switch');
      });

      it('should ignore allowlist when kill switch is off', () => {
        mockOrchestratorConfig.adapterEnabled = false;
        mockOrchestratorConfig.allowlistUsers = ['user-456'];

        const decision = service.selectEngine('session-123', 'user-456');

        expect(decision.engine).toBe(EngineSelection.LEGACY);
        expect(decision.reason).toBe('kill_switch');
      });

      it('should log debug message for kill switch decision', () => {
        mockOrchestratorConfig.adapterEnabled = false;

        service.selectEngine('session-123', 'user-456');

        expect(logger.debug).toHaveBeenCalledWith(
          expect.stringContaining('kill switch'),
          expect.objectContaining({
            sessionId: 'session-123',
            userId: 'user-456',
            decision: 'kill_switch',
          })
        );
      });
    });

    describe('2. Denylist', () => {
      it('should return LEGACY when user is in denylist', () => {
        mockOrchestratorConfig.denylistUsers = ['user-456'];

        const decision = service.selectEngine('session-123', 'user-456');

        expect(decision.engine).toBe(EngineSelection.LEGACY);
        expect(decision.reason).toBe('denylist');
      });

      it('should prioritize denylist over allowlist', () => {
        mockOrchestratorConfig.allowlistUsers = ['user-456'];
        mockOrchestratorConfig.denylistUsers = ['user-456'];

        const decision = service.selectEngine('session-123', 'user-456');

        expect(decision.engine).toBe(EngineSelection.LEGACY);
        expect(decision.reason).toBe('denylist');
      });

      it('should not apply denylist when userId is undefined', () => {
        mockOrchestratorConfig.denylistUsers = ['user-456'];
        mockOrchestratorConfig.rolloutPercentage = 100;

        const decision = service.selectEngine('session-123', undefined);

        // Should proceed to percentage check since no userId
        expect(decision.reason).not.toBe('denylist');
      });
    });

    describe('3. Allowlist', () => {
      it('should return ORCHESTRATOR when user is in allowlist', () => {
        mockOrchestratorConfig.allowlistUsers = ['user-456'];

        const decision = service.selectEngine('session-123', 'user-456');

        expect(decision.engine).toBe(EngineSelection.ORCHESTRATOR);
        expect(decision.reason).toBe('allowlist');
      });

      // NOTE: Shadow mode test removed - FSM is now canonical

      it('should not apply allowlist when userId is undefined', () => {
        mockOrchestratorConfig.allowlistUsers = ['user-456'];
        mockOrchestratorConfig.rolloutPercentage = 0;

        const decision = service.selectEngine('session-123', undefined);

        // Phase 3: percentage rollout removed, defaults to orchestrator
        expect(decision.reason).toBe('default_enabled');
      });
    });

    describe('4. Circuit Breaker', () => {
      it('should return LEGACY when circuit breaker is open', () => {
        mockOrchestratorConfig.circuitBreaker.enabled = true;
        mockOrchestratorConfig.circuitBreaker.errorThresholdPercent = 5;

        // Trip the circuit breaker
        for (let i = 0; i < 15; i++) {
          service.recordError();
        }

        const decision = service.selectEngine('session-123', 'user-456');

        expect(decision.engine).toBe(EngineSelection.LEGACY);
        expect(decision.reason).toBe('circuit_breaker');
      });

      it('should proceed to percentage when circuit breaker is closed', () => {
        mockOrchestratorConfig.circuitBreaker.enabled = true;
        mockOrchestratorConfig.rolloutPercentage = 100;

        // Record some successes to keep circuit breaker closed
        service.recordSuccess();

        const decision = service.selectEngine('session-123', 'user-456');

        expect(decision.reason).not.toBe('circuit_breaker');
        expect(decision.engine).toBe(EngineSelection.ORCHESTRATOR);
      });

      it('should bypass circuit breaker check when circuit breaker is disabled', () => {
        mockOrchestratorConfig.circuitBreaker.enabled = false;
        mockOrchestratorConfig.rolloutPercentage = 100;

        // Record errors that would trip the breaker if it were enabled
        for (let i = 0; i < 20; i++) {
          service.recordError();
        }

        const decision = service.selectEngine('session-123');

        // Circuit breaker disabled, so should proceed based on percentage
        expect(decision.reason).not.toBe('circuit_breaker');
      });
    });

    describe('5. Default Enabled (Phase 3 - Percentage Rollout Removed)', () => {
      // NOTE: Phase 3 migration removed percentage-based rollout.
      // All sessions now default to orchestrator (FSM canonical).

      it('should return ORCHESTRATOR by default (ignores rolloutPercentage)', () => {
        mockOrchestratorConfig.rolloutPercentage = 100; // Ignored in Phase 3

        const decision = service.selectEngine('any-session-id');

        expect(decision.engine).toBe(EngineSelection.ORCHESTRATOR);
        expect(decision.reason).toBe('default_enabled');
      });

      it('should return ORCHESTRATOR even when rolloutPercentage is 0 (Phase 3)', () => {
        mockOrchestratorConfig.rolloutPercentage = 0; // Ignored in Phase 3

        const decision = service.selectEngine('any-session-id');

        expect(decision.engine).toBe(EngineSelection.ORCHESTRATOR);
        expect(decision.reason).toBe('default_enabled');
      });

      // NOTE: Shadow mode test removed - FSM is now canonical
    });
  });

  describe('Phase-like configuration presets', () => {
    // NOTE: Shadow mode removed - FSM is now canonical

    it('Phase 0 / legacy-only posture forces LEGACY via kill switch', () => {
      mockOrchestratorConfig.adapterEnabled = false;
      mockOrchestratorConfig.rolloutPercentage = 100;

      const decision = service.selectEngine('any-session', 'any-user');

      expect(decision.engine).toBe(EngineSelection.LEGACY);
      expect(decision.reason).toBe('kill_switch');
    });

    it('Phase 1 / staging orchestrator-only routes typical sessions to ORCHESTRATOR', () => {
      mockOrchestratorConfig.adapterEnabled = true;
      mockOrchestratorConfig.rolloutPercentage = 100; // Ignored in Phase 3
      mockOrchestratorConfig.allowlistUsers = [];
      mockOrchestratorConfig.denylistUsers = [];
      mockOrchestratorConfig.circuitBreaker = {
        enabled: true,
        errorThresholdPercent: 5,
        errorWindowSeconds: 300,
      };

      // Keep circuit breaker closed
      service.recordSuccess();

      const decision = service.selectEngine('staging-session', 'staging-user');
      expect(decision.engine).toBe(EngineSelection.ORCHESTRATOR);
      // Phase 3: percentage rollout removed, defaults to orchestrator
      expect(decision.reason).toBe('default_enabled');
    });

    it('Phase 3 / default enabled with allowlist routes allowlisted users to ORCHESTRATOR', () => {
      mockOrchestratorConfig.adapterEnabled = true;
      mockOrchestratorConfig.rolloutPercentage = 0; // Ignored in Phase 3
      mockOrchestratorConfig.allowlistUsers = ['vip-user'];
      mockOrchestratorConfig.denylistUsers = [];
      mockOrchestratorConfig.circuitBreaker = {
        enabled: true,
        errorThresholdPercent: 5,
        errorWindowSeconds: 300,
      };

      // Allowlisted user sees ORCHESTRATOR via allowlist path
      const allowlisted = service.selectEngine('prod-session-1', 'vip-user');
      expect(allowlisted.engine).toBe(EngineSelection.ORCHESTRATOR);
      expect(allowlisted.reason).toBe('allowlist');

      // Phase 3: Non-allowlisted user also gets ORCHESTRATOR (default enabled)
      const regular = service.selectEngine('prod-session-2', 'regular-user');
      expect(regular.engine).toBe(EngineSelection.ORCHESTRATOR);
      expect(regular.reason).toBe('default_enabled');
    });

    it('Phase 3 / default enabled honors circuit breaker (percentage ignored)', () => {
      mockOrchestratorConfig.adapterEnabled = true;
      mockOrchestratorConfig.rolloutPercentage = 25; // Ignored in Phase 3
      mockOrchestratorConfig.allowlistUsers = [];
      mockOrchestratorConfig.denylistUsers = [];
      mockOrchestratorConfig.circuitBreaker = {
        enabled: true,
        errorThresholdPercent: 10,
        errorWindowSeconds: 300,
      };

      // Phase 3: All sessions should be routed to ORCHESTRATOR (percentage ignored)
      for (let i = 0; i < 10; i++) {
        const decision = service.selectEngine(`rollout-session-${i}`);
        expect(decision.engine).toBe(EngineSelection.ORCHESTRATOR);
        expect(decision.reason).toBe('default_enabled');
      }

      // Now trip the circuit breaker and confirm it forces LEGACY
      for (let i = 0; i < 20; i++) {
        service.recordError();
      }
      expect(service.isCircuitBreakerOpen()).toBe(true);

      const afterTrip = service.selectEngine('post-trip-session');
      expect(afterTrip.engine).toBe(EngineSelection.LEGACY);
      expect(afterTrip.reason).toBe('circuit_breaker');
    });
  });

  describe('Consistent Hashing (Legacy - Phase 3 now default enabled)', () => {
    // NOTE: Phase 3 migration removed percentage-based rollout.
    // These tests verify that behavior is now consistent (always orchestrator).

    it('should return consistent results for the same session ID', () => {
      mockOrchestratorConfig.rolloutPercentage = 50; // Ignored in Phase 3

      const decision1 = service.selectEngine('test-session-abc');
      const decision2 = service.selectEngine('test-session-abc');
      const decision3 = service.selectEngine('test-session-abc');

      expect(decision1.engine).toBe(decision2.engine);
      expect(decision2.engine).toBe(decision3.engine);
      expect(decision1.reason).toBe(decision2.reason);
      // Phase 3: All go to orchestrator
      expect(decision1.engine).toBe(EngineSelection.ORCHESTRATOR);
    });

    it('should return ORCHESTRATOR for all sessions (Phase 3 - percentage ignored)', () => {
      mockOrchestratorConfig.rolloutPercentage = 50; // Ignored in Phase 3

      const results = new Set<EngineSelection>();
      for (let i = 0; i < 100; i++) {
        const decision = service.selectEngine(`session-${i}`);
        results.add(decision.engine);
      }

      // Phase 3: All sessions go to ORCHESTRATOR (percentage ignored)
      expect(results.size).toBe(1);
      expect(results.has(EngineSelection.ORCHESTRATOR)).toBe(true);
    });

    it('should route all sessions to ORCHESTRATOR (Phase 3 - percentage ignored)', () => {
      mockOrchestratorConfig.rolloutPercentage = 50; // Ignored in Phase 3

      let orchestratorCount = 0;
      const totalSessions = 100;

      for (let i = 0; i < totalSessions; i++) {
        const decision = service.selectEngine(`session-${i}-${Math.random()}`);
        if (decision.engine === EngineSelection.ORCHESTRATOR) {
          orchestratorCount++;
        }
      }

      // Phase 3: All sessions go to orchestrator
      expect(orchestratorCount).toBe(totalSessions);
    });

    it('should ignore percentage boundaries (Phase 3 - all go to orchestrator)', () => {
      // Phase 3: percentage is ignored, all sessions go to orchestrator
      mockOrchestratorConfig.rolloutPercentage = 10; // Ignored

      let orchestratorCount = 0;
      const totalSessions = 100;

      for (let i = 0; i < totalSessions; i++) {
        const decision = service.selectEngine(`test-${i}`);
        if (decision.engine === EngineSelection.ORCHESTRATOR) {
          orchestratorCount++;
        }
      }

      // Phase 3: All go to orchestrator regardless of percentage
      expect(orchestratorCount).toBe(totalSessions);
    });
  });

  describe('Circuit Breaker Logic', () => {
    describe('Error Rate Tracking', () => {
      it('should start with zero error rate', () => {
        expect(service.getErrorRate()).toBe(0);
      });

      it('should calculate error rate correctly', () => {
        // Record 8 successes and 2 errors = 20% error rate
        for (let i = 0; i < 8; i++) {
          service.recordSuccess();
        }
        for (let i = 0; i < 2; i++) {
          service.recordError();
        }

        expect(service.getErrorRate()).toBe(20);
      });

      it('should record success correctly', () => {
        service.recordSuccess();
        service.recordSuccess();
        service.recordSuccess();

        const state = service.getCircuitBreakerState();
        expect(state.requestCount).toBe(3);
        expect(state.errorCount).toBe(0);
      });

      it('should record error correctly', () => {
        service.recordError();
        service.recordError();

        const state = service.getCircuitBreakerState();
        expect(state.requestCount).toBe(2);
        expect(state.errorCount).toBe(2);
      });
    });

    describe('Circuit Breaker Tripping', () => {
      it('should not trip with fewer than minimum requests', () => {
        mockOrchestratorConfig.circuitBreaker.errorThresholdPercent = 5;

        // Record 9 errors - below minimum of 10 requests
        for (let i = 0; i < 9; i++) {
          service.recordError();
        }

        expect(service.isCircuitBreakerOpen()).toBe(false);
      });

      it('should trip when error rate exceeds threshold with minimum requests', () => {
        mockOrchestratorConfig.circuitBreaker.errorThresholdPercent = 5;

        // Record 10 errors = 100% error rate > 5% threshold
        for (let i = 0; i < 10; i++) {
          service.recordError();
        }

        expect(service.isCircuitBreakerOpen()).toBe(true);
      });

      it('should trip at exactly the threshold boundary', () => {
        mockOrchestratorConfig.circuitBreaker.errorThresholdPercent = 10;

        // 89 successes + 11 errors = 11% error rate > 10% threshold
        for (let i = 0; i < 89; i++) {
          service.recordSuccess();
        }
        for (let i = 0; i < 11; i++) {
          service.recordError();
        }

        expect(service.isCircuitBreakerOpen()).toBe(true);
      });

      it('should not trip when error rate is below threshold', () => {
        mockOrchestratorConfig.circuitBreaker.errorThresholdPercent = 10;

        // 95 successes + 5 errors = 5% error rate < 10% threshold
        for (let i = 0; i < 95; i++) {
          service.recordSuccess();
        }
        for (let i = 0; i < 5; i++) {
          service.recordError();
        }

        expect(service.isCircuitBreakerOpen()).toBe(false);
      });

      it('should log error when circuit breaker opens', () => {
        mockOrchestratorConfig.circuitBreaker.errorThresholdPercent = 5;

        for (let i = 0; i < 15; i++) {
          service.recordError();
        }

        expect(logger.error).toHaveBeenCalledWith(
          expect.stringContaining('Circuit breaker OPENED'),
          expect.objectContaining({
            action: 'circuit_breaker_opened',
          })
        );
      });

      it('should not log again if already open', () => {
        mockOrchestratorConfig.circuitBreaker.errorThresholdPercent = 5;

        // Trip the breaker
        for (let i = 0; i < 15; i++) {
          service.recordError();
        }

        jest.clearAllMocks();

        // Record more errors
        for (let i = 0; i < 5; i++) {
          service.recordError();
        }

        // Should not log again since already open
        expect(logger.error).not.toHaveBeenCalledWith(
          expect.stringContaining('Circuit breaker OPENED'),
          expect.any(Object)
        );
      });

      it('should not trip when circuit breaker is disabled', () => {
        mockOrchestratorConfig.circuitBreaker.enabled = false;

        // Record many errors
        for (let i = 0; i < 100; i++) {
          service.recordError();
        }

        expect(service.isCircuitBreakerOpen()).toBe(false);
      });
    });

    describe('Window Reset', () => {
      it('should reset counts when window expires', () => {
        mockOrchestratorConfig.circuitBreaker.errorWindowSeconds = 1; // 1 second window

        // Record some errors
        for (let i = 0; i < 15; i++) {
          service.recordError();
        }

        expect(service.isCircuitBreakerOpen()).toBe(true);

        // Fast forward time past the window
        jest.useFakeTimers();
        jest.advanceTimersByTime(1100); // 1.1 seconds

        // Check state - should be reset
        expect(service.isCircuitBreakerOpen()).toBe(false);

        const state = service.getCircuitBreakerState();
        expect(state.errorCount).toBe(0);
        expect(state.requestCount).toBe(0);
        expect(state.isOpen).toBe(false);

        jest.useRealTimers();
      });

      it('should log when circuit breaker closes due to window expiry', () => {
        mockOrchestratorConfig.circuitBreaker.errorWindowSeconds = 1;

        // Trip the breaker
        for (let i = 0; i < 15; i++) {
          service.recordError();
        }

        expect(service.isCircuitBreakerOpen()).toBe(true);

        jest.clearAllMocks();

        jest.useFakeTimers();
        jest.advanceTimersByTime(1100);

        // Access state to trigger window check
        service.isCircuitBreakerOpen();

        expect(logger.info).toHaveBeenCalledWith(
          expect.stringContaining('Circuit breaker closed due to window expiry'),
          expect.any(Object)
        );

        jest.useRealTimers();
      });
    });

    describe('Manual Reset', () => {
      it('should reset circuit breaker state', () => {
        // Trip the breaker
        for (let i = 0; i < 15; i++) {
          service.recordError();
        }

        expect(service.isCircuitBreakerOpen()).toBe(true);

        service.resetCircuitBreaker();

        expect(service.isCircuitBreakerOpen()).toBe(false);
        const state = service.getCircuitBreakerState();
        expect(state.errorCount).toBe(0);
        expect(state.requestCount).toBe(0);
      });

      it('should log when manually resetting an open circuit breaker', () => {
        // Trip the breaker
        for (let i = 0; i < 15; i++) {
          service.recordError();
        }

        jest.clearAllMocks();

        service.resetCircuitBreaker();

        expect(logger.info).toHaveBeenCalledWith(
          expect.stringContaining('Circuit breaker manually reset'),
          expect.objectContaining({
            action: 'circuit_breaker_reset',
          })
        );
      });

      it('should not log when resetting an already closed circuit breaker', () => {
        // Breaker is closed by default
        jest.clearAllMocks();

        service.resetCircuitBreaker();

        expect(logger.info).not.toHaveBeenCalledWith(
          expect.stringContaining('Circuit breaker manually reset'),
          expect.any(Object)
        );
      });
    });

    describe('State Inspection', () => {
      it('should return a copy of the circuit breaker state', () => {
        service.recordError();
        service.recordSuccess();

        const state1 = service.getCircuitBreakerState();
        const state2 = service.getCircuitBreakerState();

        expect(state1).not.toBe(state2); // Different objects
        expect(state1).toEqual(state2); // Same content
      });

      it('should include all state properties', () => {
        const state = service.getCircuitBreakerState();

        expect(state).toHaveProperty('isOpen');
        expect(state).toHaveProperty('errorCount');
        expect(state).toHaveProperty('requestCount');
        expect(state).toHaveProperty('windowStart');
        expect(typeof state.isOpen).toBe('boolean');
        expect(typeof state.errorCount).toBe('number');
        expect(typeof state.requestCount).toBe('number');
        expect(typeof state.windowStart).toBe('number');
      });
    });
  });

  // NOTE: Shadow Mode tests removed - FSM is now canonical (shadow mode no longer exists)

  describe('Logging', () => {
    it('should log debug messages for each decision type', () => {
      // Test kill switch logging
      mockOrchestratorConfig.adapterEnabled = false;
      service.selectEngine('s1');
      expect(logger.debug).toHaveBeenCalled();

      jest.clearAllMocks();

      // Test denylist logging
      resetConfig();
      mockOrchestratorConfig.denylistUsers = ['user1'];
      service.selectEngine('s2', 'user1');
      expect(logger.debug).toHaveBeenCalled();

      jest.clearAllMocks();

      // Test allowlist logging
      resetConfig();
      mockOrchestratorConfig.allowlistUsers = ['user2'];
      service.selectEngine('s3', 'user2');
      expect(logger.debug).toHaveBeenCalled();

      jest.clearAllMocks();

      // Test percentage included logging
      resetConfig();
      mockOrchestratorConfig.rolloutPercentage = 100;
      service.selectEngine('s4');
      expect(logger.debug).toHaveBeenCalled();

      jest.clearAllMocks();

      // Test percentage excluded logging
      resetConfig();
      mockOrchestratorConfig.rolloutPercentage = 0;
      service.selectEngine('s5');
      expect(logger.debug).toHaveBeenCalled();
    });

    it('should include session and user in log context', () => {
      mockOrchestratorConfig.rolloutPercentage = 100;

      service.selectEngine('test-session-id', 'test-user-id');

      expect(logger.debug).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          sessionId: 'test-session-id',
          userId: 'test-user-id',
        })
      );
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty session ID', () => {
      mockOrchestratorConfig.rolloutPercentage = 50;

      // Should not throw
      const decision = service.selectEngine('');
      expect(decision).toBeDefined();
      expect([EngineSelection.LEGACY, EngineSelection.ORCHESTRATOR]).toContain(decision.engine);
    });

    it('should handle very long session IDs', () => {
      mockOrchestratorConfig.rolloutPercentage = 50;

      const longSessionId = 'a'.repeat(10000);
      const decision = service.selectEngine(longSessionId);

      expect(decision).toBeDefined();
    });

    it('should handle special characters in session ID', () => {
      mockOrchestratorConfig.rolloutPercentage = 50;

      const specialSessionId = 'session-123-!@#$%^&*()_+-=[]{}|;:,.<>?';
      const decision1 = service.selectEngine(specialSessionId);
      const decision2 = service.selectEngine(specialSessionId);

      // Should still be consistent
      expect(decision1.engine).toBe(decision2.engine);
    });

    it('should handle multiple users in allowlist/denylist', () => {
      mockOrchestratorConfig.allowlistUsers = ['user1', 'user2', 'user3'];
      mockOrchestratorConfig.denylistUsers = ['banned1', 'banned2'];

      expect(service.selectEngine('s1', 'user1').reason).toBe('allowlist');
      expect(service.selectEngine('s2', 'user2').reason).toBe('allowlist');
      expect(service.selectEngine('s3', 'banned1').reason).toBe('denylist');
      expect(service.selectEngine('s4', 'banned2').reason).toBe('denylist');
    });

    it('should handle concurrent access to circuit breaker', () => {
      mockOrchestratorConfig.circuitBreaker.errorThresholdPercent = 50;

      // Simulate concurrent access
      const operations = [];
      for (let i = 0; i < 100; i++) {
        if (i % 2 === 0) {
          operations.push(() => service.recordSuccess());
        } else {
          operations.push(() => service.recordError());
        }
      }

      // Execute all operations
      operations.forEach((op) => op());

      // State should be consistent
      const state = service.getCircuitBreakerState();
      expect(state.requestCount).toBe(100);
      expect(state.errorCount).toBe(50);
    });
  });

  describe('EngineSelection Enum', () => {
    it('should have correct enum values', () => {
      expect(EngineSelection.LEGACY).toBe('legacy');
      expect(EngineSelection.ORCHESTRATOR).toBe('orchestrator');
      // SHADOW mode removed as FSM is now canonical
    });
  });

  describe('RolloutDecision Interface', () => {
    it('should return decisions matching the interface', () => {
      const decision = service.selectEngine('session-123', 'user-456');

      expect(decision).toHaveProperty('engine');
      expect(decision).toHaveProperty('reason');
      expect(typeof decision.engine).toBe('string');
      expect(typeof decision.reason).toBe('string');
    });
  });

  describe('Service Instance', () => {
    it('should create independent instances', () => {
      const service1 = new OrchestratorRolloutService();
      const service2 = new OrchestratorRolloutService();

      // Trip circuit breaker in service1
      for (let i = 0; i < 15; i++) {
        service1.recordError();
      }

      // service2 should be unaffected
      expect(service1.isCircuitBreakerOpen()).toBe(true);
      expect(service2.isCircuitBreakerOpen()).toBe(false);
    });
  });
});

describe('Singleton Export', () => {
  it('should export a singleton instance', async () => {
    // Import the singleton - must be done dynamically to get fresh import
    const { orchestratorRollout } =
      await import('../../src/server/services/OrchestratorRolloutService');

    expect(orchestratorRollout).toBeDefined();
    expect(orchestratorRollout).toBeInstanceOf(OrchestratorRolloutService);
  });
});
