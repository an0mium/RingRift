/**
 * Rules Parity Metrics Unit Tests
 *
 * Tests for the rulesParityMetrics module including:
 * - logRulesMismatch structured logging helper
 * - recordRulesParityMismatch unified mismatch recorder
 * - CacheKeys key generator utility functions
 */

const mockRecordRulesParityMismatch = jest.fn();

// Mock MetricsService BEFORE importing the module under test
jest.mock('../../../src/server/services/MetricsService', () => ({
  getMetricsService: jest.fn(() => ({
    recordRulesParityMismatch: mockRecordRulesParityMismatch,
    rulesParityValidMismatch: { inc: jest.fn() },
    rulesParityHashMismatch: { inc: jest.fn() },
    rulesParitySMismatch: { inc: jest.fn() },
    rulesParityGameStatusMismatch: { inc: jest.fn() },
    recordCacheHit: jest.fn(),
    recordCacheMiss: jest.fn(),
  })),
}));

jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    debug: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

// Import AFTER mocks are set up
import {
  logRulesMismatch,
  recordRulesParityMismatch,
  rulesParityMetrics,
} from '../../../src/server/utils/rulesParityMetrics';
import { CacheKeys } from '../../../src/server/cache/redis';
import { logger } from '../../../src/server/utils/logger';

describe('logRulesMismatch', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should log valid mismatch with details', () => {
    const details = {
      gameId: 'game-123',
      move: 5,
      tsResult: true,
      pythonResult: false,
    };

    logRulesMismatch('valid', details);

    expect(logger.warn).toHaveBeenCalledWith('rules_parity_mismatch', {
      kind: 'valid',
      gameId: 'game-123',
      move: 5,
      tsResult: true,
      pythonResult: false,
    });
  });

  it('should log hash mismatch with details', () => {
    logRulesMismatch('hash', { tsHash: 'abc123', pythonHash: 'xyz789' });

    expect(logger.warn).toHaveBeenCalledWith('rules_parity_mismatch', {
      kind: 'hash',
      tsHash: 'abc123',
      pythonHash: 'xyz789',
    });
  });

  it('should log S mismatch', () => {
    logRulesMismatch('S', { tsS: 42, pythonS: 41 });

    expect(logger.warn).toHaveBeenCalledWith('rules_parity_mismatch', {
      kind: 'S',
      tsS: 42,
      pythonS: 41,
    });
  });

  it('should log gameStatus mismatch', () => {
    logRulesMismatch('gameStatus', { tsStatus: 'in_progress', pythonStatus: 'victory' });

    expect(logger.warn).toHaveBeenCalledWith('rules_parity_mismatch', {
      kind: 'gameStatus',
      tsStatus: 'in_progress',
      pythonStatus: 'victory',
    });
  });

  it('should log backend_fallback event', () => {
    logRulesMismatch('backend_fallback', { reason: 'python_timeout', latencyMs: 5000 });

    expect(logger.warn).toHaveBeenCalledWith('rules_parity_mismatch', {
      kind: 'backend_fallback',
      reason: 'python_timeout',
      latencyMs: 5000,
    });
  });

  it('should log shadow_error event', () => {
    logRulesMismatch('shadow_error', { error: 'Connection refused', endpoint: '/validate' });

    expect(logger.warn).toHaveBeenCalledWith('rules_parity_mismatch', {
      kind: 'shadow_error',
      error: 'Connection refused',
      endpoint: '/validate',
    });
  });

  it('should log with empty details object', () => {
    logRulesMismatch('valid', {});

    expect(logger.warn).toHaveBeenCalledWith('rules_parity_mismatch', {
      kind: 'valid',
    });
  });
});

describe('rulesParityMetrics wiring', () => {
  it('exposes MetricsService-backed counters for parity dimensions', () => {
    expect(rulesParityMetrics.validMismatch).toBeDefined();
    expect(typeof rulesParityMetrics.validMismatch.inc).toBe('function');

    expect(rulesParityMetrics.hashMismatch).toBeDefined();
    expect(typeof rulesParityMetrics.hashMismatch.inc).toBe('function');

    expect(rulesParityMetrics.sMismatch).toBeDefined();
    expect(typeof rulesParityMetrics.sMismatch.inc).toBe('function');

    expect(rulesParityMetrics.gameStatusMismatch).toBeDefined();
    expect(typeof rulesParityMetrics.gameStatusMismatch.inc).toBe('function');
  });
});

describe('recordRulesParityMismatch', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should record validation mismatch', () => {
    recordRulesParityMismatch({
      mismatchType: 'validation',
      suite: 'runtime_shadow',
    });

    expect(mockRecordRulesParityMismatch).toHaveBeenCalledWith('validation', 'runtime_shadow');
  });

  it('should record hash mismatch', () => {
    recordRulesParityMismatch({
      mismatchType: 'hash',
      suite: 'runtime_python_mode',
    });

    expect(mockRecordRulesParityMismatch).toHaveBeenCalledWith('hash', 'runtime_python_mode');
  });

  it('should record s_invariant mismatch', () => {
    recordRulesParityMismatch({
      mismatchType: 's_invariant',
      suite: 'runtime_ts',
    });

    expect(mockRecordRulesParityMismatch).toHaveBeenCalledWith('s_invariant', 'runtime_ts');
  });

  it('should record game_status mismatch with contract vectors suite', () => {
    recordRulesParityMismatch({
      mismatchType: 'game_status',
      suite: 'contract_vectors_v2',
    });

    expect(mockRecordRulesParityMismatch).toHaveBeenCalledWith(
      'game_status',
      'contract_vectors_v2'
    );
  });
});

describe('CacheKeys', () => {
  describe('user', () => {
    it('should generate user key', () => {
      expect(CacheKeys.user('user-123')).toBe('user:user-123');
    });
  });

  describe('userSession', () => {
    it('should generate user session key', () => {
      expect(CacheKeys.userSession('sess-abc')).toBe('session:sess-abc');
    });
  });

  describe('game', () => {
    it('should generate game key', () => {
      expect(CacheKeys.game('game-456')).toBe('game:game-456');
    });
  });

  describe('gameState', () => {
    it('should generate game state key', () => {
      expect(CacheKeys.gameState('game-789')).toBe('game:game-789:state');
    });
  });

  describe('userGames', () => {
    it('should generate user games key', () => {
      expect(CacheKeys.userGames('user-xyz')).toBe('user:user-xyz:games');
    });
  });

  describe('onlineUsers', () => {
    it('should generate online users key', () => {
      expect(CacheKeys.onlineUsers()).toBe('users:online');
    });
  });

  describe('gameQueue', () => {
    it('should generate game queue key for board type', () => {
      expect(CacheKeys.gameQueue('square8')).toBe('queue:square8');
    });

    it('should handle hex board queues', () => {
      expect(CacheKeys.gameQueue('hex_4p')).toBe('queue:hex_4p');
    });
  });

  describe('userStats', () => {
    it('should generate user stats key', () => {
      expect(CacheKeys.userStats('user-stats-1')).toBe('user:user-stats-1:stats');
    });
  });

  describe('leaderboard', () => {
    it('should generate leaderboard key', () => {
      expect(CacheKeys.leaderboard('square19')).toBe('leaderboard:square19');
    });
  });

  describe('tournament', () => {
    it('should generate tournament key', () => {
      expect(CacheKeys.tournament('tourney-1')).toBe('tournament:tourney-1');
    });
  });

  describe('chatHistory', () => {
    it('should generate chat history key', () => {
      expect(CacheKeys.chatHistory('game-chat-1')).toBe('chat:game-chat-1:history');
    });
  });

  describe('authLoginFailures', () => {
    it('should generate auth login failures key', () => {
      expect(CacheKeys.authLoginFailures('user@example.com')).toBe(
        'auth:login:failures:user@example.com'
      );
    });
  });

  describe('authLoginLockout', () => {
    it('should generate auth login lockout key', () => {
      expect(CacheKeys.authLoginLockout('locked@example.com')).toBe(
        'auth:login:lockout:locked@example.com'
      );
    });
  });
});
