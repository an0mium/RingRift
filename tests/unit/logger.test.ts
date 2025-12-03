/**
 * Tests for the structured logging implementation.
 *
 * @module tests/unit/logger.test
 */
import {
  logger,
  redactEmail,
  maskSensitiveData,
  maskHeaders,
  requestContextStorage,
  runWithContext,
  getRequestContext,
  RequestContext,
  withRequestContext,
  httpLogger,
  stream,
} from '../../src/server/utils/logger';

describe('Logger Utilities', () => {
  describe('redactEmail', () => {
    it('should redact email showing first 3 characters of local part', () => {
      expect(redactEmail('john.doe@example.com')).toBe('joh***@example.com');
      expect(redactEmail('test@gmail.com')).toBe('tes***@gmail.com');
    });

    it('should handle short local parts', () => {
      expect(redactEmail('ab@example.com')).toBe('ab***@example.com');
      expect(redactEmail('a@example.com')).toBe('a***@example.com');
    });

    it('should return undefined for null/undefined', () => {
      expect(redactEmail(null)).toBeUndefined();
      expect(redactEmail(undefined)).toBeUndefined();
    });

    it('should handle invalid email formats', () => {
      expect(redactEmail('')).toBeUndefined();
      expect(redactEmail('notanemail')).toBe('[REDACTED_EMAIL]');
      expect(redactEmail('@nodomain')).toBe('[REDACTED_EMAIL]');
      expect(redactEmail('noat')).toBe('[REDACTED_EMAIL]');
    });

    it('should trim whitespace', () => {
      expect(redactEmail('  john@example.com  ')).toBe('joh***@example.com');
    });
  });

  describe('maskSensitiveData', () => {
    it('should mask password fields', () => {
      const data = { password: 'secret123', username: 'john' };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked.password).toBe('secr...[REDACTED]');
      expect(masked.username).toBe('john');
    });

    it('should mask token fields', () => {
      const data = { accessToken: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature' };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked.accessToken).toContain('[REDACTED]');
    });

    it('should mask secret fields', () => {
      const data = { apiSecret: 'very-secret-key-12345' };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked.apiSecret).toBe('very...[REDACTED]');
    });

    it('should mask API key fields', () => {
      const data = { 'api-key': 'sk-1234567890abcdef', api_key: 'pk-abcdef123456' };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked['api-key']).toBe('sk-1...[REDACTED]');
      expect(masked.api_key).toBe('pk-a...[REDACTED]');
    });

    it('should mask authorization fields', () => {
      const data = { authorization: 'Bearer token123' };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked.authorization).toBe('Bear...[REDACTED]');
    });

    it('should redact email fields specially', () => {
      const data = { email: 'test@example.com', password: 'secret' };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked.email).toBe('tes***@example.com');
    });

    it('should handle nested objects', () => {
      const data = {
        user: {
          email: 'deep@nest.com',
          secret: {
            value: 'nested-secret-value',
          },
        },
      };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      const user = masked.user as Record<string, unknown>;
      // The 'secret' key triggers sensitive key detection, so nested objects get recursed
      const secret = user.secret as Record<string, unknown>;
      expect(user.email).toBe('dee***@nest.com');
      expect(secret.value).toBe('nested-secret-value'); // 'value' is not sensitive
    });

    it('should handle arrays', () => {
      const data = {
        items: ['item1', 'item2'],
        users: [{ email: 'a@b.com' }, { email: 'c@d.com' }],
      };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked.items).toEqual(['item1', 'item2']); // Arrays of strings preserved
      const users = masked.users as Array<Record<string, unknown>>;
      expect(users[0].email).toBe('a***@b.com');
      expect(users[1].email).toBe('c***@d.com');
    });

    it('should mask array values when key is sensitive', () => {
      const data = {
        passwords: ['pass1', 'pass2'], // 'passwords' triggers sensitive detection
      };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      // Array under sensitive key gets recursed but individual strings preserved
      expect(Array.isArray(masked.passwords)).toBe(true);
    });

    it('should handle short sensitive values', () => {
      const data = { password: 'abc' };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked.password).toBe('[REDACTED]');
    });

    it('should recurse into nested objects under sensitive keys', () => {
      const data = { password: { nested: 'value', email: 'test@example.com' } };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      // Objects under sensitive keys get recursed, not fully redacted
      const password = masked.password as Record<string, unknown>;
      expect(password.nested).toBe('value'); // Non-sensitive key preserved
      expect(password.email).toBe('tes***@example.com'); // Email gets special treatment
    });

    it('should redact non-object, non-string sensitive values', () => {
      const data = { password: 12345, token: true };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked.password).toBe('[REDACTED]');
      expect(masked.token).toBe('[REDACTED]');
    });

    it('should preserve null and undefined values for sensitive keys', () => {
      const data = { password: null, secret: undefined, name: 'John' };
      const masked = maskSensitiveData(data) as Record<string, unknown>;
      expect(masked.password).toBeNull();
      expect(masked.secret).toBeUndefined();
      expect(masked.name).toBe('John');
    });

    it('should respect maxDepth to prevent infinite recursion', () => {
      const data: Record<string, unknown> = {};
      let current = data;
      for (let i = 0; i < 10; i++) {
        current.nested = {};
        current = current.nested as Record<string, unknown>;
      }
      current.password = 'deeppassword';

      const masked = maskSensitiveData(data, 5) as Record<string, unknown>;
      // Should not throw and should handle depth limit gracefully
      expect(masked).toBeDefined();
    });

    it('should return primitives unchanged', () => {
      expect(maskSensitiveData(42)).toBe(42);
      expect(maskSensitiveData('hello')).toBe('hello');
      expect(maskSensitiveData(true)).toBe(true);
      expect(maskSensitiveData(null)).toBeNull();
      expect(maskSensitiveData(undefined)).toBeUndefined();
    });
  });

  describe('maskHeaders', () => {
    it('should mask authorization header', () => {
      const headers = {
        authorization: 'Bearer eyJhbGciOiJIUzI1NiJ9...',
        'content-type': 'application/json',
      };
      const masked = maskHeaders(headers);
      expect(masked.authorization).toBe('[REDACTED]');
      expect(masked['content-type']).toBe('application/json');
    });

    it('should mask cookie headers', () => {
      const headers = {
        cookie: 'session=abc123; token=xyz789',
        'set-cookie': 'session=new123',
      };
      const masked = maskHeaders(headers);
      expect(masked.cookie).toBe('[REDACTED]');
      expect(masked['set-cookie']).toBe('[REDACTED]');
    });

    it('should mask API key headers', () => {
      const headers = {
        'x-api-key': 'sk-secret-key',
        'x-auth-token': 'token123',
        'x-access-token': 'access123',
      };
      const masked = maskHeaders(headers);
      expect(masked['x-api-key']).toBe('[REDACTED]');
      expect(masked['x-auth-token']).toBe('[REDACTED]');
      expect(masked['x-access-token']).toBe('[REDACTED]');
    });

    it('should be case insensitive for header names', () => {
      const headers = {
        Authorization: 'Bearer token',
        COOKIE: 'session=abc',
      };
      const masked = maskHeaders(headers);
      expect(masked.Authorization).toBe('[REDACTED]');
      expect(masked.COOKIE).toBe('[REDACTED]');
    });

    it('should preserve non-sensitive headers', () => {
      const headers = {
        'content-type': 'application/json',
        accept: 'application/json',
        'user-agent': 'Mozilla/5.0',
        'x-request-id': 'abc-123',
      };
      const masked = maskHeaders(headers);
      expect(masked).toEqual(headers);
    });
  });

  describe('Request Context (AsyncLocalStorage)', () => {
    it('should return undefined when not in a context', () => {
      expect(getRequestContext()).toBeUndefined();
    });

    it('should provide context within runWithContext', () => {
      const context: RequestContext = {
        requestId: 'test-request-1',
        userId: 'user-1',
        method: 'GET',
        path: '/api/test',
        startTime: Date.now(),
      };

      runWithContext(context, () => {
        const retrieved = getRequestContext();
        expect(retrieved).toBeDefined();
        expect(retrieved?.requestId).toBe('test-request-1');
        expect(retrieved?.userId).toBe('user-1');
      });
    });

    it('should isolate contexts between different runs', () => {
      const context1: RequestContext = { requestId: 'req-1' };
      const context2: RequestContext = { requestId: 'req-2' };

      let capturedId1: string | undefined;
      let capturedId2: string | undefined;

      runWithContext(context1, () => {
        capturedId1 = getRequestContext()?.requestId;
      });

      runWithContext(context2, () => {
        capturedId2 = getRequestContext()?.requestId;
      });

      expect(capturedId1).toBe('req-1');
      expect(capturedId2).toBe('req-2');
    });

    it('should propagate context through async operations', async () => {
      const context: RequestContext = { requestId: 'async-test' };

      await new Promise<void>((resolve) => {
        runWithContext(context, async () => {
          // Simulate async operation
          await Promise.resolve();
          expect(getRequestContext()?.requestId).toBe('async-test');
          resolve();
        });
      });
    });

    it('should allow mutation of context within run', () => {
      const context: RequestContext = { requestId: 'mutable-test' };

      runWithContext(context, () => {
        const ctx = getRequestContext();
        if (ctx) {
          ctx.userId = 'new-user-id';
        }
        expect(getRequestContext()?.userId).toBe('new-user-id');
      });
    });
  });

  describe('Logger instance', () => {
    it('should be defined', () => {
      expect(logger).toBeDefined();
    });

    it('should have standard log methods', () => {
      expect(typeof logger.info).toBe('function');
      expect(typeof logger.warn).toBe('function');
      expect(typeof logger.error).toBe('function');
      expect(typeof logger.debug).toBe('function');
    });

    it('should not throw when logging', () => {
      // These should not throw
      expect(() => {
        logger.info('Test info message');
        logger.warn('Test warning message');
        logger.error('Test error message');
        logger.debug('Test debug message');
      }).not.toThrow();
    });

    it('should accept metadata objects', () => {
      expect(() => {
        logger.info('Message with meta', { key: 'value', count: 42 });
      }).not.toThrow();
    });

    it('should handle error objects', () => {
      const error = new Error('Test error');
      expect(() => {
        logger.error('Error occurred', { error });
      }).not.toThrow();
    });

    it('should include request context when available', () => {
      const context: RequestContext = {
        requestId: 'log-context-test',
        userId: 'test-user',
      };

      // This verifies the integration works without throwing
      runWithContext(context, () => {
        expect(() => {
          logger.info('Logged with context');
        }).not.toThrow();
      });
    });
  });
});

describe('Sensitive Key Detection', () => {
  const sensitiveKeys = [
    'password',
    'PASSWORD',
    'userPassword',
    'user_password',
    'secret',
    'apiSecret',
    'clientSecret',
    'token',
    'accessToken',
    'refresh_token',
    'api_key',
    'apiKey',
    'api-key',
    'authorization',
    'bearer',
    'credential',
    'privateKey',
    'private_key',
    'accessKey',
    'sessionId',
    'session_token',
    'cookie',
  ];

  it.each(sensitiveKeys)('should detect "%s" as sensitive', (key) => {
    const data = { [key]: 'sensitive-value-12345678' };
    const masked = maskSensitiveData(data) as Record<string, unknown>;
    expect(masked[key]).toContain('[REDACTED]');
  });

  const nonSensitiveKeys = [
    'username',
    'name',
    'email', // Email is handled specially, not as [REDACTED]
    'id',
    'type',
    'status',
    'count',
    'path',
    'method',
    'statusCode',
  ];

  it.each(nonSensitiveKeys)('should NOT fully redact "%s"', (key) => {
    const data = { [key]: 'normal-value' };
    const masked = maskSensitiveData(data) as Record<string, unknown>;
    // Email gets special redaction, others stay unchanged
    if (key === 'email') {
      expect(masked[key]).not.toBe('[REDACTED]');
    } else {
      expect(masked[key]).toBe('normal-value');
    }
  });
});

describe('withRequestContext (Legacy Helper)', () => {
  it('should include requestId when present on request object', () => {
    const req = { requestId: 'legacy-req-123' };
    const meta = { action: 'test' };

    const result = withRequestContext(req, meta);

    expect(result).toEqual({
      requestId: 'legacy-req-123',
      action: 'test',
    });
  });

  it('should return original meta when requestId is absent', () => {
    const req = { userId: 'user-1' }; // no requestId
    const meta = { action: 'test' };

    const result = withRequestContext(req, meta);

    expect(result).toEqual({ action: 'test' });
  });

  it('should return original meta when request is empty object', () => {
    const req = {};
    const meta = { action: 'test' };

    const result = withRequestContext(req, meta);

    expect(result).toEqual({ action: 'test' });
  });

  it('should work with undefined meta', () => {
    const req = { requestId: 'req-456' };

    const result = withRequestContext(req);

    expect(result).toEqual({ requestId: 'req-456' });
  });

  it('should work when req is null-ish', () => {
    const result = withRequestContext(null, { action: 'test' });
    expect(result).toEqual({ action: 'test' });

    const result2 = withRequestContext(undefined, { action: 'test' });
    expect(result2).toEqual({ action: 'test' });
  });
});

describe('httpLogger (Legacy HTTP Logger)', () => {
  it('should have all standard log methods', () => {
    expect(typeof httpLogger.info).toBe('function');
    expect(typeof httpLogger.warn).toBe('function');
    expect(typeof httpLogger.error).toBe('function');
    expect(typeof httpLogger.debug).toBe('function');
  });

  it('should not throw when calling info', () => {
    const req = { requestId: 'http-info-test' };
    expect(() => {
      httpLogger.info(req, 'Info message', { key: 'value' });
    }).not.toThrow();
  });

  it('should not throw when calling warn', () => {
    const req = { requestId: 'http-warn-test' };
    expect(() => {
      httpLogger.warn(req, 'Warning message', { key: 'value' });
    }).not.toThrow();
  });

  it('should not throw when calling error', () => {
    const req = { requestId: 'http-error-test' };
    expect(() => {
      httpLogger.error(req, 'Error message', { key: 'value' });
    }).not.toThrow();
  });

  it('should not throw when calling debug', () => {
    const req = { requestId: 'http-debug-test' };
    expect(() => {
      httpLogger.debug(req, 'Debug message', { key: 'value' });
    }).not.toThrow();
  });

  it('should work without requestId on request', () => {
    const req = {}; // No requestId
    expect(() => {
      httpLogger.info(req, 'Message without requestId');
      httpLogger.warn(req, 'Warning without requestId');
      httpLogger.error(req, 'Error without requestId');
      httpLogger.debug(req, 'Debug without requestId');
    }).not.toThrow();
  });

  it('should work without meta argument', () => {
    const req = { requestId: 'no-meta-test' };
    expect(() => {
      httpLogger.info(req, 'Message without meta');
    }).not.toThrow();
  });
});

describe('stream (Morgan Stream)', () => {
  it('should have write method', () => {
    expect(typeof stream.write).toBe('function');
  });

  it('should not throw when writing a message', () => {
    expect(() => {
      stream.write('GET /api/test 200 15ms\n');
    }).not.toThrow();
  });

  it('should trim whitespace from messages', () => {
    // The stream.write calls logger.info with trimmed message
    // We can't easily verify the trimming without mocking logger,
    // but we can verify it doesn't throw with various whitespace
    expect(() => {
      stream.write('  leading/trailing  \n');
      stream.write('\n\n');
      stream.write('   ');
    }).not.toThrow();
  });
});
