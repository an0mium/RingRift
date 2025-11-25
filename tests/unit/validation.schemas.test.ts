/**
 * Unit tests for input validation schemas and sanitization utilities.
 * These tests verify that validation constraints are enforced correctly
 * and that sanitization functions properly handle XSS vectors.
 */

import {
  RegisterSchema,
  LoginSchema,
  UpdateProfileSchema,
  CreateGameSchema,
  ChatMessageSchema,
  UUIDSchema,
  GameIdParamSchema,
  GameListingQuerySchema,
  UserSearchQuerySchema,
  LeaderboardQuerySchema,
  sanitizeString,
  sanitizeHtmlContent,
  createSanitizedStringSchema,
} from '../../src/shared/validation/schemas';

describe('Validation Schemas', () => {
  describe('UUIDSchema', () => {
    it('accepts valid UUID v4', () => {
      const validUUID = '123e4567-e89b-12d3-a456-426614174000';
      expect(UUIDSchema.safeParse(validUUID).success).toBe(true);
    });

    it('rejects invalid UUID formats', () => {
      const invalidUUIDs = ['not-a-uuid', '123e4567-e89b-12d3-a456', '', '   '];
      for (const uuid of invalidUUIDs) {
        expect(UUIDSchema.safeParse(uuid).success).toBe(false);
      }
    });
  });

  describe('GameIdParamSchema', () => {
    it('accepts valid gameId parameter', () => {
      const result = GameIdParamSchema.safeParse({
        gameId: '123e4567-e89b-12d3-a456-426614174000',
      });
      expect(result.success).toBe(true);
    });

    it('rejects missing or invalid gameId', () => {
      expect(GameIdParamSchema.safeParse({}).success).toBe(false);
      expect(GameIdParamSchema.safeParse({ gameId: 'not-valid' }).success).toBe(false);
    });
  });

  describe('GameListingQuerySchema', () => {
    it('accepts valid query parameters and provides defaults', () => {
      const result = GameListingQuerySchema.safeParse({ status: 'active' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.status).toBe('active');
        expect(result.data.limit).toBe(20);
        expect(result.data.offset).toBe(0);
      }
    });

    it('coerces string numbers to integers', () => {
      const result = GameListingQuerySchema.safeParse({ limit: '50', offset: '100' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.limit).toBe(50);
        expect(result.data.offset).toBe(100);
      }
    });

    it('rejects invalid status or out-of-range values', () => {
      expect(GameListingQuerySchema.safeParse({ status: 'invalid' }).success).toBe(false);
      expect(GameListingQuerySchema.safeParse({ limit: '1000' }).success).toBe(false);
      expect(GameListingQuerySchema.safeParse({ offset: '-1' }).success).toBe(false);
    });
  });

  describe('UserSearchQuerySchema', () => {
    it('accepts valid search query and sanitizes', () => {
      const result = UserSearchQuerySchema.safeParse({ q: 'test\x00query' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.q).toBe('testquery');
        expect(result.data.limit).toBe(10);
      }
    });

    it('rejects empty or too-long queries', () => {
      expect(UserSearchQuerySchema.safeParse({ q: '' }).success).toBe(false);
      expect(UserSearchQuerySchema.safeParse({ q: 'a'.repeat(101) }).success).toBe(false);
    });
  });

  describe('RegisterSchema', () => {
    const validData = {
      username: 'testuser',
      email: 'test@example.com',
      password: 'Password123',
      confirmPassword: 'Password123',
    };

    it('accepts valid registration data', () => {
      expect(RegisterSchema.safeParse(validData).success).toBe(true);
    });

    it('rejects invalid username (too short, invalid chars)', () => {
      expect(RegisterSchema.safeParse({ ...validData, username: 'ab' }).success).toBe(false);
      expect(RegisterSchema.safeParse({ ...validData, username: 'a'.repeat(21) }).success).toBe(false);
      expect(RegisterSchema.safeParse({ ...validData, username: 'test user' }).success).toBe(false);
    });

    it('rejects weak passwords', () => {
      expect(RegisterSchema.safeParse({ ...validData, password: 'password', confirmPassword: 'password' }).success).toBe(false);
      expect(RegisterSchema.safeParse({ ...validData, password: 'Pass1', confirmPassword: 'Pass1' }).success).toBe(false);
    });

    it('rejects mismatched passwords', () => {
      expect(RegisterSchema.safeParse({ ...validData, confirmPassword: 'Different123' }).success).toBe(false);
    });
  });

  describe('CreateGameSchema', () => {
    const validGame = {
      boardType: 'square8',
      timeControl: { initialTime: 600, increment: 10 },
    };

    it('accepts valid game creation request', () => {
      expect(CreateGameSchema.safeParse(validGame).success).toBe(true);
    });

    it('rejects invalid board type or time control', () => {
      expect(CreateGameSchema.safeParse({ ...validGame, boardType: 'invalid' }).success).toBe(false);
      expect(CreateGameSchema.safeParse({ ...validGame, timeControl: { initialTime: 30, increment: 5 } }).success).toBe(false);
    });
  });

  describe('ChatMessageSchema', () => {
    it('accepts valid chat message and trims content', () => {
      const result = ChatMessageSchema.safeParse({
        gameId: '123e4567-e89b-12d3-a456-426614174000',
        content: '  Hello  ',
        type: 'game',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.content).toBe('Hello');
      }
    });

    it('rejects empty or too-long content', () => {
      const base = { gameId: '123e4567-e89b-12d3-a456-426614174000', type: 'game' };
      expect(ChatMessageSchema.safeParse({ ...base, content: '' }).success).toBe(false);
      expect(ChatMessageSchema.safeParse({ ...base, content: 'a'.repeat(501) }).success).toBe(false);
    });
  });
});

describe('Sanitization Utilities', () => {
  describe('sanitizeString', () => {
    it('removes null bytes and trims whitespace', () => {
      expect(sanitizeString('hello\x00world')).toBe('helloworld');
      expect(sanitizeString('  hello  ')).toBe('hello');
    });

    it('handles non-string input gracefully', () => {
      expect(sanitizeString(null as any)).toBe('');
      expect(sanitizeString(undefined as any)).toBe('');
    });
  });

  describe('sanitizeHtmlContent', () => {
    it('escapes HTML special characters', () => {
      expect(sanitizeHtmlContent('<script>alert("xss")</script>')).toBe(
        '&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;'
      );
      expect(sanitizeHtmlContent('Tom & Jerry')).toBe('Tom &amp; Jerry');
    });

    it('handles empty strings', () => {
      expect(sanitizeHtmlContent('')).toBe('');
    });
  });

  describe('createSanitizedStringSchema', () => {
    it('creates schema with custom limits and applies sanitization', () => {
      const schema = createSanitizedStringSchema(50, 5);
      expect(schema.safeParse('a'.repeat(51)).success).toBe(false);
      expect(schema.safeParse('abc').success).toBe(false);
      
      const result = schema.safeParse('  hello\x00world  ');
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toBe('helloworld');
      }
    });
  });
});

describe('Security Validation', () => {
  it('rejects SQL injection attempts in UUIDs', () => {
    const attacks = ["'; DROP TABLE users; --", "1' OR '1'='1"];
    for (const attack of attacks) {
      expect(UUIDSchema.safeParse(attack).success).toBe(false);
    }
  });

  it('enforces maximum limits to prevent DoS', () => {
    expect(UserSearchQuerySchema.safeParse({ q: 'a'.repeat(200) }).success).toBe(false);
    expect(GameListingQuerySchema.safeParse({ limit: '999999' }).success).toBe(false);
  });

  it('enforces password complexity requirements', () => {
    const base = { username: 'test', email: 'test@test.com' };
    const weakPasswords = ['password', 'PASSWORD', '12345678'];
    for (const pwd of weakPasswords) {
      expect(RegisterSchema.safeParse({ ...base, password: pwd, confirmPassword: pwd }).success).toBe(false);
    }
  });
});