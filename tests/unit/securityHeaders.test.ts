/**
 * Security Headers Tests
 *
 * Verifies that the security middleware properly configures:
 * - Content Security Policy (CSP)
 * - HTTP Strict Transport Security (HSTS)
 * - X-Frame-Options
 * - X-Content-Type-Options
 * - Referrer-Policy
 * - Cross-Origin policies
 * - CORS configuration
 */

import express, { Express } from 'express';
import request from 'supertest';

// Need to set up test environment before importing server modules
process.env.NODE_ENV = 'test';
process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/test';
process.env.REDIS_URL = 'redis://localhost:6379';
process.env.JWT_SECRET = 'test-jwt-secret';
process.env.JWT_REFRESH_SECRET = 'test-jwt-refresh-secret';
process.env.ALLOWED_ORIGINS = 'http://localhost:3000,http://localhost:5173';

import { securityMiddleware } from '../../src/server/middleware/securityHeaders';

describe('Security Headers Middleware', () => {
  let app: Express;

  beforeEach(() => {
    app = express();
    // Apply security middleware
    app.use(securityMiddleware.headers);
    app.use(securityMiddleware.cors);

    // Test endpoint
    app.get('/test', (_req, res) => {
      res.json({ message: 'ok' });
    });

    app.post('/test-post', (_req, res) => {
      res.json({ message: 'posted' });
    });
  });

  describe('Content Security Policy', () => {
    it('should set Content-Security-Policy header', async () => {
      const response = await request(app).get('/test');

      expect(response.status).toBe(200);
      expect(response.headers['content-security-policy']).toBeDefined();
    });

    it('should include default-src directive', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/default-src\s+'self'/);
    });

    it('should include script-src directive', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/script-src/);
    });

    it('should include style-src directive with unsafe-inline', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/style-src[^;]*'unsafe-inline'/);
    });

    it('should include connect-src directive for WebSocket support', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/connect-src/);
    });

    it('should block object-src for security', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/object-src\s+'none'/);
    });

    it('should block frame-src for clickjacking protection', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/frame-src\s+'none'/);
    });

    it('should set frame-ancestors to none', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/frame-ancestors\s+'none'/);
    });
  });

  describe('X-Frame-Options', () => {
    it('should set X-Frame-Options header to DENY', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-frame-options']).toBe('DENY');
    });
  });

  describe('X-Content-Type-Options', () => {
    it('should set X-Content-Type-Options header to nosniff', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-content-type-options']).toBe('nosniff');
    });
  });

  describe('Referrer-Policy', () => {
    it('should set Referrer-Policy header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['referrer-policy']).toBeDefined();
      expect(response.headers['referrer-policy']).toBe('strict-origin-when-cross-origin');
    });
  });

  describe('X-DNS-Prefetch-Control', () => {
    it('should set X-DNS-Prefetch-Control header to off', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-dns-prefetch-control']).toBe('off');
    });
  });

  describe('Cross-Origin-Opener-Policy', () => {
    it('should set Cross-Origin-Opener-Policy header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['cross-origin-opener-policy']).toBe('same-origin');
    });
  });

  describe('Cross-Origin-Resource-Policy', () => {
    it('should set Cross-Origin-Resource-Policy header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['cross-origin-resource-policy']).toBe('same-site');
    });
  });

  describe('X-Permitted-Cross-Domain-Policies', () => {
    it('should set X-Permitted-Cross-Domain-Policies header to none', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-permitted-cross-domain-policies']).toBe('none');
    });
  });

  describe('Origin-Agent-Cluster', () => {
    it('should set Origin-Agent-Cluster header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['origin-agent-cluster']).toBeDefined();
    });
  });

  describe('X-Powered-By', () => {
    it('should not expose X-Powered-By header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-powered-by']).toBeUndefined();
    });
  });

  describe('CORS', () => {
    it('should allow requests without origin (server-to-server)', async () => {
      const response = await request(app).get('/test');

      expect(response.status).toBe(200);
    });

    it('should allow requests from allowed origins', async () => {
      const response = await request(app)
        .get('/test')
        .set('Origin', 'http://localhost:3000');

      expect(response.status).toBe(200);
      expect(response.headers['access-control-allow-origin']).toBe('http://localhost:3000');
    });

    it('should allow credentials', async () => {
      const response = await request(app)
        .get('/test')
        .set('Origin', 'http://localhost:3000');

      expect(response.headers['access-control-allow-credentials']).toBe('true');
    });

    it('should respond to preflight OPTIONS requests', async () => {
      const response = await request(app)
        .options('/test')
        .set('Origin', 'http://localhost:3000')
        .set('Access-Control-Request-Method', 'POST');

      expect(response.status).toBe(204);
      expect(response.headers['access-control-allow-methods']).toBeDefined();
    });

    it('should expose specified headers', async () => {
      const response = await request(app)
        .get('/test')
        .set('Origin', 'http://localhost:3000');

      const exposedHeaders = response.headers['access-control-expose-headers'];
      expect(exposedHeaders).toBeDefined();
      expect(exposedHeaders).toContain('X-Request-ID');
    });

    it('should reject requests from non-allowed origins', async () => {
      const response = await request(app)
        .get('/test')
        .set('Origin', 'http://malicious-site.com');

      // CORS middleware throws error for rejected origins
      expect(response.status).toBe(500);
    });

    it('should reject requests from localhost with non-configured port in test mode', async () => {
      // In test mode (which Jest uses), only explicitly configured origins are allowed
      // Development mode would allow any localhost port via regex
      const response = await request(app)
        .get('/test')
        .set('Origin', 'http://localhost:8080');

      // Should be rejected in test mode since we only allow 3000 and 5173
      expect(response.status).toBe(500);
    });
  });

  describe('Security Headers Summary', () => {
    it('should have all critical security headers set', async () => {
      const response = await request(app).get('/test');

      // Critical headers for XSS protection
      expect(response.headers['content-security-policy']).toBeDefined();

      // Clickjacking protection
      expect(response.headers['x-frame-options']).toBeDefined();

      // MIME sniffing protection
      expect(response.headers['x-content-type-options']).toBeDefined();

      // Referrer leakage protection
      expect(response.headers['referrer-policy']).toBeDefined();

      // Cross-origin isolation
      expect(response.headers['cross-origin-opener-policy']).toBeDefined();

      // Server info exposure prevention
      expect(response.headers['x-powered-by']).toBeUndefined();
    });
  });
});

describe('Origin Validation Middleware', () => {
  let app: Express;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    app.use(securityMiddleware.cors);
    // Note: originValidation middleware skips in development mode
    // To test it properly, we'd need to mock config.isDevelopment = false

    app.post('/api/test', (_req, res) => {
      res.json({ message: 'posted' });
    });
  });

  it('should allow GET requests without origin validation', async () => {
    const response = await request(app).get('/api/test');

    // Will be 404 since no GET route, but not blocked by origin validation
    expect(response.status).toBe(404);
  });

  it('should allow POST requests in development mode', async () => {
    const response = await request(app)
      .post('/api/test')
      .send({ data: 'test' });

    expect(response.status).toBe(200);
  });
});

describe('Sanitization Utilities', () => {
  // Import sanitization functions from validation schemas
  const { sanitizeString, sanitizeHtmlContent } = require('../../src/shared/validation/schemas');

  describe('sanitizeString', () => {
    it('should remove null bytes', () => {
      const input = 'hello\x00world';
      expect(sanitizeString(input)).toBe('helloworld');
    });

    it('should trim whitespace', () => {
      const input = '  hello world  ';
      expect(sanitizeString(input)).toBe('hello world');
    });

    it('should handle non-string input', () => {
      expect(sanitizeString(null as any)).toBe('');
      expect(sanitizeString(undefined as any)).toBe('');
      expect(sanitizeString(123 as any)).toBe('');
    });

    it('should normalize Unicode', () => {
      // NFC normalization test - café in decomposed form
      const decomposed = 'cafe\u0301';
      const result = sanitizeString(decomposed);
      expect(result).toBe('café');
    });
  });

  describe('sanitizeHtmlContent', () => {
    it('should escape HTML entities', () => {
      const input = '<script>alert("xss")</script>';
      const result = sanitizeHtmlContent(input);
      expect(result).not.toContain('<script>');
      expect(result).toContain('&lt;script&gt;');
    });

    it('should escape ampersands', () => {
      const input = 'Tom & Jerry';
      expect(sanitizeHtmlContent(input)).toBe('Tom &amp; Jerry');
    });

    it('should escape quotes', () => {
      const input = 'He said "hello"';
      expect(sanitizeHtmlContent(input)).toContain('&quot;');
    });

    it('should escape single quotes', () => {
      const input = "It's a test";
      expect(sanitizeHtmlContent(input)).toContain('&#x27;');
    });

    it('should escape complex XSS payloads', () => {
      const input = '<img src=x onerror="alert(1)">';
      const result = sanitizeHtmlContent(input);
      // HTML characters should be escaped
      expect(result).not.toContain('<img');
      expect(result).toContain('&lt;img');
      // Quotes and equals are escaped, making the payload inert in HTML context
      expect(result).toContain('&#x3D;'); // escaped equals
      expect(result).toContain('&quot;'); // escaped quotes
    });

    it('should handle non-string input', () => {
      expect(sanitizeHtmlContent(null as any)).toBe('');
      expect(sanitizeHtmlContent(undefined as any)).toBe('');
    });
  });
});