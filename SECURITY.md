# Security Policy

> **Doc Status (2025-12-11): Active**
>
> Security posture documentation for RingRift.

---

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

---

## Reporting a Vulnerability

If you discover a security vulnerability in RingRift, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Contact the maintainers privately via the repository's security advisory feature
3. Provide detailed information about the vulnerability and steps to reproduce

We aim to respond within 48 hours and provide a fix timeline within 7 days.

---

## Security Implementation Overview

RingRift implements comprehensive security measures across multiple layers:

### Authentication & Authorization

| Feature              | Implementation                          | Status |
| -------------------- | --------------------------------------- | ------ |
| Password hashing     | bcrypt (10 rounds)                      | ✅     |
| JWT access tokens    | 15-minute expiry, RS256                 | ✅     |
| Refresh tokens       | 7-day expiry, family tracking, rotation | ✅     |
| Email verification   | Token-based, 24-hour expiry             | ✅     |
| Password reset       | Secure token, 1-hour expiry             | ✅     |
| Login lockout        | Configurable threshold, Redis-backed    | ✅     |
| Session invalidation | Token version tracking                  | ✅     |

### Rate Limiting

Comprehensive rate limiting via `rate-limiter-flexible`:

| Endpoint Type         | Limit          | Window     | Block Duration |
| --------------------- | -------------- | ---------- | -------------- |
| API (anonymous)       | 50 requests    | 1 minute   | 5 minutes      |
| API (authenticated)   | 200 requests   | 1 minute   | 5 minutes      |
| Login                 | 5 attempts     | 15 minutes | 30 minutes     |
| Registration          | 3 attempts     | 1 hour     | 1 hour         |
| Password reset        | 3 attempts     | 1 hour     | 1 hour         |
| Game moves            | 100 requests   | 1 minute   | 1 minute       |
| WebSocket connections | 10 connections | 1 minute   | 5 minutes      |

All limits are configurable via environment variables (`RATE_LIMIT_*`).

### HTTP Security Headers

Via Helmet middleware:

| Header                     | Configuration                                   |
| -------------------------- | ----------------------------------------------- |
| Content-Security-Policy    | Strict CSP with script-src 'self' (production)  |
| Strict-Transport-Security  | 1 year, includeSubDomains, preload (production) |
| X-Frame-Options            | DENY                                            |
| X-Content-Type-Options     | nosniff                                         |
| Referrer-Policy            | strict-origin-when-cross-origin                 |
| X-DNS-Prefetch-Control     | off                                             |
| Cross-Origin-Opener-Policy | same-origin                                     |

### CORS Configuration

- Whitelist-based origin validation
- Credentials enabled for refresh token cookies
- Preflight caching (10 minutes)
- Origin validation middleware for state-changing requests

### Cookie Security

Refresh tokens stored in HTTP-only cookies:

| Setting  | Production | Development |
| -------- | ---------- | ----------- |
| httpOnly | true       | true        |
| secure   | true       | false       |
| sameSite | strict     | lax         |
| path     | /api/auth  | /api/auth   |

### Input Validation

- All input validated via Zod schemas
- Type-safe request handling
- Structured error responses with codes

### Database Security

- Parameterized queries via Prisma ORM
- Soft-delete support for user data
- Hashed refresh tokens stored (SHA-256)
- Password reset tokens with expiry

---

## Security Best Practices Followed

### OWASP Top 10 Mitigations

| Vulnerability             | Mitigation                           |
| ------------------------- | ------------------------------------ |
| Injection                 | Parameterized queries (Prisma)       |
| Broken Authentication     | JWT + refresh token rotation, bcrypt |
| Sensitive Data Exposure   | HTTPS, secure cookies, hashed tokens |
| XML External Entities     | Not applicable (JSON only)           |
| Broken Access Control     | Middleware-based auth checks         |
| Security Misconfiguration | Helmet, environment-based config     |
| XSS                       | CSP, React's built-in escaping       |
| Insecure Deserialization  | Zod validation                       |
| Known Vulnerabilities     | Regular dependency updates           |
| Insufficient Logging      | Winston structured logging           |

### Additional Measures

- Correlation IDs for request tracking
- Structured error responses (no stack traces in production)
- Password strength requirements (8+ characters)
- Token family tracking for refresh token reuse detection
- Graceful degradation (in-memory fallback for Redis)

---

## Environment Variables

Security-related configuration:

```bash
# JWT Configuration
JWT_SECRET=your-secret-key
JWT_EXPIRES_IN=15m

# Database
DATABASE_URL=postgresql://...

# Redis (optional, falls back to in-memory)
REDIS_URL=redis://...

# Rate Limiting (all optional, have sensible defaults)
RATE_LIMIT_API_POINTS=50
RATE_LIMIT_AUTH_LOGIN_POINTS=5
# ... see src/server/middleware/rateLimiter.ts for full list

# Login Lockout
AUTH_LOGIN_LOCKOUT_ENABLED=true
```

---

## Security Testing

Security-related tests in the test suite:

- `tests/unit/auth.*.test.ts` - Authentication flows
- `tests/unit/middleware/rateLimiter.test.ts` - Rate limiting
- `tests/unit/middleware/auth.test.ts` - Authorization middleware
- `tests/integration/WebSocket.*.test.ts` - WebSocket security

---

## Incident Response

In case of a security incident:

1. Rotate JWT secrets immediately
2. Invalidate all refresh tokens (increment all user `tokenVersion`)
3. Review audit logs for affected timeframe
4. Notify affected users if data exposure occurred
5. Document incident and remediation

---

## Related Files

- `src/server/middleware/auth.ts` - Authentication middleware
- `src/server/middleware/rateLimiter.ts` - Rate limiting
- `src/server/middleware/securityHeaders.ts` - Security headers & CORS
- `src/server/middleware/errorHandler.ts` - Error handling
- `src/server/routes/auth.ts` - Authentication endpoints
- `src/server/utils/email.ts` - Email services
