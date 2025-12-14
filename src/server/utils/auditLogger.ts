/**
 * Audit Logger for Security Events
 *
 * Provides structured logging for security-relevant events including:
 * - Authentication (login, logout, token refresh, failed attempts)
 * - Authorization (access denied, admin actions)
 * - Security events (rate limits, validation failures)
 *
 * All audit logs are written to a dedicated audit.log file and include
 * immutable fields for forensic analysis.
 */

import winston from 'winston';
import path from 'path';
import { config } from '../config';
import { redactEmail, getRequestContext } from './logger';

// ============================================================================
// Types
// ============================================================================

export type AuditEventType =
  // Authentication events
  | 'auth.login.success'
  | 'auth.login.failed'
  | 'auth.logout'
  | 'auth.token.refresh'
  | 'auth.token.revoked'
  | 'auth.register'
  | 'auth.password.reset.requested'
  | 'auth.password.reset.completed'
  | 'auth.email.verified'
  | 'auth.lockout'
  // Authorization events
  | 'authz.access.denied'
  | 'authz.role.changed'
  | 'authz.admin.action'
  // Security events
  | 'security.rate_limit.exceeded'
  | 'security.cors.rejected'
  | 'security.validation.failed'
  | 'security.suspicious.activity'
  // Data events
  | 'data.export.requested'
  | 'data.account.deleted'
  // WebSocket events
  | 'ws.connection.authenticated'
  | 'ws.connection.failed'
  | 'ws.session.terminated';

export interface AuditLogEntry {
  /** Event type for categorization and alerting */
  event: AuditEventType;
  /** Unix timestamp in milliseconds */
  timestamp: number;
  /** ISO timestamp for human readability */
  timestampIso: string;
  /** User ID (if authenticated) */
  userId?: string | undefined;
  /** Redacted email (for login/register events) */
  email?: string | undefined;
  /** Client IP address */
  ip?: string | undefined;
  /** Request ID for correlation */
  requestId?: string | undefined;
  /** User agent string */
  userAgent?: string | undefined;
  /** HTTP method */
  method?: string | undefined;
  /** Request path */
  path?: string | undefined;
  /** Event-specific details */
  details?: Record<string, unknown> | undefined;
  /** Result of the operation */
  result: 'success' | 'failure' | 'blocked';
  /** Reason for failure/block (if applicable) */
  reason?: string | undefined;
}

/** Request-like object for audit logging */
export interface AuditRequest {
  ip?: string | undefined;
  headers?: Record<string, string | string[] | undefined>;
  method?: string | undefined;
  path?: string | undefined;
  originalUrl?: string | undefined;
  requestId?: string | undefined;
}

// ============================================================================
// Audit Logger Configuration
// ============================================================================

const logsDir = path.join(process.cwd(), 'logs');

/**
 * Dedicated Winston logger for audit events.
 * Uses a separate file transport to ensure audit logs are not mixed with
 * application logs and can be easily shipped to a SIEM.
 */
const auditLogger = winston.createLogger({
  level: 'info',
  defaultMeta: {
    service: 'ringrift-api',
    logType: 'audit',
    environment: config.nodeEnv,
  },
  format: winston.format.combine(
    winston.format.timestamp({
      format: () => new Date().toISOString(),
    }),
    winston.format.json()
  ),
  transports: [
    // Dedicated audit log file
    new winston.transports.File({
      filename: path.join(logsDir, 'audit.log'),
      maxsize: 10485760, // 10MB
      maxFiles: 10, // Keep more files for audit trail
    }),
  ],
});

// Also log to console in development for visibility
if (!config.isProduction) {
  auditLogger.add(
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.printf(({ timestamp, event, userId, result, reason }) => {
          const resultColor = result === 'success' ? '\x1b[32m' : '\x1b[31m';
          const resetColor = '\x1b[0m';
          return `${timestamp} [AUDIT] ${event} | user=${userId || 'anonymous'} | ${resultColor}${result}${resetColor}${reason ? ` | reason=${reason}` : ''}`;
        })
      ),
    })
  );
}

// ============================================================================
// Audit Log Functions
// ============================================================================

/**
 * Extract client info from Express request object.
 */
function extractClientInfo(
  req: AuditRequest | undefined
): Pick<AuditLogEntry, 'ip' | 'userAgent' | 'method' | 'path'> {
  if (!req) return {};

  const forwardedFor = req.headers?.['x-forwarded-for'];
  const ip =
    (typeof forwardedFor === 'string' ? forwardedFor.split(',')[0]?.trim() : undefined) ||
    req.ip ||
    'unknown';

  const userAgent = req.headers?.['user-agent'];

  const result: Pick<AuditLogEntry, 'ip' | 'userAgent' | 'method' | 'path'> = { ip };

  if (typeof userAgent === 'string') {
    result.userAgent = userAgent;
  }
  if (req.method) {
    result.method = req.method;
  }
  const pathValue = req.originalUrl || req.path;
  if (pathValue) {
    result.path = pathValue;
  }

  return result;
}

/** Options for the audit function */
interface AuditOptions {
  userId?: string | undefined;
  email?: string | undefined;
  req?: AuditRequest | undefined;
  reason?: string | undefined;
  details?: Record<string, unknown> | undefined;
}

/**
 * Log a security audit event.
 */
export function audit(
  event: AuditEventType,
  result: AuditLogEntry['result'],
  options: AuditOptions = {}
): void {
  const now = Date.now();
  const context = getRequestContext();

  const entry: AuditLogEntry = {
    event,
    timestamp: now,
    timestampIso: new Date(now).toISOString(),
    result,
    ...extractClientInfo(options.req),
  };

  // Add optional fields only if they have values
  const userId = options.userId || context?.userId;
  if (userId) entry.userId = userId;

  if (options.email) entry.email = redactEmail(options.email);

  const requestId = options.req?.requestId || context?.requestId;
  if (requestId) entry.requestId = requestId;

  if (options.reason) entry.reason = options.reason;
  if (options.details) entry.details = options.details;

  // Log at appropriate level based on result
  if (result === 'success') {
    auditLogger.info(entry);
  } else {
    auditLogger.warn(entry);
  }
}

// ============================================================================
// Convenience Functions for Common Events
// ============================================================================

/**
 * Log a successful login.
 */
export function auditLoginSuccess(userId: string, email: string, req?: AuditRequest): void {
  audit('auth.login.success', 'success', { userId, email, req });
}

/**
 * Log a failed login attempt.
 */
export function auditLoginFailed(email: string, reason: string, req?: AuditRequest): void {
  audit('auth.login.failed', 'failure', { email, reason, req });
}

/**
 * Log a user logout.
 */
export function auditLogout(userId: string, req?: AuditRequest): void {
  audit('auth.logout', 'success', { userId, req });
}

/**
 * Log a token refresh.
 */
export function auditTokenRefresh(
  userId: string,
  req?: AuditRequest,
  details?: { familyId?: string | undefined }
): void {
  const auditDetails = details?.familyId ? { familyId: details.familyId } : undefined;
  audit('auth.token.refresh', 'success', { userId, req, details: auditDetails });
}

/**
 * Log account lockout.
 */
export function auditLockout(
  email: string,
  req?: AuditRequest,
  details?: { attempts?: number; lockoutDuration?: number }
): void {
  const auditDetails: Record<string, unknown> = {};
  if (details?.attempts !== undefined) auditDetails.attempts = details.attempts;
  if (details?.lockoutDuration !== undefined)
    auditDetails.lockoutDuration = details.lockoutDuration;

  audit('auth.lockout', 'blocked', {
    email,
    req,
    details: Object.keys(auditDetails).length > 0 ? auditDetails : undefined,
    reason: 'Too many failed attempts',
  });
}

/**
 * Log user registration.
 */
export function auditRegister(userId: string, email: string, req?: AuditRequest): void {
  audit('auth.register', 'success', { userId, email, req });
}

/**
 * Log access denied.
 */
export function auditAccessDenied(
  userId: string | undefined,
  resource: string,
  reason: string,
  req?: AuditRequest
): void {
  audit('authz.access.denied', 'blocked', {
    userId,
    reason,
    req,
    details: { resource },
  });
}

/**
 * Log admin action.
 */
export function auditAdminAction(
  userId: string,
  action: string,
  req?: AuditRequest,
  details?: Record<string, unknown>
): void {
  audit('authz.admin.action', 'success', {
    userId,
    req,
    details: { action, ...details },
  });
}

/**
 * Log rate limit exceeded.
 */
export function auditRateLimitExceeded(limiterName: string, key: string, req?: AuditRequest): void {
  audit('security.rate_limit.exceeded', 'blocked', {
    req,
    reason: `Rate limit exceeded: ${limiterName}`,
    details: { limiter: limiterName, key },
  });
}

/**
 * Log CORS rejection.
 */
export function auditCorsRejected(origin: string, req?: AuditRequest): void {
  audit('security.cors.rejected', 'blocked', {
    req,
    reason: 'CORS policy violation',
    details: { origin },
  });
}

/**
 * Log validation failure.
 */
export function auditValidationFailed(
  endpoint: string,
  errors: string[],
  req?: AuditRequest
): void {
  audit('security.validation.failed', 'failure', {
    req,
    reason: 'Input validation failed',
    details: { endpoint, errors: errors.slice(0, 5) }, // Limit error details
  });
}

/**
 * Log data export request.
 */
export function auditDataExport(userId: string, req?: AuditRequest): void {
  audit('data.export.requested', 'success', { userId, req });
}

/**
 * Log account deletion.
 */
export function auditAccountDeleted(userId: string, req?: AuditRequest): void {
  audit('data.account.deleted', 'success', { userId, req });
}

/**
 * Log WebSocket authentication.
 */
export function auditWsAuthenticated(userId: string, socketId: string, ip?: string): void {
  const req: AuditRequest | undefined = ip ? { ip, headers: {} } : undefined;
  audit('ws.connection.authenticated', 'success', {
    userId,
    details: { socketId },
    req,
  });
}

/**
 * Log WebSocket session termination.
 */
export function auditWsSessionTerminated(userId: string, reason: string, count: number): void {
  audit('ws.session.terminated', 'success', {
    userId,
    reason,
    details: { sessionsTerminated: count },
  });
}

export { auditLogger };
