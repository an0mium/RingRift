/**
 * Minimal, env-gated client-side error reporting utilities.
 *
 * This module intentionally keeps configuration surface small and
 * avoids any third-party SDKs. When disabled, all helpers are
 * cheap no-ops so they are safe to import in any client code.
 */
export interface ClientErrorContext {
  [key: string]: unknown;
}

interface NormalizedError {
  name: string;
  message: string;
  stack?: string | undefined;
}

interface ClientErrorPayload extends NormalizedError {
  context?: ClientErrorContext | undefined;
  url?: string | undefined;
  userAgent?: string | undefined;
  timestamp: string;
  type: string;
}

/** Vite injects client env vars via a synthetic __VITE_ENV__ object on globalThis. */
interface ViteEnvWindow {
  __VITE_ENV__?: Record<string, string | undefined>;
  MODE?: string;
}

// Vite exposes env variables on import.meta.env
// Access via globalThis to avoid Jest parse errors with import.meta syntax
const viteGlobal = globalThis as unknown as ViteEnvWindow;
const env: Record<string, string | undefined> = viteGlobal.__VITE_ENV__ ?? {};

const ERROR_REPORTING_ENABLED: boolean = env.VITE_ERROR_REPORTING_ENABLED === 'true';
const ERROR_REPORTING_ENDPOINT: string =
  (env.VITE_ERROR_REPORTING_ENDPOINT as string | undefined) || '/api/client-errors';
const MAX_REPORTS_PER_SESSION: number = Number(env.VITE_ERROR_REPORTING_MAX_EVENTS ?? 50);

let reportsSent = 0;
let globalHandlersInstalled = false;

export function isErrorReportingEnabled(): boolean {
  return ERROR_REPORTING_ENABLED;
}

function shouldReport(): boolean {
  if (!ERROR_REPORTING_ENABLED) return false;
  if (reportsSent >= MAX_REPORTS_PER_SESSION) return false;
  reportsSent += 1;
  return true;
}

function normalizeError(error: unknown): NormalizedError {
  if (error instanceof Error) {
    return {
      name: error.name || 'Error',
      message: error.message || 'Unknown error',
      stack: error.stack,
    };
  }

  if (typeof error === 'string') {
    return {
      name: 'Error',
      message: error,
    };
  }

  try {
    const serialized = JSON.stringify(error);
    return {
      name: 'Error',
      message: serialized,
    };
  } catch {
    return {
      name: 'Error',
      message: String(error),
    };
  }
}

function buildPayload(error: unknown, context?: ClientErrorContext): ClientErrorPayload {
  const normalized = normalizeError(error);

  let url: string | undefined;
  let userAgent: string | undefined;

  if (typeof window !== 'undefined') {
    url = window.location?.href;
  }

  if (typeof navigator !== 'undefined') {
    userAgent = navigator.userAgent;
  }

  return {
    ...normalized,
    context,
    url,
    userAgent,
    timestamp: new Date().toISOString(),
    type:
      context && 'type' in context && typeof context.type === 'string'
        ? context.type
        : 'client_error',
  };
}

export async function reportClientError(
  error: unknown,
  context?: ClientErrorContext
): Promise<void> {
  if (!shouldReport()) return;

  // Avoid throwing from the reporter; failures are best-effort only.
  try {
    if (typeof fetch !== 'function') {
      return;
    }

    const payload = buildPayload(error, context);

    // Fire-and-forget; callers do not await diagnostics.
    void fetch(ERROR_REPORTING_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      keepalive: true,
    });
  } catch (e) {
    // Swallow any errors. Optionally emit a console warning in dev builds.
    if ((env.MODE as string | undefined) === 'development') {
      console.warn('Failed to report client error', e);
    }
  }
}

/**
 * Attach window-level listeners for uncaught errors and unhandled promise
 * rejections. Safe to call multiple times; handlers are installed once.
 */
/**
 * Extract a user-friendly error message from an unknown error.
 * Handles axios-style errors with nested response.data.error.message,
 * standard Error objects, and falls back to a default message.
 */
export function extractErrorMessage(error: unknown, defaultMessage: string): string {
  if (error && typeof error === 'object') {
    const e = error as Record<string, unknown>;

    // Handle axios-style error responses
    const response = e.response as Record<string, unknown> | undefined;
    if (response) {
      const data = response.data as Record<string, unknown> | undefined;
      if (data) {
        // Nested error object: { error: { message: "..." } }
        const errorObj = data.error as Record<string, unknown> | undefined;
        if (typeof errorObj?.message === 'string') {
          return errorObj.message;
        }
        // Check for error.code for special handling
        if (typeof errorObj?.code === 'string') {
          // Return both code and message for caller to handle
        }
        // Direct message on data: { message: "..." }
        if (typeof data.message === 'string') {
          return data.message;
        }
      }
    }

    // Standard Error.message
    if (typeof e.message === 'string') {
      return e.message;
    }
  }

  // String errors
  if (typeof error === 'string') {
    return error;
  }

  return defaultMessage;
}

/**
 * Extract axios-style error code if present.
 * Returns undefined if no code is found.
 */
export function extractErrorCode(error: unknown): string | undefined {
  if (error && typeof error === 'object') {
    const e = error as Record<string, unknown>;
    const response = e.response as Record<string, unknown> | undefined;
    const data = response?.data as Record<string, unknown> | undefined;
    const errorObj = data?.error as Record<string, unknown> | undefined;
    if (typeof errorObj?.code === 'string') {
      return errorObj.code;
    }
  }
  return undefined;
}

export function setupGlobalErrorHandlers(): void {
  if (!ERROR_REPORTING_ENABLED) return;
  if (typeof window === 'undefined' || typeof window.addEventListener !== 'function') {
    return;
  }
  if (globalHandlersInstalled) return;
  globalHandlersInstalled = true;

  window.addEventListener('error', (event: ErrorEvent) => {
    // Ignore script load errors and similar noise; focus on actual Error objects.
    const error = event.error ?? event.message ?? 'Unknown window error';
    void reportClientError(error, { type: 'window_error' });
  });

  window.addEventListener('unhandledrejection', (event: PromiseRejectionEvent) => {
    void reportClientError(event.reason, { type: 'unhandledrejection' });
  });
}
