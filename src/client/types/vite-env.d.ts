/// <reference types="vite/client" />

/**
 * Extend Vite's built-in ImportMetaEnv with custom VITE_* environment variables.
 * Vite already defines MODE, DEV, PROD, SSR in its client types.
 * We only add our application-specific variables here.
 */
interface ImportMetaEnv {
  readonly VITE_ERROR_REPORTING_ENABLED?: string | undefined;
  readonly VITE_ERROR_REPORTING_ENDPOINT?: string | undefined;
  readonly VITE_ERROR_REPORTING_MAX_EVENTS?: string | undefined;
  readonly VITE_API_BASE_URL?: string | undefined;
  readonly VITE_WS_BASE_URL?: string | undefined;
}
