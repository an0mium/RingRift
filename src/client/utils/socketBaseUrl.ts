/**
 * Derive the WebSocket base URL from environment configuration.
 *
 * This helper is shared between the legacy GameContext implementation
 * and the new SocketGameConnection service so that the URL selection
 * logic has a single source of truth.
 */
export function getSocketBaseUrl(): string {
  // Vite exposes env variables on import.meta.env; in the bundled client this
  // is wired to globalThis.__VITE_ENV__ via vite.config.ts `define`.
  // Using the global avoids Jest parse issues with raw `import.meta` syntax.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const env: Record<string, string | undefined> = (globalThis as any).__VITE_ENV__ ?? {};

  const wsUrl = env.VITE_WS_URL as string | undefined;
  if (wsUrl) {
    return wsUrl.replace(/\/$/, '');
  }

  const apiUrl = env.VITE_API_URL as string | undefined;
  if (apiUrl) {
    const base = apiUrl.replace(/\/?api\/?$/, '');
    return base.replace(/\/$/, '');
  }

  if (typeof window !== 'undefined' && window.location?.origin) {
    const origin = window.location.origin;
    if (origin.startsWith('http://localhost:5173') || origin.startsWith('https://localhost:5173')) {
      return 'http://localhost:3000';
    }
    return origin;
  }

  return 'http://localhost:3000';
}
