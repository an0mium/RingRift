import { getSocketBaseUrl } from '../../../src/client/utils/socketBaseUrl';

describe('getSocketBaseUrl', () => {
  const originalViteEnv = (globalThis as any).__VITE_ENV__;

  function setViteEnv(overrides: Record<string, string | undefined>) {
    const baseEnv = { ...(originalViteEnv || {}) };
    delete (baseEnv as any).VITE_WS_URL;
    delete (baseEnv as any).VITE_API_URL;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = { ...baseEnv, ...overrides };
  }

  afterEach(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = originalViteEnv;
  });

  it('prefers VITE_WS_URL and strips trailing slash', () => {
    setViteEnv({
      VITE_WS_URL: 'http://localhost:4000/',
      VITE_API_URL: undefined,
    });

    const url = getSocketBaseUrl();
    expect(url).toBe('http://localhost:4000');
  });

  it('falls back to VITE_API_URL, removing /api suffix and trailing slash', () => {
    setViteEnv({
      VITE_WS_URL: undefined,
      VITE_API_URL: 'https://api.example.com/api/',
    });

    const url = getSocketBaseUrl();
    expect(url).toBe('https://api.example.com');
  });
});
