/**
 * Setup file for jsdom environment
 * Runs BEFORE test framework is installed
 */

// Mock import.meta for Vite-specific code
// This must be set on the global object before any modules are loaded
Object.defineProperty(global, 'importMeta', {
  value: {
    env: {
      MODE: 'test',
      DEV: false,
      PROD: false,
      SSR: false,
      VITE_API_URL: 'http://localhost:3000',
      VITE_WS_URL: 'http://localhost:3000',
    },
  },
  writable: true,
  configurable: true,
});