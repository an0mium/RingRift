/**
 * Jest Environment Setup
 * Runs BEFORE test framework is installed
 * Used for mocking global browser APIs in Node environment
 */

// Mock localStorage for Node environment
const localStorageMock = {
  getItem: (key: string) => null,
  setItem: (key: string, value: string) => {},
  removeItem: (key: string) => {},
  clear: () => {},
  length: 0,
  key: (index: number) => null,
};

// Define localStorage on global object
Object.defineProperty(global, 'localStorage', {
  value: localStorageMock,
  writable: true,
  configurable: true,
});

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error';
