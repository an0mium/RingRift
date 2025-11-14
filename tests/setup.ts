/**
 * Jest Setup File
 * Runs AFTER test framework is installed
 */

// Global test timeout
jest.setTimeout(10000);

// Mock console methods to reduce noise (optional)
// Uncomment if you want to suppress console output during tests
// global.console = {
//   ...console,
//   log: jest.fn(),
//   debug: jest.fn(),
//   info: jest.fn(),
//   warn: jest.fn(),
//   error: jest.fn(),
// };

// Add custom matchers if needed
// expect.extend({
//   // Custom matchers here
// });

// Global test cleanup
afterEach(() => {
  // Clear all mocks after each test
  jest.clearAllMocks();
});
