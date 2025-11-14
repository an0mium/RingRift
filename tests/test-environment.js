/**
 * Custom Jest Test Environment
 * Extends jest-environment-node without localStorage initialization
 */

const NodeEnvironment = require('jest-environment-node').TestEnvironment;

class CustomTestEnvironment extends NodeEnvironment {
  constructor(config, context) {
    super(config, context);
    
    // Set test environment variables
    this.global.process.env.NODE_ENV = 'test';
    this.global.process.env.LOG_LEVEL = 'error';
    
    // Mock localStorage without SecurityError
    this.global.localStorage = {
      getItem: () => null,
      setItem: () => {},
      removeItem: () => {},
      clear: () => {},
      length: 0,
      key: () => null,
    };
  }

  async setup() {
    await super.setup();
  }

  async teardown() {
    await super.teardown();
  }

  getVmContext() {
    return super.getVmContext();
  }
}

module.exports = CustomTestEnvironment;
