const JSDOMEnvironment = require('jest-environment-jsdom').default;

class CustomJSDOMEnvironment extends JSDOMEnvironment {
  constructor(config, context) {
    // Suppress noisy Node deprecation warning for punycode used by jsdom deps.
    if (!process.emitWarning.__ringriftPunycodeFilter) {
      const originalEmitWarning = process.emitWarning;
      const filtered = (warning, ...args) => {
        const message = typeof warning === 'string' ? warning : warning?.message;
        if (message && message.includes('punycode')) {
          return;
        }
        return originalEmitWarning.call(process, warning, ...args);
      };
      filtered.__ringriftPunycodeFilter = true;
      process.emitWarning = filtered;
    }

    super(config, context);
    
    // Add import.meta mock
    this.global.import = this.global.import || {};
    this.global.import.meta = {
      env: {
        MODE: 'test',
        DEV: false,
        PROD: false,
        SSR: false,
        VITE_API_URL: 'http://localhost:3000',
        VITE_WS_URL: 'http://localhost:3000',
      },
    };

    // Polyfill structuredClone for jsdom environment
    // structuredClone is available in Node 17+ but jsdom doesn't expose it by default
    // Use Node's native structuredClone if available, otherwise use JSON fallback
    if (typeof this.global.structuredClone === 'undefined') {
      if (typeof structuredClone !== 'undefined') {
        // Node 17+ has native structuredClone - use it
        this.global.structuredClone = structuredClone;
      } else {
        // Fallback for older Node versions
        this.global.structuredClone = (obj) => JSON.parse(JSON.stringify(obj));
      }
    }
  }
}

module.exports = CustomJSDOMEnvironment;
