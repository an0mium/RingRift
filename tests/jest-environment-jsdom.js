const JSDOMEnvironment = require('jest-environment-jsdom').default;

class CustomJSDOMEnvironment extends JSDOMEnvironment {
  constructor(config, context) {
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
  }
}

module.exports = CustomJSDOMEnvironment;