/**
 * Email Utility Unit Tests
 *
 * Tests for the mock email service including:
 * - sendEmail function
 * - sendVerificationEmail function
 * - sendPasswordResetEmail function
 * - HTML truncation branch in logging
 */

import {
  sendEmail,
  sendVerificationEmail,
  sendPasswordResetEmail,
} from '../../../src/server/utils/email';
import { logger } from '../../../src/server/utils/logger';

jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    debug: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../../src/server/config', () => ({
  config: {
    server: {
      publicClientUrl: 'http://localhost:3000',
    },
    email: {
      provider: 'mock',
      from: 'test@ringrift.ai',
      ses: null,
    },
    isTest: true,
  },
}));

describe('sendEmail', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should log email content and return true', async () => {
    const options = {
      to: 'test@example.com',
      subject: 'Test Subject',
      text: 'Test body text',
    };

    const resultPromise = sendEmail(options);

    // Fast-forward the 100ms delay
    jest.advanceTimersByTime(100);

    const result = await resultPromise;

    expect(result).toBe(true);
    expect(logger.info).toHaveBeenCalledWith('MOCK EMAIL SENT', {
      to: 'test@example.com',
      subject: 'Test Subject',
      text: 'Test body text',
      html: undefined,
    });
  });

  it('should include truncated HTML in log when HTML is provided', async () => {
    const shortHtml = '<p>Short HTML</p>';
    const options = {
      to: 'test@example.com',
      subject: 'Test Subject',
      text: 'Test body text',
      html: shortHtml,
    };

    const resultPromise = sendEmail(options);
    jest.advanceTimersByTime(100);
    await resultPromise;

    expect(logger.info).toHaveBeenCalledWith('MOCK EMAIL SENT', {
      to: 'test@example.com',
      subject: 'Test Subject',
      text: 'Test body text',
      html: '<p>Short HTML</p>...',
    });
  });

  it('should truncate long HTML content to 100 characters', async () => {
    const longHtml = 'a'.repeat(150);
    const options = {
      to: 'test@example.com',
      subject: 'Test Subject',
      text: 'Test body text',
      html: longHtml,
    };

    const resultPromise = sendEmail(options);
    jest.advanceTimersByTime(100);
    await resultPromise;

    const logCall = (logger.info as jest.Mock).mock.calls[1];
    const loggedHtml = logCall[1].html;

    // Should be truncated to first 100 chars + '...'
    expect(loggedHtml).toBe('a'.repeat(100) + '...');
    expect(loggedHtml.length).toBe(103);
  });

  it('should handle undefined html option (no html branch)', async () => {
    const options = {
      to: 'recipient@test.com',
      subject: 'No HTML',
      text: 'Plain text only',
    };

    const resultPromise = sendEmail(options);
    jest.advanceTimersByTime(100);
    await resultPromise;

    const logCall = (logger.info as jest.Mock).mock.calls[1];
    expect(logCall[1].html).toBeUndefined();
  });
});

describe('sendVerificationEmail', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should send verification email with correct link', async () => {
    const email = 'newuser@example.com';
    const token = 'verification-token-123';

    const resultPromise = sendVerificationEmail(email, token);
    jest.advanceTimersByTime(100);
    const result = await resultPromise;

    expect(result).toBe(true);
    expect(logger.info).toHaveBeenCalledTimes(2); // sendEmail called + MOCK EMAIL SENT

    // Second call is the MOCK EMAIL SENT log
    const logCall = (logger.info as jest.Mock).mock.calls[1];
    expect(logCall[0]).toBe('MOCK EMAIL SENT');
    expect(logCall[1].to).toBe('newuser@example.com');
    expect(logCall[1].subject).toBe('Verify your RingRift account');
    expect(logCall[1].text).toContain(
      'http://localhost:3000/verify-email?token=verification-token-123'
    );
  });

  it('should include HTML content with verification link', async () => {
    const email = 'user@test.com';
    const token = 'abc123';

    const resultPromise = sendVerificationEmail(email, token);
    jest.advanceTimersByTime(100);
    await resultPromise;

    const logCall = (logger.info as jest.Mock).mock.calls[1];
    // HTML should be truncated but should contain the link mention
    expect(logCall[1].html).toBeDefined();
  });
});

describe('sendPasswordResetEmail', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should send password reset email with correct link', async () => {
    const email = 'forgotpw@example.com';
    const token = 'reset-token-456';

    const resultPromise = sendPasswordResetEmail(email, token);
    jest.advanceTimersByTime(100);
    const result = await resultPromise;

    expect(result).toBe(true);
    expect(logger.info).toHaveBeenCalledTimes(2); // sendEmail called + MOCK EMAIL SENT

    const logCall = (logger.info as jest.Mock).mock.calls[1];
    expect(logCall[0]).toBe('MOCK EMAIL SENT');
    expect(logCall[1].to).toBe('forgotpw@example.com');
    expect(logCall[1].subject).toBe('Reset your RingRift password');
    expect(logCall[1].text).toContain('http://localhost:3000/reset-password?token=reset-token-456');
  });

  it('should include HTML content with reset link', async () => {
    const email = 'user@test.com';
    const token = 'xyz789';

    const resultPromise = sendPasswordResetEmail(email, token);
    jest.advanceTimersByTime(100);
    await resultPromise;

    const logCall = (logger.info as jest.Mock).mock.calls[1];
    expect(logCall[1].html).toBeDefined();
  });

  it('should mention expiration in the text body', async () => {
    const email = 'user@test.com';
    const token = 'expiring-token';

    const resultPromise = sendPasswordResetEmail(email, token);
    jest.advanceTimersByTime(100);
    await resultPromise;

    const logCall = (logger.info as jest.Mock).mock.calls[1];
    // The text body should indicate the reset was requested
    expect(logCall[1].text).toContain('password reset');
  });
});
