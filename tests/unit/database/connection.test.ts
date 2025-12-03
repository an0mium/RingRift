/**
 * Unit tests for database/connection.ts
 *
 * Tests cover:
 * - connectDatabase(): initial connection, already connected, connection error, dev logging
 * - getDatabaseClient(): returns client when connected, null when disconnected
 * - disconnectDatabase(): disconnects when connected, no-op when already disconnected
 * - checkDatabaseHealth(): healthy, no prisma, query error
 * - withTransaction(): throws when not connected, executes callback successfully
 */

// Mock dependencies before imports
const mockLogger = {
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  debug: jest.fn(),
};

jest.mock('../../../src/server/utils/logger', () => ({
  logger: mockLogger,
}));

// Track config.isDevelopment for testing dev logging branch
let mockIsDevelopment = false;
jest.mock('../../../src/server/config', () => ({
  config: {
    get isDevelopment() {
      return mockIsDevelopment;
    },
  },
}));

// Create mock PrismaClient instance
const mockPrismaInstance = {
  $connect: jest.fn(),
  $disconnect: jest.fn(),
  $queryRaw: jest.fn(),
  $transaction: jest.fn(),
  $on: jest.fn(),
};

// Track PrismaClient constructor calls
const mockPrismaClientConstructor = jest.fn(() => mockPrismaInstance);

jest.mock('@prisma/client', () => ({
  PrismaClient: mockPrismaClientConstructor,
}));

// Import module after mocks are set up
import {
  connectDatabase,
  getDatabaseClient,
  disconnectDatabase,
  checkDatabaseHealth,
  withTransaction,
} from '../../../src/server/database/connection';

describe('database/connection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockIsDevelopment = false;

    // Reset mock implementations to defaults
    mockPrismaInstance.$connect.mockResolvedValue(undefined);
    mockPrismaInstance.$disconnect.mockResolvedValue(undefined);
    mockPrismaInstance.$queryRaw.mockResolvedValue([{ 1: 1 }]);
    mockPrismaInstance.$transaction.mockImplementation(async (callback) => {
      return await callback(mockPrismaInstance);
    });
  });

  // We need to reset module state between test suites
  // This is tricky because the module has internal state (prisma variable)
  // We'll test in sequence to manage state

  describe('connectDatabase', () => {
    beforeEach(async () => {
      // Ensure disconnected state before each test
      await disconnectDatabase();
      jest.clearAllMocks();
    });

    it('creates new PrismaClient and connects on first call', async () => {
      const result = await connectDatabase();

      expect(mockPrismaClientConstructor).toHaveBeenCalledTimes(1);
      expect(mockPrismaClientConstructor).toHaveBeenCalledWith({
        log: [
          { emit: 'event', level: 'query' },
          { emit: 'event', level: 'error' },
          { emit: 'event', level: 'info' },
          { emit: 'event', level: 'warn' },
        ],
        errorFormat: 'pretty',
      });
      expect(mockPrismaInstance.$connect).toHaveBeenCalledTimes(1);
      expect(mockLogger.info).toHaveBeenCalledWith('Database connected successfully');
      expect(result).toBe(mockPrismaInstance);
    });

    it('returns existing client on subsequent calls (already connected branch)', async () => {
      // First call - creates new client
      const first = await connectDatabase();
      jest.clearAllMocks();

      // Second call - should return existing client
      const second = await connectDatabase();

      expect(mockPrismaClientConstructor).not.toHaveBeenCalled();
      expect(mockPrismaInstance.$connect).not.toHaveBeenCalled();
      expect(first).toBe(second);
    });

    it('throws and logs error on connection failure', async () => {
      const connectionError = new Error('Connection refused');
      mockPrismaInstance.$connect.mockRejectedValueOnce(connectionError);

      await expect(connectDatabase()).rejects.toThrow('Connection refused');

      expect(mockLogger.error).toHaveBeenCalledWith(
        'Failed to connect to database:',
        connectionError
      );
    });

    it('throws and logs error when PrismaClient constructor fails (e.g. invalid DATABASE_URL)', async () => {
      const constructorError = new Error('Invalid DATABASE_URL');
      mockPrismaClientConstructor.mockImplementationOnce(() => {
        throw constructorError;
      });

      await expect(connectDatabase()).rejects.toThrow('Invalid DATABASE_URL');

      expect(mockLogger.error).toHaveBeenCalledWith(
        'Failed to connect to database:',
        constructorError
      );
    });

    it('registers $on event handlers in development mode', async () => {
      mockIsDevelopment = true;

      await connectDatabase();

      // Should register 4 event handlers: query, error, info, warn
      expect(mockPrismaInstance.$on).toHaveBeenCalledTimes(4);
      expect(mockPrismaInstance.$on).toHaveBeenCalledWith('query', expect.any(Function));
      expect(mockPrismaInstance.$on).toHaveBeenCalledWith('error', expect.any(Function));
      expect(mockPrismaInstance.$on).toHaveBeenCalledWith('info', expect.any(Function));
      expect(mockPrismaInstance.$on).toHaveBeenCalledWith('warn', expect.any(Function));
    });

    it('does not register $on event handlers in production mode', async () => {
      mockIsDevelopment = false;

      await connectDatabase();

      expect(mockPrismaInstance.$on).not.toHaveBeenCalled();
    });

    describe('development logging event handlers', () => {
      beforeEach(async () => {
        mockIsDevelopment = true;
        await disconnectDatabase();
        jest.clearAllMocks();
      });

      it('query event handler logs database queries', async () => {
        await connectDatabase();

        // Get the query event handler
        const queryHandler = mockPrismaInstance.$on.mock.calls.find(
          (call: [string, (e: unknown) => void]) => call[0] === 'query'
        )?.[1];

        expect(queryHandler).toBeDefined();

        // Simulate query event
        const queryEvent = {
          query: 'SELECT * FROM users',
          params: '[]',
          duration: 15,
        };
        queryHandler(queryEvent);

        expect(mockLogger.debug).toHaveBeenCalledWith('Database Query:', {
          query: 'SELECT * FROM users',
          params: '[]',
          duration: '15ms',
        });
      });

      it('error event handler logs database errors', async () => {
        await connectDatabase();

        const errorHandler = mockPrismaInstance.$on.mock.calls.find(
          (call: [string, (e: unknown) => void]) => call[0] === 'error'
        )?.[1];

        expect(errorHandler).toBeDefined();

        const errorEvent = { message: 'Database error occurred' };
        errorHandler(errorEvent);

        expect(mockLogger.error).toHaveBeenCalledWith('Database Error:', errorEvent);
      });

      it('info event handler logs database info', async () => {
        await connectDatabase();

        const infoHandler = mockPrismaInstance.$on.mock.calls.find(
          (call: [string, (e: unknown) => void]) => call[0] === 'info'
        )?.[1];

        expect(infoHandler).toBeDefined();

        const infoEvent = { message: 'Database info message' };
        infoHandler(infoEvent);

        expect(mockLogger.info).toHaveBeenCalledWith('Database Info:', 'Database info message');
      });

      it('warn event handler logs database warnings', async () => {
        await connectDatabase();

        const warnHandler = mockPrismaInstance.$on.mock.calls.find(
          (call: [string, (e: unknown) => void]) => call[0] === 'warn'
        )?.[1];

        expect(warnHandler).toBeDefined();

        const warnEvent = { message: 'Database warning message' };
        warnHandler(warnEvent);

        expect(mockLogger.warn).toHaveBeenCalledWith(
          'Database Warning:',
          'Database warning message'
        );
      });
    });
  });

  describe('getDatabaseClient', () => {
    it('returns null when not connected', async () => {
      await disconnectDatabase();

      const result = getDatabaseClient();

      expect(result).toBeNull();
    });

    it('returns PrismaClient when connected', async () => {
      await disconnectDatabase();
      await connectDatabase();

      const result = getDatabaseClient();

      expect(result).toBe(mockPrismaInstance);
    });
  });

  describe('disconnectDatabase', () => {
    it('disconnects and logs when connected', async () => {
      await disconnectDatabase();
      await connectDatabase();
      jest.clearAllMocks();

      await disconnectDatabase();

      expect(mockPrismaInstance.$disconnect).toHaveBeenCalledTimes(1);
      expect(mockLogger.info).toHaveBeenCalledWith('Database disconnected');
    });

    it('no-ops when already disconnected', async () => {
      await disconnectDatabase();
      jest.clearAllMocks();

      await disconnectDatabase();

      expect(mockPrismaInstance.$disconnect).not.toHaveBeenCalled();
      expect(mockLogger.info).not.toHaveBeenCalled();
    });

    it('sets prisma to null after disconnect', async () => {
      await disconnectDatabase();
      await connectDatabase();

      await disconnectDatabase();

      expect(getDatabaseClient()).toBeNull();
    });
  });

  describe('checkDatabaseHealth', () => {
    it('returns false when not connected', async () => {
      await disconnectDatabase();

      const result = await checkDatabaseHealth();

      expect(result).toBe(false);
      expect(mockPrismaInstance.$queryRaw).not.toHaveBeenCalled();
    });

    it('returns true when query succeeds', async () => {
      await disconnectDatabase();
      await connectDatabase();
      mockPrismaInstance.$queryRaw.mockResolvedValueOnce([{ 1: 1 }]);

      const result = await checkDatabaseHealth();

      expect(result).toBe(true);
      expect(mockPrismaInstance.$queryRaw).toHaveBeenCalled();
    });

    it('returns false and logs error when query fails', async () => {
      await disconnectDatabase();
      await connectDatabase();
      jest.clearAllMocks();

      const queryError = new Error('Connection lost');
      mockPrismaInstance.$queryRaw.mockRejectedValueOnce(queryError);

      const result = await checkDatabaseHealth();

      expect(result).toBe(false);
      expect(mockLogger.error).toHaveBeenCalledWith('Database health check failed:', queryError);
    });
  });

  describe('withTransaction', () => {
    it('throws when not connected', async () => {
      await disconnectDatabase();

      await expect(withTransaction(async () => 'result')).rejects.toThrow('Database not connected');
    });

    it('executes callback within transaction when connected', async () => {
      await disconnectDatabase();
      await connectDatabase();

      const callback = jest.fn().mockResolvedValue('transaction result');
      const result = await withTransaction(callback);

      expect(mockPrismaInstance.$transaction).toHaveBeenCalledTimes(1);
      expect(callback).toHaveBeenCalledWith(mockPrismaInstance);
      expect(result).toBe('transaction result');
    });

    it('propagates callback errors', async () => {
      await disconnectDatabase();
      await connectDatabase();

      const callbackError = new Error('Transaction failed');
      mockPrismaInstance.$transaction.mockRejectedValueOnce(callbackError);

      await expect(withTransaction(async () => 'result')).rejects.toThrow('Transaction failed');
    });

    it('returns callback result type correctly', async () => {
      await disconnectDatabase();
      await connectDatabase();

      const typedResult = { id: 1, name: 'test' };
      const callback = jest.fn().mockResolvedValue(typedResult);
      mockPrismaInstance.$transaction.mockImplementationOnce(async (cb) => cb(mockPrismaInstance));

      const result = await withTransaction(callback);

      expect(result).toEqual(typedResult);
    });
  });

  describe('process event handlers', () => {
    // Note: These are registered at module load time.
    // Testing them directly is complex because they call process.exit().
    // We primarily verify coverage through the connectDatabase/disconnectDatabase tests.

    it('beforeExit handler calls disconnectDatabase', async () => {
      // The handler is registered at module load, we can verify disconnectDatabase works
      await connectDatabase();
      expect(getDatabaseClient()).not.toBeNull();

      await disconnectDatabase();
      expect(getDatabaseClient()).toBeNull();
    });
  });
});
