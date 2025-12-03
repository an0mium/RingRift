/**
 * @file PythonRulesClient.test.ts
 * @description Comprehensive unit tests for PythonRulesClient
 * covering all branch paths including error handling and metrics recording.
 */

// Mock dependencies before imports
const mockLogger = {
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  debug: jest.fn(),
};

jest.mock('../../src/server/utils/logger', () => ({
  logger: mockLogger,
}));

const mockRecordRulesError = jest.fn();
jest.mock('../../src/server/services/MetricsService', () => ({
  getMetricsService: jest.fn(() => ({
    recordRulesError: mockRecordRulesError,
  })),
}));

jest.mock('../../src/server/config', () => ({
  config: {
    aiService: {
      url: 'http://default-ai-service:8000',
      rulesTimeoutMs: 5000,
    },
  },
}));

// Mock axios
const mockAxiosPost = jest.fn();
jest.mock('axios', () => ({
  create: jest.fn(() => ({
    post: mockAxiosPost,
  })),
}));

import { PythonRulesClient } from '../../src/server/services/PythonRulesClient';
import type { GameState, Move } from '../../src/shared/types/game';
import axios from 'axios';

describe('PythonRulesClient', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should use default config URL when no baseURL provided', () => {
      new PythonRulesClient();

      expect(axios.create).toHaveBeenCalledWith({
        baseURL: 'http://default-ai-service:8000',
        timeout: 5000,
        headers: { 'Content-Type': 'application/json' },
      });
    });

    it('should use provided baseURL when passed', () => {
      new PythonRulesClient('http://custom-service:9000');

      expect(axios.create).toHaveBeenCalledWith({
        baseURL: 'http://custom-service:9000',
        timeout: 5000,
        headers: { 'Content-Type': 'application/json' },
      });
    });
  });

  describe('evaluateMove', () => {
    const mockState: GameState = {
      board: { cells: [] },
      players: [],
      currentPlayer: 0,
      gameStatus: 'active',
      turnNumber: 1,
      phase: 'placement',
    } as unknown as GameState;

    const mockMove: Move = {
      type: 'place',
      playerId: 'player1',
      position: { q: 0, r: 0 },
    } as unknown as Move;

    it('should return successful response with all fields mapped from snake_case', async () => {
      const mockResponse = {
        data: {
          valid: true,
          validation_error: undefined,
          next_state: { ...mockState, turnNumber: 2 },
          state_hash: 'abc123',
          s_invariant: 42,
          game_status: 'active',
        },
      };
      mockAxiosPost.mockResolvedValueOnce(mockResponse);

      const client = new PythonRulesClient();
      const result = await client.evaluateMove(mockState, mockMove);

      expect(mockAxiosPost).toHaveBeenCalledWith('/rules/evaluate_move', {
        game_state: mockState,
        move: mockMove,
      });

      expect(result).toEqual({
        valid: true,
        validationError: undefined,
        nextState: { ...mockState, turnNumber: 2 },
        stateHash: 'abc123',
        sInvariant: 42,
        gameStatus: 'active',
      });
    });

    it('should return invalid response with validation_error mapped', async () => {
      const mockResponse = {
        data: {
          valid: false,
          validation_error: 'Invalid move: position occupied',
          next_state: undefined,
          state_hash: undefined,
          s_invariant: undefined,
          game_status: undefined,
        },
      };
      mockAxiosPost.mockResolvedValueOnce(mockResponse);

      const client = new PythonRulesClient();
      const result = await client.evaluateMove(mockState, mockMove);

      expect(result).toEqual({
        valid: false,
        validationError: 'Invalid move: position occupied',
        nextState: undefined,
        stateHash: undefined,
        sInvariant: undefined,
        gameStatus: undefined,
      });
    });

    describe('error handling', () => {
      it('should log error and record validation metric for 400 status', async () => {
        const error = new Error('Bad Request') as Error & {
          response?: { data?: unknown; status?: number };
        };
        error.response = { status: 400, data: { detail: 'Invalid data' } };
        mockAxiosPost.mockRejectedValueOnce(error);

        const client = new PythonRulesClient();

        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow('Bad Request');

        expect(mockLogger.error).toHaveBeenCalledWith('Python rules evaluate_move failed', {
          message: 'Bad Request',
          response: { detail: 'Invalid data' },
          status: 400,
        });
        expect(mockRecordRulesError).toHaveBeenCalledWith('validation');
      });

      it('should log error and record validation metric for 422 status', async () => {
        const error = new Error('Unprocessable Entity') as Error & {
          response?: { data?: unknown; status?: number };
        };
        error.response = { status: 422, data: { detail: 'Validation failed' } };
        mockAxiosPost.mockRejectedValueOnce(error);

        const client = new PythonRulesClient();

        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow(
          'Unprocessable Entity'
        );

        expect(mockRecordRulesError).toHaveBeenCalledWith('validation');
      });

      it('should record validation metric when response contains validation_error as string', async () => {
        const error = new Error('Server Error') as Error & {
          response?: { data?: unknown; status?: number };
        };
        // Status is 500 but data is a string containing "validation_error"
        error.response = {
          status: 500,
          data: 'Response includes validation_error in the message',
        };
        mockAxiosPost.mockRejectedValueOnce(error);

        const client = new PythonRulesClient();

        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow('Server Error');

        expect(mockRecordRulesError).toHaveBeenCalledWith('validation');
      });

      it('should record validation metric when response data object contains validation_error', async () => {
        const error = new Error('Server Error') as Error & {
          response?: { data?: unknown; status?: number };
        };
        // Status is 500 but data object JSON stringifies to include "validation_error"
        error.response = {
          status: 500,
          data: { validation_error: 'Some validation issue', other: 'data' },
        };
        mockAxiosPost.mockRejectedValueOnce(error);

        const client = new PythonRulesClient();

        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow('Server Error');

        expect(mockRecordRulesError).toHaveBeenCalledWith('validation');
      });

      it('should record internal error for non-validation errors', async () => {
        const error = new Error('Internal Server Error') as Error & {
          response?: { data?: unknown; status?: number };
        };
        error.response = { status: 500, data: { detail: 'Database error' } };
        mockAxiosPost.mockRejectedValueOnce(error);

        const client = new PythonRulesClient();

        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow(
          'Internal Server Error'
        );

        expect(mockRecordRulesError).toHaveBeenCalledWith('internal');
      });

      it('should record internal error when no response object exists', async () => {
        const error = new Error('Network Error');
        // No response property - network failure
        mockAxiosPost.mockRejectedValueOnce(error);

        const client = new PythonRulesClient();

        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow('Network Error');

        expect(mockLogger.error).toHaveBeenCalledWith('Python rules evaluate_move failed', {
          message: 'Network Error',
          response: undefined,
          status: undefined,
        });
        expect(mockRecordRulesError).toHaveBeenCalledWith('internal');
      });

      it('should record internal error when response has no data', async () => {
        const error = new Error('Empty Response') as Error & {
          response?: { data?: unknown; status?: number };
        };
        error.response = { status: 503, data: null };
        mockAxiosPost.mockRejectedValueOnce(error);

        const client = new PythonRulesClient();

        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow('Empty Response');

        // hasValidationPayload checks !!err.response?.data - null is falsy
        expect(mockRecordRulesError).toHaveBeenCalledWith('internal');
      });

      it('should silently swallow metrics recording failures', async () => {
        const axiosError = new Error('Network Error') as Error & {
          response?: { data?: unknown; status?: number };
        };
        axiosError.response = undefined;
        mockAxiosPost.mockRejectedValueOnce(axiosError);

        // Make metrics service throw
        mockRecordRulesError.mockImplementationOnce(() => {
          throw new Error('Metrics service unavailable');
        });

        const client = new PythonRulesClient();

        // Should still throw the original error, not the metrics error
        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow('Network Error');

        // Metrics was attempted
        expect(mockRecordRulesError).toHaveBeenCalled();
        // But original error is thrown, metrics failure silently caught
      });

      it('should handle response with empty object as data', async () => {
        const error = new Error('Server Error') as Error & {
          response?: { data?: unknown; status?: number };
        };
        error.response = { status: 502, data: {} };
        mockAxiosPost.mockRejectedValueOnce(error);

        const client = new PythonRulesClient();

        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow('Server Error');

        // Empty object {} is truthy, JSON.stringify gives '{}' which doesn't contain validation_error
        expect(mockRecordRulesError).toHaveBeenCalledWith('internal');
      });

      it('should detect validation_error in nested object via JSON.stringify', async () => {
        const error = new Error('Server Error') as Error & {
          response?: { data?: unknown; status?: number };
        };
        error.response = {
          status: 500,
          data: {
            nested: { validation_error: 'deep issue' },
          },
        };
        mockAxiosPost.mockRejectedValueOnce(error);

        const client = new PythonRulesClient();

        await expect(client.evaluateMove(mockState, mockMove)).rejects.toThrow('Server Error');

        // JSON.stringify includes nested keys
        expect(mockRecordRulesError).toHaveBeenCalledWith('validation');
      });
    });
  });
});
