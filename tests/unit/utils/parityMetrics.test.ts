/**
 * Parity Metrics Unit Tests
 *
 * Tests for the parityMetrics module which provides a lightweight helper
 * for recording TS↔Python parity check outcomes.
 */

const mockRecordParityCheck = jest.fn();

// Mock MetricsService BEFORE importing the module under test
jest.mock('../../../src/server/services/MetricsService', () => ({
  getMetricsService: jest.fn(() => ({
    recordParityCheck: mockRecordParityCheck,
  })),
}));

// Import AFTER mocks are set up
import { recordParityBatchResult } from '../../../src/server/utils/parityMetrics';
import { getMetricsService } from '../../../src/server/services/MetricsService';

describe('recordParityBatchResult', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should record passed parity check', () => {
    recordParityBatchResult(true);

    expect(getMetricsService).toHaveBeenCalled();
    expect(mockRecordParityCheck).toHaveBeenCalledWith(true);
  });

  it('should record failed parity check', () => {
    recordParityBatchResult(false);

    expect(getMetricsService).toHaveBeenCalled();
    expect(mockRecordParityCheck).toHaveBeenCalledWith(false);
  });

  it('should silently handle errors from metrics service', () => {
    // Make getMetricsService throw an error
    (getMetricsService as jest.Mock).mockImplementationOnce(() => {
      throw new Error('Metrics service unavailable');
    });

    // Should not throw - errors are silently caught
    expect(() => recordParityBatchResult(true)).not.toThrow();
  });

  it('should silently handle errors from recordParityCheck', () => {
    // Make recordParityCheck throw an error
    mockRecordParityCheck.mockImplementationOnce(() => {
      throw new Error('Recording failed');
    });

    // Should not throw - errors are silently caught
    expect(() => recordParityBatchResult(false)).not.toThrow();
  });
});
