import type { Request, Response } from 'express';
import { handleDifficultyCalibrationTelemetry } from '../../../src/server/routes/difficultyCalibrationTelemetry';
import { getMetricsService } from '../../../src/server/services/MetricsService';

// Hoisted mock handle mirroring the rulesUxTelemetry route tests.
const mockRecordDifficultyCalibrationEvent = jest.fn();

jest.mock('../../../src/server/services/MetricsService', () => ({
  __esModule: true,
  getMetricsService: jest.fn(() => ({
    recordDifficultyCalibrationEvent: mockRecordDifficultyCalibrationEvent,
  })),
}));

describe('difficultyCalibrationTelemetry route handler', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRecordDifficultyCalibrationEvent.mockReset();
  });

  it('forwards a valid DifficultyCalibrationEventPayload to MetricsService and returns 204', () => {
    const req = {
      body: {
        type: 'difficulty_calibration_game_started',
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 4,
        isCalibrationOptIn: true,
      },
    } as Partial<Request> as Request;

    const status = jest.fn().mockReturnThis();
    const send = jest.fn();
    const res = { status, send } as Partial<Response> as Response;

    handleDifficultyCalibrationTelemetry(req, res);

    expect(getMetricsService).toHaveBeenCalledTimes(1);
    expect(mockRecordDifficultyCalibrationEvent).toHaveBeenCalledTimes(1);
    expect(mockRecordDifficultyCalibrationEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'difficulty_calibration_game_started',
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 4,
        isCalibrationOptIn: true,
      })
    );

    expect(status).toHaveBeenCalledWith(204);
    expect(send).toHaveBeenCalledWith();
  });

  it('throws a 400-level error and does not record metrics when payload is invalid', () => {
    const req = {
      body: {
        // Missing "type" and invalid difficulty to exercise validation path.
        boardType: 'square8',
        numPlayers: 2,
        difficulty: 0,
        isCalibrationOptIn: true,
      },
    } as Partial<Request> as Request;

    const res = {} as Partial<Response> as Response;

    let thrown: unknown;
    try {
      handleDifficultyCalibrationTelemetry(req, res);
    } catch (err) {
      thrown = err;
    }

    expect(thrown).toBeDefined();
    const anyError = thrown as { statusCode?: number };
    expect(anyError.statusCode).toBe(400);
    expect(mockRecordDifficultyCalibrationEvent).not.toHaveBeenCalled();
  });
});
