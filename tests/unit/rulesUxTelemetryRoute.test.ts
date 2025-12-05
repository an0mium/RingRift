import type { Request, Response } from 'express';
import { handleRulesUxTelemetry } from '../../src/server/routes/rulesUxTelemetry';
import { getMetricsService } from '../../src/server/services/MetricsService';

// Jest hoists jest.mock calls, so we can define the mock function in outer scope
// and have it injected into the mocked MetricsService module. The `mock*` prefix
// keeps Jest's ESM transform happy for out-of-scope variables.
const mockRecordRulesUxEvent = jest.fn();

jest.mock('../../src/server/services/MetricsService', () => ({
  __esModule: true,
  getMetricsService: jest.fn(() => ({
    recordRulesUxEvent: mockRecordRulesUxEvent,
  })),
}));

describe('rulesUxTelemetry route handler', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRecordRulesUxEvent.mockReset();
  });

  it('forwards a valid RulesUxEventPayload to MetricsService and returns 204', () => {
    const req = {
      body: {
        type: 'rules_help_open',
        boardType: 'square8',
        numPlayers: 2,
        topic: 'active_no_moves',
      },
    } as Partial<Request> as Request;

    const status = jest.fn().mockReturnThis();
    const send = jest.fn();
    const res = { status, send } as Partial<Response> as Response;

    handleRulesUxTelemetry(req, res);

    expect(getMetricsService).toHaveBeenCalledTimes(1);
    expect(mockRecordRulesUxEvent).toHaveBeenCalledTimes(1);
    expect(mockRecordRulesUxEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'rules_help_open',
        boardType: 'square8',
        numPlayers: 2,
        topic: 'active_no_moves',
      })
    );

    expect(status).toHaveBeenCalledWith(204);
    expect(send).toHaveBeenCalledWith();
  });

  it('throws a 400-level error and does not record metrics when payload is invalid', () => {
    const req = {
      body: {
        // Missing "type" and invalid numPlayers to exercise validation path.
        boardType: 'square8',
        numPlayers: 0,
      },
    } as Partial<Request> as Request;

    const res = {} as Partial<Response> as Response;

    let thrown: unknown;
    try {
      handleRulesUxTelemetry(req, res);
    } catch (err) {
      thrown = err;
    }

    expect(thrown).toBeDefined();
    const anyError = thrown as { statusCode?: number };
    // The route uses createError with an explicit 400 status code.
    expect(anyError.statusCode).toBe(400);
    expect(mockRecordRulesUxEvent).not.toHaveBeenCalled();
  });
});
