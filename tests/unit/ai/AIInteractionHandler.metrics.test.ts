import { AIInteractionHandler } from '../../../src/server/game/ai/AIInteractionHandler';
import {
  LineOrderChoice,
  RingEliminationChoice,
  PlayerChoiceResponse,
} from '../../../src/shared/types/game';

// Mock MetricsService to observe choice metrics calls.
const mockRecordAIChoiceRequest = jest.fn();
const mockRecordAIChoiceLatencyMs = jest.fn();

jest.mock('../../../src/server/services/MetricsService', () => ({
  getMetricsService: () => ({
    recordAIChoiceRequest: mockRecordAIChoiceRequest,
    recordAIChoiceLatencyMs: mockRecordAIChoiceLatencyMs,
  }),
}));

// Mock globalAIEngine to control service-backed choice paths.
const mockGetAIConfig = jest.fn();
const mockGetLineOrderChoice = jest.fn();
const mockGetRingEliminationChoice = jest.fn();

jest.mock('../../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    getAIConfig: (...args: any[]) => mockGetAIConfig(...args),
    getLineOrderChoice: (...args: any[]) => mockGetLineOrderChoice(...args),
    getRingEliminationChoice: (...args: any[]) => mockGetRingEliminationChoice(...args),
  },
}));

describe('AIInteractionHandler choice metrics', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('records success metrics when service returns a valid line_order choice', async () => {
    const handler = new AIInteractionHandler();
    const choice: LineOrderChoice = {
      id: 'choice-line-order',
      gameId: 'game-1',
      playerNumber: 1,
      type: 'line_order',
      prompt: 'Choose line',
      options: [
        { lineId: 'short', markerPositions: [{ x: 0, y: 0 }], moveId: 'm-short' },
        {
          lineId: 'long',
          markerPositions: [
            { x: 0, y: 0 },
            { x: 1, y: 1 },
            { x: 2, y: 2 },
          ],
          moveId: 'm-long',
        },
      ],
    };

    mockGetAIConfig.mockReturnValue({ mode: 'service', difficulty: 5 });
    mockGetLineOrderChoice.mockResolvedValue(choice.options[1]);

    const response = (await handler.requestChoice(choice)) as PlayerChoiceResponse<
      LineOrderChoice['options'][number]
    >;

    expect(response.selectedOption).toEqual(choice.options[1]);
    expect(mockRecordAIChoiceRequest).toHaveBeenCalledWith('line_order', 'success');
    expect(mockRecordAIChoiceLatencyMs).toHaveBeenCalledWith(
      'line_order',
      expect.any(Number),
      'success'
    );
  });

  it('records timeout metrics and falls back to heuristic when service times out for line_order', async () => {
    const handler = new AIInteractionHandler();
    const choice: LineOrderChoice = {
      id: 'choice-line-order-timeout',
      gameId: 'game-2',
      playerNumber: 1,
      type: 'line_order',
      prompt: 'Choose line',
      options: [
        { lineId: 'short', markerPositions: [{ x: 0, y: 0 }], moveId: 'm-short' },
        {
          lineId: 'long',
          markerPositions: [
            { x: 0, y: 0 },
            { x: 1, y: 1 },
            { x: 2, y: 2 },
            { x: 3, y: 3 },
          ],
          moveId: 'm-long',
        },
      ],
    };

    mockGetAIConfig.mockReturnValue({ mode: 'service', difficulty: 5 });
    const timeoutError: any = new Error('timeout');
    timeoutError.aiErrorType = 'timeout';
    mockGetLineOrderChoice.mockRejectedValue(timeoutError);

    const response = (await handler.requestChoice(choice)) as PlayerChoiceResponse<
      LineOrderChoice['options'][number]
    >;

    // Heuristic should pick the longest line option.
    expect(response.selectedOption).toEqual(choice.options[1]);
    expect(mockRecordAIChoiceRequest).toHaveBeenCalledWith('line_order', 'timeout');
    expect(mockRecordAIChoiceLatencyMs).toHaveBeenCalledWith(
      'line_order',
      expect.any(Number),
      'timeout'
    );
  });

  it('records fallback metrics when ring_elimination service rejects and uses local heuristic', async () => {
    const handler = new AIInteractionHandler();
    const choice: RingEliminationChoice = {
      id: 'choice-ring-elim',
      gameId: 'game-3',
      playerNumber: 1,
      type: 'ring_elimination',
      prompt: 'Eliminate ring',
      options: [
        { stackPosition: { x: 0, y: 0 }, capHeight: 1, totalHeight: 3, moveId: 'm-a' },
        { stackPosition: { x: 1, y: 1 }, capHeight: 2, totalHeight: 4, moveId: 'm-b' },
      ],
    };

    mockGetAIConfig.mockReturnValue({ mode: 'service', difficulty: 5 });
    mockGetRingEliminationChoice.mockRejectedValue(new Error('service unavailable'));

    const response = (await handler.requestChoice(choice)) as PlayerChoiceResponse<
      RingEliminationChoice['options'][number]
    >;

    // Heuristic prefers smallest capHeight.
    expect(response.selectedOption).toEqual(choice.options[0]);
    expect(mockRecordAIChoiceRequest).toHaveBeenCalledWith('ring_elimination', 'fallback');
    expect(mockRecordAIChoiceLatencyMs).toHaveBeenCalledWith(
      'ring_elimination',
      expect.any(Number),
      'fallback'
    );
  });
});
