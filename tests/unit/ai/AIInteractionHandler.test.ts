import { AIInteractionHandler } from '../../../src/server/game/ai/AIInteractionHandler';
import {
  CaptureDirectionChoice,
  LineOrderChoice,
  LineRewardChoice,
  PlayerChoice,
  PlayerChoiceResponse,
  Position,
  RegionOrderChoice,
  RingEliminationChoice,
} from '../../../src/shared/types/game';
import { globalAIEngine } from '../../../src/server/game/ai/AIEngine';
import { logger } from '../../../src/server/utils/logger';
import { createCancellationSource } from '../../../src/shared/utils/cancellation';

/**
 * Unit tests for AIInteractionHandler
 *
 * These tests validate that the handler:
 * - Implements the PlayerInteractionHandler contract
 * - Returns a valid selectedOption drawn from choice.options
 * - Applies simple, deterministic heuristics for each choice type
 */

jest.mock('../../../src/server/game/ai/AIEngine', () => {
  const mockGlobalAIEngine = {
    getLineRewardChoice: jest.fn(),
    getRingEliminationChoice: jest.fn(),
    getRegionOrderChoice: jest.fn(),
    getLineOrderChoice: jest.fn(),
    getCaptureDirectionChoice: jest.fn(),
    // By default, behave as if the AI is in `service` mode so that
    // service-backed paths remain exercised unless a test overrides
    // this mock.
    getAIConfig: jest.fn(() => ({ difficulty: 5, aiType: 'heuristic', mode: 'service' })),
  };

  return { globalAIEngine: mockGlobalAIEngine };
});

jest.mock('../../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('AIInteractionHandler', () => {
  const handler = new AIInteractionHandler();

  const baseChoice = {
    id: 'choice-1',
    gameId: 'game-1',
    playerNumber: 1,
    prompt: 'Test choice',
  } as const;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns a PlayerChoiceResponse with matching choiceId and playerNumber', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    };

    const response: PlayerChoiceResponse<unknown> = await handler.requestChoice(
      choice as PlayerChoice
    );

    expect(response.choiceId).toBe(choice.id);
    expect(response.playerNumber).toBe(choice.playerNumber);
  });

  it('prefers longer lines for line_order choices', async () => {
    const positionsA: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
    ];
    const positionsB: Position[] = [
      { x: 3, y: 3 },
      { x: 4, y: 4 },
    ];

    const choice: LineOrderChoice = {
      ...baseChoice,
      type: 'line_order',
      options: [
        { moveId: 'm-short', lineId: 'short', markerPositions: positionsB },
        { moveId: 'm-long', lineId: 'long', markerPositions: positionsA },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineOrderChoice['options'][number];

    expect(['short', 'long']).toContain(selected.lineId);
    expect(selected.markerPositions.length).toBe(3);
    expect(selected.lineId).toBe('long');
  });

  it('prefers Option 2 for line_reward_option when available (falling back from service)', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineRewardChoice['options'][number];

    expect(selected).toBe('option_2_min_collapse_no_elimination');
  });

  it('falls back to first option when option 2 is not available for line_reward_option', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: ['option_1_collapse_all_and_eliminate'],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineRewardChoice['options'][number];

    expect(selected).toBe('option_1_collapse_all_and_eliminate');
  });

  it('chooses stack with smallest capHeight for ring_elimination choices', async () => {
    const choice: RingEliminationChoice = {
      ...baseChoice,
      type: 'ring_elimination',
      options: [
        {
          moveId: 'm-a',
          stackPosition: { x: 0, y: 0 },
          capHeight: 3,
          totalHeight: 5,
        },
        {
          moveId: 'm-b',
          stackPosition: { x: 1, y: 1 },
          capHeight: 1,
          totalHeight: 4,
        },
        {
          moveId: 'm-c',
          stackPosition: { x: 2, y: 2 },
          capHeight: 1,
          totalHeight: 6,
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as RingEliminationChoice['options'][number];

    expect(selected.capHeight).toBe(1);
    // Among capHeight === 1, the option with smaller totalHeight (4) should win
    expect(selected.totalHeight).toBe(4);
    expect(selected.stackPosition).toEqual({ x: 1, y: 1 });
  });

  it('exercises totalHeight tie-breaking when later option has smaller totalHeight with same capHeight', async () => {
    const choice: RingEliminationChoice = {
      ...baseChoice,
      type: 'ring_elimination',
      options: [
        {
          moveId: 'm-a',
          stackPosition: { x: 0, y: 0 },
          capHeight: 2,
          totalHeight: 10, // First option becomes best
        },
        {
          moveId: 'm-b',
          stackPosition: { x: 1, y: 1 },
          capHeight: 2, // Same capHeight
          totalHeight: 5, // Smaller totalHeight - should become new best (line 319)
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as RingEliminationChoice['options'][number];

    expect(selected.capHeight).toBe(2);
    expect(selected.totalHeight).toBe(5);
    expect(selected.stackPosition).toEqual({ x: 1, y: 1 });
  });

  it('threads a canceled session token into globalAIEngine.getRingEliminationChoice for ring_elimination', async () => {
    const choice: RingEliminationChoice = {
      ...baseChoice,
      type: 'ring_elimination',
      options: [
        {
          moveId: 'm-a',
          stackPosition: { x: 0, y: 0 },
          capHeight: 3,
          totalHeight: 5,
        },
        {
          moveId: 'm-b',
          stackPosition: { x: 1, y: 1 },
          capHeight: 1,
          totalHeight: 4,
        },
      ],
    };

    const parentSource = createCancellationSource();
    parentSource.cancel('session_cleanup');

    const handlerWithToken = new AIInteractionHandler(parentSource.token);

    const mockEngine = globalAIEngine as unknown as {
      getRingEliminationChoice: jest.Mock;
    };

    mockEngine.getRingEliminationChoice.mockResolvedValue(choice.options[1]);

    const response = await handlerWithToken.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as RingEliminationChoice['options'][number];

    expect(mockEngine.getRingEliminationChoice).toHaveBeenCalledTimes(1);
    const callArgs = mockEngine.getRingEliminationChoice.mock.calls[0];
    const requestOptions = callArgs[3];

    expect(requestOptions).toBeDefined();
    expect(requestOptions.token).toBeDefined();
    expect(requestOptions.token.isCanceled).toBe(true);

    expect(selected).toEqual(choice.options[1]);
  });

  it('threads a canceled session token into globalAIEngine.getLineOrderChoice for line_order', async () => {
    const positionsA: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
    ];
    const positionsB: Position[] = [
      { x: 3, y: 3 },
      { x: 4, y: 4 },
    ];

    const choice: LineOrderChoice = {
      ...baseChoice,
      type: 'line_order',
      options: [
        { moveId: 'm-short', lineId: 'short', markerPositions: positionsB },
        { moveId: 'm-long', lineId: 'long', markerPositions: positionsA },
      ],
    };

    const parentSource = createCancellationSource();
    parentSource.cancel('session_cleanup');

    const handlerWithToken = new AIInteractionHandler(parentSource.token);

    const mockEngine = globalAIEngine as unknown as {
      getLineOrderChoice: jest.Mock;
    };

    mockEngine.getLineOrderChoice.mockResolvedValue(choice.options[0]);

    const response = await handlerWithToken.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineOrderChoice['options'][number];

    expect(mockEngine.getLineOrderChoice).toHaveBeenCalledTimes(1);
    const callArgs = mockEngine.getLineOrderChoice.mock.calls[0];
    const requestOptions = callArgs[3];

    expect(requestOptions).toBeDefined();
    expect(requestOptions.token).toBeDefined();
    expect(requestOptions.token.isCanceled).toBe(true);

    expect(selected.lineId).toBe('short');
  });

  it('threads a canceled session token into globalAIEngine.getCaptureDirectionChoice for capture_direction', async () => {
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
        {
          targetPosition: { x: 3, y: 3 },
          landingPosition: { x: 5, y: 5 },
          capturedCapHeight: 2,
        },
        {
          targetPosition: { x: 4, y: 4 },
          landingPosition: { x: 6, y: 6 },
          capturedCapHeight: 3,
        },
      ],
    };

    const parentSource = createCancellationSource();
    parentSource.cancel('session_cleanup');

    const handlerWithToken = new AIInteractionHandler(parentSource.token);

    const mockEngine = globalAIEngine as unknown as {
      getCaptureDirectionChoice: jest.Mock;
    };

    mockEngine.getCaptureDirectionChoice.mockResolvedValue(choice.options[0]);

    const response = await handlerWithToken.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(mockEngine.getCaptureDirectionChoice).toHaveBeenCalledTimes(1);
    const callArgs = mockEngine.getCaptureDirectionChoice.mock.calls[0];
    const requestOptions = callArgs[3];

    expect(requestOptions).toBeDefined();
    expect(requestOptions.token).toBeDefined();
    expect(requestOptions.token.isCanceled).toBe(true);

    expect(selected).toBe(choice.options[0]);
  });

  it('chooses largest region for region_order choices', async () => {
    const choice: RegionOrderChoice = {
      ...baseChoice,
      type: 'region_order',
      options: [
        {
          moveId: 'm-small',
          regionId: 'small',
          size: 3,
          representativePosition: { x: 0, y: 0 },
        },
        {
          moveId: 'm-large',
          regionId: 'large',
          size: 7,
          representativePosition: { x: 5, y: 5 },
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as RegionOrderChoice['options'][number];

    expect(selected.regionId).toBe('large');
    expect(selected.size).toBe(7);
  });

  it('still prefers the largest concrete region when a skip option is present', async () => {
    const choice: RegionOrderChoice = {
      ...baseChoice,
      type: 'region_order',
      options: [
        {
          moveId: 'm-region',
          regionId: 'region-1',
          size: 4,
          representativePosition: { x: 1, y: 1 },
        },
        {
          moveId: 'm-skip',
          regionId: 'skip',
          size: 0,
          representativePosition: { x: 0, y: 0 },
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as RegionOrderChoice['options'][number];

    expect(selected.regionId).toBe('region-1');
    expect(selected.size).toBe(4);
  });

  it('falls back to a skip-like option when no concrete regions are available', async () => {
    const choice: RegionOrderChoice = {
      ...baseChoice,
      type: 'region_order',
      options: [
        {
          moveId: 'm-skip-1',
          regionId: 'skip',
          size: 0,
          representativePosition: { x: 0, y: 0 },
        },
        {
          moveId: 'm-skip-2',
          regionId: 'meta',
          size: 0,
          representativePosition: { x: 1, y: 1 },
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as RegionOrderChoice['options'][number];

    expect(selected.moveId).toBe('m-skip-1');
    expect(['skip', 'meta']).toContain(selected.regionId);
    expect(selected.size).toBe(0);
  });

  it('threads a canceled session token into globalAIEngine.getRegionOrderChoice for region_order', async () => {
    const choice: RegionOrderChoice = {
      ...baseChoice,
      type: 'region_order',
      options: [
        {
          moveId: 'm-small',
          regionId: 'small',
          size: 3,
          representativePosition: { x: 0, y: 0 },
        },
        {
          moveId: 'm-large',
          regionId: 'large',
          size: 7,
          representativePosition: { x: 5, y: 5 },
        },
      ],
    };

    const parentSource = createCancellationSource();
    // Cancel before the AI-backed choice is requested to simulate a
    // terminated session.
    parentSource.cancel('session_cleanup');

    const handlerWithToken = new AIInteractionHandler(parentSource.token);

    const mockEngine = globalAIEngine as unknown as {
      getRegionOrderChoice: jest.Mock;
    };

    mockEngine.getRegionOrderChoice.mockResolvedValue(choice.options[1]);

    const response = await handlerWithToken.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as RegionOrderChoice['options'][number];

    expect(mockEngine.getRegionOrderChoice).toHaveBeenCalledTimes(1);
    const callArgs = mockEngine.getRegionOrderChoice.mock.calls[0];
    const requestOptions = callArgs[3];

    expect(requestOptions).toBeDefined();
    expect(requestOptions.token).toBeDefined();
    expect(requestOptions.token.isCanceled).toBe(true);

    // Despite the canceled token, a valid option returned by the service
    // should still be honoured by the handler.
    expect(selected.regionId).toBe('large');
  });

  it('prefers higher capturedCapHeight for capture_direction choices, tie-breaking by distance to centre', async () => {
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
        {
          targetPosition: { x: 3, y: 3 },
          landingPosition: { x: 5, y: 5 },
          capturedCapHeight: 2,
        },
        {
          targetPosition: { x: 4, y: 4 },
          landingPosition: { x: 6, y: 6 },
          capturedCapHeight: 3,
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(selected.capturedCapHeight).toBe(3);
    expect(selected.targetPosition).toEqual({ x: 4, y: 4 });
  });

  it('returns the only option for capture_direction with single option', async () => {
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
        {
          targetPosition: { x: 3, y: 3 },
          landingPosition: { x: 5, y: 5 },
          capturedCapHeight: 2,
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(selected.capturedCapHeight).toBe(2);
  });

  it('tie-breaks capture_direction by distance when capturedCapHeight is equal', async () => {
    // The centre is estimated as the first option's landing position (5,5),
    // so the second option with a landing closer to (5,5) will be preferred
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
        {
          targetPosition: { x: 3, y: 3 },
          landingPosition: { x: 5, y: 5 }, // This becomes the "centre" - distance 0
          capturedCapHeight: 3,
        },
        {
          targetPosition: { x: 4, y: 4 },
          landingPosition: { x: 4, y: 4 }, // Distance from (5,5) = 2
          capturedCapHeight: 3,
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(selected.capturedCapHeight).toBe(3);
    // First option has distance 0 from centre, second has distance 2, so first wins
    expect(selected.targetPosition).toEqual({ x: 3, y: 3 });
  });

  it('exercises distance tie-breaking when third option is closer than second with same capHeight', async () => {
    // The centre is estimated as the first option's landing position (5,5).
    // Option 2 becomes best (higher capHeight), then option 3 has same capHeight
    // but closer landing position, so it should become the new best.
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
        {
          targetPosition: { x: 1, y: 1 },
          landingPosition: { x: 5, y: 5 }, // Centre is (5,5)
          capturedCapHeight: 2,
        },
        {
          targetPosition: { x: 2, y: 2 },
          landingPosition: { x: 10, y: 10 }, // Distance from (5,5) = 10
          capturedCapHeight: 3,
        },
        {
          targetPosition: { x: 3, y: 3 },
          landingPosition: { x: 6, y: 5 }, // Distance from (5,5) = 1
          capturedCapHeight: 3,
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(selected.capturedCapHeight).toBe(3);
    // Third option should win because it has same capHeight as second but closer to centre
    expect(selected.targetPosition).toEqual({ x: 3, y: 3 });
  });

  it('handles positions with z coordinate for capture_direction', async () => {
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
        {
          targetPosition: { x: 3, y: 3, z: 1 } as Position,
          landingPosition: { x: 5, y: 5, z: 1 } as Position,
          capturedCapHeight: 2,
        },
        {
          targetPosition: { x: 4, y: 4, z: 0 } as Position,
          landingPosition: { x: 6, y: 6, z: 0 } as Position,
          capturedCapHeight: 3,
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(selected.capturedCapHeight).toBe(3);
  });

  it('uses AI service line_reward_option when it returns a valid option', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    };

    const mockEngine = globalAIEngine as unknown as {
      getLineRewardChoice: jest.Mock;
    };

    mockEngine.getLineRewardChoice.mockResolvedValue('option_1_collapse_all_and_eliminate');

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineRewardChoice['options'][number];

    expect(mockEngine.getLineRewardChoice).toHaveBeenCalledWith(
      choice.playerNumber,
      null,
      choice.options
    );
    // Service choice should override the local "prefer option 2" heuristic
    expect(selected).toBe('option_1_collapse_all_and_eliminate');
  });

  it('falls back to local heuristic and logs when AI service returns invalid line_reward_option', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    };

    const mockEngine = globalAIEngine as unknown as {
      getLineRewardChoice: jest.Mock;
    };

    mockEngine.getLineRewardChoice.mockResolvedValue('not_a_valid_option' as any);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineRewardChoice['options'][number];

    // Fallback heuristic still prefers option 2
    expect(selected).toBe('option_2_min_collapse_no_elimination');

    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('AI service returned invalid option for line_reward_option'),
      expect.objectContaining({
        gameId: choice.gameId,
        playerNumber: choice.playerNumber,
        choiceId: choice.id,
        choiceType: choice.type,
      })
    );
  });

  it('uses AI service line_order when it returns a valid option', async () => {
    const positionsA: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
    ];
    const positionsB: Position[] = [
      { x: 3, y: 3 },
      { x: 4, y: 4 },
    ];

    const choice: LineOrderChoice = {
      ...baseChoice,
      type: 'line_order',
      options: [
        { moveId: 'm-short', lineId: 'short', markerPositions: positionsB },
        { moveId: 'm-long', lineId: 'long', markerPositions: positionsA },
      ],
    };

    const mockEngine = globalAIEngine as unknown as {
      getLineOrderChoice: jest.Mock;
    };

    mockEngine.getLineOrderChoice.mockResolvedValue(choice.options[0]);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineOrderChoice['options'][number];

    expect(mockEngine.getLineOrderChoice).toHaveBeenCalledWith(
      choice.playerNumber,
      null,
      choice.options,
      undefined
    );
    expect(selected.lineId).toBe('short');
  });

  it('falls back to local line_order heuristic and logs when AI service returns invalid option', async () => {
    const positionsA: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
    ];
    const positionsB: Position[] = [
      { x: 3, y: 3 },
      { x: 4, y: 4 },
    ];

    const choice: LineOrderChoice = {
      ...baseChoice,
      type: 'line_order',
      options: [
        { moveId: 'm-short', lineId: 'short', markerPositions: positionsB },
        { moveId: 'm-long', lineId: 'long', markerPositions: positionsA },
      ],
    };

    const mockEngine = globalAIEngine as unknown as {
      getLineOrderChoice: jest.Mock;
    };

    mockEngine.getLineOrderChoice.mockResolvedValue({} as any);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineOrderChoice['options'][number];

    // Fallback heuristic should still pick the longer line ("long").
    expect(selected.lineId).toBe('long');

    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('AI service returned invalid option for line_order'),
      expect.objectContaining({
        gameId: choice.gameId,
        playerNumber: choice.playerNumber,
        choiceId: choice.id,
        choiceType: choice.type,
      })
    );
  });

  it('uses AI service capture_direction when it returns a valid option', async () => {
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
        {
          targetPosition: { x: 3, y: 3 },
          landingPosition: { x: 5, y: 5 },
          capturedCapHeight: 2,
        },
        {
          targetPosition: { x: 4, y: 4 },
          landingPosition: { x: 6, y: 6 },
          capturedCapHeight: 3,
        },
      ],
    };

    const mockEngine = globalAIEngine as unknown as {
      getCaptureDirectionChoice: jest.Mock;
    };

    mockEngine.getCaptureDirectionChoice.mockResolvedValue(choice.options[0]);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(mockEngine.getCaptureDirectionChoice).toHaveBeenCalledWith(
      choice.playerNumber,
      null,
      choice.options,
      undefined
    );
    expect(selected).toBe(choice.options[0]);
  });

  it('falls back to local capture_direction heuristic and logs when AI service returns invalid option', async () => {
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
        {
          targetPosition: { x: 3, y: 3 },
          landingPosition: { x: 5, y: 5 },
          capturedCapHeight: 2,
        },
        {
          targetPosition: { x: 4, y: 4 },
          landingPosition: { x: 6, y: 6 },
          capturedCapHeight: 3,
        },
      ],
    };

    const mockEngine = globalAIEngine as unknown as {
      getCaptureDirectionChoice: jest.Mock;
    };

    mockEngine.getCaptureDirectionChoice.mockResolvedValue({} as any);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    // Fallback heuristic should still prefer the higher capturedCapHeight (3).
    expect(selected.capturedCapHeight).toBe(3);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('AI service returned invalid option for capture_direction'),
      expect.objectContaining({
        gameId: choice.gameId,
        playerNumber: choice.playerNumber,
        choiceId: choice.id,
        choiceType: choice.type,
      })
    );
  });

  it('logs and throws when receiving an unknown choice type', async () => {
    const choice = {
      ...baseChoice,
      type: 'unknown_choice_type',
      options: [] as string[],
    } as unknown as PlayerChoice;

    await expect(handler.requestChoice(choice)).rejects.toThrow(
      'Unhandled PlayerChoice type: unknown_choice_type'
    );

    expect(logger.error).toHaveBeenCalledWith(
      'AIInteractionHandler received unknown choice type',
      expect.objectContaining({
        choiceId: baseChoice.id,
        playerNumber: baseChoice.playerNumber,
      })
    );
  });

  describe('AI service error fallback', () => {
    it('falls back to local heuristic when AI service throws for line_order', async () => {
      const positionsA: Position[] = [
        { x: 0, y: 0 },
        { x: 1, y: 1 },
        { x: 2, y: 2 },
      ];
      const positionsB: Position[] = [
        { x: 3, y: 3 },
        { x: 4, y: 4 },
      ];

      const choice: LineOrderChoice = {
        ...baseChoice,
        type: 'line_order',
        options: [
          { moveId: 'm-short', lineId: 'short', markerPositions: positionsB },
          { moveId: 'm-long', lineId: 'long', markerPositions: positionsA },
        ],
      };

      const mockEngine = globalAIEngine as unknown as {
        getLineOrderChoice: jest.Mock;
      };

      mockEngine.getLineOrderChoice.mockRejectedValue(new Error('Service unavailable'));

      const response = await handler.requestChoice(choice as PlayerChoice);
      const selected = response.selectedOption as LineOrderChoice['options'][number];

      // Should fall back to local heuristic (prefer longer line)
      expect(selected.lineId).toBe('long');
      expect(logger.warn).toHaveBeenCalledWith(
        'AI service unavailable for line_order; falling back to local heuristic',
        expect.objectContaining({
          gameId: choice.gameId,
          playerNumber: choice.playerNumber,
          choiceId: choice.id,
        })
      );
    });

    it('falls back to local heuristic when AI service throws for line_reward_option', async () => {
      const choice: LineRewardChoice = {
        ...baseChoice,
        type: 'line_reward_option',
        options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
      };

      const mockEngine = globalAIEngine as unknown as {
        getLineRewardChoice: jest.Mock;
      };

      mockEngine.getLineRewardChoice.mockRejectedValue(new Error('Service unavailable'));

      const response = await handler.requestChoice(choice as PlayerChoice);
      const selected = response.selectedOption as LineRewardChoice['options'][number];

      // Should fall back to local heuristic (prefer option 2)
      expect(selected).toBe('option_2_min_collapse_no_elimination');
      expect(logger.warn).toHaveBeenCalledWith(
        'AI service unavailable for line_reward_option; falling back to local heuristic',
        expect.objectContaining({
          gameId: choice.gameId,
          playerNumber: choice.playerNumber,
          choiceId: choice.id,
        })
      );
    });

    it('falls back to local heuristic when AI service throws for ring_elimination', async () => {
      const choice: RingEliminationChoice = {
        ...baseChoice,
        type: 'ring_elimination',
        options: [
          {
            moveId: 'm-a',
            stackPosition: { x: 0, y: 0 },
            capHeight: 3,
            totalHeight: 5,
          },
          {
            moveId: 'm-b',
            stackPosition: { x: 1, y: 1 },
            capHeight: 1,
            totalHeight: 4,
          },
        ],
      };

      const mockEngine = globalAIEngine as unknown as {
        getRingEliminationChoice: jest.Mock;
      };

      mockEngine.getRingEliminationChoice.mockRejectedValue(new Error('Service unavailable'));

      const response = await handler.requestChoice(choice as PlayerChoice);
      const selected = response.selectedOption as RingEliminationChoice['options'][number];

      // Should fall back to local heuristic (prefer smallest capHeight)
      expect(selected.capHeight).toBe(1);
      expect(logger.warn).toHaveBeenCalledWith(
        'AI service unavailable for ring_elimination; falling back to local heuristic',
        expect.objectContaining({
          gameId: choice.gameId,
          playerNumber: choice.playerNumber,
          choiceId: choice.id,
        })
      );
    });

    it('falls back to local heuristic when AI service throws for region_order', async () => {
      const choice: RegionOrderChoice = {
        ...baseChoice,
        type: 'region_order',
        options: [
          {
            moveId: 'm-small',
            regionId: 'small',
            size: 3,
            representativePosition: { x: 0, y: 0 },
          },
          {
            moveId: 'm-large',
            regionId: 'large',
            size: 7,
            representativePosition: { x: 5, y: 5 },
          },
        ],
      };

      const mockEngine = globalAIEngine as unknown as {
        getRegionOrderChoice: jest.Mock;
      };

      mockEngine.getRegionOrderChoice.mockRejectedValue(new Error('Service unavailable'));

      const response = await handler.requestChoice(choice as PlayerChoice);
      const selected = response.selectedOption as RegionOrderChoice['options'][number];

      // Should fall back to local heuristic (prefer largest region)
      expect(selected.regionId).toBe('large');
      expect(logger.warn).toHaveBeenCalledWith(
        'AI service unavailable for region_order; falling back to local heuristic',
        expect.objectContaining({
          gameId: choice.gameId,
          playerNumber: choice.playerNumber,
          choiceId: choice.id,
        })
      );
    });

    it('falls back to local heuristic when AI service throws for capture_direction', async () => {
      const choice: CaptureDirectionChoice = {
        ...baseChoice,
        type: 'capture_direction',
        options: [
          {
            targetPosition: { x: 3, y: 3 },
            landingPosition: { x: 5, y: 5 },
            capturedCapHeight: 2,
          },
          {
            targetPosition: { x: 4, y: 4 },
            landingPosition: { x: 6, y: 6 },
            capturedCapHeight: 3,
          },
        ],
      };

      const mockEngine = globalAIEngine as unknown as {
        getCaptureDirectionChoice: jest.Mock;
      };

      mockEngine.getCaptureDirectionChoice.mockRejectedValue(new Error('Service unavailable'));

      const response = await handler.requestChoice(choice as PlayerChoice);
      const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

      // Should fall back to local heuristic (prefer higher capHeight)
      expect(selected.capturedCapHeight).toBe(3);
      expect(logger.warn).toHaveBeenCalledWith(
        'AI service unavailable for capture_direction; falling back to local heuristic',
        expect.objectContaining({
          gameId: choice.gameId,
          playerNumber: choice.playerNumber,
          choiceId: choice.id,
        })
      );
    });
  });

  describe('empty options error handling', () => {
    it('throws and logs error for line_order with empty options', async () => {
      const choice: LineOrderChoice = {
        ...baseChoice,
        type: 'line_order',
        options: [],
      };

      await expect(handler.requestChoice(choice as PlayerChoice)).rejects.toThrow(
        'PlayerChoice[line_order] must have at least one option'
      );

      expect(logger.error).toHaveBeenCalledWith(
        'AIInteractionHandler received line_order choice with no options',
        expect.objectContaining({
          choiceId: choice.id,
          choiceType: choice.type,
          playerNumber: choice.playerNumber,
        })
      );
    });

    it('throws and logs error for line_reward_option with empty options', async () => {
      const choice: LineRewardChoice = {
        ...baseChoice,
        type: 'line_reward_option',
        options: [],
      };

      await expect(handler.requestChoice(choice as PlayerChoice)).rejects.toThrow(
        'PlayerChoice[line_reward_option] must have at least one option'
      );

      expect(logger.error).toHaveBeenCalledWith(
        'AIInteractionHandler received line_reward_option choice with no options',
        expect.objectContaining({
          choiceId: choice.id,
          choiceType: choice.type,
          playerNumber: choice.playerNumber,
        })
      );
    });

    it('throws and logs error for ring_elimination with empty options', async () => {
      const choice: RingEliminationChoice = {
        ...baseChoice,
        type: 'ring_elimination',
        options: [],
      };

      await expect(handler.requestChoice(choice as PlayerChoice)).rejects.toThrow(
        'PlayerChoice[ring_elimination] must have at least one option'
      );

      expect(logger.error).toHaveBeenCalledWith(
        'AIInteractionHandler received ring_elimination choice with no options',
        expect.objectContaining({
          choiceId: choice.id,
          choiceType: choice.type,
          playerNumber: choice.playerNumber,
        })
      );
    });

    it('throws and logs error for region_order with empty options', async () => {
      const choice: RegionOrderChoice = {
        ...baseChoice,
        type: 'region_order',
        options: [],
      };

      await expect(handler.requestChoice(choice as PlayerChoice)).rejects.toThrow(
        'PlayerChoice[region_order] must have at least one option'
      );

      expect(logger.error).toHaveBeenCalledWith(
        'AIInteractionHandler received region_order choice with no options',
        expect.objectContaining({
          choiceId: choice.id,
          choiceType: choice.type,
          playerNumber: choice.playerNumber,
        })
      );
    });

    it('throws and logs error for capture_direction with empty options', async () => {
      const choice: CaptureDirectionChoice = {
        ...baseChoice,
        type: 'capture_direction',
        options: [],
      };

      await expect(handler.requestChoice(choice as PlayerChoice)).rejects.toThrow(
        'PlayerChoice[capture_direction] must have at least one option'
      );

      expect(logger.error).toHaveBeenCalledWith(
        'AIInteractionHandler received capture_direction choice with no options',
        expect.objectContaining({
          choiceId: choice.id,
          choiceType: choice.type,
          playerNumber: choice.playerNumber,
        })
      );
    });
  });
});
