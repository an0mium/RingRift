import {
  AIEngine,
  AI_DIFFICULTY_PRESETS,
  AIType as InternalAIType,
} from '../../src/server/game/ai/AIEngine';
import {
  getAIServiceClient,
  AIType as ServiceAIType,
  MoveResponse,
  RingEliminationChoiceResponsePayload,
  RegionOrderChoiceResponsePayload,
} from '../../src/server/services/AIServiceClient';
import {
  GameState,
  Move,
  AIProfile,
  RingEliminationChoice,
  RegionOrderChoice,
} from '../../src/shared/types/game';

jest.mock('../../src/server/services/AIServiceClient');

describe('AIEngine service integration (profile-driven)', () => {
  it('getAIMove calls AIServiceClient.getAIMove with mapped ai_type and forwards the returned move', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeMove: Move = {
      id: 'svc-move-1',
      type: 'move_ring',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const fakeResponse: MoveResponse = {
      move: fakeMove,
      evaluation: 0.42,
      thinking_time_ms: 1234,
      ai_type: 'minimax',
      difficulty: 7,
    };

    const fakeClient = {
      getAIMove: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 7,
      mode: 'service',
      aiType: 'minimax',
    };

    engine.createAIFromProfile(1, profile);

    const gameState = {
      currentPhase: 'movement',
    } as unknown as GameState;

    const move = await engine.getAIMove(1, gameState);

    expect(fakeClient.getAIMove).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getAIMove.mock.calls[0];
    expect(callArgs[0]).toBe(gameState);
    expect(callArgs[1]).toBe(1);
    expect(callArgs[2]).toBe(7);
    expect(callArgs[3]).toBe(ServiceAIType.MINIMAX);

    expect(move).toBe(fakeMove);
  });

  it('getRingEliminationChoice calls AIServiceClient.getRingEliminationChoice and returns the selected option', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const options: RingEliminationChoice['options'] = [
      {
        stackPosition: { x: 0, y: 0 },
        capHeight: 3,
        totalHeight: 5,
        moveId: 'opt-1',
      },
      {
        stackPosition: { x: 1, y: 1 },
        capHeight: 1,
        totalHeight: 4,
        moveId: 'opt-2',
      },
    ];

    const fakeResponse: RingEliminationChoiceResponsePayload = {
      selectedOption: options[1],
      aiType: 'heuristic',
      difficulty: 5,
    };

    const fakeClient = {
      getRingEliminationChoice: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 5,
      mode: 'service',
      aiType: 'heuristic',
    };

    engine.createAIFromProfile(2, profile);

    const selected = await engine.getRingEliminationChoice(2, null, options);

    expect(fakeClient.getRingEliminationChoice).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getRingEliminationChoice.mock.calls[0];
    expect(callArgs[0]).toBeNull();
    expect(callArgs[1]).toBe(2);
    expect(callArgs[2]).toBe(5);
    expect(callArgs[3]).toBe(ServiceAIType.HEURISTIC);
    expect(callArgs[4]).toBe(options);

    expect(selected).toBe(options[1]);
  });

  it('getRegionOrderChoice calls AIServiceClient.getRegionOrderChoice and returns the selected option', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const options: RegionOrderChoice['options'] = [
      {
        regionId: 'small',
        size: 3,
        representativePosition: { x: 0, y: 0 },
        moveId: 'reg-1',
      },
      {
        regionId: 'large',
        size: 7,
        representativePosition: { x: 5, y: 5 },
        moveId: 'reg-2',
      },
    ];

    const fakeResponse: RegionOrderChoiceResponsePayload = {
      selectedOption: options[1],
      aiType: 'heuristic',
      difficulty: 6,
    };

    const fakeClient = {
      getRegionOrderChoice: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 6,
      mode: 'service',
      aiType: 'heuristic',
    };

    engine.createAIFromProfile(3, profile);

    const selected = await engine.getRegionOrderChoice(3, null, options);

    expect(fakeClient.getRegionOrderChoice).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getRegionOrderChoice.mock.calls[0];
    expect(callArgs[0]).toBeNull();
    expect(callArgs[1]).toBe(3);
    expect(callArgs[2]).toBe(6);
    expect(callArgs[3]).toBe(ServiceAIType.HEURISTIC);
    expect(callArgs[4]).toBe(options);

    expect(selected).toBe(options[1]);
  });

  it('AI_DIFFICULTY_PRESETS matches canonical mapping for difficulties 1–10', () => {
    const expected: Record<
      number,
      { aiType: InternalAIType; randomness: number; thinkTime: number }
    > = {
      1: { aiType: InternalAIType.RANDOM, randomness: 0.5, thinkTime: 150 },
      2: { aiType: InternalAIType.RANDOM, randomness: 0.3, thinkTime: 200 },
      3: { aiType: InternalAIType.HEURISTIC, randomness: 0.2, thinkTime: 250 },
      4: { aiType: InternalAIType.HEURISTIC, randomness: 0.1, thinkTime: 300 },
      5: { aiType: InternalAIType.HEURISTIC, randomness: 0.05, thinkTime: 350 },
      6: { aiType: InternalAIType.MINIMAX, randomness: 0.02, thinkTime: 400 },
      7: { aiType: InternalAIType.MINIMAX, randomness: 0.01, thinkTime: 450 },
      8: { aiType: InternalAIType.MINIMAX, randomness: 0.0, thinkTime: 500 },
      9: { aiType: InternalAIType.MCTS, randomness: 0.0, thinkTime: 600 },
      10: { aiType: InternalAIType.MCTS, randomness: 0.0, thinkTime: 700 },
    };

    for (const [key, expectedPreset] of Object.entries(expected)) {
      const difficulty = Number(key);
      const preset = AI_DIFFICULTY_PRESETS[difficulty];

      expect(preset.aiType).toBe(expectedPreset.aiType);
      expect(preset.randomness).toBeCloseTo(expectedPreset.randomness);
      expect(preset.thinkTime).toBe(expectedPreset.thinkTime);
    }
  });

  it('createAIFromProfile uses preset aiType/randomness/thinkTime when aiType is omitted', () => {
    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 7,
      mode: 'service',
    };

    engine.createAIFromProfile(1, profile);
    const config = engine.getAIConfig(1);

    expect(config).toBeDefined();
    expect(config!.aiType).toBe(InternalAIType.MINIMAX);
    expect(config!.randomness).toBeCloseTo(0.01);
    expect(config!.thinkTime).toBe(450);
  });
});
