import { AI_DIFFICULTY_PRESETS, AIEngine, AIType } from '../../src/server/game/ai/AIEngine';

describe('AI difficulty ladder (canonical mapping)', () => {
  it('keeps the 1–10 difficulty presets stable and canonical', () => {
    const expected: Record<
      number,
      { aiType: AIType; randomness: number; thinkTime: number; profileId: string }
    > = {
      1: { aiType: AIType.RANDOM, randomness: 0.5, thinkTime: 150, profileId: 'v1-random-1' },
      2: { aiType: AIType.HEURISTIC, randomness: 0.3, thinkTime: 200, profileId: 'v1-heuristic-2' },
      3: { aiType: AIType.MINIMAX, randomness: 0.15, thinkTime: 1800, profileId: 'v1-minimax-3' },
      4: {
        aiType: AIType.MINIMAX,
        randomness: 0.08,
        thinkTime: 2800,
        profileId: 'v1-minimax-4-nnue',
      },
      5: {
        aiType: AIType.DESCENT,
        randomness: 0.05,
        thinkTime: 4000,
        profileId: 'ringrift_best_sq8_2p',
      },
      6: {
        aiType: AIType.DESCENT,
        randomness: 0.02,
        thinkTime: 5500,
        profileId: 'ringrift_best_sq8_2p',
      },
      7: { aiType: AIType.MCTS, randomness: 0.0, thinkTime: 7500, profileId: 'v1-mcts-7' },
      8: {
        aiType: AIType.MCTS,
        randomness: 0.0,
        thinkTime: 9600,
        profileId: 'ringrift_best_sq8_2p',
      },
      9: {
        aiType: AIType.GUMBEL_MCTS,
        randomness: 0.0,
        thinkTime: 12600,
        profileId: 'ringrift_best_sq8_2p',
      },
      10: {
        aiType: AIType.GUMBEL_MCTS,
        randomness: 0.0,
        thinkTime: 16000,
        profileId: 'ringrift_best_sq8_2p',
      },
    };

    for (let difficulty = 1; difficulty <= 10; difficulty += 1) {
      const preset = AI_DIFFICULTY_PRESETS[difficulty];
      expect(preset).toBeDefined();

      const expectedPreset = expected[difficulty];
      expect(preset.aiType).toBe(expectedPreset.aiType);
      expect(preset.randomness).toBeCloseTo(expectedPreset.randomness, 6);
      expect(preset.thinkTime).toBe(expectedPreset.thinkTime);
      expect(preset.profileId).toBe(expectedPreset.profileId);
    }
  });

  it('creates per-player configs consistent with the canonical presets', () => {
    const engine = new AIEngine();

    for (let difficulty = 1; difficulty <= 10; difficulty += 1) {
      engine.createAIFromProfile(1, { difficulty, mode: 'service' });
      const config = engine.getAIConfig(1);
      expect(config).toBeDefined();

      const preset = AI_DIFFICULTY_PRESETS[difficulty];
      expect(config!.difficulty).toBe(difficulty);
      expect(config!.aiType).toBe(preset.aiType);
      expect(config!.randomness).toBeCloseTo(preset.randomness ?? 0, 6);
      expect(config!.thinkTime).toBe(preset.thinkTime);
    }
  });
});
