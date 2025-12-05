import api from '../../src/client/services/api';
import {
  sendDifficultyCalibrationEvent,
  storeDifficultyCalibrationSession,
  getDifficultyCalibrationSession,
  clearDifficultyCalibrationSession,
} from '../../src/client/utils/difficultyCalibrationTelemetry';
import type { DifficultyCalibrationEventPayload } from '../../src/shared/telemetry/difficultyCalibrationEvents';
import type { BoardType } from '../../src/shared/types/game';

jest.mock('../../src/client/services/api', () => {
  const post = jest.fn();
  return {
    __esModule: true,
    default: { post },
  };
});

const mockedApi = api as unknown as { post: jest.MockedFunction<typeof api.post> };

describe('difficultyCalibrationTelemetry helper', () => {
  beforeEach(() => {
    mockedApi.post.mockReset();
    // Reset synthetic Vite env used by the helper
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {};
    if (typeof window !== 'undefined' && window.sessionStorage) {
      window.sessionStorage.clear();
    }
  });

  it('sends a DifficultyCalibrationEventPayload to /telemetry/difficulty-calibration when enabled', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {
      VITE_DIFFICULTY_CALIBRATION_TELEMETRY_ENABLED: 'true',
    };

    const event: DifficultyCalibrationEventPayload = {
      type: 'difficulty_calibration_game_started',
      boardType: 'square8' as BoardType,
      numPlayers: 2,
      difficulty: 4,
      isCalibrationOptIn: true,
    };

    mockedApi.post.mockResolvedValueOnce({ data: {} } as any);

    await expect(sendDifficultyCalibrationEvent(event)).resolves.toBeUndefined();

    expect(mockedApi.post).toHaveBeenCalledTimes(1);
    expect(mockedApi.post).toHaveBeenCalledWith('/telemetry/difficulty-calibration', event);
  });

  it('swallows errors from the underlying HTTP call and still resolves', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {
      VITE_DIFFICULTY_CALIBRATION_TELEMETRY_ENABLED: 'true',
    };

    const event: DifficultyCalibrationEventPayload = {
      type: 'difficulty_calibration_game_completed',
      boardType: 'square8' as BoardType,
      numPlayers: 2,
      difficulty: 6,
      isCalibrationOptIn: true,
      result: 'win',
      movesPlayed: 87,
    };

    mockedApi.post.mockRejectedValueOnce(new Error('network failure'));

    await expect(sendDifficultyCalibrationEvent(event)).resolves.toBeUndefined();

    expect(mockedApi.post).toHaveBeenCalledTimes(1);
  });

  it('respects the global enable flag and becomes a no-op when disabled', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {
      VITE_DIFFICULTY_CALIBRATION_TELEMETRY_ENABLED: 'false',
    };

    const event: DifficultyCalibrationEventPayload = {
      type: 'difficulty_calibration_game_started',
      boardType: 'square8' as BoardType,
      numPlayers: 2,
      difficulty: 2,
      isCalibrationOptIn: true,
    };

    await sendDifficultyCalibrationEvent(event);

    expect(mockedApi.post).not.toHaveBeenCalled();
  });

  it('persists, reads, and clears calibration sessions via sessionStorage', () => {
    const gameId = 'game-abc';
    const session = {
      boardType: 'square8' as BoardType,
      numPlayers: 2,
      difficulty: 4,
      isCalibrationOptIn: true,
    };

    // Initially nothing stored
    expect(getDifficultyCalibrationSession(gameId)).toBeNull();

    storeDifficultyCalibrationSession(gameId, session);

    const loaded = getDifficultyCalibrationSession(gameId);
    expect(loaded).toEqual(session);

    clearDifficultyCalibrationSession(gameId);

    expect(getDifficultyCalibrationSession(gameId)).toBeNull();
  });

  it('returns null for malformed session data in storage', () => {
    const gameId = 'game-bad';

    if (typeof window !== 'undefined' && window.sessionStorage) {
      window.sessionStorage.setItem(
        `rr_difficulty_calibration_game:${gameId}`,
        JSON.stringify({
          boardType: 123,
          numPlayers: 'not-a-number',
          difficulty: 'x',
          isCalibrationOptIn: true,
        })
      );
    }

    const loaded = getDifficultyCalibrationSession(gameId);
    expect(loaded).toBeNull();
  });
});
