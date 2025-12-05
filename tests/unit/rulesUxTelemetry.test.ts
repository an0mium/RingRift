import api from '../../src/client/services/api';
import { sendRulesUxEvent } from '../../src/client/utils/rulesUxTelemetry';
import type { RulesUxEventPayload } from '../../src/shared/telemetry/rulesUxEvents';

jest.mock('../../src/client/services/api', () => {
  const post = jest.fn();
  return {
    __esModule: true,
    default: { post },
  };
});

const mockedApi = api as unknown as { post: jest.MockedFunction<typeof api.post> };

describe('rulesUxTelemetry.sendRulesUxEvent', () => {
  beforeEach(() => {
    // Reset axios mock
    mockedApi.post.mockReset();
    // Reset synthetic Vite env used by the helper
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {};
  });

  it('sends a RulesUxEventPayload to /telemetry/rules-ux via the shared API client', async () => {
    // Enable telemetry and disable sampling so the event is always sent.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {
      VITE_RULES_UX_TELEMETRY_ENABLED: 'true',
      VITE_RULES_UX_HELP_OPEN_SAMPLE_RATE: '1',
    };

    const event: RulesUxEventPayload = {
      type: 'rules_help_open',
      boardType: 'square8',
      numPlayers: 2,
      aiDifficulty: 5,
      topic: 'active_no_moves',
    };

    mockedApi.post.mockResolvedValueOnce({ data: {} } as any);

    await expect(sendRulesUxEvent(event)).resolves.toBeUndefined();

    expect(mockedApi.post).toHaveBeenCalledTimes(1);
    const [path, payload] = mockedApi.post.mock.calls[0];
    expect(path).toBe('/telemetry/rules-ux');
    // Payload should contain the original event fields plus enrichment metadata.
    expect(payload).toEqual(
      expect.objectContaining({
        type: 'rules_help_open',
        boardType: 'square8',
        numPlayers: 2,
        aiDifficulty: 5,
        topic: 'active_no_moves',
      })
    );
    expect(typeof payload.ts).toBe('string');
    expect(typeof payload.clientPlatform).toBe('string');
  });

  it('swallows errors from the underlying HTTP call and still resolves', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {
      VITE_RULES_UX_TELEMETRY_ENABLED: 'true',
      VITE_RULES_UX_HELP_OPEN_SAMPLE_RATE: '1',
    };

    const event: RulesUxEventPayload = {
      type: 'rules_help_open',
      boardType: 'square8',
      numPlayers: 2,
      topic: 'forced_elimination',
    };

    mockedApi.post.mockRejectedValueOnce(new Error('network failure'));

    await expect(sendRulesUxEvent(event)).resolves.toBeUndefined();

    expect(mockedApi.post).toHaveBeenCalledTimes(1);
  });

  it('respects the global enable flag and becomes a no-op when disabled', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {
      VITE_RULES_UX_TELEMETRY_ENABLED: 'false',
      VITE_RULES_UX_HELP_OPEN_SAMPLE_RATE: '1',
    };

    const event: RulesUxEventPayload = {
      type: 'rules_help_open',
      boardType: 'square8',
      numPlayers: 2,
      topic: 'active_no_moves',
    };

    await sendRulesUxEvent(event);

    expect(mockedApi.post).not.toHaveBeenCalled();
  });

  it('applies sampling to rules_help_open events when sample rate is 0 (never sends)', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {
      VITE_RULES_UX_TELEMETRY_ENABLED: 'true',
      VITE_RULES_UX_HELP_OPEN_SAMPLE_RATE: '0',
    };

    const event: RulesUxEventPayload = {
      type: 'rules_help_open',
      boardType: 'square8',
      numPlayers: 2,
      topic: 'active_no_moves',
    };

    await sendRulesUxEvent(event);

    expect(mockedApi.post).not.toHaveBeenCalled();
  });

  it('does not sample non-help events even when help-open sample rate is 0', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).__VITE_ENV__ = {
      VITE_RULES_UX_TELEMETRY_ENABLED: 'true',
      VITE_RULES_UX_HELP_OPEN_SAMPLE_RATE: '0',
    };

    const event: RulesUxEventPayload = {
      type: 'rules_undo_churn',
      boardType: 'square8',
      numPlayers: 2,
      undoStreak: 4,
    };

    mockedApi.post.mockResolvedValueOnce({ data: {} } as any);

    await sendRulesUxEvent(event);

    expect(mockedApi.post).toHaveBeenCalledTimes(1);
    const [path, payload] = mockedApi.post.mock.calls[0];

    expect(path).toBe('/telemetry/rules-ux');
    // Non-help events should still be sent even when help-open sampling is 0.
    // The payload is enriched with telemetry metadata, so we assert on a subset.
    expect(payload).toEqual(
      expect.objectContaining({
        type: 'rules_undo_churn',
        boardType: 'square8',
        numPlayers: 2,
        undoStreak: 4,
      })
    );
  });
});
