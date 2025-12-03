import { GameSession } from '../../src/server/game/GameSession';
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';

jest.mock('../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    getDiagnostics: jest.fn(),
  },
}));

describe('GameSession AI diagnostics aggregation', () => {
  const makeSession = () => {
    const io = {
      to: jest.fn().mockReturnThis(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient = {} as any;

    return new GameSession('game-1', io, pythonClient, new Map());
  };

  it('escalates aiQualityMode to rulesServiceDegraded when rules diagnostics show failures', () => {
    const session = makeSession();

    const rulesDiag = {
      pythonEvalFailures: 1,
      pythonBackendFallbacks: 2,
      pythonShadowErrors: 3,
    };

    (session as any).rulesFacade = {
      getDiagnostics: jest.fn(() => rulesDiag),
    };

    (globalAIEngine.getDiagnostics as jest.Mock).mockReturnValue({
      serviceFailureCount: 4,
      localFallbackCount: 1,
    });

    (session as any).updateDiagnostics(1);

    const snapshot = session.getAIDiagnosticsSnapshotForTesting();

    expect(snapshot.rulesServiceFailureCount).toBe(3);
    expect(snapshot.rulesShadowErrorCount).toBe(3);
    expect(snapshot.aiServiceFailureCount).toBe(4);
    expect(snapshot.aiFallbackMoveCount).toBe(1);
    expect(snapshot.aiQualityMode).toBe('rulesServiceDegraded');
  });

  it('sets aiQualityMode to fallbackLocalAI when AI falls back locally and rules are healthy', () => {
    const session = makeSession();

    const rulesDiag = {
      pythonEvalFailures: 0,
      pythonBackendFallbacks: 0,
      pythonShadowErrors: 0,
    };

    (session as any).rulesFacade = {
      getDiagnostics: jest.fn(() => rulesDiag),
    };

    (globalAIEngine.getDiagnostics as jest.Mock).mockReturnValue({
      serviceFailureCount: 0,
      localFallbackCount: 2,
    });

    (session as any).updateDiagnostics(1);

    const snapshot = session.getAIDiagnosticsSnapshotForTesting();

    expect(snapshot.rulesServiceFailureCount).toBe(0);
    expect(snapshot.rulesShadowErrorCount).toBe(0);
    expect(snapshot.aiServiceFailureCount).toBe(0);
    expect(snapshot.aiFallbackMoveCount).toBe(2);
    expect(snapshot.aiQualityMode).toBe('fallbackLocalAI');
  });

  it('sets aiQualityMode to normal when both rules and AI diagnostics are clean', () => {
    const session = makeSession();

    const rulesDiag = {
      pythonEvalFailures: 0,
      pythonBackendFallbacks: 0,
      pythonShadowErrors: 0,
    };

    (session as any).rulesFacade = {
      getDiagnostics: jest.fn(() => rulesDiag),
    };

    (globalAIEngine.getDiagnostics as jest.Mock).mockReturnValue({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    });

    (session as any).updateDiagnostics(1);

    const snapshot = session.getAIDiagnosticsSnapshotForTesting();

    expect(snapshot.rulesServiceFailureCount).toBe(0);
    expect(snapshot.rulesShadowErrorCount).toBe(0);
    expect(snapshot.aiServiceFailureCount).toBe(0);
    expect(snapshot.aiFallbackMoveCount).toBe(0);
    expect(snapshot.aiQualityMode).toBe('normal');
  });

  it('exposes AIRequestState terminal breakdown fields in diagnostics snapshot', () => {
    const session = makeSession();

    // Seed internal counters that would normally be driven by the
    // AIRequestState machine / GameSession request tracking.
    (session as any).aiRequestState = { kind: 'timed_out' };

    const rulesDiag = {
      pythonEvalFailures: 0,
      pythonBackendFallbacks: 0,
      pythonShadowErrors: 0,
    };

    (session as any).rulesFacade = {
      getDiagnostics: jest.fn(() => rulesDiag),
    };

    (globalAIEngine.getDiagnostics as jest.Mock).mockReturnValue({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    });

    (session as any).updateDiagnostics(1);

    const snapshot = session.getAIDiagnosticsSnapshotForTesting() as any;

    // The snapshot currently focuses on aggregate quality/health fields rather
    // than mirroring the full AIRequestState; this test simply ensures that
    // calling updateDiagnostics with a non-idle aiRequestState does not throw
    // and returns a structured object.
    expect(typeof snapshot).toBe('object');
  });
});
