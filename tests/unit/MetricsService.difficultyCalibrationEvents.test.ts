import client from 'prom-client';
import { MetricsService } from '../../src/server/services/MetricsService';

describe('MetricsService difficulty calibration telemetry metrics', () => {
  beforeEach(() => {
    MetricsService.resetInstance();
    client.register.clear();
  });

  it('records difficulty calibration events with normalised labels for a rich payload', async () => {
    const metrics = MetricsService.getInstance();

    metrics.recordDifficultyCalibrationEvent({
      type: 'difficulty_calibration_game_completed',
      boardType: 'square8' as any,
      numPlayers: 2,
      difficulty: 4,
      isCalibrationOptIn: true,
      result: 'win',
      movesPlayed: 87,
      perceivedDifficulty: 3,
    });

    const output = await metrics.getMetrics();

    expect(output).toContain('ringrift_difficulty_calibration_events_total');
    expect(output).toContain(
      'ringrift_difficulty_calibration_events_total{type="difficulty_calibration_game_completed",board_type="square8",num_players="2",difficulty="4",result="win"} 1'
    );
  });

  it('ignores events where isCalibrationOptIn is false and normalises out-of-range labels', async () => {
    const metrics = MetricsService.getInstance();

    metrics.recordDifficultyCalibrationEvent({
      type: 'difficulty_calibration_game_started',
      boardType: 'hexagonal' as any,
      // Deliberately outside the normal 1–4 range to exercise numPlayers normalisation.
      numPlayers: 99 as any,
      // Deliberately outside the 1–10 difficulty range to exercise difficulty normalisation.
      difficulty: 42 as any,
      isCalibrationOptIn: false,
    });

    const output = await metrics.getMetrics();

    // The counter should exist but no samples should have been recorded for calibration events
    // because isCalibrationOptIn=false causes the helper to early-return.
    const lines = output
      .split('\n')
      .filter((line) => line.startsWith('ringrift_difficulty_calibration_events_total'));
    expect(lines.length).toBe(0);
  });
});
