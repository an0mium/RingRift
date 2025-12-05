import client from 'prom-client';
import { MetricsService } from '../../src/server/services/MetricsService';

describe('MetricsService rules-UX telemetry metrics', () => {
  beforeEach(() => {
    MetricsService.resetInstance();
    client.register.clear();
  });

  it('records rules-UX events with normalised labels for a rich payload', async () => {
    const metrics = MetricsService.getInstance();

    metrics.recordRulesUxEvent({
      type: 'rules_help_open',
      boardType: 'square8' as any,
      numPlayers: 2,
      aiDifficulty: 5,
      topic: 'active_no_moves',
      rulesConcept: 'anm_intro',
      scenarioId: 'scenario-1',
      weirdStateType: 'active-no-moves-movement',
      undoStreak: 3,
      repeatCount: 2,
      secondsSinceWeirdState: 10,
    });

    const output = await metrics.getMetrics();

    expect(output).toContain('ringrift_rules_ux_events_total');
    expect(output).toContain(
      'ringrift_rules_ux_events_total{type="rules_help_open",board_type="square8",num_players="2",ai_difficulty="5",topic="active_no_moves",rules_concept="anm_intro",weird_state_type="active-no-moves-movement"} 1'
    );
  });

  it('falls back to safe label values when optional fields are missing or invalid', async () => {
    const metrics = MetricsService.getInstance();

    metrics.recordRulesUxEvent({
      type: 'rules_help_open',
      boardType: 'hexagonal' as any,
      // Deliberately outside the normal 1-4 range to exercise num_players normalisation.
      numPlayers: 99,
      // Omit aiDifficulty and provide empty strings for other optional fields.
      topic: '',
      rulesConcept: '',
      scenarioId: '',
      weirdStateType: '',
    });

    const output = await metrics.getMetrics();

    expect(output).toContain(
      'ringrift_rules_ux_events_total{type="rules_help_open",board_type="hexagonal",num_players="unknown",ai_difficulty="none",topic="none",rules_concept="none",weird_state_type="none"} 1'
    );
  });
});
