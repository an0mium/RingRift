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
      'ringrift_rules_ux_events_total{event_type="rules_help_open",rules_context="none",source="unknown",board_type="square8",num_players="2",difficulty="5",is_ranked="unknown",is_sandbox="unknown"} 1'
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
      'ringrift_rules_ux_events_total{event_type="rules_help_open",rules_context="none",source="unknown",board_type="hexagonal",num_players="unknown",difficulty="unknown",is_ranked="unknown",is_sandbox="unknown"} 1'
    );
  });
});
