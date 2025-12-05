import type { RulesUxContext } from '../telemetry/rulesUxEvents';
import {
  TEACHING_SCENARIOS,
  type RulesConcept,
  type TeachingScenarioMetadata,
} from './teachingScenarios';

/**
 * Helpers for mapping curated teaching scenarios and rules concepts
 * onto the low-cardinality RulesUxContext tags used by telemetry.
 *
 * This module is shared between the client (TeachingOverlay, sandbox
 * hosts) and the server-side metrics pipeline so both sides agree on
 * how curated rules teaching content rolls up into semantic
 * rules_context buckets.
 */
export function getRulesUxContextForRulesConcept(
  concept: RulesConcept
): RulesUxContext | undefined {
  switch (concept) {
    case 'anm_forced_elimination':
      return 'anm_forced_elimination';
    case 'territory_mini_region':
      return 'territory_mini_region';
    case 'territory_multi_region_budget':
      return 'territory_multi_region';
    case 'line_vs_territory_multi_phase':
      // This flow touches both line rewards and territory processing;
      // attribute it to the more conservative territory_multi_region
      // bucket for telemetry aggregation.
      return 'territory_multi_region';
    case 'capture_chain_mandatory':
      return 'capture_chain_mandatory';
    case 'landing_on_own_marker':
      return 'landing_on_own_marker';
    case 'structural_stalemate':
      return 'structural_stalemate';
    case 'last_player_standing':
      return 'last_player_standing';
    default:
      return undefined;
  }
}

/**
 * Derive the canonical RulesUxContext for a single teaching scenario
 * metadata entry. Prefers the explicit telemetryRulesContext field
 * when present and falls back to mapping the scenario's rulesConcept.
 */
export function getRulesUxContextForTeachingScenario(
  scenario: TeachingScenarioMetadata
): RulesUxContext | undefined {
  if (scenario.telemetryRulesContext) {
    return scenario.telemetryRulesContext;
  }
  return getRulesUxContextForRulesConcept(scenario.rulesConcept);
}

/**
 * Look up the RulesUxContext for a teaching scenario by its
 * scenarioId.
 *
 * This helper is intentionally narrow – it only knows about the
 * curated TEACHING_SCENARIOS catalogue and is used by TeachingOverlay
 * plus sandbox telemetry when only a scenario id is available.
 */
export function getRulesUxContextForScenarioId(scenarioId: string): RulesUxContext | undefined {
  const meta = TEACHING_SCENARIOS.find((s) => s.scenarioId === scenarioId);
  if (!meta) return undefined;
  return getRulesUxContextForTeachingScenario(meta);
}
