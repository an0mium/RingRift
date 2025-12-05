import type { BoardType } from '../types/game';
import type { RulesWeirdStateReasonCode } from '../engine/weirdStateReasons';
import type { RulesUxContext } from '../telemetry/rulesUxEvents';

/**
 * Shared metadata types for rules teaching scenarios.
 *
 * This mirrors the abstract shape defined in docs/UX_RULES_TEACHING_SCENARIOS.md §2
 * and is consumed by both client hosts (TeachingOverlay, sandbox) and tests.
 */
export type RulesConcept =
  | 'anm_forced_elimination'
  | 'territory_mini_region'
  | 'territory_multi_region_budget'
  | 'line_vs_territory_multi_phase'
  | 'capture_chain_mandatory'
  | 'landing_on_own_marker'
  | 'structural_stalemate'
  | 'last_player_standing';

export type TeachingStepKind = 'guided' | 'interactive';

export interface TeachingScenarioMetadata {
  scenarioId: string;
  rulesConcept: RulesConcept;
  flowId: string;
  stepIndex: number;
  stepKind: TeachingStepKind;

  rulesDocAnchor?: string;
  uxWeirdStateReasonCode?: RulesWeirdStateReasonCode;
  telemetryRulesContext?: RulesUxContext;

  recommendedBoardType: BoardType;
  recommendedNumPlayers: 2 | 3 | 4;
  showInTeachingOverlay: boolean;
  showInSandboxPresets: boolean;
  showInTutorialCarousel: boolean;

  learningObjectiveShort: string;
  difficultyTag?: 'intro' | 'intermediate' | 'advanced';
}

/**
 * Minimal initial flow: Forced Elimination loop & Active–No–Moves (fe_loop_intro).
 *
 * Board positions for these scenarios are defined separately in curated scenario
 * JSON or fixtures. For now the scenarioId fields serve as stable ids that can be
 * wired to concrete boards in a later Code-mode pass.
 */
export const TEACHING_SCENARIOS: readonly TeachingScenarioMetadata[] = [
  {
    scenarioId: 'teaching.fe_loop.step_1',
    rulesConcept: 'anm_forced_elimination',
    flowId: 'fe_loop_intro',
    stepIndex: 1,
    stepKind: 'guided',
    rulesDocAnchor: 'ringrift_complete_rules.md#forced-elimination-when-blocked',
    uxWeirdStateReasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
    telemetryRulesContext: 'anm_forced_elimination',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: true,
    learningObjectiveShort:
      'Recognise when you have no legal placements, movements, or captures and forced elimination will apply.',
    difficultyTag: 'intro',
  },
  {
    scenarioId: 'teaching.fe_loop.step_2',
    rulesConcept: 'anm_forced_elimination',
    flowId: 'fe_loop_intro',
    stepIndex: 2,
    stepKind: 'interactive',
    rulesDocAnchor: 'ringrift_complete_rules.md#forced-elimination-when-blocked',
    uxWeirdStateReasonCode: 'ANM_MOVEMENT_FE_BLOCKED',
    telemetryRulesContext: 'anm_forced_elimination',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: true,
    showInTutorialCarousel: true,
    learningObjectiveShort: 'Execute a forced elimination when no real move exists.',
    difficultyTag: 'intro',
  },
  {
    scenarioId: 'teaching.fe_loop.step_3',
    rulesConcept: 'anm_forced_elimination',
    flowId: 'fe_loop_intro',
    stepIndex: 3,
    stepKind: 'interactive',
    rulesDocAnchor: 'ringrift_complete_rules.md#forced-elimination-when-blocked',
    uxWeirdStateReasonCode: 'FE_SEQUENCE_CURRENT_PLAYER',
    telemetryRulesContext: 'anm_forced_elimination',
    recommendedBoardType: 'square8',
    recommendedNumPlayers: 2,
    showInTeachingOverlay: true,
    showInSandboxPresets: false,
    showInTutorialCarousel: false,
    learningObjectiveShort:
      'See how repeated forced elimination over multiple turns can shrink your stacks toward plateau or Last Player Standing.',
    difficultyTag: 'intermediate',
  },
];
