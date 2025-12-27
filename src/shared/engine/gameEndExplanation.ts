import type { BoardType } from '../types/game';
import type { RulesWeirdStateReasonCode } from './weirdStateReasons';
import type { RulesUxContext } from '../telemetry/rulesUxEvents';

/**
 * Outcome category for a completed game.
 *
 * This is a narrow, UX-oriented projection of GameResult.reason.
 */
export type GameEndOutcomeType =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'structural_stalemate'
  | 'resignation'
  | 'timeout'
  | 'abandonment';

/**
 * High-level victory reason aligned with UX_RULES_CONCEPTS_INDEX.
 */
export type GameEndVictoryReasonCode =
  | 'victory_ring_majority'
  | 'victory_territory_majority'
  | 'victory_last_player_standing'
  | 'victory_structural_stalemate_tiebreak'
  | 'victory_resignation'
  | 'victory_timeout'
  | 'victory_abandonment';

/**
 * Dimension of the structural stalemate tiebreak ladder.
 */
export type GameEndTiebreakKind =
  | 'territory_spaces'
  | 'eliminated_rings'
  | 'markers'
  | 'last_real_action';

export interface GameEndPlayerScoreBreakdown {
  playerId: string;
  eliminatedRings: number;
  territorySpaces: number;
  markers: number;
  /**
   * Board- or mode-specific numeric fields (e.g. bonus points).
   */
  extra?: Record<string, number>;
}

export interface GameEndTiebreakStep {
  kind: GameEndTiebreakKind;
  /**
   * Player id that won this ladder step, or null if still tied at this step.
   */
  winnerPlayerId: string | null;
  /**
   * Comparable numeric value per player for this ladder dimension.
   */
  valuesByPlayer: Record<string, number>;
}

/**
 * Alias into the shared RulesWeirdStateReasonCode vocabulary.
 */
export type GameEndRulesWeirdStateReasonCode = RulesWeirdStateReasonCode;

/**
 * Rules-context tags used for both explanations and telemetry.
 *
 * These are aligned with RulesUxContext but kept under a local alias
 * to decouple consumers from telemetry implementation details.
 */
export type GameEndRulesContextTag = RulesUxContext;

export interface GameEndWeirdStateContext {
  /**
   * Canonical weird-state reason codes involved in this ending.
   */
  reasonCodes: GameEndRulesWeirdStateReasonCode[];
  /**
   * Primary reason code to highlight in UX and telemetry (if any).
   */
  primaryReasonCode?: GameEndRulesWeirdStateReasonCode;
  /**
   * One or more semantic rules-context tags associated with this ending.
   */
  rulesContextTags?: GameEndRulesContextTag[];
  /**
   * Optional teaching topics suggested by this weird-state combination.
   *
   * These are opaque ids (e.g. "teaching.active_no_moves") – no
   * teaching types are imported here to keep this module lightweight.
   */
  teachingTopicIds?: string[];
}

export interface GameEndRulesReference {
  /**
   * Primary rules anchor for this ending (e.g. "RR-CANON R170").
   */
  rulesSpecAnchor?: string;
  /**
   * Optional list of rules-doc links expressed as simple string identifiers
   * (for example "RULES_CANONICAL_SPEC.md#170-ring-elimination-victory").
   */
  rulesDocsLinks?: string[];
}

export interface GameEndTeachingLink {
  /**
   * High-level teaching topics relevant to this ending.
   */
  teachingTopics?: string[];
  /**
   * Recommended curated flows or scenario ids.
   */
  recommendedFlows?: string[];
}

export interface GameEndUxCopyKeys {
  /**
   * HUD / compact text lookup key.
   */
  shortSummaryKey: string;
  /**
   * VictoryModal / TeachingOverlay lookup key.
   */
  detailedSummaryKey?: string;
}

export interface GameEndDebugInfo {
  /**
   * Raw outcome or engine result used to derive this explanation.
   */
  rawOutcome?: unknown;
  /**
   * Free-form notes for debugging / parity tooling.
   */
  notes?: string;
}

export interface GameEndExplanation {
  // 1) Top-level metadata
  gameId?: string;
  boardType: BoardType;
  numPlayers: number;
  winnerPlayerId: string | null;
  outcomeType: GameEndOutcomeType;
  victoryReasonCode: GameEndVictoryReasonCode;
  // 2) Victory and scoring detail
  scoreBreakdown?: Record<string, GameEndPlayerScoreBreakdown>;
  tiebreakSteps?: GameEndTiebreakStep[];
  primaryConceptId?: string;
  // 3) Rules references
  rulesReferences?: GameEndRulesReference[];
  // 4) Weird-state and teaching context
  weirdStateContext?: GameEndWeirdStateContext;
  teaching?: GameEndTeachingLink;
  // 5) UX copy
  uxCopy: GameEndUxCopyKeys;
  // 6) Telemetry hints
  telemetry?: {
    rulesContextTags?: GameEndRulesContextTag[];
    weirdStateReasonCodes?: GameEndRulesWeirdStateReasonCode[];
  };
  // 7) Developer-only debugging information
  debugInfo?: GameEndDebugInfo;
}

/**
 * Source shape used by builders to construct a GameEndExplanation.
 */
export interface GameEndExplanationSource {
  gameId?: string;
  boardType: BoardType;
  numPlayers: number;
  winnerPlayerId: string | null;
  outcomeType: GameEndOutcomeType;
  victoryReasonCode: GameEndVictoryReasonCode;
  scoreBreakdown?: Record<string, GameEndPlayerScoreBreakdown>;
  tiebreakSteps?: GameEndTiebreakStep[];
  primaryConceptId?: string;
  rulesReferences?: GameEndRulesReference[];
  weirdStateContext?: GameEndWeirdStateContext;
  teaching?: GameEndTeachingLink;
  /**
   * Optional semantic rules-context tags chosen by the caller.
   *
   * These will be merged with any tags present on weirdStateContext.
   */
  telemetryTags?: GameEndRulesContextTag[];
  uxCopy: GameEndUxCopyKeys;
}

/**
 * Minimal engine-level view of a completed game used to construct
 * UX-facing GameEndExplanation instances without coupling to full
 * GameResult / engine internals.
 */
export interface GameEndEngineView {
  gameId?: string;
  boardType: BoardType;
  numPlayers: number;
  winnerPlayerId: string | null;
  outcomeType: GameEndOutcomeType;
  victoryReasonCode: GameEndVictoryReasonCode;
  scoreBreakdown?: Record<string, GameEndPlayerScoreBreakdown>;
  tiebreakSteps?: GameEndTiebreakStep[];
  /**
   * Optional higher-level concept identifier for this ending, aligned
   * with docs/ux/UX_RULES_CONCEPTS_INDEX.md (e.g. "anm_fe_core",
   * "lps_real_actions", "structural_stalemate", "territory_mini_regions").
   */
  primaryConceptId?: string;
  /**
   * Optional rules references already derived by engine / host layers.
   */
  rulesReferences?: GameEndRulesReference[];
  /**
   * Optional weird-state context associated with this game ending.
   */
  weirdStateContext?: GameEndWeirdStateContext;
}

function uniq<T>(values: readonly T[] | undefined): T[] | undefined {
  if (!values || values.length === 0) {
    return undefined;
  }
  const seen = new Set<T>();
  const result: T[] = [];
  for (const value of values) {
    if (!seen.has(value)) {
      seen.add(value);
      result.push(value);
    }
  }
  return result.length > 0 ? result : undefined;
}

function normalizeArray<T>(values: T[] | undefined | null): T[] | undefined {
  if (!values || values.length === 0) {
    return undefined;
  }
  return values;
}

export function buildGameEndExplanation(source: GameEndExplanationSource): GameEndExplanation {
  const combinedRulesContextTags = uniq<GameEndRulesContextTag>([
    ...(source.weirdStateContext?.rulesContextTags ?? []),
    ...(source.telemetryTags ?? []),
  ]);

  const weirdStateReasonCodes = uniq<GameEndRulesWeirdStateReasonCode>(
    source.weirdStateContext?.reasonCodes
  );

  const telemetry: GameEndExplanation['telemetry'] =
    combinedRulesContextTags || weirdStateReasonCodes
      ? {
          ...(combinedRulesContextTags ? { rulesContextTags: combinedRulesContextTags } : {}),
          ...(weirdStateReasonCodes ? { weirdStateReasonCodes } : {}),
        }
      : undefined;

  const explanation: GameEndExplanation = {
    boardType: source.boardType,
    numPlayers: source.numPlayers,
    winnerPlayerId: source.winnerPlayerId,
    outcomeType: source.outcomeType,
    victoryReasonCode: source.victoryReasonCode,
    uxCopy: source.uxCopy,
  };

  if (source.gameId !== undefined) {
    explanation.gameId = source.gameId;
  }

  const scoreBreakdown =
    source.scoreBreakdown && Object.keys(source.scoreBreakdown).length > 0
      ? source.scoreBreakdown
      : undefined;
  if (scoreBreakdown !== undefined) {
    explanation.scoreBreakdown = scoreBreakdown;
  }

  const tiebreakSteps = normalizeArray(source.tiebreakSteps ?? undefined);
  if (tiebreakSteps !== undefined) {
    explanation.tiebreakSteps = tiebreakSteps;
  }

  if (source.primaryConceptId !== undefined) {
    explanation.primaryConceptId = source.primaryConceptId;
  }

  const rulesReferences = normalizeArray(source.rulesReferences ?? undefined);
  if (rulesReferences !== undefined) {
    explanation.rulesReferences = rulesReferences;
  }

  if (source.weirdStateContext !== undefined) {
    explanation.weirdStateContext = source.weirdStateContext;
  }

  if (source.teaching !== undefined) {
    explanation.teaching = source.teaching;
  }

  if (telemetry !== undefined) {
    explanation.telemetry = telemetry;
  }

  // debugInfo is intentionally omitted by the builder; callers that need to
  // attach debug metadata can do so in a separate step.

  return explanation;
}

export function buildGameEndExplanationFromEngineView(
  view: GameEndEngineView,
  extra: {
    teaching?: GameEndTeachingLink;
    telemetryTags?: GameEndRulesContextTag[];
    uxCopy: GameEndUxCopyKeys;
  }
): GameEndExplanation {
  const source: GameEndExplanationSource = {
    boardType: view.boardType,
    numPlayers: view.numPlayers,
    winnerPlayerId: view.winnerPlayerId,
    outcomeType: view.outcomeType,
    victoryReasonCode: view.victoryReasonCode,
    uxCopy: extra.uxCopy,
  };

  if (view.gameId !== undefined) {
    source.gameId = view.gameId;
  }
  if (view.scoreBreakdown !== undefined) {
    source.scoreBreakdown = view.scoreBreakdown;
  }
  if (view.tiebreakSteps !== undefined) {
    source.tiebreakSteps = view.tiebreakSteps;
  }
  if (view.primaryConceptId !== undefined) {
    source.primaryConceptId = view.primaryConceptId;
  }
  if (view.rulesReferences !== undefined) {
    source.rulesReferences = view.rulesReferences;
  }
  if (view.weirdStateContext !== undefined) {
    source.weirdStateContext = view.weirdStateContext;
  }
  if (extra.teaching !== undefined) {
    source.teaching = extra.teaching;
  }
  if (extra.telemetryTags !== undefined) {
    source.telemetryTags = extra.telemetryTags;
  }

  return buildGameEndExplanation(source);
}
