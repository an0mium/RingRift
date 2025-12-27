# UX Rules Game-End Explanation Model Spec

> **Doc Status (2025-12-05): Active – W‑UX‑7 (Structured "Why Did the Game End?" Explanation Model)**
>
> **Role:** Define a single structured payload, `GameEndExplanation`, that explains **why** a RingRift game ended and can be consumed consistently by HUD, VictoryModal, sandbox, and TeachingOverlay.
>
> **Source of truth:** [`src/shared/engine/gameEndExplanation.ts`](../../src/shared/engine/gameEndExplanation.ts)
>
> **Inputs:**
>
> - Weakest‑aspect assessment: [`WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md`](../archive/assessments/WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md)
> - Rules concept vocabulary: [`UX_RULES_CONCEPTS_INDEX.md`](UX_RULES_CONCEPTS_INDEX.md)
> - Weird‑state reasons: [`UX_RULES_WEIRD_STATES_SPEC.md`](UX_RULES_WEIRD_STATES_SPEC.md) and [`weirdStateReasons.ts`](../../src/shared/engine/weirdStateReasons.ts)
> - Teaching flows: [`UX_RULES_TEACHING_SCENARIOS.md`](UX_RULES_TEACHING_SCENARIOS.md) and [`teachingScenarios.ts`](../../src/shared/teaching/teachingScenarios.ts)
> - Telemetry taxonomy: [`UX_RULES_TELEMETRY_SPEC.md`](UX_RULES_TELEMETRY_SPEC.md) and [`rulesUxEvents.ts`](../../src/shared/telemetry/rulesUxEvents.ts)
> - Canonical rules: [`RULES_CANONICAL_SPEC.md`](../../RULES_CANONICAL_SPEC.md)
> - Tests: [`GameEndExplanation.builder.test.ts`](../../tests/unit/GameEndExplanation.builder.test.ts), [`GameEndExplanation.fromEngineView.test.ts`](../../tests/unit/GameEndExplanation.fromEngineView.test.ts)

---

## 1. Scope and goals

This spec exists because players regularly ask _“why did the game end?”_ in ANM/FE, LPS, structural stalemate, and mini‑region territory outcomes. The model provides a single, structured explanation object so that HUD, VictoryModal, sandbox, and TeachingOverlay **do not re-derive** these details independently.

Goals:

- One **shared explanation payload** for all end‑of‑game surfaces.
- Clear linkage between engine outcomes, weird‑state reasons, rules references, and teaching topics.
- Low‑cardinality telemetry tags tied to rules concepts rather than UI surfaces.

Non‑goals:

- Defining UI layouts or localized copy strings (see [`UX_RULES_COPY_SPEC.md`](UX_RULES_COPY_SPEC.md)).
- Changing rules semantics (see [`RULES_CANONICAL_SPEC.md`](../../RULES_CANONICAL_SPEC.md)).

---

## 2. Canonical types (aligned to code)

The TypeScript definitions below mirror the canonical source file (`gameEndExplanation.ts`). Only the most relevant fields are shown; optional fields are marked.

```ts
type GameEndOutcomeType =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'structural_stalemate'
  | 'resignation'
  | 'timeout'
  | 'abandonment';

type GameEndVictoryReasonCode =
  | 'victory_ring_majority'
  | 'victory_territory_majority'
  | 'victory_last_player_standing'
  | 'victory_structural_stalemate_tiebreak'
  | 'victory_resignation'
  | 'victory_timeout'
  | 'victory_abandonment';

type GameEndTiebreakKind = 'territory_spaces' | 'eliminated_rings' | 'markers' | 'last_real_action';

type GameEndRulesContextTag =
  | 'anm_forced_elimination'
  | 'structural_stalemate'
  | 'last_player_standing'
  | 'territory_mini_region'
  | 'territory_multi_region'
  | 'line_reward_exact'
  | 'line_reward_overlength'
  | 'line_vs_territory_multi_phase'
  | 'capture_chain_mandatory'
  | 'landing_on_own_marker'
  | 'pie_rule_swap'
  | 'placement_cap'
  | string;

type GameEndRulesWeirdStateReasonCode =
  | 'ANM_MOVEMENT_FE_BLOCKED'
  | 'ANM_LINE_NO_ACTIONS'
  | 'ANM_TERRITORY_NO_ACTIONS'
  | 'FE_SEQUENCE_CURRENT_PLAYER'
  | 'STRUCTURAL_STALEMATE_TIEBREAK'
  | 'LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS'
  | string;

type GameEndPlayerScoreBreakdown = {
  playerId: string;
  eliminatedRings: number;
  territorySpaces: number;
  markers: number;
  extra?: Record<string, number>;
};

type GameEndTiebreakStep = {
  kind: GameEndTiebreakKind;
  winnerPlayerId: string | null;
  valuesByPlayer: Record<string, number>;
};

type GameEndWeirdStateContext = {
  reasonCodes: GameEndRulesWeirdStateReasonCode[];
  primaryReasonCode?: GameEndRulesWeirdStateReasonCode;
  rulesContextTags?: GameEndRulesContextTag[];
  teachingTopicIds?: string[];
};

type GameEndRulesReference = {
  rulesSpecAnchor?: string; // e.g. "RR-CANON-R173"
  rulesDocsLinks?: string[]; // e.g. "RULES_CANONICAL_SPEC.md#173-..."
};

type GameEndTeachingLink = {
  teachingTopics?: string[]; // TeachingOverlay topic ids
  recommendedFlows?: string[]; // Curated flow ids
};

type GameEndUxCopyKeys = {
  shortSummaryKey: string; // HUD / compact banner
  detailedSummaryKey?: string; // VictoryModal / TeachingOverlay
};

type GameEndExplanation = {
  gameId?: string;
  boardType: 'square8' | 'square19' | 'hex8' | 'hexagonal';
  numPlayers: number;
  winnerPlayerId: string | null;
  outcomeType: GameEndOutcomeType;
  victoryReasonCode: GameEndVictoryReasonCode;
  scoreBreakdown?: Record<string, GameEndPlayerScoreBreakdown>;
  tiebreakSteps?: GameEndTiebreakStep[];
  primaryConceptId?: string; // From UX_RULES_CONCEPTS_INDEX
  rulesReferences?: GameEndRulesReference[];
  weirdStateContext?: GameEndWeirdStateContext;
  teaching?: GameEndTeachingLink;
  uxCopy: GameEndUxCopyKeys;
  telemetry?: {
    rulesContextTags?: GameEndRulesContextTag[];
    weirdStateReasonCodes?: GameEndRulesWeirdStateReasonCode[];
  };
  debugInfo?: {
    rawOutcome?: unknown;
    notes?: string;
  };
};
```

Notes:

- `primaryConceptId` should align with concept ids in [`UX_RULES_CONCEPTS_INDEX.md`](UX_RULES_CONCEPTS_INDEX.md).
- `rulesContextTags` should align with `RulesUxContext` in [`rulesUxEvents.ts`](../../src/shared/telemetry/rulesUxEvents.ts).
- `reasonCodes` should align with [`RulesWeirdStateReasonCode`](../../src/shared/engine/weirdStateReasons.ts).

---

## 3. Builder behavior and invariants

Two helpers construct `GameEndExplanation` instances:

- `buildGameEndExplanation(source)` – for fully populated payloads.
- `buildGameEndExplanationFromEngineView(view, extra)` – for minimal engine views plus UX extras.

Key builder behavior (tested in `GameEndExplanation.*.test.ts`):

- **Telemetry merge:** `telemetry.rulesContextTags` is a de‑duplicated union of `source.weirdStateContext.rulesContextTags` and `source.telemetryTags`.
- **Reason codes:** `telemetry.weirdStateReasonCodes` is a de‑duplicated copy of `weirdStateContext.reasonCodes`.
- **Optional fields:** `scoreBreakdown`, `tiebreakSteps`, `rulesReferences`, `weirdStateContext`, `teaching`, and `telemetry` are omitted when empty.
- **Debug info:** `debugInfo` is intentionally not populated by the builder and should be added by callers if needed.

This model is **semantics‑first**: all rendering logic should derive from the explanation payload instead of re‑deriving engine outcomes.

---

## 4. Mapping guidance

### 4.1 Weird‑state context

Use weird‑state reason codes and rules‑context tags to link confusing outcomes to teaching content:

- Reason codes → [`UX_RULES_WEIRD_STATES_SPEC.md`](UX_RULES_WEIRD_STATES_SPEC.md)
- Teaching topic ids → [`TeachingOverlay`](../../src/client/components/TeachingOverlay.tsx)

If a game ends due to ANM/FE, LPS, or structural stalemate, ensure `weirdStateContext` is populated with the appropriate reason codes and tags.

### 4.2 Rules references

`rulesReferences` may include:

- `rulesSpecAnchor` (e.g., `RR-CANON-R173`)
- `rulesDocsLinks` (e.g., `RULES_CANONICAL_SPEC.md#173-stalemate`)

This allows VictoryModal or TeachingOverlay to link to a canonical source without embedding ad‑hoc URLs.

### 4.3 Primary concept

Set `primaryConceptId` when a specific high‑risk concept is the primary driver for confusion (e.g., `anm_fe_core`, `structural_stalemate`, `territory_mini_regions`). Use the concept ids from [`UX_RULES_CONCEPTS_INDEX.md`](UX_RULES_CONCEPTS_INDEX.md).

---

## 5. Examples

### 5.1 Ring‑elimination (no weird state)

```json
{
  "boardType": "square8",
  "numPlayers": 2,
  "winnerPlayerId": "P1",
  "outcomeType": "ring_elimination",
  "victoryReasonCode": "victory_ring_majority",
  "scoreBreakdown": {
    "P1": { "playerId": "P1", "eliminatedRings": 19, "territorySpaces": 4, "markers": 12 },
    "P2": { "playerId": "P2", "eliminatedRings": 6, "territorySpaces": 2, "markers": 10 }
  },
  "uxCopy": { "shortSummaryKey": "game_end.ring_elimination.short" }
}
```

### 5.2 LPS with ANM/FE context

```json
{
  "boardType": "square8",
  "numPlayers": 3,
  "winnerPlayerId": "P2",
  "outcomeType": "last_player_standing",
  "victoryReasonCode": "victory_last_player_standing",
  "primaryConceptId": "lps_real_actions",
  "weirdStateContext": {
    "reasonCodes": [
      "ANM_MOVEMENT_FE_BLOCKED",
      "FE_SEQUENCE_CURRENT_PLAYER",
      "LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS"
    ],
    "primaryReasonCode": "LAST_PLAYER_STANDING_EXCLUSIVE_REAL_ACTIONS",
    "rulesContextTags": ["anm_forced_elimination", "last_player_standing"],
    "teachingTopicIds": ["teaching.active_no_moves", "teaching.forced_elimination"]
  },
  "teaching": { "teachingTopics": ["teaching.active_no_moves"] },
  "uxCopy": {
    "shortSummaryKey": "game_end.lps.short",
    "detailedSummaryKey": "game_end.lps.with_anm_fe.detailed"
  }
}
```

### 5.3 Structural stalemate tiebreak

```json
{
  "boardType": "square19",
  "numPlayers": 3,
  "winnerPlayerId": "P3",
  "outcomeType": "structural_stalemate",
  "victoryReasonCode": "victory_structural_stalemate_tiebreak",
  "tiebreakSteps": [
    {
      "kind": "territory_spaces",
      "winnerPlayerId": "P3",
      "valuesByPlayer": { "P1": 70, "P2": 65, "P3": 72 }
    }
  ],
  "weirdStateContext": {
    "reasonCodes": ["STRUCTURAL_STALEMATE_TIEBREAK"],
    "primaryReasonCode": "STRUCTURAL_STALEMATE_TIEBREAK",
    "rulesContextTags": ["structural_stalemate"]
  },
  "uxCopy": {
    "shortSummaryKey": "game_end.structural_stalemate.short",
    "detailedSummaryKey": "game_end.structural_stalemate.tiebreak.detailed"
  }
}
```

---

## 6. Maintenance notes

- When adding new weird‑state reason codes, update both [`weirdStateReasons.ts`](../../src/shared/engine/weirdStateReasons.ts) and this spec.
- When new rules‑UX concepts are introduced, add them to [`UX_RULES_CONCEPTS_INDEX.md`](UX_RULES_CONCEPTS_INDEX.md) and use them as `primaryConceptId` values here.
- Keep `GameEndExplanation` examples in sync with the builder tests under `tests/unit/`.
