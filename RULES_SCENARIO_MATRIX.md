# RingRift Rules / FAQ → Test Scenario Matrix

**Purpose:** This file is the canonical map from the **rules documents** to the **Jest test suites**.

It answers:

- “For this rule / FAQ example, where is the corresponding test?”
- “Which parts of the rules are fully covered vs partially covered vs not yet encoded?”

It is meant to evolve alongside:

- `ringrift_complete_rules.md`
- `ringrift_compact_rules.md`
- `RULES_ANALYSIS_PHASE2.md`
- `tests/README.md`
- `CURRENT_STATE_ASSESSMENT.md` / `KNOWN_ISSUES.md`

Status legend:

- **COVERED** – there is at least one explicit, named scenario test for this rule/FAQ cluster.
- **PARTIAL** – behaviour is exercised indirectly (e.g. via mechanics/unit tests or broader scenarios), but there is not yet a direct, rule‑tagged scenario.
- **PLANNED** – no meaningful coverage yet; scenario suite still to be added.

> **Naming convention:** When adding new tests, prefer `describe` / `it` names that reference the rule or FAQ explicitly, e.g. `Q15_3_1_180_degree_reversal` or `Rules_11_2_LineReward_Option1VsOption2`.

---

## 0. Index by Rules Sections

This is a high‑level map from rule sections to clusters below.

| Rules section(s)                            | Cluster                                                           |
| ------------------------------------------- | ----------------------------------------------------------------- |
| §4.x, FAQ 15.2, FAQ 24                      | Turn sequence & forced elimination                                |
| §8.2–8.3, FAQ 2–3                           | Non‑capture movement & markers                                    |
| §9–10, FAQ 5–6, 9, 12, 14, 15.3.1–15.3.2    | Overtaking captures & chain patterns                              |
| §11, FAQ 7, 22                              | Line formation, graduated rewards, line ordering                  |
| §12, FAQ 10, 15, 20, 23                     | Territory disconnection, self‑elimination, border colour, chains  |
| §13, FAQ 11, 18, 21, 24                     | Victory conditions & stalemate ladder                             |
| §4.5, 9–12 (choice hooks), FAQ 7, 15, 22–23 | PlayerChoice flows (line order/reward, elimination, region order) |
| §4, 8–13 (via GameTrace + S‑invariant)      | Backend ↔ sandbox parity & progress invariant                    |

The following sections break these down in more detail.

---

## 1. Turn sequence & forced elimination

**Rules/FAQ:**

- `ringrift_complete_rules.md` §4.x (Turn Structure)
- Compact rules §2.2–2.3 (movement phase & forced elimination)
- FAQ 15.2 (flowchart of a turn), FAQ 24 (forced elimination when blocked)

| Coverage    | Scenario / intent                                                                  | Jest file(s)                                                                                                               | Engines | Notes                                                                                                                                                     |
| ----------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Full turn sequence including forced elimination and skipping dead players          | `tests/unit/GameEngine.turnSequence.scenarios.test.ts`                                                                     | Backend | Encodes multi‑player cases where some players are blocked with stacks and others are out of material; aligns with compact rules §2.2–2.3 and FAQ 15.2/24. |
| **COVERED** | Forced elimination & structural stalemate resolution (S‑invariant, no stacks left) | `tests/scenarios/ForcedEliminationAndStalemate.test.ts`                                                                    | Backend | Scenario‑style tests for forced elimination chains and the final stalemate ladder (converting rings in hand to eliminated rings).                         |
| **PARTIAL** | Per‑player action availability checks (`hasValidActions`, `resolveBlockedState…`)  | `tests/unit/GameEngine.aiSimulation.test.ts`, `tests/unit/GameEngine.aiSimulation.*.debug.test.ts`                         | Backend | AI simulations hit edge‑case blocked states and call `resolveBlockedStateForCurrentPlayerForTesting`; used as diagnostics.                                |
| **PARTIAL** | Sandbox turn structure ("place then move", mixed human/AI)                         | `tests/unit/ClientSandboxEngine.mixedPlayers.test.ts`, `tests/unit/ClientSandboxEngine.placementForcedElimination.test.ts` | Sandbox | Validates the unified place‑then‑move semantics and forced elimination in the client‑local engine.                                                        |

**Planned additions**

- **PLANNED:** Add a small, rule‑tagged matrix in `GameEngine.turnSequence.scenarios.test.ts` so each scenario is keyed explicitly to §4.x / FAQ 15.2 / FAQ 24.

---

## 2. Non‑capture movement & marker interaction

**Rules/FAQ:**

- `ringrift_complete_rules.md` §8.2–8.3
- Compact rules §3.1–3.2
- FAQ 2–3 (basic movement & markers)

| Coverage    | Scenario / intent                                                                      | Jest file(s)                                                                                             | Engines | Notes                                                                                                                                 |
| ----------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Minimum distance ≥ stack height; path blocking; landing restrictions (no enemy marker) | `tests/unit/RuleEngine.movementCapture.test.ts`                                                          | Backend | Core unit tests validate the movement geometry and blocking rules for non‑capture movement; used by both backend and sandbox engines. |
| **COVERED** | Marker creation, flipping opponent markers, collapsing own markers along paths         | `tests/unit/RuleEngine.movementCapture.test.ts`                                                          | Backend | Confirms departure marker placement, path‑marker flipping/collapse, and behaviour on landing into own marker (self‑elimination hook). |
| **COVERED** | Sandbox parity: movement, markers, mandatory move after placement                      | `tests/unit/ClientSandboxEngine.invariants.test.ts`, `tests/unit/ClientSandboxEngine.moveParity.test.ts` | Sandbox | Sandbox has dedicated parity/invariant checks for movement + S‑invariant; enforced via client‑local engine.                           |

**Planned additions**

- **COVERED:** `tests/unit/RuleEngine.movement.scenarios.test.ts` – minimum-distance and blocking examples for FAQ 2–3 on `square8`; extend with additional board types as needed.

---

## 3. Overtaking captures & chain patterns

**Rules/FAQ:**

- `ringrift_complete_rules.md` §9–10 (Overtaking Capture, Chain Overtaking)
- Compact rules §4.1–4.3
- FAQ 5–6, 9, 12, 14, 15.3.1–15.3.2 (180° reversal, cyclic captures)

| Coverage    | Scenario / intent                                                                                      | Jest file(s)                                                                                                           | Engines              | Notes                                                                                                             |
| ----------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Basic overtaking capture validation (geometry, capHeight checks, own/other stacks, markers along path) | `tests/unit/RuleEngine.movementCapture.test.ts`                                                                        | Backend (RuleEngine) | Validates single‑segment captures against the compact spec (segment geometry and capHeight constraints).          |
| **COVERED** | Chain capture enforcement, including 180° reversal and re‑capturing same target                        | `tests/unit/GameEngine.chainCapture.test.ts`                                                                           | Backend (GameEngine) | Exercises `chainCaptureState` and the engine‑driven continuation loop for several sequences, including reversals. |
| **COVERED** | PlayerChoice for capture direction (multi‑option chains), backend enumeration of follow‑ups            | `tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts`                                                          | Backend (GameEngine) | Uses `getCaptureOptionsFromPosition` + `CaptureDirectionChoice` to test multi‑branch chain captures.              |
| **COVERED** | WebSocket flow for capture_direction in multi‑branch scenarios                                         | `tests/unit/GameEngine.captureDirectionChoiceWebSocketIntegration.test.ts`                                             | Backend + WebSocket  | End‑to‑end test covering `CaptureDirectionChoice` over WebSockets and subsequent chain segments.                  |
| **COVERED** | Sandbox parity for chain capture, including deterministic capture selection in AI                      | `tests/unit/ClientSandboxEngine.chainCapture.test.ts`, `tests/unit/ClientSandboxEngine.aiMovementCaptures.test.ts`     | Sandbox              | Validates sandbox chain logic and AI’s deterministic selection (lexicographically smallest landing).              |
| **COVERED** | Complex, rule‑/FAQ‑style chain capture examples (180° reverse, cycles) on square boards                | `tests/scenarios/ComplexChainCaptures.test.ts`                                                                         | Backend (scenario)   | Scenario‑style tests that encode key FAQ 15.3.\* examples directly; reference for future Rust parity.             |
| **PARTIAL** | Hexagonal chain capture patterns (cross‑board comparison)                                              | `tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts`, `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts` | Backend (hex)        | Focused on hex‑specific geometry and multi‑layer stacks; more explicit FAQ‑tagged names would help.               |

**Planned additions**

- **PLANNED:** Add explicit `describe` blocks in `ComplexChainCaptures.test.ts` keyed to FAQ 15.3.1/15.3.2 text so the mapping is one‑to‑one.

---

## 4. Line formation & graduated rewards

**Rules/FAQ:**

- `ringrift_complete_rules.md` §11 (Lines & Graduated Rewards)
- Compact rules §5.1–5.3
- FAQ 7, 22

| Coverage    | Scenario / intent                                                                                    | Jest file(s)                                                                                                                         | Engines              | Notes                                                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Line detection & collapse for square and hex boards                                                  | `tests/unit/BoardManager.territoryDisconnection.test.ts` (indirect via line helpers), `tests/unit/ClientSandboxEngine.lines.test.ts` | Backend + Sandbox    | Sandbox suite validates line detection/processing in the client engine; BoardManager helpers are reused backend‑side. |
| **COVERED** | Graduated rewards: Option 1 vs Option 2 (collapse all + elimination vs min collapse, no elimination) | `tests/unit/GameEngine.lines.scenarios.test.ts`                                                                                      | Backend (GameEngine) | Tests specific cases where line length > required length and both options are available.                              |
| **COVERED** | AI‑driven choice for line rewards via Python service                                                 | `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`, `tests/unit/AIEngine.placementMetadata.test.ts`               | Backend + AI         | Confirms `LineRewardChoice` is forwarded to AI service with correct metadata, plus fallbacks.                         |
| **COVERED** | WebSocket‑driven line order and reward choices for human players                                     | `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`, `tests/unit/PlayerInteractionManager.test.ts`                  | Backend + WebSocket  | End‑to‑end coverage of line ordering + reward options over sockets.                                                   |

**Planned additions**

- **PLANNED:** Add a dedicated `LineReward` section in `GameEngine.lines.scenarios.test.ts` where each test name includes the corresponding rule section / FAQ (e.g. `Rules_11_3_OverlengthLine_ChooseOption2`).

---

## 5. Territory disconnection & chain reactions

**Rules/FAQ:**

- `ringrift_complete_rules.md` §12 (Area Disconnection & Collapse)
- Compact rules §6.1–6.4
- FAQ 10, 15, 20, 23

| Coverage    | Scenario / intent                                                                                      | Jest file(s)                                                                                                                         | Engines                | Notes                                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- | -------------------------------------------------------------------------------------------------------- |
| **COVERED** | Territory region discovery and isDisconnected flags (square8/square19/hex)                             | `tests/unit/BoardManager.territoryDisconnection.test.ts`, `tests/unit/BoardManager.territoryDisconnection.hex.test.ts`               | Backend (BoardManager) | Validates adjacency and region discovery semantics, including border/representation criteria.            |
| **COVERED** | Engine‑level processing of disconnected regions (collapse, elimination, self‑elimination prerequisite) | `tests/unit/GameEngine.territoryDisconnection.test.ts`, `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`                   | Backend (GameEngine)   | Ensures `canProcessDisconnectedRegion` and `processOneDisconnectedRegion` follow compact rules §6.3–6.4. |
| **COVERED** | Client sandbox parity for territory disconnection (square + hex)                                       | `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts`, `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts` | Sandbox                | Confirms sandbox uses same semantics; important for visual debugging via `/sandbox`.                     |
| **COVERED** | Region order PlayerChoice (choosing which disconnected region to process first)                        | `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`, `tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts`             | Backend + Sandbox      | Both backend and sandbox interaction paths for `RegionOrderChoice` are tested.                           |
| **COVERED** | Multi‑step territory chain reactions and self‑elimination prerequisite in composed scenarios           | `tests/unit/GameEngine.territory.scenarios.test.ts`, `tests/scenarios/LineAndTerritory.test.ts`                                      | Backend (scenario)     | Encodes combined line+territory steps and verifies that self‑elimination constraints are enforced.       |

**Planned additions**

- **PLANNED:** Explicit FAQ‑tagged examples in `territory.scenarios.test.ts` for Q15, Q20, Q23 with comments referencing the diagrams/positions from the rules doc.

---

## 6. Victory conditions & stalemate

**Rules/FAQ:**

- `ringrift_complete_rules.md` §13 (Victory Conditions), §7.4 (Stalemate Resolution)
- Compact rules §7.1–7.4, §9 (progress invariant)
- FAQ 11, 18, 21, 24

| Coverage    | Scenario / intent                                                                  | Jest file(s)                                                                           | Engines | Notes                                                                                                                     |
| ----------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Sandbox ring‑elimination and territory‑majority victories                          | `tests/unit/ClientSandboxEngine.victory.test.ts`                                       | Sandbox | Confirms that the local engine detects ring/territory victories per compact rules.                                        |
| **PARTIAL** | Backend victory reasons and final scores (winner, `gameResult.reason`, ratings)    | `tests/integration/FullGameFlow.test.ts`, `tests/unit/GameEngine.aiSimulation.test.ts` | Backend | AI‑vs‑AI flow ends with a terminal state; explicit rule/FAQ mapping is still thin.                                        |
| **PARTIAL** | Stalemate ladder priorities (territory > eliminated rings > markers > last action) | `tests/scenarios/ForcedEliminationAndStalemate.test.ts`                                | Backend | Scenario suite covers forced elimination and terminal states; tie‑break ordering could use more explicit, isolated tests. |

**Planned additions**

- **PARTIAL:** `tests/unit/GameEngine.victory.scenarios.test.ts` – backend ring‑elimination and territory‑control examples are implemented; add last‑player‑standing and stalemate tiebreaker scenarios to complete coverage.

---

## 7. PlayerChoice flows (engine, WebSocket, AI service, sandbox)

**Rules/FAQ:**

- `ringrift_complete_rules.md` §4.5, §10.3, §11–12 (places where choices are surfaced)
- PlayerChoice types: `LineOrderChoice`, `LineRewardChoice`, `RingEliminationChoice`, `RegionOrderChoice`, `CaptureDirectionChoice`
- FAQ 7 (line choice), 15 (region choice), 22–23 (line/territory details)

| Coverage    | Scenario / intent                                                                                    | Jest file(s)                                                                                                                                                  | Layer(s)                  | Notes                                                                                     |
| ----------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------- |
| **COVERED** | Core PlayerInteractionManager wiring (request/response lifecycle)                                    | `tests/unit/PlayerInteractionManager.test.ts`                                                                                                                 | Backend interaction layer | Validates registration, choice routing, and error paths.                                  |
| **COVERED** | WebSocket interaction handler (mapping `player_choice_required`/`player_choice_response` to manager) | `tests/unit/WebSocketInteractionHandler.test.ts`, `tests/unit/WebSocketServer.aiTurn.integration.test.ts`                                                     | Backend + WebSocket       | Covers human and AI choice flows via sockets.                                             |
| **COVERED** | AIInteractionHandler & AIEngine service calls for line reward, ring elimination, region order        | `tests/unit/AIInteractionHandler.test.ts`, `tests/unit/AIEngine.serviceClient.test.ts`, `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts` | Backend + AI service      | Confirms service usage + fallbacks and ensures options metadata matches the compact spec. |
| **COVERED** | CaptureDirectionChoice for multi‑branch chains (backend + WebSocket)                                 | `tests/unit/GameEngine.captureDirectionChoice.test.ts`, `tests/unit/GameEngine.captureDirectionChoiceWebSocketIntegration.test.ts`                            | Backend + WebSocket       | Ties capture direction choices back into the chain capture loop.                          |
| **COVERED** | Sandbox choice flows for lines, region order, elimination                                            | `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`, `tests/unit/ClientSandboxEngine.lines.test.ts`, `tests/unit/ClientSandboxEngine.victory.test.ts`  | Sandbox                   | Exercises local AI/human choices in the client‑local engine.                              |

**Planned additions**

- **PLANNED:** Explicit rule/FAQ references in the choice‑centric tests (e.g. note in `regionOrderChoice` tests which FAQ disconnection example they encode).

---

## 8. Backend ↔ sandbox parity & progress invariant

**Rules/FAQ:**

- Compact rules §9 (S invariant), progress commentary in `ringrift_compact_rules.md` §9
- `RULES_ANALYSIS_PHASE2.md` §4 (Progress & Termination)

| Coverage    | Scenario / intent                                                                                                  | Jest file(s)                                                                                                  | Engines           | Notes                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------- |
| **COVERED** | Trace parity between sandbox AI games and backend replays (semantic comparison of moves & phases)                  | `tests/unit/Backend_vs_Sandbox.traceParity.test.ts`, `tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts` | Backend + Sandbox | Uses `GameTrace` and `tests/utils/traces.ts` to compare step‑by‑step state.                 |
| **COVERED** | AI‑parallel debug runs (backend & sandbox) for seeded games, including mismatch logging and S‑snapshot comparisons | `tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts`, `tests/utils/traces.ts`                              | Backend + Sandbox | Heavy diagnostic harness; some seeds are tracked as known P0.2 issues.                      |
| **PARTIAL** | Sandbox AI simulation S‑invariant and stall detection                                                              | `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`, `tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts` | Sandbox           | Used as diagnostics; not CI‑blocking; ensures S is non‑decreasing and flags stalling seeds. |

**Planned additions**

- **PLANNED:** When P0.2 semantic gaps are resolved, promote a subset of parity tests to **hard CI gates** and add a short, rule‑tagged comment block at the top of each parity test file referencing compact rules §9.

---

## 9. How to extend this matrix

1. **When adding a new scenario test:**
   - Decide which rule/FAQ it encodes.
   - Add a row under the appropriate cluster table here.
   - Include:
     - Rule/FAQ references.
     - A brief description.
     - The Jest file path and, if useful, the `describe`/`it` name.
     - Engine(s) it exercises (Backend, Sandbox, WebSocket, AI service).
     - Coverage status: start as **PARTIAL**; upgrade to **COVERED** when the rule is clearly and directly asserted.

2. **When discovering a rules gap or bug:**
   - Add a **PLANNED** row for the missing scenario.
   - Link to any `KNOWN_ISSUES.md` entries.
   - Once fixed and tested, update status to **COVERED**.

3. **When modifying rules docs:**
   - If a rule section is changed or a FAQ is added/removed, scan this matrix for references to keep them in sync.

This file, together with `tests/README.md`, should be treated as the **single source of truth** for how RingRift’s formal rules map to concrete, executable tests.
