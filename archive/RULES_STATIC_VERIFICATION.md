# RingRift Rules Static Verification Report

## 1. Introduction & Methodology

This report statically verifies the current implementation of the RingRift rules engine
against the canonical rules defined in [`../RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md)
and the mapping in [`../docs/rules/RULES_IMPLEMENTATION_MAPPING.md`](../docs/rules/RULES_IMPLEMENTATION_MAPPING.md).

### 1.1 Scope and Inputs

- Canonical rules: RR-CANON R001–R191 in [`../RULES_CANONICAL_SPEC.md`](../RULES_CANONICAL_SPEC.md).
- Implementation mapping: [`../docs/rules/RULES_IMPLEMENTATION_MAPPING.md`](../docs/rules/RULES_IMPLEMENTATION_MAPPING.md).
- Shared TypeScript engine (primary rules semantics), including helpers such as
  [`TypeScript.core`](../src/shared/engine/core.ts),
  [`TypeScript.movementLogic`](../src/shared/engine/movementLogic.ts),
  [`TypeScript.captureLogic`](../src/shared/engine/captureLogic.ts),
  [`TypeScript.lineDetection`](../src/shared/engine/lineDetection.ts),
  [`TypeScript.lineDecisionHelpers`](../src/shared/engine/lineDecisionHelpers.ts),
  [`TypeScript.territoryDetection`](../src/shared/engine/territoryDetection.ts),
  [`TypeScript.territoryProcessing`](../src/shared/engine/territoryProcessing.ts),
  [`TypeScript.territoryDecisionHelpers`](../src/shared/engine/territoryDecisionHelpers.ts),
  [`TypeScript.turnLogic`](../src/shared/engine/turnLogic.ts),
  `TypeScript.victoryLogic`.
- Server orchestration:
  [`TypeScript.BoardManager`](../src/server/game/BoardManager.ts),
  [`TypeScript.GameEngine`](../src/server/game/GameEngine.ts),
  [`TypeScript.RuleEngine`](../src/server/game/RuleEngine.ts),
  [`TypeScript.advanceGameForCurrentPlayer`](../src/server/game/turn/TurnEngine.ts),
  line helpers in `TypeScript.lineProcessing`,
  territory helpers in `TypeScript.territoryProcessing`.
- Client sandbox:
  [`TypeScript.ClientSandboxEngine`](../src/client/sandbox/ClientSandboxEngine.ts),
  sandbox turn engine in `TypeScript.sandboxTurnEngine`,
  and other sandbox helpers under `src/client/sandbox`.

### 1.2 Methodology

- Use [`../docs/rules/RULES_IMPLEMENTATION_MAPPING.md`](../docs/rules/RULES_IMPLEMENTATION_MAPPING.md) as an index, but
  derive semantics directly from code in the shared engine, backend orchestration, and sandbox.
- Treat shared-engine helpers as the primary rules implementation; backend and sandbox should
  be thin wrappers around these.
- For each RR-CANON rule cluster:
  - Identify primary functions/modules that realise the semantics.
  - Verify preconditions, legality checks, and data transformations.
  - Verify order-of-operations and invariants (exclusivity of stacks/markers/collapsed, ring conservation).
  - Classify each rule’s implementation as:
    - Exact / faithful implementation.
    - Minor divergence / benign deviation.
    - Major divergence / likely defect.
- Check cross-host consistency:
  - Backend vs sandbox vs Python AI/rules integration where applicable.

### 1.3 Rule Clusters

- Board & entities (R001–R003, R020–R023, R030–R031, R040–R041, R050–R052, R060–R062).
- Turn / phases / forced elimination (R070–R072, R100).
- Placement & skip (R080–R082).
- Non-capture movement & markers (R090–R092).
- Overtaking capture & chains (R101–R103).
- Lines & graduated rewards (R120–R122).
- Territory disconnection & region processing (R140–R145).
- Victory, termination, randomness & S-invariant (R170–R173, R190–R191).

---

## 2. Cluster-by-Cluster Static Verification

Each subsection documents:

- RR-CANON rules covered.
- Primary implementation references.
- Verification summary.
- Detailed findings by rule or theme.
- Divergences and severity / impact.

### 2.1 Board & entities (R001–R003, R020–R023, R030–R031, R040–R041, R050–R052, R060–R062)

#### 2.1.1 Primary implementation references

- Board and game types, including `BoardType`, `BoardState`, `GameState`, territory and line
  structures: [`../src/shared/types/game.ts`](../src/shared/types/game.ts).
- Geometry and distance helpers (`calculateDistance`, `getPathPositions`, movement directions):
  [`TypeScript.core`](../src/shared/engine/core.ts).
- Backend board manager and geometry:
  [`TypeScript.BoardManager`](../src/server/game/BoardManager.ts)
  (including `createBoard`, `generateValidPositions`, adjacency helpers, and board invariants).
- Sandbox board creation and geometry:
  [`TypeScript.ClientSandboxEngine`](../src/client/sandbox/ClientSandboxEngine.ts)
  (methods `createEmptyBoard`, `isValidPosition`, `isCollapsedSpace`, marker/stack helpers).
- Territory adjacency / region detection:
  [`TypeScript.territoryDetection`](../src/shared/engine/territoryDetection.ts).

#### 2.1.2 Verification summary

- **Board geometry and coordinate ranges**: The implementation correctly realises the
  square-8, square-19, and hex-11 boards and uses appropriate adjacency (Moore, Von Neumann,
  hex) for movement, lines, and territory. This is consistent across shared engine, backend,
  and sandbox.
- **Exclusivity of occupancy states (stack / marker / collapsed)**: Backend and sandbox both
  enforce mutual exclusivity between stacks, markers, and collapsed spaces, with explicit
  invariant checks and repairs.
- **Ring conservation and per-player totals**: Ring placement, movement, captures, eliminations,
  and territory effects consistently update per-player and global ring counts in both backend
  and sandbox. The S-invariant is derived from these counts (see cluster 2.8).
- **Board “repair” behaviour** in [`TypeScript.BoardManager.assertBoardInvariants()`](../src/server/game/BoardManager.ts)
  runs on each stack/marker/collapsed write and will silently remove overlapping markers in
  production as well as tests. This is intended as a guard against bugs, not part of core
  rules semantics, but it can hide the presence of illegal intermediate states.

Overall, the Board & entities cluster is implemented **faithfully**, with a **minor diagnostic
divergence** around automatic repair of stack/marker overlaps.

#### 2.1.3 Detailed findings

##### R001–R003: Board types, size, and coordinate system

- Shared types define three board types with explicit sizes and total spaces in
  [`../src/shared/types/game.ts`](../src/shared/types/game.ts):
  - `square8`: 8×8 grid, Moore adjacency for movement, specific line-length and territory config.
  - `square19`: 19×19 grid, analogous configuration.
  - `hexagonal`: axial/cube coordinates with radius `size - 1` (e.g. size 11 → radius 10).
- Backend coordinate validity:
  - [`TypeScript.BoardManager.generateValidPositions()`](../src/server/game/BoardManager.ts)
    constructs:
    - For `hexagonal`: standard hex disk using cube coordinates with distance ≤ radius.
    - For square: `0 ≤ x,y < size` grid.
  - [`TypeScript.BoardManager.isValidPosition()`](../src/server/game/BoardManager.ts) checks
    membership in this set.
- Sandbox coordinate validity:
  - [`TypeScript.ClientSandboxEngine.isValidPosition()`](../src/client/sandbox/ClientSandboxEngine.ts)
    replicates exactly the same geometry and distance conditions using `BOARD_CONFIGS`.

**Assessment**: Exact / faithful implementation.

##### R020–R023: Adjacency definitions

- Adjacency types live in `AdjacencyType` and `BOARD_CONFIGS` in
  [`../src/shared/types/game.ts`](../src/shared/types/game.ts):
  - `movementAdjacency` vs `lineAdjacency` vs `territoryAdjacency` are configured per board.
- Backend adjacency helpers:
  - [`TypeScript.BoardManager.getNeighbors()`](../src/server/game/BoardManager.ts) dispatches
    to:
    - `getHexagonalNeighbors` (6-neighbour hex).
    - `getMooreNeighbors` (8-neighbour square).
    - `getVonNeumannNeighbors` (4-neighbour square).
  - [`TypeScript.BoardManager.buildAdjacencyGraph()`](../src/server/game/BoardManager.ts)
    precomputes adjacency graph using `territoryAdjacency`, in line with R141.
  - [`TypeScript.BoardManager.getAdjacentPositions()`](../src/server/game/BoardManager.ts)
    provides another adjacency helper with explicit `AdjacencyType` parameter.
- Shared engine movement directions:
  - [`TypeScript.getMovementDirectionsForBoardType()`](../src/shared/engine/core.ts)
    returns:
    - 8 Moore directions for squares.
    - 6 cube directions for hex.
- Territory detection:
  - [`TypeScript.territoryDetection.findDisconnectedRegions()`](../src/shared/engine/territoryDetection.ts)
    uses `BOARD_CONFIGS[board.type].territoryAdjacency` via a shared helper, fully aligned
    with `BoardManager`’s adjacency graph.

**Assessment**: Exact / faithful implementation of adjacency for movement, lines, and territory.

##### R030–R031, R040–R041: Entities and exclusive occupancy

- Shared `BoardState` in [`../src/shared/types/game.ts`](../src/shared/types/game.ts) has:
  - `stacks: Map<string, RingStack>`
  - `markers: Map<string, MarkerInfo>`
  - `collapsedSpaces: Map<string, number>`
  - `territories`, `formedLines`, `eliminatedRings`.
- Backend invariants:
  - [`TypeScript.BoardManager.assertBoardInvariants()`](../src/server/game/BoardManager.ts)
    enforces:
    1. No stacks on collapsed spaces.
    2. No stack+marker coexistence.
    3. No marker+collapsed coexistence.
  - It **first performs a “repair pass”** (lines 97–123):
    - Deletes any marker that overlaps a stack.
    - Deletes any marker on a collapsed space.
    - Logs a diagnostic.
    - Then runs stricter checks, throwing only in strict/test mode.
  - Stack/marker/collapsed writes go through:
    - [`TypeScript.BoardManager.setMarker()`](../src/server/game/BoardManager.ts) – refuses to
      place on collapsed territory and deletes any stack at that key before placing the marker.
    - [`TypeScript.BoardManager.collapseMarker()`](../src/server/game/BoardManager.ts) and
      [`TypeScript.BoardManager.setCollapsedSpace()`](../src/server/game/BoardManager.ts) –
      delete stacks and markers before marking collapsed.
    - [`TypeScript.BoardManager.setStack()`](../src/server/game/BoardManager.ts) – logs and
      deletes any marker at the key before writing the stack.
  - Sandbox equivalents:
    - [`TypeScript.ClientSandboxEngine.setMarker()`](../src/client/sandbox/ClientSandboxEngine.ts),
      [`TypeScript.ClientSandboxEngine.collapseMarker()`](../src/client/sandbox/ClientSandboxEngine.ts),
      and direct manipulation of `board.stacks`/`board.markers` mirror backend semantics and
      enforce the same exclusivity.
    - Test-only invariant helper
      [`TypeScript.ClientSandboxEngine.assertBoardInvariants()`](../src/client/sandbox/ClientSandboxEngine.ts)
      checks for overlaps and throws under `NODE_ENV === 'test'`.

**Assessment**: Exact occupancy exclusivity semantics for all normal rules paths.

**Minor divergence**:

- The **automatic repair step** in
  [`TypeScript.BoardManager.assertBoardInvariants()`](../src/server/game/BoardManager.ts) runs in
  all environments (not gated on `BOARD_INVARIANTS_STRICT`) and can silently delete markers if an
  illegal overlap is created by a bug. This behaviour is _not_ described in RR-CANON (those
  states are supposed to be unreachable). It is a debugging safeguard, but if triggered in live
  games it will deterministically favour stacks over markers.

**Severity**: Low for game semantics (only affects already-illegal states), Medium for debugging
(violations may be “repaired” rather than surfaced as hard failures in production).

##### R050–R052, R060–R062: Ring counts, conservation, and tracking

- Global and per-player ring counts:
  - `GameState.totalRingsInPlay`, `totalRingsEliminated`, `players[*].ringsInHand`,
    `players[*].eliminatedRings`, and `board.eliminatedRings` live in
    [`../src/shared/types/game.ts`](../src/shared/types/game.ts).
- Placement:
  - Backend placement in [`TypeScript.GameEngine.applyMove()`](../src/server/game/GameEngine.ts)
    for `place_ring`:
    - Adds `placementCount` rings for the moving player to top of a stack (or creates a new stack).
    - Decrements `player.ringsInHand` by `placementCount`, clamped at zero.
    - Does **not** change `totalRingsInPlay` (it is treated as the initial total).
- Movement and capture:
  - Movement helpers (`move_stack`/`move_ring`) in
    [`TypeScript.GameEngine.applyMove()`](../src/server/game/GameEngine.ts) re-arrange rings
    between stacks using `boardManager.setStack()`/`removeStack()` with no creation/destruction.
  - Overtaking capture in [`TypeScript.GameEngine.performOvertakingCapture()`](../src/server/game/GameEngine.ts):
    - Moves a single captured ring from target stack to bottom of attacker stack.
    - Adjusts target stack or deletes it.
    - Does not alter elimination counts.
- Elimination:
  - Line/territory-related elimination and forced/explicit elimination increment:
    - `gameState.totalRingsEliminated`.
    - `board.eliminatedRings[player]`.
    - `player.eliminatedRings`.
  - Examples:
    - [`TypeScript.GameEngine.eliminateFromStack()`](../src/server/game/GameEngine.ts),
      [`TypeScript.GameEngine.eliminateTopRingAt()`](../src/server/game/GameEngine.ts),
      line-processing elimination in
      `TypeScript.lineProcessing.eliminateFromStack()`,
      territory processing in
      [`TypeScript.GameEngine.processDisconnectedRegionCore()`](../src/server/game/GameEngine.ts)
      and `TypeScript.territoryProcessing.processOneDisconnectedRegion()`.
- Sandbox:
  - Mirrors the same placement, capture, and elimination logic via:
    - [`TypeScript.ClientSandboxEngine.tryPlaceRings()`](../src/client/sandbox/ClientSandboxEngine.ts),
    - capture application in
      ~~`TypeScript.sandboxMovementEngine.applyCaptureSegmentWithHooks()`~~ (legacy, removed; capture chains now orchestrated via [`TypeScript.ClientSandboxEngine.performCaptureChainInternal`](../src/client/sandbox/ClientSandboxEngine.ts)),
    - elimination helpers such as
      [`TypeScript.ClientSandboxEngine.forceEliminateCap()`](../src/client/sandbox/ClientSandboxEngine.ts).

**Assessment**: Ring conservation and accounting are faithfully implemented along all live
backend and sandbox rules paths.

#### 2.1.4 Divergences and severity for Board & entities

- **Exact / faithful implementation**
  - Board geometry and adjacency (R001–R003, R020–R023).
  - Occupancy exclusivity for stacks/markers/collapsed (R030–R031, R040–R041).
  - Ring conservation and accounting (R050–R052, R060–R062) for placement, movement, capture,
    line and territory effects.

- **Minor divergence / benign deviation**
  - **Board repair semantics**:
    - Location:
      [`TypeScript.BoardManager.assertBoardInvariants()`](../src/server/game/BoardManager.ts).
    - RR-CANON IDs: R030–R031, R040–R041 (exclusivity rules).
    - Description: Violations of exclusivity are “repaired” by deleting markers in the presence
      of stacks/collapsed spaces, even in non-test environments. The canonical rules treat such
      states as unreachable; the engine defines a deterministic repair policy.
    - Severity: Low for gameplay (only affects already-invalid states), Medium for debugging
      (might hide root causes in production logs rather than failing hard).

- **Major divergence / likely defect**
  - None identified in this cluster.

**Cluster severity & impact**: **Low** – The core rules for board geometry, entity types,
exclusivity, and ring conservation are implemented faithfully. The only deviation is in
diagnostic repair behaviour for illegal states.

---

### 2.2 Turn / phases / forced elimination (R070–R072, R100)

#### 2.2.1 Primary implementation references

- Shared turn logic state machine:
  [`TypeScript.advanceTurnAndPhase`](../src/shared/engine/turnLogic.ts).
- Backend turn engine wrapper and delegates:
  [`TypeScript.advanceGameForCurrentPlayer`](../src/server/game/turn/TurnEngine.ts),
  [`TypeScript.TurnEngine.updatePerTurnStateAfterMove()`](../src/server/game/turn/TurnEngine.ts),
  forced elimination helper
  [`TypeScript.turn.processForcedElimination()`](../src/server/game/turn/TurnEngine.ts).
- Backend game engine integration:
  - Per-turn placement state:
    [`TypeScript.GameEngine.updatePerTurnStateAfterMove()`](../src/server/game/GameEngine.ts).
  - High-level turn advance:
    [`TypeScript.GameEngine.advanceGame()`](../src/server/game/GameEngine.ts).
  - Test-only blocked-state resolver:
    [`TypeScript.GameEngine.resolveBlockedStateForCurrentPlayerForTesting()`](../src/server/game/GameEngine.ts).
- Backend rules move enumeration and validation:
  [`TypeScript.RuleEngine.getValidMoves()`](../src/server/game/RuleEngine.ts),
  including:
  - Placement enumeration
    [`TypeScript.RuleEngine.getValidRingPlacements()`](../src/server/game/RuleEngine.ts),
  - Movement enumeration
    [`TypeScript.RuleEngine.getValidStackMovements()`](../src/server/game/RuleEngine.ts),
  - Capture enumeration
    [`TypeScript.RuleEngine.getValidCaptures()`](../src/server/game/RuleEngine.ts),
  - Skip-placement validation
    [`TypeScript.RuleEngine.validateSkipPlacement()`](../src/server/game/RuleEngine.ts).
- Sandbox turn engine and forced elimination:
  `TypeScript.sandboxTurnEngine.startTurnForCurrentPlayerSandbox()`,
  `TypeScript.sandboxTurnEngine.maybeProcessForcedEliminationForCurrentPlayerSandbox()`,
  `TypeScript.sandboxTurnEngine.advanceTurnAndPhaseForCurrentPlayerSandbox()`.

#### 2.2.2 Verification summary

- **Turn phase ordering (R070–R071)**:
  - Shared [`TypeScript.advanceTurnAndPhase`](../src/shared/engine/turnLogic.ts) defines a
    canonical phase flow:
    - `ring_placement` → `movement` (when any placement/movement/capture is available)
      or directly to `line_processing` when movement/capture are impossible.
    - `movement`, `capture`, and `chain_capture` always advance to `line_processing`.
    - `line_processing` always advances to `territory_processing`.
    - `territory_processing` ends the turn and advances to the next player’s starting phase.
  - Backend [`TypeScript.advanceGameForCurrentPlayer`](../src/server/game/turn/TurnEngine.ts)
    and sandbox
    `TypeScript.advanceTurnAndPhaseForCurrentPlayerSandbox()`
    both delegate to this shared state machine, differing only in how they implement
    delegates and side-effect hooks.
  - This matches RR-CANON’s phase ladder: placement → movement/capture → line → territory →
    next player.

- **Forced elimination eligibility (R072)**:
  - The shared engine consults delegates `hasAnyPlacement`, `hasAnyMovement`, and
    `hasAnyCapture` to determine whether a player is **completely blocked**.
  - Backend delegates:
    - `hasValidPlacements()` in
      [`TypeScript.TurnEngine.hasValidPlacements`](../src/server/game/turn/TurnEngine.ts)
      uses `RuleEngine.getValidMoves()` from a ring_placement view and checks for any
      `place_ring` **or** `skip_placement` move.
    - `hasValidMovements()` and `hasValidCaptures()` use `RuleEngine.getValidMoves()` from
      `movement` and `capture` views respectively, filtered by `mustMoveFromStackKey` when
      applicable.
  - Sandbox delegates:
    - `hasAnyMovement` / `hasAnyCapture` in
      `TypeScript.sandboxTurnEngine.advanceTurnAndPhaseForCurrentPlayerSandbox()`
      use reachability via `hasAnyLegalMoveOrCaptureFrom`.
    - `maybeProcessForcedEliminationForCurrentPlayerSandbox()` implements an additional
      guard that checks for both placements (respecting ring caps and no-dead-placement) and
      moves/captures before applying forced elimination.
  - In both backend and sandbox, **forced elimination is only applied when**:
    - The player controls stacks, and
    - They have no legal placement (place_ring or skip_placement in backend, no legal
      placement positions in sandbox), **and**
    - They have no legal movement/capture from any stack.

- **Forced elimination behaviour (R100)**:
  - Backend forced elimination is executed by
    [`TypeScript.turn.processForcedElimination()`](../src/server/game/turn/TurnEngine.ts).
    - Chooses a stack among the player’s stacks:
      - Prefers stacks with `capHeight > 0`.
      - Among these, selects the stack with the **smallest** capHeight.
      - If no cap exists (degenerate tests), uses the first stack.
    - Calls `hooks.eliminatePlayerRingOrCap(playerNumber, bestStack.position)`, which
      backend wires to
      [`TypeScript.GameEngine.eliminatePlayerRingOrCap()`](../src/server/game/GameEngine.ts),
      eliminating the entire cap and updating elimination counters.
    - After elimination, victory is re-evaluated via `RuleEngine.checkGameEnd()` and
      `hooks.endGame()`.
  - Sandbox forced elimination:
    - Uses `forceEliminateCapOnBoard` in
      [`TypeScript.ClientSandboxEngine.forceEliminateCap()`](../src/client/sandbox/ClientSandboxEngine.ts)
      and the same cap semantics in
      `TypeScript.sandboxTurnEngine.maybeProcessForcedEliminationForCurrentPlayerSandbox()`.

**Assessment**: The phase ladder and forced elimination logic match R070–R072 and R100 closely
for both backend and sandbox. The main differences are in **surface-level representation** of
optional placement (backend exposes a `skip_placement` Move; sandbox does not) and some
sandbox heuristics for starting phase selection. These are minor to medium risk from a
parity/UX standpoint but do not represent a direct violation of canonical legality semantics.

#### 2.2.3 Detailed findings

##### R070–R071: Phase ordering and interactions

- Shared phase logic in [`TypeScript.turnLogic`](../src/shared/engine/turnLogic.ts):
  - From `ring_placement`:
    - Delegates decide whether movement/capture are available after placement:
      - If any `hasAnyMovement` or `hasAnyCapture` returns true, the next phase becomes
        `movement`.
      - If not, phase skips directly to `line_processing`.
  - From `movement`, `capture`, `chain_capture`:
    - Always advance to `line_processing` once the move (and any capture chain) is done.
  - From `line_processing`:
    - Always advance to `territory_processing`.
  - From `territory_processing`:
    - Ends the turn for the current player, then:
      - Uses `getNextPlayerNumber` to propose the next player.
      - For that player, uses `hasAnyPlacement`/`hasAnyMovement`/`hasAnyCapture` and
        `getPlayerStacks` to determine whether forced elimination applies or normal
        turn start occurs.
- Backend integration via [`TypeScript.GameEngine.advanceGame()`](../src/server/game/GameEngine.ts):
  - Passes `eliminatePlayerRingOrCap` and `endGame` hooks into
    [`TypeScript.advanceGameForCurrentPlayer`](../src/server/game/turn/TurnEngine.ts).
  - After `advanceTurnAndPhase`, GameEngine may run
    `stepAutomaticPhasesForTesting()` to skip automatic bookkeeping phases when no
    decisions exist (for parity harnesses).
- Sandbox integration:
  - Rather than calling `advanceTurnAndPhase` directly at start-of-turn, the sandbox
    uses:
    - `TypeScript.sandboxTurnEngine.startTurnForCurrentPlayerSandbox()`
      to:
      - Re-check victory.
      - Loop over players, applying `maybeProcessForcedEliminationForCurrentPlayerSandbox`.
      - Set `currentPhase` to `ring_placement` if the player has rings in hand, otherwise
        `movement`.
    - `TypeScript.sandboxTurnEngine.advanceTurnAndPhaseForCurrentPlayerSandbox()`
      to run the shared state machine when the sandbox has explicitly set
      `currentPhase` to `'territory_processing'` at the end of automatic consequences.

**Assessment**: Backend and shared-engine ordering is fully consistent with RR-CANON.
Sandbox reuses the same core state machine but wraps it with additional heuristics for
start-of-turn; those heuristics appear aligned with the rules but are more brittle
(see deviations below).

##### R072: Eligibility for forced elimination

- Backend eligibility check:
  - Defined by the shared `advanceTurnAndPhase` via delegates:
    - `hasAnyPlacement` from
      [`TypeScript.TurnEngine.hasValidPlacements`](../src/server/game/turn/TurnEngine.ts):
      - Constructs a ring_placement view of the state and calls
        [`TypeScript.RuleEngine.getValidMoves()`](../src/server/game/RuleEngine.ts).
      - Returns true if any move of type `place_ring` **or** `skip_placement` exists.
    - `hasAnyMovement` from
      [`TypeScript.TurnEngine.hasValidMovements`](../src/server/game/turn/TurnEngine.ts):
      - Constructs a movement-phase view and uses `RuleEngine.getValidMoves`.
      - Filters for `move_stack`/`move_ring`/`build_stack`.
      - Respects `mustMoveFromStackKey` in the same way as GameEngine’s
        `getValidMoves`.
    - `hasAnyCapture` from
      [`TypeScript.TurnEngine.hasValidCaptures`](../src/server/game/turn/TurnEngine.ts):
      - Similar pattern, with phase forced to `capture` and filtering for
        `overtaking_capture`.
  - Forced elimination is applied in the shared engine only when:
    - The player has at least one stack (via `getPlayerStacks`).
    - **No placement, movement, or capture moves** exist in the views above.
- Sandbox eligibility:
  - In core turn advancement (phase transitions) the sandbox defers to
    [`TypeScript.advanceTurnAndPhase`](../src/shared/engine/turnLogic.ts) via
    `TypeScript.sandboxTurnEngine.advanceTurnAndPhaseForCurrentPlayerSandbox()`,
    with `applyForcedElimination` wired to sandbox elimination and victory checks.
  - At the _start_ of a player’s turn, sandbox explicitly calls
    `TypeScript.sandboxTurnEngine.maybeProcessForcedEliminationForCurrentPlayerSandbox()`:
    - Computes stacks for the current player.
    - Approximates ring cap usage in the same way as the sandbox AI (counting all rings in
      controlled stacks).
    - Uses `hasAnyLegalMoveOrCaptureFrom` to detect whether any stack can move/capture.
    - Uses `enumerateLegalRingPlacements` for no-dead-placement-aware placements.
    - If there are no placements (subject to caps) and no moves/captures but stacks exist,
      forced elimination is triggered via `hooks.forceEliminateCap`, else the turn starts
      normally.
  - This matches the RR-CANON requirement “forced elimination only when all legal placements,
    moves, and captures are absent” for all observable cases.

**Assessment**: Forced elimination **eligibility** is implemented faithfully in both hosts.
Backend uses `skip_placement` as a legal no-op action only when placements are optional,
which is consistent with the rules. Sandbox uses no-dead-placement-aware placement
enumeration and movement reachability.

##### R100: Forced elimination behaviour and crediting

- Backend behaviour:
  - [`TypeScript.turn.processForcedElimination()`](../src/server/game/turn/TurnEngine.ts):
    - Computes player stacks via `BoardManager.getPlayerStacks`.
    - Returns immediately if the player has no stacks (no place to eliminate from).
    - Otherwise:
      - Searches for a stack with `capHeight > 0` and minimal cap.
      - Falls back to the first stack if no caps are found.
      - Calls `hooks.eliminatePlayerRingOrCap(playerNumber, bestStack.position)`.
  - [`TypeScript.GameEngine.eliminatePlayerRingOrCap()`](../src/server/game/GameEngine.ts) and
    [`TypeScript.GameEngine.eliminateFromStack()`](../src/server/game/GameEngine.ts):
    - Eliminate the **entire cap** (all consecutive top rings of controlling color).
    - Update `totalRingsEliminated`, `board.eliminatedRings[player]`, and
      `player.eliminatedRings`.
    - Remove the stack if it becomes empty; otherwise recompute stack’s `capHeight` and
      controlling player.

- Sandbox behaviour:
  - Forced elimination in
    [`TypeScript.ClientSandboxEngine.forceEliminateCap()`](../src/client/sandbox/ClientSandboxEngine.ts)
    uses `forceEliminateCapOnBoard`, which:
    - Computes capHeight with `calculateCapHeight`.
    - Eliminates the entire cap from a chosen stack.
    - Updates elimination counters in `board.eliminatedRings`, `players[*].eliminatedRings`,
      and `totalRingsEliminated`.
  - Turn-level orchestration:
    - `TypeScript.sandboxTurnEngine.maybeProcessForcedEliminationForCurrentPlayerSandbox()`
      chooses when to call `forceEliminateCap` and then advances to the next player (with
      reset per-turn state).

**Assessment**: Forced elimination consistently removes a full cap and credits eliminations
to the blocked player in both backend and sandbox, matching R100.

##### Additional backend helper: resolveBlockedStateForCurrentPlayerForTesting

- [`TypeScript.GameEngine.resolveBlockedStateForCurrentPlayerForTesting()`](../src/server/game/GameEngine.ts)
  is a **test-only** safety helper used when tests detect an impossible state:
  - If in an interactive phase and `getValidMoves()` returns no moves, it:
    1. Attempts to hand the turn to a player with a legal action.
    2. If none exist, applies repeated forced eliminations until:
       - No stacks remain (bare-board stalemate), or
       - A player has a legal action.
    3. Uses S-invariant-inspired bounds to avoid infinite loops.
  - This helper is not used in live backend flows; it is safe as a diagnostic tool and
    explicitly references invariant/stalemate rules.

**Assessment**: Test-only helper, no divergence for production rules.

#### 2.2.4 Divergences and severity for Turn / phases / forced elimination

- **Exact / faithful implementation**
  - Phase ordering and transition conditions in the shared engine and backend:
    - R070–R071: placement → movement/capture → line → territory → next turn.
  - Forced elimination eligibility:
    - R072: Only when no legal placements, movements, or captures exist.
  - Forced elimination behaviour:
    - R100: Eliminates an entire cap, credited to the blocked player.

- **Minor divergence / benign deviation**
  - **Absence of explicit `skip_placement` in the sandbox**:
    - Location: Sandbox does not expose a `skip_placement` Move; instead, it treats
      placement as a direct board mutation via
      [`TypeScript.ClientSandboxEngine.tryPlaceRings()`](../src/client/sandbox/ClientSandboxEngine.ts)
      and uses `enumerateLegalRingPlacements` for AI.
    - RR-CANON IDs: R070–R072, R080 (placement optionality).
    - Description:
      - Backend expresses “optional placement” as presence of both `place_ring` and
        `skip_placement` Moves in ring_placement.
      - Sandbox never encodes skip_placement as a Move; instead, when no placements are
        possible but movement is available, the intended way forward is to start movement
        and ignore placement entirely.
      - In practice, sandbox phase management still allows the AI and UI to move even when
        `ringsInHand > 0` and no legal placements exist, but this is accomplished through
        heuristic flow rather than explicit no-op moves.
    - Severity: Low–Medium.
      - Low for core legality (sandbox does not permit illegal placements).
      - Medium for parity/tracing: canonical logs/harnesses that expect a `skip_placement`
        Move will not see one in sandbox traces.

- **Potential medium-risk divergence (sandbox start-of-turn phase selection)**
  - Location:
    `TypeScript.sandboxTurnEngine.startTurnForCurrentPlayerSandbox()`.
  - RR-CANON IDs: R070–R072.
  - Description:
    - After applying `maybeProcessForcedEliminationForCurrentPlayerSandbox`, the sandbox
      chooses the starting phase solely based on `ringsInHand`:
      - If `ringsInHand > 0` → `ring_placement`.
      - Else → `movement`.
    - This is slightly different from backend semantics, where the shared state machine
      can conceptually detect “no legal placements” and then rely on movement/capture
      phases and `skip_placement` as appropriate.
    - In pathological positions where:
      - `ringsInHand > 0`, but
      - **no** legal placements pass no-dead-placement and ring-cap constraints,
      - and movement/capture is available,
        the sandbox would still report `currentPhase === 'ring_placement'`. Human
        interaction code (`handleHumanCellClick`) only implements actions for
        `ring_placement` and `movement` separately, which could make these turns
        awkward to play manually (AI still finds moves via movement hooks).
  - Severity: Medium (sandbox-only).
    - Does not violate core legality, but **can create UX/parity surprises** for sandbox
      users and tools that interpret `currentPhase` literally.
    - Mitigation: tests should ensure such edge cases either cannot occur or are handled
      by advancing to `movement` when no placements exist.

**Cluster severity & impact**: **Medium (sandbox), Low (backend)** – Backend faithfully
implements the canonical phase ladder and forced elimination semantics. Sandbox semantics
are equivalent from a legality perspective but encode optional placement and phase choice
less explicitly, which can affect parity tooling and rare UX edge cases.

---

### 2.3 Placement & skip (R080–R082)

#### 2.3.1 Primary implementation references

- Shared placement validator:
  - [`TypeScript.validatePlacementOnBoard()`](../src/shared/engine/validators/PlacementValidator.ts).
  - [`TypeScript.validatePlacement()`](../src/shared/engine/validators/PlacementValidator.ts).
  - [`TypeScript.validateSkipPlacement()`](../src/shared/engine/validators/PlacementValidator.ts).
- Backend RuleEngine and move enumeration:
  - [`TypeScript.RuleEngine.validateRingPlacement()`](../src/server/game/RuleEngine.ts).
  - [`TypeScript.RuleEngine.getValidRingPlacements()`](../src/server/game/RuleEngine.ts).
  - [`TypeScript.RuleEngine.validateSkipPlacement()`](../src/server/game/RuleEngine.ts).
  - Ring_placement branch of
    [`TypeScript.RuleEngine.getValidMoves()`](../src/server/game/RuleEngine.ts).
- Backend GameEngine placement application:
  - `place_ring` and `skip_placement` in
    [`TypeScript.GameEngine.applyMove()`](../src/server/game/GameEngine.ts).
  - Per-turn placement state in
    [`TypeScript.GameEngine.updatePerTurnStateAfterMove()`](../src/server/game/GameEngine.ts).
- Sandbox placement:
  - Hypothetical placement + no-dead-placement:
    [`TypeScript.sandboxPlacement.createHypotheticalBoardWithPlacement()`](../src/client/sandbox/sandboxPlacement.ts),
    [`TypeScript.sandboxPlacement.hasAnyLegalMoveOrCaptureFrom()`](../src/client/sandbox/sandboxPlacement.ts).
  - Placement enumeration:
    [`TypeScript.sandboxPlacement.enumerateLegalRingPlacements()`](../src/client/sandbox/sandboxPlacement.ts).
  - Human/AI application path:
    [`TypeScript.ClientSandboxEngine.tryPlaceRings()`](../src/client/sandbox/ClientSandboxEngine.ts),
    `handleHumanCellClick` in
    [`TypeScript.ClientSandboxEngine`](../src/client/sandbox/ClientSandboxEngine.ts),
    AI hooks in `maybeRunAITurnSandbox`.

#### 2.3.2 Verification summary

- The shared validator [`TypeScript.validatePlacementOnBoard()`](../src/shared/engine/validators/PlacementValidator.ts)
  encodes canonical placement semantics:
  - Board geometry > valid positions only.
  - No placement on collapsed spaces or markers.
  - Per-player ring cap and rings-in-hand constraints.
  - Single-ring-per-action onto existing stacks; 1–3 rings onto empty cells.
  - No-dead-placement: resulting stack must have at least one legal move or capture via
    [`TypeScript.hasAnyLegalMoveOrCaptureFromOnBoard()`](../src/shared/engine/core.ts).
- Backend RuleEngine and GameEngine use this shared validator for:
  - Placement validation in `validateRingPlacement`.
  - Placement enumeration in `getValidRingPlacements`.
  - Legality of `place_ring` Moves in `getValidMoves`.
- Skip-Placement semantics (R080):
  - Shared validator `validateSkipPlacement` and backend
    [`TypeScript.RuleEngine.validateSkipPlacement()`](../src/server/game/RuleEngine.ts) align:
    - Only during `ring_placement`.
    - Player must have rings in hand.
    - Player must control at least one stack.
    - At least one controlled stack must have any legal move or capture from the resulting board.
- Sandbox placement helpers (`enumerateLegalRingPlacements` when provided with a
  `PlacementContext`) delegate to the same `validatePlacementOnBoard`, giving sandbox AI and
  parity tooling **identical** placement legality to the backend.
- Sandbox human interaction logic in `tryPlaceRings` reproduces the same constraints:
  - No placement on collapsed spaces or markers.
  - At most one ring onto an existing stack per action; arbitrary count on empty but gated by
    rings in hand.
  - No-dead-placement by explicitly building a hypothetical board and calling
    `hasAnyLegalMoveOrCaptureFrom`.

Overall, the placement & skip rules (R080–R082) are implemented **faithfully** across the
shared engine and backend. Sandbox is semantically aligned but models `skip_placement` and
phase choice more implicitly (as already noted under 2.2).

#### 2.3.3 Detailed findings

##### R080: Mandatory vs optional vs forbidden placement; skip_placement

- Core rules in shared placement validator:
  - `validatePlacementOnBoard` enforces:
    - Valid target position on board.
    - No placement on collapsed territory (`board.collapsedSpaces.has(posKey)`).
    - No placement on markers (`board.markers.has(posKey)`).
    - Respect of global capacity:
      - `ringsPerPlayerCap` and `ringsInHand` combine into `maxAvailableGlobal`.
    - Per-cell cap:
      - `perCellCap = 1` for existing stacks.
      - `perCellCap = 3` for empty cells.
    - Requested `count` must be between 1 and `maxPlacementCount`.
    - No-dead-placement: resulting hypothetical stack must have at least one legal
      move/capture via `hasAnyLegalMoveOrCaptureFromOnBoard`.
- Backend optional vs mandatory vs forbidden semantics:
  - **Mandatory placement**:
    - When in `ring_placement` and
      - `RuleEngine.getValidRingPlacements` returns at least one `place_ring`, and
      - `validateSkipPlacement` rejects `skip_placement` (e.g. player has no controlled
        stacks or no stack has legal move/capture),
      - then `RuleEngine.getValidMoves` exposes only `place_ring` options.
    - TurnEngine’s `hasValidPlacements` sees at least one `place_ring` and **may** also
      see `skip_placement` if it is legal; otherwise only `place_ring`.
  - **Optional placement**:
    - `validateSkipPlacement` in shared validator and backend allows `skip_placement` when:
      - Phase is `ring_placement`.
      - Player has rings in hand.
      - Player controls at least one stack.
      - At least one controlled stack has any legal move or capture via
        `hasAnyLegalMoveOrCaptureFromOnBoard`.
    - When both `place_ring` and `skip_placement` exist in `getValidMoves`, placement is
      optional. AI uses `chooseLocalMoveFromCandidates` in
      [`TypeScript.localAIMoveSelection`](../src/shared/engine/localAIMoveSelection.ts) to
      randomly choose between placement and non-placement options, weighted by counts.
  - **Forbidden placement**:
    - `validatePlacementOnBoard` rejects placements that:
      - Violate capacity (`NO_RINGS_AVAILABLE`).
      - Target collapsed spaces or markers.
      - Use invalid counts (on existing stacks or empty cells).
      - Would create a dead stack (`NO_LEGAL_MOVES`).
    - When **no** placements pass validation and `validateSkipPlacement` still passes (player
      has rings and controlled stacks with moves), `getValidMoves` exposes _only_
      `skip_placement`. In this case, the **only legal action in ring_placement** is to skip,
      and the engine transitions to movement/capture.
- Sandbox:
  - Does not encode `skip_placement` as a Move. Instead:
    - `ClientSandboxEngine.tryPlaceRings` is the only placement entry point.
    - If placement is impossible (e.g. no-dead-placement violation), it returns `false` and
      leaves board and phase unchanged; the caller (AI or UI) then proceeds to movement.
    - `enumerateLegalRingPlacements` with a `PlacementContext` yields exactly the same
      legal placement cells as `validatePlacementOnBoard`, and is used by:
      - AI helpers in `maybeRunAITurnSandbox`.
      - Diagnostic and parity harnesses.

**Assessment**: Semantics of mandatory vs optional vs forbidden placement are fully consistent
with R080 in the backend and shared engine. Sandbox is behaviourally equivalent but does not
expose a first-class `skip_placement` Move.

##### R081–R082: Multi-ring placement and stack placement; no-dead-placement

- Multi-ring placement rules:
  - Shared validator:
    - On existing stacks: `perCellCap = 1`, and any `requestedCount > 1` yields
      `INVALID_COUNT` with an explanatory `reason`.
    - On empty cells: `perCellCap = 3`, and `1 ≤ count ≤ 3` (subject to capacity).
  - Backend enumeration:
    - In [`TypeScript.RuleEngine.getValidRingPlacements()`](../src/server/game/RuleEngine.ts),
      for each position:
      - Determines if the cell is occupied (`isOccupied`).
      - Computes `maxPerPlacement = 1` on stacks, or `≤ 3` on empty cells, but also capped by
        `maxAvailableGlobal`.
      - Iterates `count` from 1..`maxPerPlacement` and calls
        `validatePlacementOnBoard(board, pos, count, baseCtx)`.
      - Only yields moves when validation succeeds.
  - Sandbox:
    - `tryPlaceRings`:
      - When placing onto an existing stack, caps effective count at 1
        (`maxPerPlacement = isOccupied ? 1 : maxFromHand`).
      - For empty cells, allows up to `maxFromHand` rings (no explicit per-cell cap), but
        because `BOARD_CONFIGS[boardType].ringsPerPlayer` typically equals the total rings
        in hand, this matches canonical behaviour in all realistic games.
      - No-dead-placement is enforced by building a hypothetical board using
        `createHypotheticalBoardWithPlacement` and using
        `hasAnyLegalMoveOrCaptureFrom` (which wraps `hasAnyLegalMoveOrCaptureFromOnBoard`).
- No-dead-placement guarantee:
  - Shared validator always constructs a hypothetical stack and calls
    [`TypeScript.hasAnyLegalMoveOrCaptureFromOnBoard()`](../src/shared/engine/core.ts) for the
    exact target position.
  - This helper:
    - Enumerates simple non-capture moves via movement directions and path constraints.
    - Enumerates capture moves via `enumerateCaptureMoves` and `validateCaptureSegmentOnBoard`.
    - Returns true if **any** legal move or capture exists from that stack.
  - Backend and sandbox use that same shared helper (either directly or via their own thin
    wrappers), so **any accepted placement is guaranteed to leave at least one legal move or
    capture from the resulting stack**.

**Assessment**: R081–R082 (stack placement and no-dead-placement) are implemented faithfully
in shared and backend code. Sandbox mirrors these constraints; the only difference is the
absence of an explicit per-cell 3-ring cap on empty cells, but that is effectively enforced
by initial ring supply in all supported board configs.

##### Per-player ring cap approximation

- `validatePlacementOnBoard` and
  [`TypeScript.RuleEngine.getValidRingPlacements()`](../src/server/game/RuleEngine.ts) both
  approximate “rings on board for this player” as the **sum of stack heights** for stacks
  where `controllingPlayer === player`.
- This may slightly over-approximate the number of rings of that colour in presence of
  mixed-colour stacks (captured rings of other players), but:
  - Each player starts with `ringsPerPlayer` rings of their colour in hand.
  - Placement only ever consumes that initial supply (no rings are created).
  - Thus, placements cannot actually exceed the physical ring inventory even when the cap
    computation is approximate.
- Effect on rules:
  - In pathological positions where a player has:
    - Few rings of their own colour on the board,
    - Many captured rings of other players in stacks they control,
    - And rings still in hand,
      the per-player cap could be considered reached earlier than strictly necessary,
      preventing additional placements that RR-CANON might still allow.
  - However, such positions are rare and already near ring-cap saturation.

**Assessment**: This is a **benign approximation** of “rings on board” that may be slightly
more restrictive than strictly required by RR-CANON but does not impact ring conservation.
Severity: Low.

#### 2.3.4 Divergences and severity for Placement & skip

- **Exact / faithful implementation**
  - Placement legality (on non-collapsed, non-marker cells).
  - Per-action multi-ring limits: 1 ring onto stacks; 1–3 onto empty cells (backend/shared).
  - No-dead-placement via `hasAnyLegalMoveOrCaptureFromOnBoard`.
  - Skip-Placement legality (only when rings in hand and at least one controlled stack with
    a legal move/capture).

- **Minor divergence / benign deviation**
  - **Skip-placement representation in sandbox**:
    - As discussed under 2.2, sandbox does not expose `skip_placement` as an explicit Move.
    - RR-CANON semantics (optional vs mandatory vs forbidden placement) are still respected
      in practice; the divergence is in how they are surfaced to clients/parity tools.
    - Severity: Low–Medium (mostly parity/UX).
  - **Per-player ring cap approximation**:
    - Counting all rings on stacks controlled by a player (including captured rings of other
      colours) as part of their cap.
    - This can only make placement _more_ restrictive, not less, and does not threaten ring
      conservation.
    - Severity: Low.

- **Major divergence / likely defect**
  - None identified in this cluster.

**Cluster severity & impact**: **Low–Medium** – Core placement semantics and no-dead-placement
are implemented faithfully. Slight differences in how optional placement and per-player caps
are modelled in sandbox vs backend are primarily parity/UX concerns.

---

### 2.4 Non-capture movement & markers (R090–R092)

_Detailed analysis to be filled in by subsequent revisions._

### 2.5 Overtaking capture & chains (R101–R103 + R100 interaction)

_Detailed analysis to be filled in by subsequent revisions._

### 2.6 Lines & graduated rewards (R120–R122)

_Detailed analysis to be filled in by subsequent revisions._

### 2.7 Territory disconnection & region processing (R140–R145)

_Detailed analysis to be filled in by subsequent revisions._

### 2.8 Victory, termination, randomness & S-invariant (R170–R173, R190–R191)

_Detailed analysis to be filled in by subsequent revisions._

---

## 3. Cross-Host Consistency (TS backend, sandbox, Python)

This section summarises how the three main hosts (TS backend, TS sandbox, and Python
GameEngine) align on the core turn/phase semantics, forced elimination, and the new
strict "no-move" invariant.

### 3.1 TS backend ↔ sandbox

The TypeScript backend and client sandbox are intentionally kept in close lockstep by:

- Sharing core helpers under `src/shared/engine/**` for movement, capture, lines,
  territory, turn logic, and victory.
- Wiring both hosts through the same `advanceTurnAndPhase` state machine in
  [`TypeScript.turnLogic`](../src/shared/engine/turnLogic.ts), with different delegate
  implementations but the same abstract contract:
  - From `ring_placement` → `movement` when placements/movements/captures exist, else
    `line_processing`.
  - From `movement` / `capture` / `chain_capture` → `line_processing`.
  - From `line_processing` → `territory_processing`.
  - From `territory_processing` → end-of-turn, rotating to the next player and applying
    forced elimination when appropriate.

Differences are primarily in representation and UX rather than core legality:

- **Skip-placement**: the backend exposes a first-class `skip_placement` Move when
  placement is optional, whereas the sandbox treats "skip" as simply not placing and
  proceeding directly to movement.
- **Start-of-turn phase selection**: the backend always runs through the shared
  `advanceTurnAndPhase` ladder, while the sandbox heuristically chooses
  `ring_placement` vs `movement` at the beginning of a turn based on `ringsInHand` and
  a forced-elimination pre-pass. This can create UX/parity edge cases but does not
  change which actions are ultimately legal for a given player.

From a rules-correctness perspective, both hosts:

- Use the same placement, movement, capture, line, and territory validators.
- Apply forced elimination only when a player controls stacks but has **no** legal
  placements, movements, or captures.
- Credit eliminations and territory consistently via shared helpers.

### 3.2 TS backend ↔ Python GameEngine (turns, forced elimination, strict invariant)

The Python `GameEngine` in `ai-service/app/game_engine.py`
implements a semantically equivalent rules engine for AI/training purposes. Its turn and
phase semantics are intentionally aligned with the TS backend:

- **Move generation**:
  - `GameEngine.get_valid_moves` mirrors `RuleEngine.getValidMoves`, including:
    - Phase-specific branches for `ring_placement`, `movement`, `capture`,
      `chain_capture`, `line_processing`, and `territory_processing`.
    - Per-turn `must_move_from_stack_key` behaviour identical to
      `TurnEngine.updatePerTurnStateAfterMove` and
      `GameEngine.updatePerTurnStateAfterMove` in TS.
- **Forced elimination eligibility**:
  - Python defines `_has_valid_placements`, `_has_valid_movements`, and
    `_has_valid_captures` which reuse the same placement/movement/capture generators
    that power `get_valid_moves`.
  - `_has_valid_actions` returns true if **any** of the above are available, exactly
    mirroring the TS TurnEngine delegates `hasValidPlacements`, `hasValidMovements`, and
    `hasValidCaptures`.
  - `_get_forced_elimination_moves` exposes a FORCED_ELIMINATION move only when the
    player controls at least one stack and `_has_valid_actions` returns false, matching
    the TS condition for `processForcedElimination`.
- **Forced elimination behaviour**:
  - `_apply_forced_elimination` eliminates the entire cap from the chosen stack using
    `_eliminate_top_ring_at`, updating `board.eliminated_rings`,
    `total_rings_eliminated`, and per-player `eliminated_rings` exactly like
    `GameEngine.eliminateFromStack` / `eliminateTopRingAt` in TS.
  - `_perform_forced_elimination_for_player`, invoked by `_end_turn`, picks one of the
    eligible forced-elimination moves and then re-checks victory, matching TS
    `processForcedElimination` + `checkGameEnd`.
- **Turn rotation and eliminated-player skipping**:
  - `_end_turn` rotates `current_player` through `players` in table order, skipping any
    player who has **no stacks and no rings in hand**. This matches the TS TurnEngine
    behaviour that effectively treats fully eliminated players as permanently
    inactive when searching for the next player with material.
  - For the first player with material:
    - If they have rings in hand, Python starts them in `RING_PLACEMENT`.
    - Otherwise, starts them in `MOVEMENT`, with an immediate forced-elimination pass
      if they have stacks but no actions. This matches TS
      `advanceGameForCurrentPlayer` delegates and their use inside
      `GameEngine.advanceGame`.

### 3.3 Python strict "no-move" invariant vs TS assumptions

The Python engine adds an explicit **strict invariant** that is conceptually present but
not enforced as a hard runtime check in TS:

> Any state with `game_status == ACTIVE` for `current_player` must offer at least one
> legal **action** for that player: either an interactive move or a forced-elimination
> step when the player still controls stacks.

This is implemented by `_assert_active_player_has_legal_action`, which is invoked at the
end of `GameEngine.apply_move` when the environment flag
`RINGRIFT_STRICT_NO_MOVE_INVARIANT` (and module-level `STRICT_NO_MOVE_INVARIANT`) are
enabled. The invariant is defined as:

- Compute `legal_moves = GameEngine.get_valid_moves(state, current_player)` for the
  current phase.
- Compute `forced_elims = _get_forced_elimination_moves(state, current_player)`.
- If `legal_moves` or `forced_elims` is non-empty, the invariant holds.
- If both are empty:
  - **If the current player has no stacks and no rings in hand**:
    - Call `_end_turn(state)` once to rotate to the next player with material,
      mirroring TS turn-skipping for fully eliminated players.
    - If the game ends during rotation, the invariant is considered satisfied.
    - Otherwise, recompute `legal_moves` and `forced_elims` for the new
      `current_player`, and only fail if both remain empty.
  - Otherwise (player still has material but no actions and no forced elimination is
    available), log an `active_no_moves_p*_*.json` snapshot and raise a `RuntimeError`.

This invariant corresponds to the TS expectation that the shared `advanceTurnAndPhase`
state machine **never** leaves an interactive or decision phase in a state where the
active player has neither moves nor forced elimination available, and that
fully-eliminated players are skipped when searching for the next player with material.
The Python invariant makes this contract **executable**:

- It surfaced historical bugs in:
  - Movement + forced elimination gating.
  - Line-processing phase exit logic.
  - Territory-processing and self-elimination sequences.
  - Late rotation away from fully eliminated players.
- Each unique structural pattern discovered by strict self-play soaks has been turned
  into a regression test under `ai-service/tests/invariants/` and
  `ai-service/tests/parity/`:
  - `test_active_no_moves_movement_forced_elimination_regression.py` (movement /
    forced-elimination entry).
  - `test_active_no_moves_movement_fully_eliminated_regression.py` (ACTIVE state
    with a fully eliminated current_player, now corrected by defensive turn
    rotation).
  - `test_active_no_moves_movement_placements_only_regression.py` (MOVEMENT phase
    with no movement/capture but legal ring placements, now treated as having a
    global action instead of a strict failure).
  - `test_active_no_moves_territory_processing_regression.py` (territory-processing
    with no follow-up decisions).
  - `test_active_no_moves_line_processing_regression.py` (line-processing phase exit).

Strict-invariant self-play soaks and the CI-friendly stability test in
`ai-service/tests/test_self_play_stability.py` now run with this invariant enabled by
default, providing a dynamic check that the Python implementation continues to respect
TS turn/phase semantics and eliminated-player skipping across 2p/3p/4p square boards and
hex configurations.

---

## 4. Discrepancies & Risks

_A consolidated list of issues (RR-CANON IDs, locations, severity, impact) will be compiled
after all clusters are populated. For now, see:_

- **Board & entities (2.1)**:
  - Board invariant repair deletes markers in illegal stack/marker or
    marker/collapsed overlaps (Low–Medium severity).
- **Turn / phases / forced elimination (2.2)**:
  - Sandbox lacks explicit `skip_placement` Moves and uses heuristics for phase
    selection (Medium sandbox-only severity).
- **Placement & skip (2.3)**:
  - Per-player ring cap approximated via total rings on stacks controlled by a player,
    including captured rings; slightly more restrictive than necessary (Low severity).

---

## 5. Recommendations for Code and Test Changes

_Detailed recommendations will follow once all clusters are populated. Early candidates:_

- Add explicit assertions or metrics to detect when
  [`TypeScript.BoardManager.assertBoardInvariants()`](../src/server/game/BoardManager.ts)
  performs repairs in production, and treat this as a serious diagnostic event.
- Consider adding a sandbox-level `skip_placement` representation (even if handled
  internally) or adjusting `startTurnForCurrentPlayerSandbox` to start in `movement`
  automatically when no legal placements exist, to better mirror backend semantics.
- Evaluate whether per-player ring-cap checks should distinguish between rings of a
  player’s own colour vs captured rings, or clarify the canonical spec to match the
  current implementation.
- Maintain the Python strict "no-move" invariant as a required property for all
  ACTIVE states, and continue to mine any new `active_no_moves_p*_*.json` snapshots
  into dedicated regression tests. Where a snapshot reveals a TS↔Python mismatch,
  treat the TS shared engine as canonical and adjust Python (or the invariant) to
  match its semantics.

---

## 4. Discrepancies & Risks

_A consolidated list of issues (RR-CANON IDs, locations, severity, impact) will be compiled
after all clusters are populated. For now, see:_

- **Board & entities (2.1)**:
  - Board invariant repair deletes markers in illegal stack/marker or
    marker/collapsed overlaps (Low–Medium severity).
- **Turn / phases / forced elimination (2.2)**:
  - Sandbox lacks explicit `skip_placement` Moves and uses heuristics for phase
    selection (Medium sandbox-only severity).
- **Placement & skip (2.3)**:
  - Per-player ring cap approximated via total rings on stacks controlled by a player,
    including captured rings; slightly more restrictive than necessary (Low severity).

---

## 5. Recommendations for Code and Test Changes

_Detailed recommendations will follow once all clusters are populated. Early candidates:_

- Add explicit assertions or metrics to detect when
  [`TypeScript.BoardManager.assertBoardInvariants()`](../src/server/game/BoardManager.ts)
  performs repairs in production, and treat this as a serious diagnostic event.
- Consider adding a sandbox-level `skip_placement` representation (even if handled
  internally) or adjusting `startTurnForCurrentPlayerSandbox` to start in `movement`
  automatically when no legal placements exist, to better mirror backend semantics.
- Evaluate whether per-player ring-cap checks should distinguish between rings of a
  player’s own colour vs captured rings, or clarify the canonical spec to match the
  current implementation.
