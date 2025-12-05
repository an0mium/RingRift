# UX Rules Copy Spec (HUD & Sandbox)

> Canonical UX copy for RingRift client surfaces that explain rules in HUD, Sandbox, TeachingOverlay, Onboarding, and curated scenarios. This spec is semantics-first and must remain aligned with RR‑CANON and the current engine.

## 1. Purpose and scope

This document defines **canonical UX copy** for rules-related client surfaces. It is the single reference for:

- HUD victory conditions and ring/territory stats.
- Sandbox phase copy (movement, capture, chain, lines, territory).
- TeachingOverlay topics for movement, capture, chains, lines, territory, and victory.
- Onboarding victory summary.
- Curated sandbox scenarios' `rulesSnippet` text.

Semantics must always match:

- [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md:1)
- [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1)
- [`ringrift_compact_rules.md`](ringrift_compact_rules.md:1)
- [`docs/ACTIVE_NO_MOVES_BEHAVIOUR.md`](docs/ACTIVE_NO_MOVES_BEHAVIOUR.md:1)
- [`docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md`](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md:1)

## 2. Terminology & invariants

**Rings vs markers vs territory**

- _Ring_ – a physical piece on a stack. Still “in play” unless eliminated.
- _Marker_ – a marker token on a space (used to form lines). Lines are **markers only**, never rings.
- _Territory space_ – a collapsed space you permanently own. Territory victory counts territory spaces.

**Elimination vs capture**

- _Captured ring_ – taken by overtaking capture and moved to the **bottom** of your stack. It **stays in play** and does **not** directly count toward elimination victory.
- _Eliminated ring_ – permanently removed from play and credited to the player who caused it (via movement onto markers, line rewards, territory processing, forced elimination, etc).

**Global elimination threshold (R060–R061)**

- Let `totalRingsInGame` be the total ring supply at start.
- The global elimination threshold is `floor(totalRingsInGame / 2) + 1`.
- A player wins by **Ring Elimination** when their credited eliminated rings **exceed 50% of all rings in the game**, not just one opponent.

**Territory threshold (R062, R140–R145)**

- Let `totalSpaces` be the board’s cell count.
- Territory victory threshold is `floor(totalSpaces / 2) + 1`.
- Territory victory uses **territory spaces**, not raw markers or stacks.

**Real moves vs forced elimination (R070–R072, R170–R173)**

- _Real moves_ are placements, movements, and captures.
- Forced elimination and automatic line/territory processing are **not** real moves for LPS.
- Last‑Player‑Standing is determined only by availability of real moves across a full round.

## 3. Victory conditions copy

This section defines the canonical labels and one‑liners used by:

- [`VictoryConditionsPanel`](src/client/components/GameHUD.tsx:843)
- HUD summary rows in [`GameHUD.GameHUDFromViewModel()`](src/client/components/GameHUD.tsx:984)
- Victory messaging in [`VictoryModal`](src/client/components/VictoryModal.tsx:365) and [`toVictoryViewModel()`](src/client/adapters/gameViewModels.ts:1311)

### 3.1 Ring Elimination

**HUD label**

- `"Ring Elimination"`

**HUD one‑liner**

- `"Win by eliminating more than half of all rings in play."`

**Tooltip (multi‑line)**

- Line 1: `"You win Ring Elimination when the rings you have eliminated exceed 50% of all rings in the game."`
- Line 2: `"Only *eliminated* rings count – captured rings you carry in stacks still remain in play."`
- Line 3: `"Eliminations can come from movement onto markers, line rewards, territory processing, or forced elimination."`

**TeachingOverlay victory topic – elimination**
Short description (used in [`TeachingOverlay.TEACHING_CONTENT.victory_elimination`](src/client/components/TeachingOverlay.tsx:96)):

- `"Win by eliminating more than half of all rings in the game – not just one opponent’s set. Eliminated rings are permanently removed; captured rings you carry in stacks do not count toward this threshold."`

### 3.2 Territory Control

**HUD label**

- `"Territory Control"`

**HUD one‑liner**

- `"Win by controlling more than half of all board spaces as territory."`

**Tooltip (multi‑line)**

- Line 1: `"Territory spaces are collapsed cells you permanently own."`
- Line 2: `"If your territory spaces exceed 50% of all board spaces during territory processing, you win immediately."`
- Line 3: `"Claiming a region usually requires eliminating rings from a stack you control outside that region (mandatory self‑elimination cost)."`

**TeachingOverlay victory topic – territory**
Description (used in [`TeachingOverlay.TEACHING_CONTENT.victory_territory`](src/client/components/TeachingOverlay.tsx:107)):

- `"Win by owning more than half of all board spaces as Territory. Territory comes from collapsing marker lines and resolving disconnected regions, and once a space becomes Territory it can’t be captured back."`

### 3.3 Last Player Standing (LPS)

**HUD label**

- `"Last Player Standing"`

**HUD one‑liner**

- `"Win if, after a full round, you are the only player with any real moves (placements, movements, or captures)."`

**Tooltip (multi‑line)**

- Line 1: `"Real moves are placements, movements, and captures – forced elimination does not count."`
- Line 2: `"If for a full round only you have any real moves available, you win by Last Player Standing."`

**TeachingOverlay victory topic – stalemate / LPS**
Description (used in [`TeachingOverlay.TEACHING_CONTENT.victory_stalemate`](src/client/components/TeachingOverlay.tsx:118)):

- `"Last Player Standing happens when, after a full round of turns, you are the only player who can still make real moves (placements, movements, or captures). Forced eliminations and automatic territory processing do not prevent LPS."`

## 4. Movement semantics

**Canonical rule (R090–R092)**

- A stack with height `H` moves in a straight line.
- It must move **at least** `H` spaces and may move farther if the path stays legal.
- The path cannot pass through other stacks or collapsed territory spaces; markers do not block movement.
- The landing cell must not contain a stack or collapsed territory; it may contain a marker.
- If you land on a marker, the top ring of your moving stack is eliminated and credited to you.

**TeachingOverlay – Stack Movement description**
Used in [`TeachingOverlay.TEACHING_CONTENT.stack_movement`](src/client/components/TeachingOverlay.tsx:36):

- `"Move a stack you control (your ring on top) in a straight line at least as many spaces as the stack’s height. You can keep going farther as long as the path has no stacks or territory spaces blocking you; markers are allowed and may eliminate your top ring when you land on them."`

**Sandbox phase copy – Movement**
Used in [`SandboxGameHost.PHASE_COPY.movement.summary`](src/client/pages/SandboxGameHost.tsx:238):

- `"Pick a stack and move it in a straight line at least as far as its height, and farther if the path stays clear (stacks and territory block; markers do not)."`

## 5. Capture and chain capture semantics

**Basic overtaking capture (R101–R102)**

- Capture is always a **jump** over exactly one target stack along a straight line to a landing cell.
- Your stack’s cap height must be ≥ the target stack’s cap height.
- The path from origin to landing may not cross other stacks or collapsed territory spaces (other than the single target stack).
- The landing cell must be empty or contain markers (no stack, no collapsed territory).
- You remove the **top ring** from the target stack and place it on the **bottom** of your stack: captured rings stay in play.

**Chain capture (R103)**

- Starting a capture is **optional**.
- Once you take a capture segment, chain continuation is **mandatory**: if another capture exists from your new position, you must keep capturing.
- When multiple capture directions are available, you choose which capture to take next.
- The chain ends only when no legal capture segments remain.

**TeachingOverlay – Capturing description**
Used in [`TeachingOverlay.TEACHING_CONTENT.capturing`](src/client/components/TeachingOverlay.tsx:48):

- `"To capture, jump over an adjacent opponent stack in a straight line and land on the empty space just beyond it. You take the top ring from the jumped stack and add it to the bottom of your own stack. Captured rings stay in play – only later *eliminations* move rings out of the game."`

**TeachingOverlay – Chain Capture description**
Used in [`TeachingOverlay.TEACHING_CONTENT.chain_capture`](src/client/components/TeachingOverlay.tsx:60):

- `"If your capturing stack can jump again after a capture, you are in a chain capture. Starting the first capture is optional, but once the chain begins you must keep capturing as long as any capture is available. When several jumps exist, you choose which target to take next."`

**Sandbox phase copy – Capture / Chain Capture**
Used in [`SandboxGameHost.PHASE_COPY.capture`](src/client/pages/SandboxGameHost.tsx:244) and [`SandboxGameHost.PHASE_COPY.chain_capture`](src/client/pages/SandboxGameHost.tsx:249):

- Capture phase label: `"Capture"`
- Capture summary: `"Start an overtaking capture by jumping over an adjacent stack and landing on an empty or marker space beyond it."`
- Chain Capture label: `"Chain Capture"`
- Chain Capture summary: `"Continue the capture chain: you must keep jumping while any capture exists, but you choose which capture direction to take next."`

## 6. Lines and rewards

**Core semantics (R120–R122)**

- Lines are formed by **markers**, not rings.
- A contiguous run of your markers of length ≥ the board’s configured `lineLength` is a completed line.
- **Exact‑length line:** all markers in the line must collapse into your territory, and you must eliminate one ring (or an entire cap) from a stack you control.
- **Overlength line:** you choose between:
  - (A) Collapse the whole line into territory **and** eliminate a ring/cap, or
  - (B) Collapse only a contiguous segment of length `lineLength` into territory and **skip elimination**.

**TeachingOverlay – Lines description**
Used in [`TeachingOverlay.TEACHING_CONTENT.line_bonus`](src/client/components/TeachingOverlay.tsx:72):

- `"Lines are built from your markers. When a straight line of your markers reaches the minimum length for this board, it becomes a scoring line: you collapse markers in that line into permanent Territory and, on many boards, must pay a ring elimination cost from a stack you control."`

**TeachingOverlay – Lines tips**
Replace any legacy “ring to hand” text with:

- `"Exact‑length lines always collapse fully into Territory and usually require you to eliminate a ring from one of your stacks."`
- `"Overlength lines can trade safety for value: you may collapse a shorter scoring segment with no elimination, or collapse the full line and pay the ring cost."`

**Curated scenarios – line‑focused rulesSnippet template**
For scenarios like `learn.lines.formation.Rules_11_2_Q7_Q20` in [`curated.json`](src/client/public/scenarios/curated.json:175):

- `"When a contiguous line of your markers reaches the minimum scoring length, it immediately becomes a line to resolve. You choose how to reward each completed line: collapse markers into permanent Territory, and on many boards pay a small ring‑elimination cost from one of your stacks. Rings are never returned to hand as a line reward."`

## 7. Territory and disconnected regions

**Core semantics (R140–R145)**

- Territory processing examines **disconnected regions** of stacks and markers for each player.
- When you process a region you control:
  - All spaces in that region become your Territory spaces (collapsed).
  - All rings in the region are eliminated and credited to you.
  - You must eliminate rings from a stack you control **outside** the region to pay the cost, if required by board config.
- Territory victory is checked whenever territory processing completes and your territory spaces exceed the threshold.

**TeachingOverlay – Territory description**
Used in [`TeachingOverlay.TEACHING_CONTENT.territory`](src/client/components/TeachingOverlay.tsx:84):

- `"Territory spaces are collapsed cells that you permanently own. When a disconnected region of your pieces is processed, all of its spaces become your Territory and its rings are eliminated, often at the cost of eliminating a ring from one of your other stacks. If your Territory passes more than half of the board, you win immediately."`

**Sandbox phase copy – Territory Processing**
Used in [`SandboxGameHost.PHASE_COPY.territory_processing.summary`](src/client/pages/SandboxGameHost.tsx:258):

- `"Resolve disconnected regions into permanent Territory, eliminating rings in that region and paying any required self‑elimination cost from your other stacks."`

## 8. Sandbox phase‑copy summary (movement, capture, chains, lines, territory)

The canonical short summaries used in [`SandboxGameHost.PHASE_COPY`](src/client/pages/SandboxGameHost.tsx:227):

- **Ring Placement – label**: `"Ring Placement"`
  - summary: `"Place new rings or add to existing stacks while keeping at least one real move available for your next turn."`
- **Movement – label**: `"Movement"`
  - summary: `"Pick a stack and move it in a straight line at least as far as its height, and farther if the path stays clear (stacks and territory block; markers do not)."`
- **Capture – label**: `"Capture"`
  - summary: `"Start an overtaking capture by jumping over an adjacent stack and landing on an empty or marker space beyond it."`
- **Chain Capture – label**: `"Chain Capture"`
  - summary: `"Continue the capture chain: you must keep jumping while any capture exists, but you choose which capture direction to take next."`
- **Line Processing – label**: `"Line Completion"`
  - summary: `"Resolve completed marker lines into Territory and choose whether to take or skip any ring‑elimination reward."`
- **Territory Processing – label**: `"Territory Claim"`
  - summary: `"Evaluate disconnected regions, collapse them into Territory, and pay any required self‑elimination cost; territory wins are checked here."`

## 9. Surface‑ID to component mapping

This section ties each text block in this spec to its implementation surface.

### 9.1 HUD & victory surfaces

- `hud.victory.ring_elimination.label` → [`VictoryConditionsPanel`](src/client/components/GameHUD.tsx:843) elimination row label.
- `hud.victory.ring_elimination.tooltip` → [`VictoryConditionsPanel`](src/client/components/GameHUD.tsx:843) elimination tooltip.
- `hud.victory.territory.label` and `.tooltip` → same component, territory row.
- `hud.victory.lps.label` and `.tooltip` → same component, Last Player Standing row.
- `hud.stats.rings_eliminated.label` →
  - [`RingStats`](src/client/components/GameHUD.tsx:405) elimination label.
  - [`RingStatsFromVM`](src/client/components/GameHUD.tsx:552) elimination label.
  - [`CompactScoreSummary`](src/client/components/GameHUD.tsx:933) "Rings Eliminated" label.
- `victory_modal.ring_elimination.description` → [`getVictoryMessage()`](src/client/adapters/gameViewModels.ts:1408) branch for `ring_elimination`.
- `victory_modal.table.rings_eliminated.header` → [`FinalStatsTable`](src/client/components/VictoryModal.tsx:108) elimination column header.

### 9.2 Teaching & onboarding surfaces

- `teaching.ring_placement` → [`TEACHING_CONTENT.ring_placement`](src/client/components/TeachingOverlay.tsx:23).
- `teaching.stack_movement` → [`TEACHING_CONTENT.stack_movement`](src/client/components/TeachingOverlay.tsx:36).
- `teaching.capturing` → [`TEACHING_CONTENT.capturing`](src/client/components/TeachingOverlay.tsx:48).
- `teaching.chain_capture` → [`TEACHING_CONTENT.chain_capture`](src/client/components/TeachingOverlay.tsx:60).
- `teaching.lines` → [`TEACHING_CONTENT.line_bonus`](src/client/components/TeachingOverlay.tsx:72).
- `teaching.territory` → [`TEACHING_CONTENT.territory`](src/client/components/TeachingOverlay.tsx:84).
- `teaching.victory_elimination` → [`TEACHING_CONTENT.victory_elimination`](src/client/components/TeachingOverlay.tsx:96).
- `teaching.victory_territory` → [`TEACHING_CONTENT.victory_territory`](src/client/components/TeachingOverlay.tsx:107).
- `teaching.victory_stalemate` → [`TEACHING_CONTENT.victory_stalemate`](src/client/components/TeachingOverlay.tsx:118).
- `onboarding.victory.elimination` → Ring Elimination card in [`OnboardingModal.VictoryStep`](src/client/components/OnboardingModal.tsx:76).

### 9.3 Sandbox & curated scenarios

- `sandbox.phase.movement.summary` → [`PHASE_COPY.movement.summary`](src/client/pages/SandboxGameHost.tsx:238).
- `sandbox.phase.capture.summary` → [`PHASE_COPY.capture.summary`](src/client/pages/SandboxGameHost.tsx:244).
- `sandbox.phase.chain_capture.summary` → [`PHASE_COPY.chain_capture.summary`](src/client/pages/SandboxGameHost.tsx:249).
- `sandbox.phase.line_processing.summary` → [`PHASE_COPY.line_processing.summary`](src/client/pages/SandboxGameHost.tsx:254).
- `sandbox.phase.territory_processing.summary` → [`PHASE_COPY.territory_processing.summary`](src/client/pages/SandboxGameHost.tsx:258).
- `scenario.learn.movement.basics.rulesSnippet` → `learn.movement.basics` in [`curated.json`](src/client/public/scenarios/curated.json:59).
- `scenario.learn.capture.chain.rulesSnippet` → `learn.capture.chain` in [`curated.json`](src/client/public/scenarios/curated.json:117).
- `scenario.learn.lines.formation.rulesSnippet` → `learn.lines.formation.Rules_11_2_Q7_Q20` in [`curated.json`](src/client/public/scenarios/curated.json:175).

All changes to HUD, TeachingOverlay, OnboardingModal, SandboxGameHost, and curated scenarios **must** remain consistent with this document and the canonical rules references in §0.
