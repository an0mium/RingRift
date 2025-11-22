# RingRift Strategic Roadmap

**Version:** 3.0
**Last Updated:** November 22, 2025
**Status:** Engine/Rules Beta (Playable, not yet fully stable)
**Philosophy:** Robustness, Parity, and Scale

---

## 🎯 Executive Summary

**Current State:** Engine/Rules Beta (Core loop implemented; unified `chain_capture`/`continue_capture_segment` model live on the backend; shared TypeScript rules engine under `src/shared/engine/` established as the canonical rules source with an initial backend-vs-shared parity harness in [`RefactoredEngineParity.test.ts`](tests/unit/RefactoredEngineParity.test.ts:1); sandbox + AI parity and end-to-end termination still under active work—see [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md:1) and [`TODO.md`](TODO.md:73) for current P0 gaps).
**Goal:** Production-Ready Multiplayer Game
**Timeline:** 4-8 weeks to v1.0 (assuming P0 rules/parity/termination items are resolved first)
**Strategy:** Harden testing & parity → Polish UX → Expand Multiplayer Features

> Note on rules authority: when there is any question about what the correct
> behaviour _should_ be (for example, when parity harnesses or engines
> disagree), the ultimate source of canonical truth is the rules
> documentation—[`ringrift_complete_rules.md`](ringrift_complete_rules.md)
> and, where applicable, [`ringrift_compact_rules.md`](ringrift_compact_rules.md).
> Code, tests, and parity fixtures are expected to converge toward those
> documents rather than redefine the rules.

---

## 🚀 Strategic Phases

### **PHASE 1: Core Playability (COMPLETED)** ✅

- [x] Game Engine & Rules (Movement, Capture, Lines, Territory)
- [x] Board Management (8x8, 19x19, Hex)
- [x] Basic AI Integration (Service + Fallback)
- [x] Frontend Basics (Board, Lobby, HUD, Victory)
- [x] Infrastructure (Docker, DB, WebSocket)

### **PHASE 2: Robustness & Testing (IN PROGRESS)**

**Priority:** P0 - CRITICAL
**Goal:** Ensure 100% rule compliance and stability

#### 2.1 Comprehensive Scenario Testing

- [ ] Build test matrix for all FAQ edge cases (see `RULES_SCENARIO_MATRIX.md`)
- [ ] Implement scenario tests for complex chain captures (180° reversals, cycles)
- [ ] Verify all board types (especially Hexagonal edge cases)

#### 2.2 Sandbox Stage 2

- [x] Stabilize client-local sandbox with unified “place then move” turn semantics
      for both human and AI seats (including mixed games), and automatic local AI
      turns when it is an AI player’s move. Implemented in the browser-only
      sandbox via `ClientSandboxEngine` and the `/sandbox` path of `GamePage`,
      with coverage from `ClientSandboxEngine.mixedPlayers` tests.
- [ ] Ensure parity between backend and sandbox engines and improve AI-vs-AI
      termination behaviour using the sandbox AI simulation diagnostics
      (`ClientSandboxEngine.aiSimulation` with `RINGRIFT_ENABLE_SANDBOX_AI_SIM=1`),
      as tracked in P0.2 / P1.4 of `KNOWN_ISSUES.md`.

#### 2.3 Rules Engine Parity (Python/TS)

- [x] Implement `RulesBackendFacade` to abstract engine selection.
- [x] Implement `PythonRulesClient` for AI service communication.
- [x] Verify core mechanics parity (Movement, Capture, Lines, Territory) in Python engine.
- [ ] Enable `RINGRIFT_RULES_MODE=shadow` in staging/CI to collect parity metrics.

### **PHASE 3: Multiplayer Polish**

**Priority:** P1 - HIGH
**Goal:** Seamless online experience

#### 3.1 Spectator Mode

- [ ] UI for watching active games
- [ ] Real-time updates for spectators

#### 3.2 Social Features

- [ ] In-game chat
- [ ] User profiles and stats
- [ ] Leaderboards

#### 3.3 Matchmaking

- [ ] Automated matchmaking queue
- [ ] ELO-based matching

### **PHASE 4: Advanced AI**

**Priority:** P2 - MEDIUM
**Goal:** Challenging opponents for all skill levels

#### 4.1 Machine Learning

- [ ] Train neural network models
- [ ] Deploy advanced models to Python service

#### 4.2 Advanced Heuristics

- [ ] Implement MCTS/Minimax for intermediate levels

---

## 🔗 Alignment with TODO Tracks

The high-level phases above correspond to the more detailed execution tracks
and checklists in `TODO.md`:

- **Phase 2: Robustness & Testing** ↔ **Track 1** (Rules/FAQ Scenario Matrix &
  Parity Hardening) and parts of **Track 3** (Sandbox as a Rules Lab).
- **Phase 3: Multiplayer Polish** ↔ **Track 2** (Multiplayer Lifecycle &
  HUD/UX) and parts of **Track 5** (Persistence, Replays, and Stats).
- **Phase 4: Advanced AI** ↔ **Track 4** (Incremental AI Improvements &
  Observability) and **P2** items in `AI_IMPROVEMENT_PLAN.md`.

For day-to-day planning, treat `TODO.md` (including the
"Consolidated Execution Tracks & Plan" section) as the canonical, granular
list of tasks that roll up into these phases.

---

## 📊 Success Metrics for v1.0

1.  **Reliability:** >99.9% uptime, zero critical bugs.
2.  **Performance:** AI moves <1s, UI updates <16ms.
3.  **Engagement:** Users completing full games without errors.
4.  **Compliance:** 100% pass rate on rule scenario matrix.

---

## 📋 Implementation Roadmap (Post-Audit)

**P0: Rules Fidelity & Parity (Critical)**

- **Fix Forced Elimination Divergence:** Resolve auto-select vs player choice discrepancies.
- **Chain Capture Edge Cases:** Handle cyclic captures and reversals correctly.
- **Unified Move Model:** Complete the refactor to the unified move model.

**P1: AI Robustness & Intelligence (High Value)**

- **Wire Up Minimax/MCTS:** Uncomment and fix the AI factory to enable advanced strategies.
- **Fix RNG Determinism:** Implement per-game seeding for reproducible AI behavior.
- **AI Move Rejection Fallback:** Implement a tiered fallback system for invalid AI moves.

**P2: UX Polish & Multiplayer (User Experience)**

- **HUD Improvements:** Add phase indicators, timers, and better state visualization.
- **Victory Modal:** Wire up `game_over` events to the victory modal.
- **Lobby Game List:** Display waiting games correctly in the lobby.
