# Frontend UX Progress Log

This document tracks shipped UX/GUI improvements so work is not duplicated or lost across iterations.

## 2025-12-11 — Controls & Accessibility Polish

**Shipped**

- Fixed the “`?` does nothing when the board is focused” dead-end by wiring `BoardView` help requests into hosts and avoiding unconditional default-prevention when no handler is present.
- Added consistent global keyboard shortcuts in game hosts:
  - `?` toggles **Board controls & shortcuts**
  - `M` toggles mute
  - `F` toggles fullscreen
  - `R` opens the resign confirmation (backend games only)
- Reduced keyboard friction on the board:
  - Roving tabindex for cells (Tab does not traverse every cell)
  - `Home`/`End` jump-to-start/end navigation
  - Screen-reader cell labels use canonical coordinates via `formatPosition()` (e.g. `a1`)
- Hardened `BoardControlsOverlay` as a proper modal (focus trap, Escape closes, focus restore).

**Primary code paths**

- `src/client/pages/BackendGameHost.tsx`
- `src/client/pages/SandboxGameHost.tsx`
- `src/client/components/BoardView.tsx`
- `src/client/components/BoardControlsOverlay.tsx`
- `src/client/hooks/useKeyboardNavigation.ts`
- `src/client/components/ResignButton.tsx`

**Docs updated**

- `docs/ACCESSIBILITY.md`
- `KNOWN_ISSUES.md` (P1.1 capability updates)

## 2025-12-12 — Sidebar Declutter & Help Consolidation

**Shipped**

- Consolidated the in-game keyboard shortcuts/help surface: removed the unused `KeyboardShortcutsHelp` component/tests so `BoardControlsOverlay` is the single `?` help overlay.
- Reduced sidebar density in both hosts via a persisted “Advanced” toggle:
  - Backend: `ringrift_backend_sidebar_show_advanced` (“Advanced diagnostics”) keeps the move log visible and hides history/evaluation by default.
  - Sandbox: `ringrift_sandbox_sidebar_show_advanced` (“Advanced panels”) hides replays/logs/recording by default; touch controls show on mobile or when advanced panels are open.
- Backend diagnostics log now includes a dedicated Last-Player-Standing entry when the game ends by LPS.

**Primary code paths**

- `src/client/pages/BackendGameHost.tsx`
- `src/client/pages/SandboxGameHost.tsx`
- `src/client/components/BoardControlsOverlay.tsx`

**Docs updated**

- `docs/ux/UX_RULES_WEIRD_STATES_SPEC.md`
- `docs/ux/UX_RULES_TELEMETRY_SPEC.md`
- `CONTRIBUTING.md`
- `docs/planning/IMPROVEMENT.md`
- `docs/planning/WAVE_2025_12.md`

## 2025-12-15 — Spectator UI Polish (P1-UX-02)

**Shipped**

- Enhanced [`SpectatorHUD.tsx`](../../src/client/components/SpectatorHUD.tsx:1) with complete game state display and educational features for non-playing observers:
  - Clear "Spectator Mode" banner with proper ARIA labels (`role="banner"`, `aria-label="Spectator Mode - You are watching this game"`)
  - Live indicator dot for connection status
  - Helpful message: "Moves are disabled while spectating. Use the teaching topics below to learn game mechanics."
  - Current phase, turn number, and move number displayed
  - Current player indicator with ring color
  - Player standings with rings in hand, eliminated rings, and territory count
  - `VictoryConditionsPanel` for educational value about win conditions
  - `TeachingTopicButtons` for quick access to learn game mechanics (movement, capture, lines, territory, victory conditions)
  - `TeachingOverlay` integration for comprehensive in-depth learning
  - Evaluation graph and move analysis panel in collapsible "Analysis & Insights" section
  - Recent moves with annotations
- Verified spectators cannot trigger player actions:
  - SpectatorHUD has no action buttons (no resign, no choice dialogs)
  - BoardView correctly disables all cell interactions when `isSpectator` is true
  - BackendGameHost properly hides ChoiceDialog and ResignButton for spectators

**Primary code paths**

- `src/client/components/SpectatorHUD.tsx` — Enhanced with TeachingOverlay, VictoryConditionsPanel, ARIA labels
- `src/client/components/TeachingOverlay.tsx` — Already supports spectator integration via useTeachingOverlay hook
- `src/client/components/BoardView.tsx` — Properly disables interactions for spectators
- `src/client/pages/BackendGameHost.tsx` — Properly hides player-only UI for spectators

**Tests verified**

- `tests/unit/components/SpectatorHUD.test.tsx` — Passes
- `tests/unit/components/GameHUD.spectator.test.tsx` — Passes
- `tests/unit/GameSession.spectatorFlow.test.ts` — Passes
- `tests/unit/GameSession.spectatorLateJoin.test.ts` — Passes
- All 142 GameHUD/TeachingOverlay tests pass

## Next Candidates (Not Yet Implemented)

- Migrate more game-end UX copy to rely on `GameEndExplanation` as the single source of truth (HUD/Victory/Teaching consistency).
