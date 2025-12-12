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

## Next Candidates (Not Yet Implemented)

- Migrate more game-end UX copy to rely on `GameEndExplanation` as the single source of truth (HUD/Victory/Teaching consistency).
