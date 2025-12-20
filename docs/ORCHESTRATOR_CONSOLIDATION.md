# Orchestrator Consolidation Notes

Goal: keep all rules/phase transitions routed through the canonical
`turnOrchestrator.ts` entry points and eliminate legacy orchestration paths.

## Canonical entry points (SSoT)

- `src/shared/engine/orchestration/turnOrchestrator.ts`
  - `processTurn`, `processTurnAsync`
  - `validateMove`, `getValidMoves`, `hasValidMoves`
  - `applyMoveForReplay`

## Deprecated/legacy orchestration

- `src/shared/engine/orchestration/phaseStateMachine.ts`
  - Marked deprecated; should not be used by host adapters.
  - Exported via `src/shared/engine/orchestration/index.ts` for backward
    compatibility only.

## Adapters and host usage to keep aligned

- Server:
  - `src/server/game/turn/TurnEngineAdapter.ts`
  - `src/server/game/GameEngine.ts`
- Client (sandbox):
  - `src/client/sandbox/SandboxOrchestratorAdapter.ts`
  - `src/client/sandbox/ClientSandboxEngine.ts`
- Shared engine adapters:
  - `src/shared/engine/fsm/FSMAdapter.ts` (wraps `turnOrchestrator` for FSM use)

## Follow-up checks (to execute in order)

1. Confirm no adapter imports `phaseStateMachine.ts` directly.
2. Verify server + sandbox both route replay and move validation through
   `turnOrchestrator` helpers.
3. Add a focused integration test that replays a short, canonical move sequence
   via both server and sandbox adapters and asserts identical phase outcomes.
