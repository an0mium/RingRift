# AI Ladder + Gumbel MCTS Integration Plan

Status: in_progress (2025-12-21)
Owner: Tool-driven agent
Scope: AI ladder selection, AI service instantiation, docs alignment

## Goals

1. Make Gumbel MCTS a first-class AIType in the AI service (no stranded code).
2. Decide whether D9/D10 defaults should move from Descent to Gumbel MCTS.
3. Keep TS and Python ladder mappings aligned and documented.

## Plan

1. **AI service integration**
   - Wire `AIType.GUMBEL_MCTS` in the AI factory.
   - Treat Gumbel as a neural-capable search engine for gating, caching, and
     ladder artifact checks.
2. **Ladder alignment decision**
   - Compare TS ladder defaults (`src/server/game/ai/AIEngine.ts`) with Python
     canonical profiles (`ai-service/app/main.py`) and per-board ladder config
     (`ai-service/app/config/ladder_config.py`).
   - Decide: keep Descent for D9/D10 or switch to Gumbel MCTS.
3. **Apply alignment**
   - If switching defaults: update TS + Python profiles, then update docs
     (AI ladder tables, runbooks, budgets).
   - If keeping defaults: document the rationale and add explicit override
     guidance for Gumbel usage.

## Progress

- [x] Step 1: AI service now instantiates Gumbel MCTS and treats it as
      neural-capable for gating/caching/ladder artifact checks.
- [ ] Step 2: Ladder alignment decision captured.
- [ ] Step 3: Apply alignment + docs/tests as needed.
