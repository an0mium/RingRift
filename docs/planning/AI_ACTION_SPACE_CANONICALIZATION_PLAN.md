# AI Action-Space Canonicalization Plan

> Doc Status (2025-12-22): Active
>
> Purpose: Align legacy neural-net policy encoding/decoding with canonical
> move types while preserving compatibility with legacy checkpoints.
>
> Scope: `ai-service/app/ai/_neural_net_legacy.py`, canonical move encoders,
> and targeted AI tests.

## Objectives

- Ensure MAX_N policy head encoding/decoding uses canonical move types.
- Normalize legacy aliases to canonical names at model I/O boundaries.
- Keep legacy modules isolated while preventing legacy move emissions in
  active AI flows.

## Plan (Phase 1)

1. Normalize move types in legacy MAX_N encoding using
   `convert_legacy_move_type`, removing direct dependence on legacy names.
2. Decode line/territory actions to canonical move types
   (`process_line`, `eliminate_rings_from_stack`).
3. Update AI tests that still rely on legacy move names.
4. Record status in `ai-service/docs/STRANDED_FEATURES.md`.

## Success Criteria

- MAX_N encoding accepts canonical move types without legacy-only branches.
- Decoded moves emitted by `NeuralNetAI` use canonical names.
- AI tests reference canonical move types.

## Follow-ups (Phase 2+)

- Address hex policy special-action compression (distinct indices).
- Retire MAX_N policy head path once all checkpoints migrate to
  board-specific policy sizes.
- Expand parity/fixtures to cover canonical line/territory move types in
  NN policy paths.
