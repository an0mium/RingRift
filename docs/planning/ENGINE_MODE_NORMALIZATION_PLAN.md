# Engine Mode Normalization Plan

> Doc Status: Active
>
> Scope: selfplay metadata + analytics readers
>
> Owner: TBD

## Goals

- Ensure `engine_mode` metadata uses canonical hyphenated values.
- Accept legacy underscore aliases in readers to avoid drift during migration.

## Plan

- [x] Update JSONL writers to emit canonical `engine_mode` values (gumbel-mcts, policy-only).
- [x] Normalize `engine_mode` ingestion in analytics and training readers.
- [x] Spot-check for remaining underscore `engine_mode` outputs in ai-service scripts.

## Notes

- Legacy alias handling uses `normalize_engine_mode` from `app.training.selfplay_config`.
