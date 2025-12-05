# Legacy Self-Play Databases (Pre Line-Length Fix)

**Date archived:** 2025-12-05

## Bug Description

These databases were generated with a bug in `board_manager.py:find_all_lines()`:

```python
# BUG: Always used min_length=4
min_length = 4 if board.type == BoardType.SQUARE8 else 4
```

Per RR-CANON-R120, the correct line length for square8 with 3-4 players is **3**, not 4.

## Impact

- 3-marker lines were NOT detected in 3-player and 4-player square8 games
- `state_after` JSON values are incorrect (missing collapsed spaces, wrong phases)
- Territory claiming from lines did not trigger

## Affected Files

- `coverage_selfplay.db` - Mixed 2p/3p games, 3p games affected
- `selfplay_square8_3p.db` - All games affected
- `selfplay_square8_4p.db` - All games affected
- `square8_3p.db` - All games affected
- `square8_4p.db` - All games affected

## DO NOT USE FOR

- Parity testing
- Training neural networks
- Validating rules implementations

## Safe to Delete

These files are kept only for historical reference. They can be safely deleted.
