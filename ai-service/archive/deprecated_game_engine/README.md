# Deprecated Game Engine - Migration Notes

This directory contains documentation for the legacy game engine deprecation.

**Note:** The actual `_game_engine_legacy.py` file remains in `app/` due to internal
dependencies (relative imports to board_manager, etc.). It cannot be moved without
breaking those dependencies. The deprecation is enforced via the facade pattern.

## Current Status

**Location:** `app/_game_engine_legacy.py` (4,479 lines)
**Deprecation:** Active (via facade in `app/game_engine/__init__.py`)
**Removal Target:** Q2 2026

## Architecture

The deprecation uses a **facade pattern**:

1. `app/game_engine/__init__.py` - Public API, lazy imports from legacy with warnings suppressed
2. `app/_game_engine_legacy.py` - Actual implementation (DO NOT import directly)
3. Direct imports to `_game_engine_legacy` emit `DeprecationWarning`

## Replacement

Use the canonical game engine package:

```python
# Canonical API (use this)
from app.game_engine import GameEngine, PhaseRequirement

# For legacy replay compatibility only
from app.rules.legacy.replay_compatibility import LegacyReplayEngine
```

## Files Still Using Legacy Module

- `app/rules/legacy/replay_compatibility.py` - Legitimate use for replaying pre-canonical games
- `app/game_engine/__init__.py` - Facade that re-exports with deprecation warnings

## Migration Path

1. **New code**: Always use `from app.game_engine import GameEngine`
2. **Existing code**: Update imports from `app._game_engine_legacy` to `app.game_engine`
3. **Legacy replays**: Use `app.rules.legacy.replay_compatibility` module

## See Also

- `docs/specs/LEGACY_RULES_DIFF.md` - Differences between legacy and canonical rules
- `docs/architecture/ARCHITECTURE_NAMING.md` - Naming conventions (tracks deprecation)
- `RULES_CANONICAL_SPEC.md` - Canonical rules specification
