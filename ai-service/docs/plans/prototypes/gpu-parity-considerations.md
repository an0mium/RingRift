# GPU Parity Considerations - Territory Victory V2

## Overview

The GPU selfplay engine (`app/ai/gpu_parallel_games.py`) uses vectorized tensor operations for efficient batch game simulation. The territory victory rule change requires updates to maintain CPU/GPU parity.

## Current GPU Implementation

### Territory Counting

Currently, the GPU tracks territory via:

- `is_collapsed: (batch_size, board_size, board_size)` - boolean tensor for collapsed spaces
- `territory_owner: (batch_size, board_size, board_size)` - player number owning each territory

Territory count per player is derived by:

```python
# Count collapsed spaces owned by each player
for p in range(1, num_players + 1):
    territory_count[p] = (territory_owner == p).sum(dim=[1, 2])
```

### Current Victory Check

The current territory victory check (if any) uses the 50% threshold:

```python
threshold = total_spaces // 2 + 1
for p in range(num_players):
    if territory_counts[:, p] >= threshold:
        # Mark victory
```

## Required Changes

### 1. Threshold Calculation

**Change from:**

```python
threshold = total_spaces // 2 + 1
```

**Change to:**

```python
threshold = total_spaces // num_players + 1
```

**Risk: LOW** - Simple arithmetic change.

### 2. Dual Condition Check

**New logic required:**

```python
# For each player
for p in range(num_players):
    player_territory = territory_counts[:, p]  # (batch_size,)

    # Condition 1: meets threshold
    meets_threshold = player_territory >= threshold  # (batch_size,)

    # Condition 2: more than opponents combined
    opponents_total = territory_counts.sum(dim=1) - player_territory  # (batch_size,)
    dominates = player_territory > opponents_total  # (batch_size,)

    # Victory requires BOTH conditions
    has_victory = meets_threshold & dominates  # (batch_size,)
```

**Risk: MEDIUM** - New tensor operations but straightforward.

### 3. Vectorized Implementation

```python
def check_territory_victory_batch(
    self,
    territory_owner: torch.Tensor,  # (batch_size, board_size, board_size)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized territory victory check for GPU batch.

    Returns:
        has_victory: (batch_size,) bool - True if territory victory achieved
        winner: (batch_size,) int - Player number of winner (0 if no victory)
    """
    batch_size = territory_owner.shape[0]
    device = territory_owner.device

    # Count territory per player: (batch_size, num_players)
    territory_counts = torch.zeros(
        batch_size, self.num_players,
        dtype=torch.int32, device=device
    )
    for p in range(1, self.num_players + 1):
        territory_counts[:, p-1] = (territory_owner == p).sum(dim=[1, 2])

    # Threshold: floor(total_spaces / num_players) + 1
    total_spaces = self.board_size ** 2  # Adjust for hex
    threshold = (total_spaces // self.num_players) + 1

    # Check each player
    has_victory = torch.zeros(batch_size, dtype=torch.bool, device=device)
    winner = torch.zeros(batch_size, dtype=torch.int32, device=device)

    for p in range(self.num_players):
        player_territory = territory_counts[:, p]

        # Condition 1: meets threshold
        meets_threshold = player_territory >= threshold

        # Condition 2: total territory of all players
        total_territory = territory_counts.sum(dim=1)
        opponents_total = total_territory - player_territory
        dominates = player_territory > opponents_total

        # Victory requires both conditions
        player_victory = meets_threshold & dominates

        # Track first winner (don't overwrite existing victories)
        new_victory = player_victory & ~has_victory
        winner = torch.where(new_victory, p + 1, winner)
        has_victory = has_victory | player_victory

    return has_victory, winner
```

**Risk: MEDIUM** - Tensor operations need careful testing.

## Parity Verification

### Test Strategy

1. **Unit Tests**: Verify GPU implementation matches CPU for known test cases
2. **Property Tests**: Random game states, compare CPU vs GPU results
3. **Replay Tests**: Run selfplay replays through both engines

### Parity Test Implementation

```python
def test_territory_victory_parity():
    """Verify GPU matches CPU for territory victory detection."""
    import torch
    from app.game_engine import GameEngine
    from app.ai.gpu_parallel_games import BatchGameState

    test_cases = [
        # (territory_counts, expected_winner)
        ({1: 33, 2: 10}, 1),     # 2p clear win
        ({1: 22, 2: 15, 3: 15}, None),  # 3p threshold met, no dominance
        ({1: 31, 2: 15, 3: 15}, 1),     # 3p dominant
    ]

    for territory_counts, expected in test_cases:
        num_players = len(territory_counts)

        # CPU check
        cpu_result = cpu_check_territory_victory(territory_counts, num_players)

        # GPU check (single game batch)
        gpu_state = create_gpu_state_with_territory(territory_counts)
        gpu_result = gpu_state.check_territory_victory_batch()

        assert cpu_result == gpu_result[0].item(), \
            f"Parity failure: CPU={cpu_result}, GPU={gpu_result}"
```

## Edge Cases

### 1. Hexagonal Board

Hexagonal boards use a different coordinate system. Territory counting must handle:

- Axial/cube coordinates
- Non-square board shape
- Embedding size vs actual spaces (25x25 grid vs 469 valid hexes)

```python
# For hex boards, count only valid hexagonal positions
if self.board_type == 'hexagonal':
    # Mask for valid hex positions within radius
    valid_mask = self._get_hex_valid_mask()  # (board_size, board_size)
    territory_counts = []
    for p in range(1, self.num_players + 1):
        owned = (territory_owner == p) & valid_mask
        territory_counts.append(owned.sum(dim=[1, 2]))
```

### 2. Empty Board

When no territory exists, all counts are 0:

- Threshold check: 0 < threshold → no one qualifies
- Dominance check: 0 > 0 → false
- Result: No territory victory (correct)

### 3. Tie in Territory

If multiple players meet threshold but none dominates:

- P1: 30, P2: 20, P3: 14 on square8 3p (threshold=22)
- P1: 30 >= 22 but 30 < 34 (opponents combined)
- P2: 20 < 22 (threshold not met)
- Result: No territory victory (correct - must dominate)

### 4. All Territory Collapsed

Extreme case where entire board is territory:

- On square8 2p: 64 total spaces, threshold=33
- P1: 40, P2: 24
- P1 wins: 40 >= 33 AND 40 > 24

## Performance Considerations

### Memory

No significant additional memory required:

- `territory_counts`: (batch_size, num_players) - negligible
- Intermediate tensors for boolean masks - minimal

### Compute

Slightly more compute than current (additional sum and comparison):

- Old: 1 comparison per player
- New: 1 sum + 2 comparisons per player

Estimated overhead: <1% of total selfplay time.

## Migration Plan

1. **Phase 1**: Implement behind feature flag `TERRITORY_VICTORY_V2=false`
2. **Phase 2**: Enable for GPU selfplay with parity verification
3. **Phase 3**: Enable for all engines after validation
4. **Phase 4**: Remove feature flag, update documentation

## Risk Summary

| Component              | Risk Level | Mitigation                        |
| ---------------------- | ---------- | --------------------------------- |
| Threshold calculation  | Low        | Simple arithmetic, easy to verify |
| Dual condition logic   | Medium     | Comprehensive test coverage       |
| Tensor operations      | Medium     | Parity tests against CPU          |
| Hex board handling     | Medium     | Specific hex test cases           |
| Performance            | Low        | Negligible overhead               |
| Backward compatibility | Medium     | Feature flag for gradual rollout  |

## Files to Modify

1. `app/ai/gpu_parallel_games.py`:
   - Update `check_victory_conditions()` or equivalent
   - Add `check_territory_victory_batch()` method

2. `app/ai/gpu_batch.py` (if separate):
   - Similar updates for batch victory checking

3. Tests:
   - `tests/test_gpu_parallel_games.py` - add territory victory tests
   - `tests/test_gpu_cpu_parity.py` - add parity verification
