# ADR-005: GPU Vectorized Self-Play Engine

**Status**: Accepted
**Date**: December 2025
**Author**: RingRift Team

## Context

Self-play is the primary source of training data for RingRift's neural network models.
The original implementation used CPU-based sequential game simulation, which became
a bottleneck as the cluster scaled to 40+ nodes. Training was data-starved because
game generation couldn't keep pace with GPU training throughput.

Key constraints:

1. Need 100% parity with TypeScript game engine (source of truth)
2. Typical game: 50-100 moves, each requiring legal move enumeration
3. Target: 10,000+ games per hour on a single H100/GH200

## Decision

Implement a GPU-vectorized game engine (`app/ai/gpu_parallel_games.py`) that:

1. **Batched game state**: Represents multiple games (64-512) as tensor batches
2. **Vectorized move generation**: Computes legal moves for all games in parallel
3. **Parity testing**: Validated against TypeScript via replay-based tests
4. **Hybrid MCTS**: Gumbel MCTS tree search with neural network batch evaluation

### Architecture

```
gpu_parallel_games.py     # Main orchestrator (2,089 lines)
├── gpu_game_state.py     # Batch state management
├── gpu_move_generation.py # Vectorized move enumeration
├── gpu_move_application.py # Vectorized move application
└── gpu_game_termination.py # Victory/draw detection
```

### Key Design Choices

1. **Keep `.item()` calls minimal**: Reduced from 80+ to ~14 total
2. **Stay on GPU**: Minimize CPU-GPU transfers during simulation
3. **Batch-first tensors**: Shape `[batch_size, ...]` for all game data
4. **Progressive vectorization**: Started with hybrid CPU/GPU, migrated hot paths

## Consequences

### Positive

- **6.56x speedup** vs CPU at batch size 500 (production-ready)
- **100% parity** verified against TypeScript (10K seeds tested)
- **Scales with batch size**: Larger batches = more GPU utilization
- **Enables Gumbel MCTS**: Neural network evaluation amortized across batch

### Negative

- **Complexity**: ~4,000 lines of tensor manipulation code
- **Debugging difficulty**: Batched bugs harder to trace than sequential
- **Memory pressure**: 512 games × game state tensors = significant VRAM
- **Diminishing returns**: Beyond batch 512, CPU overhead dominates

### Performance Profile

| Batch Size | Speedup vs CPU | Games/Hour (H100) |
| ---------- | -------------- | ----------------- |
| 64         | 2.1x           | ~3,000            |
| 128        | 3.4x           | ~5,500            |
| 256        | 4.8x           | ~8,000            |
| 512        | 6.5x           | ~11,000           |

## Implementation Notes

### Remaining `.item()` Calls

- `gpu_parallel_games.py`: 1 call (statistics only)
- `gpu_move_generation.py`: 1 call (max_dist calculation)
- `gpu_move_application.py`: ~12 calls (attack move handling)

Further optimization would yield 10-15x speedup but requires significant
refactoring of attack move logic to be fully tensor-native.

### Parity Testing

```bash
# Verify GPU engine matches TypeScript
python scripts/check_ts_python_replay_parity.py --db data/games/canonical_hex8.db
```

Tests replay games from TypeScript-generated databases and verify move-by-move
that the GPU engine produces identical states.

## Related ADRs

- ADR-003: PFSP opponent selection (uses GPU selfplay for training data)
- ADR-004: Quality gate feedback loop (adjusts selfplay parameters)

## Files

- `app/ai/gpu_parallel_games.py` - Main GPU game runner
- `app/ai/gpu_move_generation.py` - Vectorized legal move enumeration
- `app/ai/gpu_move_application.py` - Vectorized move application
- `scripts/check_ts_python_replay_parity.py` - Parity verification
