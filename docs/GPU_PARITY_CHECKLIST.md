# GPU Parity Checklist

Purpose: keep GPU-accelerated pipelines in lockstep with CPU rules and ensure
deterministic outcomes for identical seeds and inputs.

## Phase Coverage Targets

- Ring placement: placement masks + legal move generation parity.
- Movement: stack moves + overtaking capture parity (including recovery moves).
- Capture / chain capture: continuation availability and move sequencing parity.
- Line processing: line detection + line choice option parity.
- Territory processing: region detection + elimination choices parity.
- Forced elimination: stack selection and elimination counts parity.
- No-action moves: `no_*_action` emission parity when legal actions are absent.

## Fixture Requirements

- Minimal board states for each phase with a deterministic RNG seed.
- At least one multi-player fixture (3p) for rotation/skip logic.
- One forced-elimination fixture with multiple stacks of differing cap height.
- One territory fixture with multiple regions and conflicting ownership.

## Parity Assertions

- Same set of legal move types and counts for each fixture.
- Same selected move under fixed seed and deterministic search settings.
- Same resulting phase and player after applying the move.
- Same canonical state hash (or a documented, intentional delta).

## Execution Notes

- Skip GPU parity tests when CUDA/MPS is unavailable.
- Keep GPU/CPU parity tests small and deterministic; avoid large soaks here.
- Use parity harnesses for large-scale regression once unit fixtures pass.
