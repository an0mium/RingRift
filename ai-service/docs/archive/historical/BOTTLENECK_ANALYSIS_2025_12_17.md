# Training Pipeline Bottleneck Analysis (2025-12-17)

## Executive Summary

Analysis of the training pipeline identified a **CRITICAL data flow mismatch** that was causing training to see 0 new games despite active selfplay generating hundreds of games. This has been fixed.

> **Note:** `multi_config_training_loop.py` was later removed. Current entry points are
> `scripts/run_training_loop.py` and `scripts/run_multiconfig_nnue_training.py`.

## Bottlenecks Identified

### 1. DATA FLOW MISMATCH (CRITICAL - FIXED)

**Severity**: Critical - Training completely unable to incorporate new selfplay data

**Symptoms**:

- Training logs showed `(+0/X)` pattern meaning 0 new games available
- 61+ selfplay processes running but training not seeing any data
- GPU utilization low due to lack of training work

**Root Cause**:
Training loop (`multi_config_training_loop.py`) only searched these directories:

- `data/selfplay/canonical` (doesn't exist)
- `data/games` (empty JSONL files)

But selfplay scripts write to:

- `data/selfplay/gpu/` (GPU selfplay)
- `data/selfplay/mcts_*` (MCTS policy selfplay)
- `data/selfplay/hybrid_*` (Hybrid selfplay)
- `data/selfplay/daemon_*` (Daemon selfplay)
- `data/selfplay/cluster_*` (Cluster selfplay)

**Fix Applied**:

1. Updated `CONFIG_JSONL_DIRS` to include all actual selfplay directories
2. Added `auto_discover_jsonl_dirs()` function to dynamically find JSONL directories
3. Updated `get_jsonl_counts()` to use `get_dynamic_jsonl_dirs()` which combines static config with auto-discovered directories

**Impact**: Training will now see 300+ games that were previously invisible

### 2. GPU UNDERUTILIZATION (22%)

**Severity**: Medium - Reduces training throughput

**Symptoms**:

- `nvidia-smi` showed only 22% GPU utilization during training
- Training runs slower than expected

**Root Cause**:

- StreamingDataLoader uses memory-mapped files with single-threaded batch construction
- `get_batch()` iterates one sample at a time, causing CPU bottleneck

**Mitigations Already In Place**:

- `prefetch_loader` provides background prefetching (enabled by default)
- `pin_memory=True` for faster CPU->GPU transfers
- `prefetch_count=2` for double buffering

**Potential Future Improvements**:

1. Increase `prefetch_count` to 3-4 for deeper pipeline
2. Add parallel batch construction in `get_batch()`
3. Consider HDF5 format which supports true parallel read
4. Use `RINGRIFT_DATALOADER_WORKERS` env var for non-streaming loaders

### 3. MODEL VALIDATION OVERHEAD

**Severity**: Medium - Delays training startup

**Symptoms**:

- 2992 model files scanned at startup (153GB total)
- "Scanning X model files..." message takes significant time

**Root Cause**:

- Full directory scan of all `.pth` files at startup
- No caching of validation results between runs

**Recommendation**:

1. Implement incremental validation (only validate new/modified models)
2. Cache validation results with file timestamps
3. Use background validation for non-critical checks

### 4. EMPTY JSONL FILES

**Severity**: Low - Wasted I/O

**Symptoms**:

- Many selfplay JSONL files show 0 lines
- Caused by processes starting but not completing games

**Root Cause**:

- Selfplay on RAM disk not syncing before termination
- Processes killed before flushing buffers

**Recommendation**:

1. Ensure selfplay writes with immediate flush (already implemented)
2. Configure RAM disk sync daemon with shorter intervals
3. Add file size check to skip empty files during training

## Data Flow Architecture

```
Selfplay Processes              Training Loop
==================              =============

GPU Selfplay                       |
  -> data/selfplay/gpu/*.jsonl     |
                                   |
MCTS Selfplay                      |
  -> data/selfplay/mcts_*/*.jsonl  |---> multi_config_training_loop.py
                                   |     (now uses get_dynamic_jsonl_dirs())
Hybrid Selfplay                    |
  -> data/selfplay/hybrid_*/*.jsonl|
                                   |
Daemon Selfplay                    |
  -> data/selfplay/daemon_*/*.jsonl|
```

## Files Modified

1. `multi_config_training_loop.py` (removed):
   - Updated `CONFIG_JSONL_DIRS` to include actual selfplay directories
   - Added `auto_discover_jsonl_dirs()` for dynamic directory discovery
   - Added `get_dynamic_jsonl_dirs()` to combine static and discovered directories
   - Updated `get_jsonl_counts()` to use dynamic discovery

## Verification

After applying fixes, run:

```bash
# Check discovered JSONL directories
python -c "
import sys
sys.path.insert(0, '.')
from scripts.multi_config_training_loop import auto_discover_jsonl_dirs, get_dynamic_jsonl_dirs
print('Discovered directories:')
for name, files in auto_discover_jsonl_dirs().items():
    print(f'  {name}: {len(files)} files')
print()
print('Square8 2p dirs:', get_dynamic_jsonl_dirs('square8', 2))
"

# Monitor training loop for new game detection
# multi_config_training_loop.py (removed)
# Should now show (+N/threshold) where N > 0
```

## Related Documentation

- [../../training/TRAINING_PIPELINE.md](../../training/TRAINING_PIPELINE.md) - Overall pipeline architecture
- [../../training/TRAINING_INTERNALS.md](../../training/TRAINING_INTERNALS.md) - Internal training modules
- [../../training/UNIFIED_AI_LOOP.md](../../training/UNIFIED_AI_LOOP.md) - Orchestrator documentation
