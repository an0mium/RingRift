# RingRift AI Service

The RingRift AI service is a FastAPI application that serves AI move selection,
choice handling, and position evaluation, plus a large collection of scripts
for self-play, training, and cluster orchestration.

## Quick Start (FastAPI service)

```bash
cd ai-service
./setup.sh
./run.sh
```

- API docs: `http://localhost:8001/docs`
- Health: `http://localhost:8001/health`
- Readiness: `http://localhost:8001/ready`
- Metrics: `http://localhost:8001/metrics`

## API Reference

- Canonical API reference: `ai-service/docs/API_REFERENCE.md`
- Core endpoints: `/ai/move`, `/ai/evaluate`, `/ai/choice/*`, `/api/replay/*`

## Canonical Automation Entry Points

### Primary Orchestrator

**`scripts/master_loop.py`** - canonical automation entry point for
self-play -> sync -> training -> evaluation -> promotion.

```bash
# Start the master loop (foreground)
python scripts/master_loop.py

# Watch status without starting the loop
python scripts/master_loop.py --watch

# Check status
python scripts/master_loop.py --status

# Legacy unified loop (explicit opt-in)
RINGRIFT_UNIFIED_LOOP_LEGACY=1 python scripts/unified_ai_loop.py --start
```

### Canonical Data + Parity

- `scripts/generate_canonical_selfplay.py` - canonical self-play generator + gates
- `scripts/run_canonical_selfplay_parity_gate.py` - parity gate
- `scripts/check_ts_python_replay_parity.py` - TS<->Python replay parity

## Script Inventory

For the full inventory, see `scripts/INDEX.md` and `scripts/README.md`.

### Cluster Management

- `cluster_health_check.py` - Cluster health snapshot
- `cluster_watchdog.py` - Host process watchdog
- `cluster_worker.py` - Worker node implementation

### Training

- `run_training_loop.py` - Automated training loop
- `run_self_play_soak.py` - Self-play data generation
- `export_replay_dataset.py` - Export replay data to NPZ datasets

### Evaluation

- `run_model_elo_tournament.py` - Model Elo tournaments
- `run_gauntlet.py` - Evaluation gauntlet
- `run_tournament.py` - Tournament runner

## Data & Rules SSoT

- Canonical rules: `RULES_CANONICAL_SPEC.md` (root)
- Canonical data registry: `ai-service/TRAINING_DATA_REGISTRY.md`
- Python engine mirrors TS rules under `ai-service/app/game_engine/`

## Environment Variables

Key flags (non-exhaustive):

- `RINGRIFT_TRACE_DEBUG` - Enable detailed tracing
- `RINGRIFT_SKIP_SHADOW_CONTRACTS` - Skip shadow contract validation
- `RINGRIFT_CONFIG_PATH` - Override config path
- `RINGRIFT_UNIFIED_LOOP_LEGACY` - Enable legacy unified loop
- `RINGRIFT_TRAINED_HEURISTIC_PROFILES` - Override heuristic profiles JSON

Full references:

- `ai-service/docs/ENV_REFERENCE.md`
- `ai-service/docs/ENV_REFERENCE_COMPREHENSIVE.md`
