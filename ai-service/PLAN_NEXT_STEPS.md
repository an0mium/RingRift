# RingRift AI Service: Project Plan (2025-12-20)

This plan prioritizes long-term correctness, data integrity, and cluster utilization.
All items assume canonical rules and TS engine parity are non-negotiable.

## Lane 1: Orchestration Truth and State Reconciliation (Complete)

Goal: One authoritative view of job states and utilization across Slurm/Vast/P2P.

### Completed

- Sync Slurm job states into the unified scheduler DB before reporting status.
- Add backend-derived Slurm running/pending counts to status JSON.
- Add `cluster_submit.py sync-jobs` to reconcile job states on a schedule.
- Add best-effort Vast/P2P reconciliation and backend-derived counts.
- Update status output + monitoring loop to surface backend counts.
- Add a job-state sync daemon wrapper.
- Add timestamps for job state transitions and unit tests for reconciliation.

### Next Tasks

1. Extend reconciliation to mark completed jobs explicitly when backend data allows it.
2. Add integration tests for `cluster_submit.py status --json` alignment with backends.
3. Wire the sync daemon into the default cluster startup flow.

## Lane 2: Canonical Data Pipeline Hardening

Goal: Canonical data only enters training, everywhere.

### Completed

- Added shared canonical source helper in `app/training/canonical_sources.py`.
- Updated dataset export entrypoints to enforce registry-backed canonical checks.
- Consolidated `validate_canonical_training_sources.py` to use the shared helper.

### Next Tasks

1. Extend gating to any remaining DB-driven scripts not yet covered.
2. Update `TRAINING_DATA_REGISTRY.md` and add health summary checks to automation.
3. Add a small CLI to verify canonical status and provenance for a DB path.

## Lane 3: AI Determinism, Registry, and Correctness

Goal: Reproducible AI decisions across CPU/GPU and a single registry of AI types.

Tasks:

1. Centralize RNG seeding and ensure it is threaded through every AI implementation.
2. Add an AI registry with explicit versioning and capability flags (including IG-GMO).
3. Add deterministic replay tests for a few representative AI policies.
4. Ensure AI metrics and model selection logic use the same registry metadata.

## Lane 4: Data Distribution and NFS Consolidation

Goal: Fast, reliable model/data sync across all nodes with clear provenance.

Tasks:

1. Integrate aria2-based distribution into the SyncCoordinator as a first-class option.
2. Prefer NFS-backed paths when present; avoid local caches as the default source.
3. Add content-hash verification and a "data freshness" report per host.
4. Standardize model distribution to use the same manifest format across scripts.

## Lane 5: Observability and Governance

Goal: Detect parity drift and data issues early, with actionable signals.

Tasks:

1. Add metrics for parity failures, canonical gate failures, and sync lag.
2. Add alerts for prolonged idle nodes and exploding pending queues.
3. Provide a concise "cluster health" summary CLI for daily ops.
