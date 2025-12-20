# RingRift AI Service: Project Plan (2025-12-20)

This plan prioritizes long-term correctness, data integrity, and cluster utilization.
All items assume canonical rules and TS engine parity are non-negotiable.

## Lane 1: Orchestration Truth and State Reconciliation (In Progress)

Goal: One authoritative view of job states and utilization across Slurm/Vast/P2P.

### Completed

- Sync Slurm job states into the unified scheduler DB before reporting status.
- Add backend-derived Slurm running/pending counts to status JSON.
- Add `cluster_submit.py sync-jobs` to reconcile job states on a schedule.

### Next Tasks

1. Add state transition timestamps in the unified DB.
   - Set `started_at` when a job transitions to `running`.
   - Set `finished_at` when a job transitions to terminal or `unknown`.
2. Add best-effort reconciliation for non-Slurm backends.
   - Vast: track instance running counts and mark stale jobs `unknown`.
   - P2P: use `/pipeline/status` and job history to infer in-flight work.
3. Extend the non-JSON status view to show backend-derived counts side-by-side.
4. Add unit tests for Slurm state reconciliation to prevent regressions.
5. Add a small daemon wrapper (or reuse `sync-jobs`) as the canonical reconciling loop
   used by monitoring and orchestration scripts.

## Lane 2: Canonical Data Pipeline Hardening

Goal: Canonical data only enters training, everywhere.

Tasks:

1. Introduce a shared helper that validates DB names and runs canonical gates.
2. Gate training entrypoints to canonical DBs and emit clear failure messages.
3. Update `TRAINING_DATA_REGISTRY.md` and add health summary checks to automation.
4. Add a small CLI to verify canonical status and provenance for a DB path.

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
