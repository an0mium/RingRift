# RingRift Documentation Index

> **Last Updated:** 2025-12-02
> **Maintenance:** Update this index when adding/removing docs
> **Total Documents:** 110+ files cataloged

This comprehensive index catalogs all documentation in the `docs/` directory and project root, organized by topic.  
For a high-level overview, see [INDEX.md](INDEX.md).

---

## Table of Contents

- [A. Getting Started](#a-getting-started)
- [B. Architecture & Design](#b-architecture--design)
- [C. Operations & Deployment](#c-operations--deployment)
- [D. Testing & Quality](#d-testing--quality)
- [E. Monitoring & Observability](#e-monitoring--observability)
- [F. AI & Training](#f-ai--training)
- [G. Rules & Game Logic](#g-rules--game-logic)
- [H. Assessment Reports (By Pass)](#h-assessment-reports-by-pass)
- [I. Incident Documentation](#i-incident-documentation)
- [J. Draft/Work-in-Progress](#j-draftwork-in-progress)
- [K. Supplementary Documentation](#k-supplementary-documentation)
- [Finding Documentation](#finding-documentation)

---

## A. Getting Started

### [README.md](../README.md)

**Status:** Active  
**Last Updated:** 2025-12-01  
**Purpose:** Project overview, features, current status, and quick navigation to key documentation. Entry point for new developers.  
**Related Docs:** [QUICKSTART.md](../QUICKSTART.md), [CONTRIBUTING.md](../CONTRIBUTING.md), [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md)

### [QUICKSTART.md](../QUICKSTART.md)

**Status:** Active  
**Last Updated:** 2025-11-30  
**Purpose:** Step-by-step setup guide for local development (backend, frontend, AI service). Docker and Python virtual environment instructions.  
**Related Docs:** [README.md](../README.md), [docs/DEPLOYMENT_REQUIREMENTS.md](DEPLOYMENT_REQUIREMENTS.md), [ai-service/README.md](../ai-service/README.md)

### [CONTRIBUTING.md](../CONTRIBUTING.md)

**Status:** Active  
**Last Updated:** 2025-11-27  
**Purpose:** Contribution guidelines, code style, PR process, CI policy, and development priorities. Includes historical development phases.  
**Related Docs:** [TODO.md](../TODO.md), [KNOWN_ISSUES.md](../KNOWN_ISSUES.md), [docs/SUPPLY_CHAIN_AND_CI_SECURITY.md](SUPPLY_CHAIN_AND_CI_SECURITY.md)

### [docs/INDEX.md](INDEX.md)

**Status:** Active  
**Last Updated:** 2025-11-30  
**Purpose:** High-level documentation overview with quick links to essential documents. Organized by topic with navigation aids.  
**Related Docs:** This file (DOCUMENTATION_INDEX.md), [PROJECT_GOALS.md](../PROJECT_GOALS.md)

---

## B. Architecture & Design

### [ARCHITECTURE_ASSESSMENT.md](../ARCHITECTURE_ASSESSMENT.md)

**Status:** Active (with historical content)  
**Last Updated:** 2025-11-27  
**Purpose:** Comprehensive architecture assessment and roadmap. Documents the strangler fig migration to canonical turn orchestrator (Phases 1-4 complete).  
**Related Docs:** [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md), [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md), [docs/CANONICAL_ENGINE_API.md](CANONICAL_ENGINE_API.md)

### [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md)

**Status:** Active (with historical appendix)  
**Last Updated:** 2025-11-23  
**Purpose:** AI service architecture, algorithms, training pipeline, integration with backend. Includes difficulty ladder, RNG determinism, and fallback strategies.  
**Related Docs:** [ai-service/README.md](../ai-service/README.md), [docs/AI_TRAINING_AND_DATASETS.md](AI_TRAINING_AND_DATASETS.md), [docs/AI_TRAINING_PREPARATION_GUIDE.md](AI_TRAINING_PREPARATION_GUIDE.md)

### [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md)

**Status:** Active (with historical content)  
**Last Updated:** 2025-11-26  
**Purpose:** Rules engine architecture, Python↔TypeScript parity, orchestrator rollout strategy. Documents mutator-first mode and shadow contracts.  
**Related Docs:** [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md), [RULES_IMPLEMENTATION_MAPPING.md](../RULES_IMPLEMENTATION_MAPPING.md), [docs/ORCHESTRATOR_ROLLOUT_PLAN.md](ORCHESTRATOR_ROLLOUT_PLAN.md)

### [docs/MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md)

**Status:** Active (with historical appendix)  
**Last Updated:** 2025-11-30  
**Purpose:** Catalog of module responsibilities in `src/shared/engine/`. Maps helpers, aggregates, orchestrator, and contracts with concern types and dependencies.  
**Related Docs:** [docs/CANONICAL_ENGINE_API.md](CANONICAL_ENGINE_API.md), [docs/DOMAIN_AGGREGATE_DESIGN.md](DOMAIN_AGGREGATE_DESIGN.md), [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md)

### [docs/CANONICAL_ENGINE_API.md](CANONICAL_ENGINE_API.md)

**Status:** Active  
**Last Updated:** 2025-11-26  
**Purpose:** Canonical engine public API specification. Documents Move, PendingDecision, PlayerChoice lifecycle, and WebSocket integration.  
**Related Docs:** [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md), [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md), [docs/MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md)

### [docs/DOMAIN_AGGREGATE_DESIGN.md](DOMAIN_AGGREGATE_DESIGN.md)

**Status:** Active  
**Purpose:** Design document for domain aggregates (Placement, Movement, Capture, Line, Territory, Victory) in the canonical orchestrator architecture.  
**Related Docs:** [docs/MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md), [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md)

### [docs/STATE_MACHINES.md](STATE_MACHINES.md)

**Status:** Active  
**Purpose:** Shared session/AI/choice/connection state machines over the canonical API. Documents phase transitions and lifecycle flows.  
**Related Docs:** [docs/CANONICAL_ENGINE_API.md](CANONICAL_ENGINE_API.md), [docs/P18.3-1_DECISION_LIFECYCLE_SPEC.md](P18.3-1_DECISION_LIFECYCLE_SPEC.md)

### [docs/TOPOLOGY_MODES.md](TOPOLOGY_MODES.md)

**Status:** Active  
**Purpose:** Application topology modes (single instance, multi-instance) and their operational implications. Documents sticky session requirements.  
**Related Docs:** [docs/DEPLOYMENT_REQUIREMENTS.md](DEPLOYMENT_REQUIREMENTS.md), [docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)

---

## C. Operations & Deployment

### [docs/DEPLOYMENT_REQUIREMENTS.md](DEPLOYMENT_REQUIREMENTS.md)

**Status:** Active  
**Last Updated:** 2025-11-27  
**Purpose:** Canonical deployment requirements and environment configuration guide across development, staging, and production. Covers infrastructure, health checks, resource limits, and validation.  
**Related Docs:** [docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md), [docs/SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md), [docs/OPERATIONS_DB.md](OPERATIONS_DB.md)

### [docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)

**Status:** Active  
**Last Updated:** 2025-11-30  
**Purpose:** Complete reference for all environment variables with defaults, ranges, validation rules, and security considerations. Organized by category with production checklist.  
**Related Docs:** [docs/DEPLOYMENT_REQUIREMENTS.md](DEPLOYMENT_REQUIREMENTS.md), [docs/SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md), [.env.example](../.env.example)

### [docs/OPERATIONS_DB.md](OPERATIONS_DB.md)

**Status:** Active  
**Last Updated:** 2025-11-27  
**Purpose:** Database operations playbook covering Postgres management, Prisma migrations, backups, and disaster recovery across all environments.  
**Related Docs:** [docs/DEPLOYMENT_REQUIREMENTS.md](DEPLOYMENT_REQUIREMENTS.md), [docs/runbooks/DATABASE_MIGRATION.md](runbooks/DATABASE_MIGRATION.md), [docs/runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md](runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md)

### [docs/SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md)

**Status:** Active  
**Last Updated:** 2025-11-27  
**Purpose:** Best practices for managing secrets (JWT, database, Redis) across environments. Includes generation, rotation procedures, and security checklist.  
**Related Docs:** [docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md), [docs/SECURITY_THREAT_MODEL.md](SECURITY_THREAT_MODEL.md), [docs/runbooks/SECRETS_ROTATION_DRILL.md](runbooks/SECRETS_ROTATION_DRILL.md)

### [docs/ORCHESTRATOR_ROLLOUT_PLAN.md](ORCHESTRATOR_ROLLOUT_PLAN.md)

**Status:** Active  
**Last Updated:** 2025-11-30  
**Purpose:** Orchestrator-first rollout and legacy rules shutdown plan (Track A). Defines environment phases 0-4, SLOs, and feature flag matrices.  
**Related Docs:** [docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md](runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md), [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md), [docs/STRICT_INVARIANT_SOAKS.md](STRICT_INVARIANT_SOAKS.md)

### [docs/ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md](ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md)

**Status:** Active  
**Purpose:** Completion plan for orchestrator migration with cleanup tasks and verification steps.  
**Related Docs:** [docs/ORCHESTRATOR_ROLLOUT_PLAN.md](ORCHESTRATOR_ROLLOUT_PLAN.md), [ARCHITECTURE_ASSESSMENT.md](../ARCHITECTURE_ASSESSMENT.md)

### [docs/SHARED_ENGINE_CONSOLIDATION_PLAN.md](SHARED_ENGINE_CONSOLIDATION_PLAN.md)

**Status:** Active  
**Purpose:** Plan for consolidating shared engine components and eliminating duplication across backend and sandbox.  
**Related Docs:** [docs/MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md), [ARCHITECTURE_ASSESSMENT.md](../ARCHITECTURE_ASSESSMENT.md)

### [docs/DEPENDENCY_UPGRADE_PLAN.md](DEPENDENCY_UPGRADE_PLAN.md)

**Status:** Active  
**Purpose:** Wave-based Node/TypeScript and Python dependency upgrade strategy with test guardrails and log-handling conventions.  
**Related Docs:** [CONTRIBUTING.md](../CONTRIBUTING.md), [docs/SUPPLY_CHAIN_AND_CI_SECURITY.md](SUPPLY_CHAIN_AND_CI_SECURITY.md)

### [docs/ALERTING_THRESHOLDS.md](ALERTING_THRESHOLDS.md)

**Status:** Active  
**Purpose:** Complete alert configuration and rationale for Prometheus alerts. Maps alerts to runbooks and incident guides.  
**Related Docs:** [monitoring/prometheus/alerts.yml](../monitoring/prometheus/alerts.yml), [docs/incidents/INDEX.md](incidents/INDEX.md), [docs/runbooks/INDEX.md](runbooks/INDEX.md)

### Runbooks (docs/runbooks/)

Operational runbooks for deploying and managing RingRift. See [docs/runbooks/INDEX.md](runbooks/INDEX.md) for complete index.

#### Deployment Runbooks

- **[DEPLOYMENT_INITIAL.md](runbooks/DEPLOYMENT_INITIAL.md)** - First-time environment setup
- **[DEPLOYMENT_ROUTINE.md](runbooks/DEPLOYMENT_ROUTINE.md)** - Standard release procedures
- **[DEPLOYMENT_ROLLBACK.md](runbooks/DEPLOYMENT_ROLLBACK.md)** - How to revert to previous versions
- **[DEPLOYMENT_SCALING.md](runbooks/DEPLOYMENT_SCALING.md)** - How to scale services up/down

#### Database Runbooks

- **[DATABASE_MIGRATION.md](runbooks/DATABASE_MIGRATION.md)** - Prisma migration procedures
- **[DATABASE_BACKUP_AND_RESTORE_DRILL.md](runbooks/DATABASE_BACKUP_AND_RESTORE_DRILL.md)** - Non-destructive Postgres backup/restore drill
- **[DATABASE_DOWN.md](runbooks/DATABASE_DOWN.md)** - Database outage response
- **[DATABASE_PERFORMANCE.md](runbooks/DATABASE_PERFORMANCE.md)** - Database performance issues

#### Security & Operational Drills

- **[SECRETS_ROTATION_DRILL.md](runbooks/SECRETS_ROTATION_DRILL.md)** - JWT and database credential rotation drill for staging
- **[ORCHESTRATOR_ROLLOUT_RUNBOOK.md](runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md)** - Operator runbook for orchestrator rollout, rollback, and incident response

#### AI Service Runbooks

- **[AI_SERVICE_DOWN.md](runbooks/AI_SERVICE_DOWN.md)** - AI service outage response
- **[AI_ERRORS.md](runbooks/AI_ERRORS.md)** - AI error diagnosis and resolution
- **[AI_FALLBACK.md](runbooks/AI_FALLBACK.md)** - AI fallback behavior and quality degradation
- **[AI_PERFORMANCE.md](runbooks/AI_PERFORMANCE.md)** - AI performance troubleshooting

#### Service Health Runbooks

- **[SERVICE_DEGRADATION.md](runbooks/SERVICE_DEGRADATION.md)** - Service degradation response
- **[SERVICE_OFFLINE.md](runbooks/SERVICE_OFFLINE.md)** - Complete service outage
- **[HIGH_ERROR_RATE.md](runbooks/HIGH_ERROR_RATE.md)** - High error rate incidents
- **[HIGH_LATENCY.md](runbooks/HIGH_LATENCY.md)** - Latency issues
- **[HIGH_MEMORY.md](runbooks/HIGH_MEMORY.md)** - Memory exhaustion
- **[EVENT_LOOP_LAG.md](runbooks/EVENT_LOOP_LAG.md)** - Node.js event loop lag
- **[RESOURCE_LEAK.md](runbooks/RESOURCE_LEAK.md)** - Resource leak detection

#### Redis Runbooks

- **[REDIS_DOWN.md](runbooks/REDIS_DOWN.md)** - Redis outage response
- **[REDIS_PERFORMANCE.md](runbooks/REDIS_PERFORMANCE.md)** - Redis performance issues

#### Game & WebSocket Runbooks

- **[GAME_HEALTH.md](runbooks/GAME_HEALTH.md)** - Game health monitoring
- **[GAME_PERFORMANCE.md](runbooks/GAME_PERFORMANCE.md)** - Game performance issues
- **[WEBSOCKET_ISSUES.md](runbooks/WEBSOCKET_ISSUES.md)** - WebSocket connectivity problems
- **[WEBSOCKET_SCALING.md](runbooks/WEBSOCKET_SCALING.md)** - WebSocket scaling procedures
- **[NO_ACTIVITY.md](runbooks/NO_ACTIVITY.md)** - No active games detected
- **[NO_TRAFFIC.md](runbooks/NO_TRAFFIC.md)** - No HTTP traffic detected

#### Rules & Security Runbooks

- **[RULES_PARITY.md](runbooks/RULES_PARITY.md)** - Rules parity violation response
- **[RATE_LIMITING.md](runbooks/RATE_LIMITING.md)** - Rate limiting incidents

---

## D. Testing & Quality

### [docs/TEST_CATEGORIES.md](TEST_CATEGORIES.md)

**Status:** Active  
**Last Updated:** 2025-12-01  
**Purpose:** Defines test categories (CI-gated, environment-gated, diagnostic, E2E, Python). Explains what "all tests passing" means for different contexts.  
**Related Docs:** [tests/README.md](../tests/README.md), [docs/TEST_INFRASTRUCTURE.md](TEST_INFRASTRUCTURE.md), [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)

### [docs/TEST_INFRASTRUCTURE.md](TEST_INFRASTRUCTURE.md)

**Status:** Active  
**Purpose:** Test infrastructure components and utilities including MultiClientCoordinator, NetworkSimulator, TimeController, and test fixtures.  
**Related Docs:** [docs/TEST_CATEGORIES.md](TEST_CATEGORIES.md), [docs/CONTRACT_VECTORS_DESIGN.md](CONTRACT_VECTORS_DESIGN.md), [tests/README.md](../tests/README.md)

### [docs/CONTRACT_VECTORS_DESIGN.md](CONTRACT_VECTORS_DESIGN.md)

**Status:** Active (derived)  
**Last Updated:** 2025-12-01  
**Purpose:** Design specification for extended v2 contract test vectors exercising long-tail, multi-phase, and hex-specific rules scenarios for TS↔Python parity.  
**Related Docs:** [docs/INVARIANTS_AND_PARITY_FRAMEWORK.md](INVARIANTS_AND_PARITY_FRAMEWORK.md), [docs/PYTHON_PARITY_REQUIREMENTS.md](PYTHON_PARITY_REQUIREMENTS.md), [tests/fixtures/contract-vectors/v2/](../tests/fixtures/contract-vectors/v2/)

### [docs/INVARIANTS_AND_PARITY_FRAMEWORK.md](INVARIANTS_AND_PARITY_FRAMEWORK.md)

**Status:** Active (derived, meta-framework)  
**Last Updated:** 2025-11-30  
**Purpose:** Catalogue of rules-level invariants (INV-_) and TS↔Python↔host parity expectations (PARITY-_), including enforcement mechanisms, CI mappings, and alerts.  
**Related Docs:** [docs/STRICT_INVARIANT_SOAKS.md](STRICT_INVARIANT_SOAKS.md), [docs/PYTHON_PARITY_REQUIREMENTS.md](PYTHON_PARITY_REQUIREMENTS.md), [docs/ACTIVE_NO_MOVES_BEHAVIOUR.md](ACTIVE_NO_MOVES_BEHAVIOUR.md)

### [docs/PYTHON_PARITY_REQUIREMENTS.md](PYTHON_PARITY_REQUIREMENTS.md)

**Status:** Active  
**Last Updated:** 2025-11-26  
**Purpose:** TS↔Python function/type parity matrix, shadow contract verification, and extended contract vector specifications (P18.5-1).  
**Related Docs:** [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md), [docs/CONTRACT_VECTORS_DESIGN.md](CONTRACT_VECTORS_DESIGN.md), [ai-service/tests/contracts/](../ai-service/tests/contracts/)

### [RULES_SCENARIO_MATRIX.md](../RULES_SCENARIO_MATRIX.md)

**Status:** Active  
**Last Updated:** 2025-11-30  
**Purpose:** Canonical map from rules/FAQ documents to concrete Jest test suites. Tracks coverage for movement, captures, lines, territory, victory, and parity.  
**Related Docs:** [ringrift_complete_rules.md](../ringrift_complete_rules.md), [tests/README.md](../tests/README.md), [docs/TEST_CATEGORIES.md](TEST_CATEGORIES.md)

### [docs/PARITY_SEED_TRIAGE.md](PARITY_SEED_TRIAGE.md)

**Status:** Active  
**Purpose:** Seed-specific debugging and parity triage guide for backend vs sandbox divergences.  
**Related Docs:** [docs/INVARIANTS_AND_PARITY_FRAMEWORK.md](INVARIANTS_AND_PARITY_FRAMEWORK.md), [RULES_SCENARIO_MATRIX.md](../RULES_SCENARIO_MATRIX.md)

### [docs/E2E_AUTH_AND_GAME_FLOW_TEST_STABILIZATION_SUMMARY.md](E2E_AUTH_AND_GAME_FLOW_TEST_STABILIZATION_SUMMARY.md)

**Status:** Active  
**Purpose:** Summary of E2E test stabilization work for authentication and game flow tests.  
**Related Docs:** [docs/TEST_CATEGORIES.md](TEST_CATEGORIES.md), [tests/README.md](../tests/README.md)

### [docs/P18.18_SKIPPED_TEST_TRIAGE.md](P18.18_SKIPPED_TEST_TRIAGE.md)

**Status:** Active  
**Purpose:** Triage and analysis of skipped tests with prioritization and resolution plans.  
**Related Docs:** [KNOWN_ISSUES.md](../KNOWN_ISSUES.md), [docs/TEST_CATEGORIES.md](TEST_CATEGORIES.md)

---

## E. Monitoring & Observability

### [docs/STRICT_INVARIANT_SOAKS.md](STRICT_INVARIANT_SOAKS.md)

**Status:** Active  
**Last Updated:** 2025-11-30  
**Purpose:** Invariant and orchestrator soak strategy including CI short/long soak profiles, Python strict no-move invariants, and anomaly detection.  
**Related Docs:** [docs/INVARIANTS_AND_PARITY_FRAMEWORK.md](INVARIANTS_AND_PARITY_FRAMEWORK.md), [docs/ORCHESTRATOR_ROLLOUT_PLAN.md](ORCHESTRATOR_ROLLOUT_PLAN.md), [docs/ALERTING_THRESHOLDS.md](ALERTING_THRESHOLDS.md)

### [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md)

**Status:** Active (roadmap & SLOs)  
**Last Updated:** 2025-11-30  
**Purpose:** Phased strategic roadmap with performance/scale SLOs, success metrics, and canonical load scenarios (P-01). Operational monitoring section includes dashboards and metrics.  
**Related Docs:** [PROJECT_GOALS.md](../PROJECT_GOALS.md), [TODO.md](../TODO.md), [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md)

---

## F. AI & Training

### [docs/AI_TRAINING_AND_DATASETS.md](AI_TRAINING_AND_DATASETS.md)

**Status:** Active  
**Purpose:** AI service training pipelines, self-play and territory dataset generation, JSONL schemas, CLI usage, and seed behavior.  
**Related Docs:** [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md), [docs/AI_TRAINING_PREPARATION_GUIDE.md](AI_TRAINING_PREPARATION_GUIDE.md), [docs/AI_TRAINING_ASSESSMENT_FINAL.md](AI_TRAINING_ASSESSMENT_FINAL.md)

### [docs/AI_TRAINING_PREPARATION_GUIDE.md](AI_TRAINING_PREPARATION_GUIDE.md)

**Status:** Active  
**Purpose:** RingRift-specific pre-training checklist, global memory budgeting, and infrastructure configuration for neural network and heuristic training.  
**Related Docs:** [docs/AI_TRAINING_AND_DATASETS.md](AI_TRAINING_AND_DATASETS.md), [docs/AI_TRAINING_ASSESSMENT_FINAL.md](AI_TRAINING_ASSESSMENT_FINAL.md), [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md)

### [docs/AI_TRAINING_ASSESSMENT_FINAL.md](AI_TRAINING_ASSESSMENT_FINAL.md)

**Status:** Active  
**Purpose:** Training assessment covering memory limits, bug fixes, empirical results, and evaluation methodologies that motivated training guidelines.  
**Related Docs:** [docs/AI_TRAINING_PREPARATION_GUIDE.md](AI_TRAINING_PREPARATION_GUIDE.md), [docs/AI_TRAINING_AND_DATASETS.md](AI_TRAINING_AND_DATASETS.md)

### [docs/AI_LARGE_BOARD_PERFORMANCE_ASSESSMENT.md](AI_LARGE_BOARD_PERFORMANCE_ASSESSMENT.md)

**Status:** Active  
**Purpose:** Assessment of AI performance on large boards (square19, hexagonal) with optimization recommendations.  
**Related Docs:** [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md), [docs/AI_TRAINING_AND_DATASETS.md](AI_TRAINING_AND_DATASETS.md)

---

## G. Rules & Game Logic

### [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md)

**Status:** Active  
**Last Updated:** 2025-11-26  
**Purpose:** Canonical rules specification with stable RR-CANON-RXXX rule IDs. Normalizes narrative sources into precise, implementation-ready constraints including active-no-moves semantics (R200-R207).  
**Related Docs:** [ringrift_complete_rules.md](../ringrift_complete_rules.md), [ringrift_compact_rules.md](../ringrift_compact_rules.md), [RULES_IMPLEMENTATION_MAPPING.md](../RULES_IMPLEMENTATION_MAPPING.md)

### [ringrift_complete_rules.md](../ringrift_complete_rules.md)

**Status:** Active - Authoritative Rulebook  
**Purpose:** Complete, narrative game rules for players and designers. The authoritative rulebook with examples, FAQ (Q1-Q24), and detailed explanations.  
**Related Docs:** [ringrift_compact_rules.md](../ringrift_compact_rules.md), [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md), [ringrift_simple_human_rules.md](../ringrift_simple_human_rules.md)

### [ringrift_compact_rules.md](../ringrift_compact_rules.md)

**Status:** Active  
**Purpose:** Compact, implementation-oriented rules specification for engine/AI authors. Parameterized by board type with formal semantics.  
**Related Docs:** [ringrift_complete_rules.md](../ringrift_complete_rules.md), [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md)

### [ringrift_simple_human_rules.md](../ringrift_simple_human_rules.md)

**Status:** Active
**Purpose:** Simplified human-readable rules summary for quick reference.
**Related Docs:** [ringrift_complete_rules.md](../ringrift_complete_rules.md)

### [docs/GAME_COMPARISON_ANALYSIS.md](GAME_COMPARISON_ANALYSIS.md)

**Status:** Active
**Last Updated:** 2025-12-02
**Purpose:** Comparative analysis of RingRift rules vs other abstract strategy games (YINSH, DVONN, TZAAR, Tak, Go). Assesses design influences, shared mechanics, and uniqueness (~65-70% novel combination).
**Related Docs:** [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md), [ringrift_complete_rules.md](../ringrift_complete_rules.md)

### [RULES_IMPLEMENTATION_MAPPING.md](../RULES_IMPLEMENTATION_MAPPING.md)

**Status:** Active  
**Last Updated:** 2025-11-26  
**Purpose:** Maps canonical RR-CANON rules to implementation (TS/Python) and provides inverse mapping from code to rules. Traceability document for rules changes.  
**Related Docs:** [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md), [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md), [docs/MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md)

### [docs/RULES_ENGINE_SURFACE_AUDIT.md](RULES_ENGINE_SURFACE_AUDIT.md)

**Status:** Active  
**Purpose:** Audit of rules engine surface APIs and responsibilities across shared engine, backend, and sandbox.  
**Related Docs:** [docs/MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md), [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md)

### [docs/RULES_SSOT_MAP.md](RULES_SSOT_MAP.md)

**Status:** Active  
**Purpose:** Maps single sources of truth (SSoT) for rules, lifecycle, and contracts with boundaries and priorities.  
**Related Docs:** [docs/CANONICAL_ENGINE_API.md](CANONICAL_ENGINE_API.md), [docs/SSOT_BANNER_GUIDE.md](SSOT_BANNER_GUIDE.md)

### [docs/SSOT_BANNER_GUIDE.md](SSOT_BANNER_GUIDE.md)

**Status:** Active  
**Purpose:** Guide for maintaining SSoT alignment banners in documentation files.  
**Related Docs:** [docs/RULES_SSOT_MAP.md](RULES_SSOT_MAP.md)

### [docs/ACTIVE_NO_MOVES_BEHAVIOUR.md](ACTIVE_NO_MOVES_BEHAVIOUR.md)

**Status:** Active  
**Purpose:** Comprehensive catalogue of active-no-moves (ANM) scenarios and semantics. Documents ANM-SCEN-01 through ANM-SCEN-08 with test mappings.  
**Related Docs:** [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md), [docs/INVARIANTS_AND_PARITY_FRAMEWORK.md](INVARIANTS_AND_PARITY_FRAMEWORK.md), [ai-service/tests/invariants/](../ai-service/tests/invariants/)

### [docs/API_REFERENCE.md](API_REFERENCE.md)

**Status:** Active  
**Purpose:** REST API overview, endpoints, error codes, and examples for HTTP API surface.  
**Related Docs:** [docs/CANONICAL_ENGINE_API.md](CANONICAL_ENGINE_API.md), [docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)

### [docs/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md](GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md)

**Status:** Active  
**Purpose:** Plan for integrating game replay database with sandbox for testing and analysis.  
**Related Docs:** [ai-service/docs/GAME_REPLAY_DATABASE_SPEC.md](../ai-service/docs/GAME_REPLAY_DATABASE_SPEC.md)

---

## H. Assessment Reports (By Pass)

Chronologically ordered comprehensive project assessments from PASS8 through PASS21.

### [docs/PASS8_ASSESSMENT_REPORT.md](PASS8_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-15  
**Purpose:** Post-architecture remediation assessment. Weakest aspect: Rules parity. Hardest problem: Territory disambiguation.  
**Related Docs:** [docs/PASS9_ASSESSMENT_REPORT.md](PASS9_ASSESSMENT_REPORT.md)

### [docs/PASS9_ASSESSMENT_REPORT.md](PASS9_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-16  
**Purpose:** Post-parity hardening assessment. Focus on rules correctness and test infrastructure.  
**Related Docs:** [docs/PASS8_ASSESSMENT_REPORT.md](PASS8_ASSESSMENT_REPORT.md), [docs/PASS10_VERIFICATION_REPORT.md](PASS10_VERIFICATION_REPORT.md)

### [docs/PASS10_VERIFICATION_REPORT.md](PASS10_VERIFICATION_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-17  
**Purpose:** Verification report following PASS9 recommendations.  
**Related Docs:** [docs/PASS9_ASSESSMENT_REPORT.md](PASS9_ASSESSMENT_REPORT.md), [docs/PASS11_ASSESSMENT_REPORT.md](PASS11_ASSESSMENT_REPORT.md)

### [docs/PASS11_ASSESSMENT_REPORT.md](PASS11_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-18  
**Purpose:** Assessment focusing on test stabilization and parity improvements.  
**Related Docs:** [docs/PASS10_VERIFICATION_REPORT.md](PASS10_VERIFICATION_REPORT.md), [docs/PASS12_ASSESSMENT_REPORT.md](PASS12_ASSESSMENT_REPORT.md)

### [docs/PASS12_ASSESSMENT_REPORT.md](PASS12_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-19  
**Purpose:** Comprehensive assessment of orchestrator integration progress.  
**Related Docs:** [docs/PASS11_ASSESSMENT_REPORT.md](PASS11_ASSESSMENT_REPORT.md), [docs/PASS13_ASSESSMENT_REPORT.md](PASS13_ASSESSMENT_REPORT.md)

### [docs/PASS13_ASSESSMENT_REPORT.md](PASS13_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-20  
**Purpose:** Assessment of Phase 2 orchestrator rollout completion.  
**Related Docs:** [docs/PASS12_ASSESSMENT_REPORT.md](PASS12_ASSESSMENT_REPORT.md), [docs/PASS14_ASSESSMENT_REPORT.md](PASS14_ASSESSMENT_REPORT.md)

### [docs/PASS14_ASSESSMENT_REPORT.md](PASS14_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-21  
**Purpose:** Assessment focusing on contract test implementation.  
**Related Docs:** [docs/PASS13_ASSESSMENT_REPORT.md](PASS13_ASSESSMENT_REPORT.md), [docs/PASS15_ASSESSMENT_REPORT.md](PASS15_ASSESSMENT_REPORT.md)

### [docs/PASS15_ASSESSMENT_REPORT.md](PASS15_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-22  
**Purpose:** Assessment of Phase 3 adapter migration and Python parity.  
**Related Docs:** [docs/PASS14_ASSESSMENT_REPORT.md](PASS14_ASSESSMENT_REPORT.md), [docs/PASS16_ASSESSMENT_REPORT.md](PASS16_ASSESSMENT_REPORT.md)

### [docs/PASS16_ASSESSMENT_REPORT.md](PASS16_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-23  
**Purpose:** Assessment of Phase 4 Python contract test completion.  
**Related Docs:** [docs/PASS15_ASSESSMENT_REPORT.md](PASS15_ASSESSMENT_REPORT.md), [docs/PASS17_ASSESSMENT_REPORT.md](PASS17_ASSESSMENT_REPORT.md)

### [docs/PASS17_ASSESSMENT_REPORT.md](PASS17_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-24  
**Purpose:** Orchestrator rollout, invariants/parity framework establishment, AI healthchecks implementation.  
**Related Docs:** [docs/PASS16_ASSESSMENT_REPORT.md](PASS16_ASSESSMENT_REPORT.md), [docs/PASS18_ASSESSMENT_REPORT.md](PASS18_ASSESSMENT_REPORT.md)

### [docs/PASS18_ASSESSMENT_REPORT.md](PASS18_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-26  
**Purpose:** First-pass global assessment. Weakest aspect: TS rules/host integration parity. Hardest problem: Contract vector maintenance.  
**Related Docs:** [docs/PASS18_REMEDIATION_PLAN.md](PASS18_REMEDIATION_PLAN.md), [docs/PASS18_ASSESSMENT_REPORT_PASS3.md](PASS18_ASSESSMENT_REPORT_PASS3.md)

### [docs/PASS18_REMEDIATION_PLAN.md](PASS18_REMEDIATION_PLAN.md)

**Status:** Historical  
**Purpose:** Detailed remediation plan following PASS18 assessment findings.  
**Related Docs:** [docs/PASS18_ASSESSMENT_REPORT.md](PASS18_ASSESSMENT_REPORT.md)

### [docs/PASS18_WORKING_NOTES.md](PASS18_WORKING_NOTES.md)

**Status:** Historical  
**Purpose:** Working notes and detailed findings from PASS18 assessment process.  
**Related Docs:** [docs/PASS18_ASSESSMENT_REPORT.md](PASS18_ASSESSMENT_REPORT.md)

### [docs/PASS18A_ASSESSMENT_REPORT.md](PASS18A_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-27  
**Purpose:** Post-ANM/termination remediation and test stabilization assessment.  
**Related Docs:** [docs/PASS18_ASSESSMENT_REPORT.md](PASS18_ASSESSMENT_REPORT.md), [docs/PASS18B_ASSESSMENT_REPORT.md](PASS18B_ASSESSMENT_REPORT.md)

### [docs/PASS18B_ASSESSMENT_REPORT.md](PASS18B_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-28  
**Purpose:** Second-pass assessment focusing on type safety and testing improvements.  
**Related Docs:** [docs/PASS18A_ASSESSMENT_REPORT.md](PASS18A_ASSESSMENT_REPORT.md), [docs/PASS18C_ASSESSMENT_REPORT.md](PASS18C_ASSESSMENT_REPORT.md)

### [docs/PASS18C_ASSESSMENT_REPORT.md](PASS18C_ASSESSMENT_REPORT.md)

**Status:** Historical  
**Completion Date:** 2025-11-29  
**Purpose:** Third-pass assessment. Post-accessibility & type safety remediation. 0 TS errors, 55 ARIA attributes.  
**Related Docs:** [docs/PASS18B_ASSESSMENT_REPORT.md](PASS18B_ASSESSMENT_REPORT.md), [docs/PASS18_ASSESSMENT_REPORT_PASS3.md](PASS18_ASSESSMENT_REPORT_PASS3.md)

### [docs/PASS18_ASSESSMENT_REPORT_PASS3.md](PASS18_ASSESSMENT_REPORT_PASS3.md)

**Status:** Historical  
**Completion Date:** 2025-11-29  
**Purpose:** Comprehensive third-pass global assessment. Weakest aspect: Frontend UX (3.5/5). Hardest problem: Test suite cleanup.  
**Related Docs:** [docs/PASS18C_ASSESSMENT_REPORT.md](PASS18C_ASSESSMENT_REPORT.md), [docs/PASS19A_ASSESSMENT_REPORT.md](PASS19A_ASSESSMENT_REPORT.md)

### [docs/PASS19A_ASSESSMENT_REPORT.md](PASS19A_ASSESSMENT_REPORT.md)

**Status:** Active  
**Completion Date:** 2025-11-30  
**Purpose:** Latest comprehensive assessment (2,709 tests passing, 0 failing, 63 ARIA attrs). Post-code cleanup & documentation refresh. Weakest: Frontend UX (3.5/5).  
**Related Docs:** [docs/PASS18_ASSESSMENT_REPORT_PASS3.md](PASS18_ASSESSMENT_REPORT_PASS3.md), [docs/PASS19B_ASSESSMENT_REPORT.md](PASS19B_ASSESSMENT_REPORT.md)

### [docs/PASS19B_ASSESSMENT_REPORT.md](PASS19B_ASSESSMENT_REPORT.md)

**Status:** Active  
**Purpose:** CI-gated test health summary and E2E test infrastructure assessment.  
**Related Docs:** [docs/PASS19A_ASSESSMENT_REPORT.md](PASS19A_ASSESSMENT_REPORT.md), [docs/PASS20_ASSESSMENT.md](PASS20_ASSESSMENT.md)

### [docs/PASS20_ASSESSMENT.md](PASS20_ASSESSMENT.md)

**Status:** Active  
**Completion Date:** 2025-12-01  
**Purpose:** Extended/diagnostic Jest profile analysis including jest-results.json with 72 failing tests categorization.  
**Related Docs:** [docs/PASS19B_ASSESSMENT_REPORT.md](PASS19B_ASSESSMENT_REPORT.md), [docs/PASS20_COMPLETION_SUMMARY.md](PASS20_COMPLETION_SUMMARY.md)

### [docs/PASS20_COMPLETION_SUMMARY.md](PASS20_COMPLETION_SUMMARY.md)

**Status:** Active  
**Completion Date:** 2025-12-01  
**Purpose:** Summary of PASS20 completion including 29 subtasks, orchestrator Phase 3, ~1,118 lines legacy code removal, and test stabilization achievements.  
**Related Docs:** [docs/PASS20_ASSESSMENT.md](PASS20_ASSESSMENT.md), [docs/PASS21_ASSESSMENT_REPORT.md](PASS21_ASSESSMENT_REPORT.md)

### [docs/PASS21_ASSESSMENT_REPORT.md](PASS21_ASSESSMENT_REPORT.md)

**Status:** Active  
**Completion Date:** 2025-12-01  
**Purpose:** Most recent comprehensive assessment. Weakest aspect: Operations & Observability (2.5/5). Hardest problem: Production-scale validation. Critical finding: Missing observability infrastructure.  
**Related Docs:** [docs/PASS20_COMPLETION_SUMMARY.md](PASS20_COMPLETION_SUMMARY.md), [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md)

### P18 Task Documentation

- **[docs/P18.1-1_CAPTURE_TERRITORY_HOST_MAP.md](P18.1-1_CAPTURE_TERRITORY_HOST_MAP.md)** - Capture and territory host implementation mapping
- **[docs/P18.2-1_AI_RNG_PATHS.md](P18.2-1_AI_RNG_PATHS.md)** - AI RNG code path analysis and determinism requirements
- **[docs/P18.3-1_DECISION_LIFECYCLE_SPEC.md](P18.3-1_DECISION_LIFECYCLE_SPEC.md)** - Decision lifecycle specification for PlayerChoice flows
- **[docs/P18.4-3_ORCHESTRATOR_STAGING_REPORT.md](P18.4-3_ORCHESTRATOR_STAGING_REPORT.md)** - Orchestrator staging deployment report
- **[docs/P18.5-3_ORCHESTRATOR_EXTENDED_VECTOR_SOAK_REPORT.md](P18.5-3_ORCHESTRATOR_EXTENDED_VECTOR_SOAK_REPORT.md)** - Extended vector soak testing results
- **[docs/P18.5-4_SWAP_SIDES_PARITY_REPORT.md](P18.5-4_SWAP_SIDES_PARITY_REPORT.md)** - Swap sides (pie rule) parity validation report

---

## I. Incident Documentation

### [docs/incidents/INDEX.md](incidents/INDEX.md)

**Status:** Active  
**Last Updated:** 2025-11-25  
**Purpose:** Incident response guide index with alert-to-response mapping, severity levels, communication templates, and on-call responsibilities.  
**Related Docs:** [docs/ALERTING_THRESHOLDS.md](ALERTING_THRESHOLDS.md), [docs/runbooks/INDEX.md](runbooks/INDEX.md)

### Incident Response Guides

- **[TRIAGE_GUIDE.md](incidents/TRIAGE_GUIDE.md)** - Initial triage procedures for any incident
- **[AVAILABILITY.md](incidents/AVAILABILITY.md)** - Service availability and degradation incidents (DatabaseDown, HighErrorRate, etc.)
- **[LATENCY.md](incidents/LATENCY.md)** - Performance and latency incidents
- **[RESOURCES.md](incidents/RESOURCES.md)** - Memory, CPU, and resource exhaustion incidents
- **[AI_SERVICE.md](incidents/AI_SERVICE.md)** - AI service failures and fallback incidents
- **[SECURITY.md](incidents/SECURITY.md)** - Rate limiting and security-related incidents
- **[POST_MORTEM_TEMPLATE.md](incidents/POST_MORTEM_TEMPLATE.md)** - Post-incident review template

### [docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md](INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md)

**Status:** Active  
**Purpose:** Incident report and fix for TerritoryMutator divergence between Python and TS engines. Documents root cause, resolution, and preventative measures.  
**Related Docs:** [ai-service/tests/test_territory_forced_elimination_divergence.py](../ai-service/tests/test_territory_forced_elimination_divergence.py), [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md)

---

## J. Draft/Work-in-Progress

Documents under active development or pending final review.

### [docs/drafts/LEGACY_CODE_ELIMINATION_PLAN.md](drafts/LEGACY_CODE_ELIMINATION_PLAN.md)

**Status:** Draft  
**Purpose:** Plan for eliminating remaining legacy code patterns post-orchestrator rollout.  
**Related Docs:** [docs/ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md](ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md)

### [docs/drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md](drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md)

**Status:** Draft  
**Purpose:** Feature flag design for orchestrator rollout (now largely superseded by implemented flags).  
**Related Docs:** [docs/ORCHESTRATOR_ROLLOUT_PLAN.md](ORCHESTRATOR_ROLLOUT_PLAN.md), [docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)

### [docs/drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md](drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md)

**Status:** Draft/Historical  
**Purpose:** Backend and sandbox adapter migration report (mostly complete).  
**Related Docs:** [ARCHITECTURE_ASSESSMENT.md](../ARCHITECTURE_ASSESSMENT.md)

### [docs/drafts/RULES_ENGINE_CONSOLIDATION_DESIGN.md](drafts/RULES_ENGINE_CONSOLIDATION_DESIGN.md)

**Status:** Draft/Historical  
**Purpose:** Original consolidation design document (design reference, partially historical).  
**Related Docs:** [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md), [ARCHITECTURE_ASSESSMENT.md](../ARCHITECTURE_ASSESSMENT.md)

---

## K. Supplementary Documentation

Detailed analysis and clarifications supporting the main documentation.

### [docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md](supplementary/RULES_CONSISTENCY_EDGE_CASES.md)

**Status:** Active  
**Purpose:** Edge case analysis and handling for rules consistency. Documents ambiguous scenarios and canonical interpretations.  
**Related Docs:** [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md), [ringrift_complete_rules.md](../ringrift_complete_rules.md)

### [docs/supplementary/RULES_RULESET_CLARIFICATIONS.md](supplementary/RULES_RULESET_CLARIFICATIONS.md)

**Status:** Active  
**Purpose:** Clarifications for ambiguous rules and edge cases discovered during implementation.  
**Related Docs:** [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md), [docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md](supplementary/RULES_CONSISTENCY_EDGE_CASES.md)

### [docs/supplementary/AI_IMPROVEMENT_BACKLOG.md](supplementary/AI_IMPROVEMENT_BACKLOG.md)

**Status:** Active  
**Purpose:** Backlog of AI improvements grouped by theme (difficulty ladder, RNG/determinism, search performance, NN robustness, behavior tuning).  
**Related Docs:** [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md), [docs/AI_TRAINING_AND_DATASETS.md](AI_TRAINING_AND_DATASETS.md)

### [docs/supplementary/RULES_DOCS_UX_AUDIT.md](supplementary/RULES_DOCS_UX_AUDIT.md)

**Status:** Active  
**Purpose:** UX audit of rules documentation identifying clarity improvements and navigation enhancements.  
**Related Docs:** [ringrift_complete_rules.md](../ringrift_complete_rules.md), [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md)

### [docs/supplementary/RULES_TERMINATION_ANALYSIS.md](supplementary/RULES_TERMINATION_ANALYSIS.md)

**Status:** Active  
**Purpose:** Analysis of game termination guarantees and termination ladder (finite termination proof).  
**Related Docs:** [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md), [docs/INVARIANTS_AND_PARITY_FRAMEWORK.md](INVARIANTS_AND_PARITY_FRAMEWORK.md)

---

## Additional Important Documents

### Project Planning & Status

#### [PROJECT_GOALS.md](../PROJECT_GOALS.md)

**Status:** Active - Canonical Goals  
**Last Updated:** 2025-11-27  
**Purpose:** Canonical project goals, success criteria, and scope boundaries. Defines ruleset design goals including emergent complexity and human-computer competitive balance.  
**Related Docs:** [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md), [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md)

#### [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md)

**Status:** Active  
**Last Updated:** 2025-12-01  
**Purpose:** Factual, code-verified status snapshot. Current implementation state, test counts, component scores, and verified code status.  
**Related Docs:** [TODO.md](../TODO.md), [KNOWN_ISSUES.md](../KNOWN_ISSUES.md), [WEAKNESS_ASSESSMENT_REPORT.md](../WEAKNESS_ASSESSMENT_REPORT.md)

#### [TODO.md](../TODO.md)

**Status:** Active  
**Last Updated:** 2025-11-30  
**Purpose:** Phase-structured task tracker with consolidated execution tracks and detailed implementation checklists.  
**Related Docs:** [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md), [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)

#### [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)

**Status:** Active  
**Last Updated:** 2025-12-01  
**Purpose:** Current bugs, gaps, and prioritization (P0/P1/P2). Specific issues with implementation notes.  
**Related Docs:** [TODO.md](../TODO.md), [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md)

#### [WEAKNESS_ASSESSMENT_REPORT.md](../WEAKNESS_ASSESSMENT_REPORT.md)

**Status:** Active  
**Purpose:** Latest comprehensive weakness assessment identifying weakest aspects and hardest problems across all PASS reports.  
**Related Docs:** [docs/PASS21_ASSESSMENT_REPORT.md](PASS21_ASSESSMENT_REPORT.md), [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md)

### Security & Privacy

#### [docs/SECURITY_THREAT_MODEL.md](SECURITY_THREAT_MODEL.md)

**Status:** Active (with historical/aspirational content)  
**Last Updated:** 2025-11-27  
**Purpose:** Security threat model covering assets, trust boundaries, attacker profiles, and S-05 security hardening backlog (S-05.A through S-05.F).  
**Related Docs:** [docs/SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md), [docs/DATA_LIFECYCLE_AND_PRIVACY.md](DATA_LIFECYCLE_AND_PRIVACY.md), [docs/SUPPLY_CHAIN_AND_CI_SECURITY.md](SUPPLY_CHAIN_AND_CI_SECURITY.md)

#### [docs/DATA_LIFECYCLE_AND_PRIVACY.md](DATA_LIFECYCLE_AND_PRIVACY.md)

**Status:** Active  
**Purpose:** Data inventory, retention/anonymization policies, and account deletion/export workflows (S-05.E implementation).  
**Related Docs:** [docs/SECURITY_THREAT_MODEL.md](SECURITY_THREAT_MODEL.md), [docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)

#### [docs/SUPPLY_CHAIN_AND_CI_SECURITY.md](SUPPLY_CHAIN_AND_CI_SECURITY.md)

**Status:** Active  
**Purpose:** Supply-chain & CI/CD threat overview, current controls vs gaps, and S-05.F implementation tracks for dependency, CI, Docker, and secret-management hardening.  
**Related Docs:** [docs/SECURITY_THREAT_MODEL.md](SECURITY_THREAT_MODEL.md), [CONTRIBUTING.md](../CONTRIBUTING.md), [.github/workflows/ci.yml](../.github/workflows/ci.yml)

### Legacy & Archive

#### [archive/](../archive/)

**Status:** Historical Archive  
**Purpose:** Contains historical assessments, reports, and design documents preserved for context. Not actively maintained but useful for understanding project evolution.

**Key Archived Documents:**

- `FINAL_ARCHITECT_REPORT.md` - Comprehensive architecture analysis (historical)
- `FINAL_RULES_AUDIT_REPORT.md` - Rules audit findings (historical)
- `PHASE1_REMEDIATION_PLAN.md` - Historical Phase 1 plan (superseded)
- `PHASE3_ADAPTER_MIGRATION_REPORT.md` - Historical adapter migration (superseded)
- `PHASE4_PYTHON_CONTRACT_TEST_REPORT.md` - Historical Python contract tests (superseded)
- `AI_ASSESSMENT_REPORT.md` - Historical AI assessment
- `CODEBASE_EVALUATION.md` - Historical codebase evaluation (superseded by ARCHITECTURE_ASSESSMENT.md)

---

## Finding Documentation

### By Topic

- **Getting Started:** See section [A. Getting Started](#a-getting-started)
- **Architecture:** See section [B. Architecture & Design](#b-architecture--design)
- **Deployment & Operations:** See section [C. Operations & Deployment](#c-operations--deployment)
- **Testing:** See section [D. Testing & Quality](#d-testing--quality)
- **Monitoring:** See section [E. Monitoring & Observability](#e-monitoring--observability)
- **AI & Training:** See section [F. AI & Training](#f-ai--training)
- **Rules:** See section [G. Rules & Game Logic](#g-rules--game-logic)
- **Incidents:** See section [I. Incident Documentation](#i-incident-documentation)

### By Audience

**New Contributors:**

1. Start with [README.md](../README.md) for project overview
2. Follow [QUICKSTART.md](../QUICKSTART.md) for environment setup
3. Read [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
4. Review [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md) for current status
5. Check [TODO.md](../TODO.md) for active tasks

**DevOps Engineers:**

1. Review [docs/DEPLOYMENT_REQUIREMENTS.md](DEPLOYMENT_REQUIREMENTS.md) for infrastructure needs
2. Study [docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) for configuration
3. Familiarize with [docs/runbooks/INDEX.md](runbooks/INDEX.md) for operational procedures
4. Review [docs/SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md) for security
5. Check [docs/ALERTING_THRESHOLDS.md](ALERTING_THRESHOLDS.md) for monitoring

**QA Engineers:**

1. Start with [docs/TEST_CATEGORIES.md](TEST_CATEGORIES.md) for test organization
2. Review [docs/TEST_INFRASTRUCTURE.md](TEST_INFRASTRUCTURE.md) for test utilities
3. Study [RULES_SCENARIO_MATRIX.md](../RULES_SCENARIO_MATRIX.md) for rules coverage
4. Check [docs/INVARIANTS_AND_PARITY_FRAMEWORK.md](INVARIANTS_AND_PARITY_FRAMEWORK.md) for parity expectations
5. Review [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) for known bugs

**AI Researchers:**

1. Start with [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md) for AI system overview
2. Review [docs/AI_TRAINING_AND_DATASETS.md](AI_TRAINING_AND_DATASETS.md) for training pipelines
3. Study [docs/AI_TRAINING_PREPARATION_GUIDE.md](AI_TRAINING_PREPARATION_GUIDE.md) for setup
4. Check [docs/AI_LARGE_BOARD_PERFORMANCE_ASSESSMENT.md](AI_LARGE_BOARD_PERFORMANCE_ASSESSMENT.md) for optimization
5. Review [ai-service/README.md](../ai-service/README.md) for Python service details

**Rules Designers:**

1. Read [ringrift_complete_rules.md](../ringrift_complete_rules.md) - the authoritative rulebook
2. Review [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md) for formal specifications
3. Study [RULES_IMPLEMENTATION_MAPPING.md](../RULES_IMPLEMENTATION_MAPPING.md) for code mapping
4. Check [docs/ACTIVE_NO_MOVES_BEHAVIOUR.md](ACTIVE_NO_MOVES_BEHAVIOUR.md) for edge cases
5. Review [docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md](supplementary/RULES_CONSISTENCY_EDGE_CASES.md)

**System Architects:**

1. Review [ARCHITECTURE_ASSESSMENT.md](../ARCHITECTURE_ASSESSMENT.md) for overall architecture
2. Study [docs/CANONICAL_ENGINE_API.md](CANONICAL_ENGINE_API.md) for API design
3. Review [docs/MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md) for module organization
4. Check [docs/STATE_MACHINES.md](STATE_MACHINES.md) for lifecycle flows
5. Study [docs/ORCHESTRATOR_ROLLOUT_PLAN.md](ORCHESTRATOR_ROLLOUT_PLAN.md) for migration strategy

### By File Type

**Markdown Documentation:** All `.md` files in project root and `docs/` directory  
**Test Fixtures:** `tests/fixtures/` directory  
**Contract Vectors:** `tests/fixtures/contract-vectors/v2/`  
**Configuration:** `.env.example`, `docker-compose.yml`, `docker-compose.staging.yml`  
**Monitoring Config:** `monitoring/prometheus/`, `monitoring/grafana/`

### By Status

**Active (Current):**

- All documents in sections A-G
- Recent PASS reports (PASS19A, PASS19B, PASS20, PASS21)
- All runbooks and incident guides
- Current supplementary docs

**Historical (Reference Only):**

- PASS8 through PASS18C reports (evolution history)
- Documents in `archive/` directory
- Some draft documents superseded by active versions

**Draft (Under Development):**

- Documents in `docs/drafts/` directory
- Work-in-progress specifications

### Quick Search Tips

**Finding Rules Information:**

```bash
# Search for specific rule references
grep -r "RR-CANON-R" docs/ RULES_*.md
grep -r "FAQ Q" ringrift_*.md docs/
```

**Finding Implementation Details:**

```bash
# Search for module documentation
grep -r "Primary Responsibility" docs/MODULE_RESPONSIBILITIES.md
grep -r "Status: Active" docs/
```

**Finding Operational Procedures:**

```bash
# List all runbooks
ls docs/runbooks/
# Search incident guides
ls docs/incidents/
```

### Common Documentation Queries

**"How do I deploy to production?"**

- [docs/DEPLOYMENT_REQUIREMENTS.md](DEPLOYMENT_REQUIREMENTS.md) - Infrastructure requirements
- [docs/runbooks/DEPLOYMENT_ROUTINE.md](runbooks/DEPLOYMENT_ROUTINE.md) - Deployment procedure
- [docs/SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md) - Secret handling

**"How do I run tests?"**

- [docs/TEST_CATEGORIES.md](TEST_CATEGORIES.md) - Test organization
- [tests/README.md](../tests/README.md) - Comprehensive testing guide
- [QUICKSTART.md](../QUICKSTART.md) - Running tests during development

**"How are rules implemented?"**

- [RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md) - Formal specifications
- [RULES_IMPLEMENTATION_MAPPING.md](../RULES_IMPLEMENTATION_MAPPING.md) - Code mapping
- [docs/MODULE_RESPONSIBILITIES.md](MODULE_RESPONSIBILITIES.md) - Module catalog

**"What's the current project status?"**

- [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md) - Verified current state
- [docs/PASS21_ASSESSMENT_REPORT.md](PASS21_ASSESSMENT_REPORT.md) - Latest assessment
- [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) - Current bugs and gaps
- [TODO.md](../TODO.md) - Active tasks

**"How do I set up the AI service?"**

- [QUICKSTART.md](../QUICKSTART.md) - Quick setup (Section 2)
- [ai-service/README.md](../ai-service/README.md) - Detailed AI service guide
- [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md) - AI architecture overview
- [docs/AI_TRAINING_AND_DATASETS.md](AI_TRAINING_AND_DATASETS.md) - Training pipelines

**"How do I respond to an alert?"**

- [docs/incidents/INDEX.md](incidents/INDEX.md) - Alert-to-response mapping
- [docs/runbooks/INDEX.md](runbooks/INDEX.md) - Runbook index
- [docs/ALERTING_THRESHOLDS.md](ALERTING_THRESHOLDS.md) - Alert configurations

### Cross-Reference Map

**Rules Documentation Chain:**

```
ringrift_complete_rules.md (narrative)
    ↓
ringrift_compact_rules.md (implementation-oriented)
    ↓
RULES_CANONICAL_SPEC.md (formal RR-CANON-RXXX)
    ↓
RULES_IMPLEMENTATION_MAPPING.md (code mapping)
    ↓
src/shared/engine/** (implementation)
```

**Architecture Documentation Chain:**

```
PROJECT_GOALS.md (goals & scope)
    ↓
ARCHITECTURE_ASSESSMENT.md (architecture overview)
    ↓
RULES_ENGINE_ARCHITECTURE.md (rules engine)
    ↓
AI_ARCHITECTURE.md (AI system)
    ↓
docs/CANONICAL_ENGINE_API.md (API spec)
    ↓
docs/MODULE_RESPONSIBILITIES.md (module catalog)
```

**Operational Documentation Chain:**

```
DEPLOYMENT_REQUIREMENTS.md (infrastructure)
    ↓
ENVIRONMENT_VARIABLES.md (configuration)
    ↓
SECRETS_MANAGEMENT.md (security)
    ↓
docs/runbooks/INDEX.md (procedures)
    ↓
docs/incidents/INDEX.md (incident response)
```

**Testing Documentation Chain:**

```
TEST_CATEGORIES.md (organization)
    ↓
TEST_INFRASTRUCTURE.md (utilities)
    ↓
RULES_SCENARIO_MATRIX.md (coverage map)
    ↓
INVARIANTS_AND_PARITY_FRAMEWORK.md (invariants)
    ↓
CONTRACT_VECTORS_DESIGN.md (parity vectors)
```

---

## Document Update Guidelines

When adding or modifying documentation:

1. **Add to Index:** Update this file with the new document in the appropriate section
2. **Set Status:** Mark as Active, Draft, or Historical
3. **Add Date:** Include Last Updated date in document header
4. **Link Related:** Add cross-references to related documents
5. **Update TOC:** Ensure Table of Contents is current
6. **Follow SSoT:** Respect SSoT alignment banners in documents
7. **Update INDEX.md:** Also update the high-level [docs/INDEX.md](INDEX.md) if needed

### SSoT Precedence

When documents conflict:

1. **Rules Semantics:** RULES_CANONICAL_SPEC.md + src/shared/engine/\*\* win
2. **Lifecycle/API:** docs/CANONICAL_ENGINE_API.md + shared types/schemas win
3. **Current Status:** CURRENT_STATE_ASSESSMENT.md + code win over assessments
4. **Configuration:** docs/ENVIRONMENT_VARIABLES.md + src/server/config/\*\* win

---

**Index Version:** 1.0  
**Generated:** 2025-12-01  
**Maintained By:** Architecture Team  
**Next Review:** When major documentation changes occur
