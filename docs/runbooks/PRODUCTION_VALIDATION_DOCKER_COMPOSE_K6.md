# Production Validation via docker-compose + k6

> **Doc Status (2025-12-01): Active Runbook**  
> **Role:** Step‑by‑step guide for validating a RingRift build at production‑like scale using the standard `docker-compose.yml` stack plus the k6 load harness under `tests/load/`.
>
> **SSoT alignment:** This runbook is derived from:
>
> - `docker-compose.yml` – canonical local/staging stack definition (app, Postgres, Redis, AI service, Prometheus, Alertmanager, Grafana, nginx).
> - `tests/load/README.md` – k6 scenario definitions and SLO thresholds.
> - `monitoring/prometheus/prometheus.yml` and `monitoring/prometheus/alerts.yml` – scrape and alert rules.
> - `monitoring/grafana/dashboards/*.json` – Game Performance, Rules Correctness, and System Health dashboards.
> - `docs/PASS22_ASSESSMENT_REPORT.md` – PASS22 production validation goals and success criteria.
>
> **Precedence:** Docker compose files, k6 configs, and alert/dashboard definitions are authoritative for how the stack behaves. This document explains **how to run validation**; if procedures here diverge from configs or scripts, update the runbook to match code and infra.

---

## 1. Bring Up the Staging‑Like Stack

From the project root:

```bash
docker compose up -d
```

Verify that core services are healthy:

```bash
curl -s http://localhost:3000/health | jq      # App HTTP
curl -s http://localhost:3001/health | jq      # App WebSocket/HTTP proxy (if exposed)
curl -s http://localhost:8001/health | jq      # AI service
```

Confirm monitoring components:

- Prometheus UI: http://localhost:9090
- Grafana UI: http://localhost:3002 (default admin password from `GRAFANA_PASSWORD` or `admin`)
- Alertmanager UI: http://localhost:9093

> **Environment sanity checks (from `docker-compose.yml`):**
>
> - Ensure you are **not** using the placeholder JWT secrets in real environments:
>   - `JWT_SECRET`, `JWT_REFRESH_SECRET` should be set via `.env` / secrets manager.
> - Verify topology flags:
>   - `RINGRIFT_APP_TOPOLOGY=single` for single‑instance app.
>   - `RINGRIFT_RULES_MODE=ts` (default) unless running explicit python‑mode diagnostics.
> - Check service URLs:
>   - `AI_SERVICE_URL=http://ai-service:8001` (internal) is reachable from the app container.

---

## 2. Run k6 Load Scenarios Against docker‑compose

All k6 scenarios are defined under `tests/load/` and documented in `tests/load/README.md`.

### 2.1 Install Dependencies

On the host:

```bash
cd tests/load
npm install
```

Ensure k6 is available:

```bash
k6 version
```

or use the Docker image:

```bash
docker pull grafana/k6:latest
```

### 2.2 Smoke Validation

From `tests/load/`:

```bash
npm run validate:syntax
npm run validate:dry-run

# Quick smoke on creation + concurrent games
npm run test:smoke:game-creation
npm run test:smoke:concurrent-games
```

Target the docker‑compose app:

```bash
BASE_URL=http://localhost:3001 k6 run scenarios/game-creation.js
```

### 2.3 Full Load Profile

Run all four primary scenarios sequentially (recommended for staging validation):

```bash
cd tests/load
npm run test:load:all
```

Or individually:

```bash
BASE_URL=http://localhost:3001 k6 run scenarios/game-creation.js
BASE_URL=http://localhost:3001 k6 run scenarios/concurrent-games.js
BASE_URL=http://localhost:3001 k6 run scenarios/player-moves.js
BASE_URL=http://localhost:3001 \
WS_URL=ws://localhost:3001 \
  k6 run scenarios/websocket-stress.js
```

Align against SLOs from `tests/load/config/thresholds.json` and `monitoring/prometheus/alerts.yml` (move latency, HTTP latency, error rates).

---

## 3. Use Grafana Dashboards as the Red/Green Signal

Open Grafana (http://localhost:3002) and consult:

1. **Game Performance** dashboard
   - Validate during load:
     - `ringrift_game_move_latency_seconds` – P95/P99 under thresholds for each `board_type`.
     - `ringrift_games_active` – expected concurrency (e.g., ≥100 games in concurrent‑games scenario).
     - `ringrift_game_session_abnormal_termination_total` – no unexpected spikes (abandonment/timeouts) under normal test flows.

2. **Rules Correctness** dashboard
   - Confirm:
     - `ringrift_invariant_violations_total` remains zero.
     - Parity/contract error metrics stay flat during test runs.

3. **System Health** dashboard
   - Watch:
     - CPU/memory for app, ai‑service, Postgres, and Redis.
     - `ringrift_websocket_reconnection_total` and connection gauges during WebSocket stress.
     - HTTP error rates and saturations.

> **Green run:** All four k6 scenarios complete without breaches of:
>
> - Latency/error SLOs (per k6 output and Prometheus alerts).
> - Move latency and active‑games metrics in Grafana.
> - Invariant/parity metrics (no violations).

---

## 4. Tuning Environment & Resource Limits

If k6 or dashboards reveal issues:

1. **App / AI service memory or CPU saturation**
   - Adjust `deploy.resources.limits` and `reservations` for `app` and `ai-service` in `docker-compose.yml`.
   - Re‑run the most problematic scenario (often `concurrent-games` or `player-moves`) to confirm headroom.

2. **Postgres bottlenecks**
   - Check Postgres CPU and I/O; if consistently high:
     - Increase `postgres` memory limits in `docker-compose.yml`.
     - Consider adjusting connection pool size in `src/server/config/unified.ts` / Prisma config.

3. **WebSocket saturation**
   - Use the WebSocket stress scenario plus the System Health dashboard:
     - If connection failures or high error rates appear well below target (e.g., <500 connections), investigate:
       - `WEBSOCKET_ISSUES.md` and `WEBSOCKET_SCALING.md`.
       - Nginx config (`nginx.conf`) and any upstream timeouts.

Record any permanent limit changes back into:

- `docker-compose.yml` (for local/staging).
- `docs/planning/DEPLOYMENT_REQUIREMENTS.md` and `docs/architecture/TOPOLOGY_MODES.md` for production expectations.

---

## 5. Capturing Results in PASS22 / Baseline Docs

After a successful validation run:

1. Update the **Production Validation** section in `docs/PASS22_ASSESSMENT_REPORT.md` / `PASS22_COMPLETION_SUMMARY.md` with:
   - Date and commit hash used for validation.
   - Scenarios executed (smoke vs full load).
   - High‑level metrics summary (e.g., P95 move latency, max concurrent games).

2. Add baseline ranges to:
   - `docs/runbooks/GAME_HEALTH.md` – expected `ringrift_games_active`, abnormal termination rates under healthy load.
   - `docs/runbooks/GAME_PERFORMANCE.md` – typical P95/P99 move latencies per board type at target scale.

This closes the PASS22 “Production Validation” gap by pairing the existing dashboards and k6 harness with an explicit, repeatable procedure and documented baselines.
