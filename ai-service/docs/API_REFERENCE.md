# AI Service API Reference

> **Doc Status (2025-12-29): Active**
>
> **SSoT:** `ai-service/app/main.py` and `ai-service/app/routes/*.py`.
> Use the FastAPI interactive docs at `http://localhost:8001/docs` for
> request/response schemas.

## FastAPI Service (port 8001)

### Service Metadata + Health

| Method | Endpoint   | Purpose                                    |
| ------ | ---------- | ------------------------------------------ |
| GET    | `/`        | Service info (`name`, `status`, `version`) |
| GET    | `/health`  | Health check (200 healthy, 503 degraded)   |
| GET    | `/ready`   | Readiness check (200/503)                  |
| GET    | `/live`    | Liveness check (200 when process alive)    |
| GET    | `/metrics` | Prometheus metrics (text format)           |

### AI Move & Evaluation

| Method | Endpoint                | Purpose                                |
| ------ | ----------------------- | -------------------------------------- |
| POST   | `/ai/move`              | Select a single AI move                |
| POST   | `/ai/moves_batch`       | Batch move selection                   |
| POST   | `/ai/evaluate`          | Evaluate a position                    |
| POST   | `/ai/evaluate_position` | Evaluate with detailed score breakdown |

### AI Choice Endpoints

These mirror TypeScript decision heuristics used during choice phases.

| Method | Endpoint                        | Purpose                        |
| ------ | ------------------------------- | ------------------------------ |
| POST   | `/ai/choice/line_reward_option` | Choose line reward option      |
| POST   | `/ai/choice/ring_elimination`   | Choose ring elimination target |
| POST   | `/ai/choice/region_order`       | Choose region order            |
| POST   | `/ai/choice/line_order`         | Choose line order              |
| POST   | `/ai/choice/capture_direction`  | Choose capture direction       |

### Rules Evaluation

| Method | Endpoint               | Purpose                                      |
| ------ | ---------------------- | -------------------------------------------- |
| POST   | `/rules/evaluate_move` | Validate move and return next state + hashes |

### Replay API (Mounted)

The replay router is mounted at `/api/replay`.

| Method | Endpoint                              | Purpose                        |
| ------ | ------------------------------------- | ------------------------------ |
| GET    | `/api/replay/games`                   | List games with filters        |
| GET    | `/api/replay/games/{game_id}`         | Fetch metadata for a game      |
| GET    | `/api/replay/games/{game_id}/state`   | Reconstruct state at a move    |
| GET    | `/api/replay/games/{game_id}/moves`   | List moves in a game           |
| GET    | `/api/replay/games/{game_id}/choices` | List choice records for a game |
| GET    | `/api/replay/stats`                   | DB statistics                  |
| POST   | `/api/replay/games`                   | Store a game (sandbox tooling) |

### Model & Cache Metadata

| Method | Endpoint          | Purpose                                            |
| ------ | ----------------- | -------------------------------------------------- |
| GET    | `/ai/models`      | Report currently loaded model artifacts            |
| GET    | `/ai/cache/stats` | AI instance cache stats                            |
| DELETE | `/ai/cache`       | Clear AI cache (**admin**, requires `X-Admin-Key`) |

### Admin / Operations (requires `X-Admin-Key`)

| Method | Endpoint                        | Purpose                     |
| ------ | ------------------------------- | --------------------------- |
| GET    | `/admin/health/coordinators`    | Coordinator health snapshot |
| GET    | `/admin/health/full`            | Full health snapshot        |
| GET    | `/admin/sync/status`            | Data sync status            |
| POST   | `/admin/sync/trigger`           | Trigger a data sync         |
| POST   | `/admin/sync/data-server/start` | Start data server           |
| POST   | `/admin/sync/data-server/stop`  | Stop data server            |
| GET    | `/admin/velocity`               | Training velocity metrics   |

### Internal

| Method | Endpoint                  | Purpose                                     |
| ------ | ------------------------- | ------------------------------------------- |
| GET    | `/internal/ladder/health` | Ladder tiers + artifact availability report |

## Optional Routers (Not Mounted by Default)

The following routers live under `ai-service/app/routes/` but are not included
in `main.py` unless `include_all_routes(app)` is wired manually:

- `cluster.py` -> `/api/cluster/*`
- `training.py` -> `/api/training/*`

If you expose these endpoints, document the deployment in
`README.md` and keep this reference in sync.

## Related Services (Separate Processes)

The P2P orchestrator is a standalone service started via
`scripts/p2p_orchestrator.py` (default port 8770). Its API surface is documented
in `../app/p2p/README.md` and `infrastructure/`.
