# Production vs Training Infrastructure

This document explains the separation between RingRift's production deployment and AI training infrastructure.

## Overview

RingRift has two distinct deployment concerns:

| Concern        | Directory       | Purpose                   | Daemons | Memory |
| -------------- | --------------- | ------------------------- | ------- | ------ |
| **Production** | `ai-inference/` | Serve AI moves to players | 0       | ~500MB |
| **Training**   | `ai-service/`   | Train new AI models       | 132     | 2-8GB  |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION DEPLOYMENT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Frontend   │◄──►│  Game Server │◄──►│ AI Inference │       │
│  │   (React)    │    │  (Node.js)   │    │  (FastAPI)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                          Pre-trained             │
│                                          Models (.pth)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING INFRASTRUCTURE                      │
├─────────────────────────────────────────────────────────────────┤
│  (Runs separately, NOT part of production deployment)            │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    P2P Mesh  │◄──►│   Selfplay   │──► │   Training   │       │
│  │   (41 nodes) │    │   Workers    │    │   Pipeline   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               132 Coordination Daemons                    │   │
│  │  (Sync, Evaluation, Distribution, Health, Recovery, etc.) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│                    New Models (.pth) ──────────────────────────►│
│                                           Copy to production     │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
RingRift/
├── ai-inference/           # PRODUCTION: Minimal inference service
│   ├── app/main.py         #   ~150 lines FastAPI server
│   ├── models/             #   Pre-trained .pth files
│   ├── Dockerfile          #   Production container
│   └── README.md           #   Deployment docs
│
├── ai-service/             # TRAINING: Full infrastructure
│   ├── app/coordination/   #   132 daemon types
│   ├── scripts/p2p/        #   P2P mesh orchestration
│   ├── scripts/master_loop.py  # Main automation
│   └── ...                 #   ~235K lines total
│
└── docker-compose.production.yml  # Uses ai-inference
└── docker-compose.yml             # Uses ai-service (development)
```

## When to Use Each

### Production Deployment (`ai-inference/`)

Use for:

- Serving the game to players
- Production website/app
- Low resource environments
- Simple deployment

```bash
# Production deployment
docker compose -f docker-compose.production.yml up -d
```

### Training Infrastructure (`ai-service/`)

Use for:

- Training new models
- Running selfplay
- Distributed GPU cluster
- Model evaluation and promotion

```bash
# Development/training
cd ai-service
python scripts/master_loop.py
```

## Model Workflow

1. **Training** (ai-service): Generate models via selfplay + training pipeline
2. **Evaluation** (ai-service): Gauntlet evaluation, promotion decisions
3. **Copy** to production: `cp ai-service/models/canonical_*.pth ai-inference/models/`
4. **Deploy** (ai-inference): Production servers load pre-trained models

## Resource Comparison

| Metric        | ai-inference  | ai-service         |
| ------------- | ------------- | ------------------ |
| Lines of code | ~150          | ~235,000           |
| Dependencies  | 4 packages    | 50+ packages       |
| Daemons       | 0             | 132                |
| Memory usage  | 500MB-1GB     | 2-8GB              |
| Startup time  | <5 seconds    | 30+ seconds        |
| GPU required  | No (optional) | Yes (for training) |
| Network       | None          | P2P mesh           |

## API Compatibility

Both services expose the same `/move` endpoint:

```bash
# Same API, different backend
curl -X POST http://localhost:8001/move \
  -H "Content-Type: application/json" \
  -d '{"game_state": {...}, "player_number": 1, "difficulty": 3}'
```

The game server (`AI_SERVICE_URL=http://ai-service:8001`) works with either backend.

## Deployment Checklist

### Production (ai-inference)

- [ ] Copy canonical models: `cp ai-service/models/canonical_*.pth ai-inference/models/`
- [ ] Build container: `docker build -f ai-inference/Dockerfile -t ringrift-ai .`
- [ ] Deploy with `docker-compose.production.yml`
- [ ] Verify health: `curl http://localhost:8001/health`

### Training (ai-service)

- [ ] Configure cluster: `ai-service/config/distributed_hosts.yaml`
- [ ] Start P2P: `python scripts/p2p_orchestrator.py`
- [ ] Run training: `python scripts/master_loop.py`
- [ ] Monitor: `python -m app.distributed.cluster_monitor --watch`

## FAQ

**Q: Why not use ai-service in production?**

A: The full ai-service includes 132 daemons, P2P networking, and training infrastructure that is not needed for serving moves. This adds unnecessary complexity, memory usage, and attack surface.

**Q: Can I use GPU acceleration in production?**

A: Yes. If the production server has a GPU, ai-inference will use it automatically for faster inference at higher difficulty levels (MCTS/Gumbel).

**Q: How do I update production models?**

A: After training new models in ai-service:

1. Evaluate with gauntlet
2. Copy to ai-inference/models/
3. Restart ai-inference (or use rolling deployment)

**Q: What about the existing ai-service Docker image?**

A: The `ghcr.io/an0mium/ringrift-ai` image contains the full ai-service. For minimal production deployment, build from `ai-inference/Dockerfile` instead.
