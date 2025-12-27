# CLAUDE.md - AI Assistant Context for RingRift

This file provides context for AI assistants working on this codebase.

## What is RingRift?

A multiplayer territory control board game where players place pieces to claim territory. Features:

- Multiple board geometries (square, hexagonal) and sizes
- 2-4 player support
- Neural network AI opponents trained via self-play
- Real-time multiplayer with matchmaking

## Repository Structure

```
RingRift/
├── src/                    # TypeScript source
│   ├── client/            # React frontend
│   ├── server/            # Node.js game server
│   └── shared/            # Shared game engine (SOURCE OF TRUTH for rules)
│       ├── engine/        # Core game logic
│       └── types/         # Type definitions
├── ai-service/            # Python ML pipeline (see ai-service/CLAUDE.md)
│   ├── app/              # Core modules
│   ├── scripts/          # CLI tools
│   └── data/             # Databases and training data
├── tests/                 # Integration tests
└── config/               # Configuration files
```

## Key Principle: TypeScript is Source of Truth

The game rules are defined in `src/shared/engine/`. The Python `ai-service` **mirrors** these rules for training. When rules change:

1. Update TypeScript first
2. Update Python to match
3. Run parity tests to verify they agree

## Quick Start Commands

```bash
# Frontend development
npm run dev:client

# Backend server
npm run dev:server

# AI service (Python)
cd ai-service
python -m app.training.train --help

# Run tests
npm test                           # TypeScript tests
cd ai-service && pytest           # Python tests
```

## Board Configurations

| Type        | Sizes                 | Description             |
| ----------- | --------------------- | ----------------------- |
| `square8`   | 8x8 (64 cells)        | Standard square board   |
| `square19`  | 19x19 (361 cells)     | Large square (Go-sized) |
| `hex8`      | radius 4 (61 cells)   | Small hexagonal         |
| `hexagonal` | radius 12 (469 cells) | Large hexagonal         |

All board types support 2, 3, or 4 players.

## Common Workflows

### Train a New Model

```bash
cd ai-service

# 1. Export training data from game databases
python scripts/export_replay_dataset.py \
  --use-discovery --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# 2. Train the model
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz
```

### Transfer Learning (2p to 4p)

```bash
cd ai-service

# Resize value head for 4-player model
python scripts/transfer_2p_to_4p.py \
  --source models/my_2p_model.pth \
  --output models/my_4p_init.pth \
  --board-type square8

# Train with transferred weights
python -m app.training.train \
  --board-type square8 --num-players 4 \
  --init-weights models/my_4p_init.pth \
  --data-path data/training/sq8_4p.npz
```

### Check Game Data Quality

```bash
cd ai-service
python -m app.training.data_quality --db data/games/selfplay.db
```

### Verify TS/Python Parity

```bash
cd ai-service
python scripts/check_ts_python_replay_parity.py --db data/games/my_games.db
```

## Cluster Infrastructure

RingRift uses a P2P mesh network for distributed training across ~52 configured nodes (Dec 2025).

### Active Cluster (Dec 2025)

**Lambda Labs account terminated Dec 2025. All Lambda nodes permanently removed.**

| Provider | Nodes | GPUs                               | Status |
| -------- | ----- | ---------------------------------- | ------ |
| Vast.ai  | ~30   | RTX 5090, 4090, 3090, A40, 4060 Ti | Active |
| RunPod   | 6     | H100, A100, L40S, RTX 3090 Ti      | Active |
| Nebius   | 4     | H100 (80GB), L40S                  | Active |
| Vultr    | 3     | A100 (20GB vGPU)                   | Active |
| Hetzner  | 4     | CPU only (P2P voters)              | Active |
| Local    | 2     | Mac Studio M3 (coordinator)        | Active |

### Cluster Management

```bash
# Check cluster status via P2P
curl -s http://localhost:8770/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Leader: {d.get(\"leader_id\")}"); print(f"Alive: {d.get(\"alive_peers\")}")'

# Or use the monitor
cd ai-service && python -m app.distributed.cluster_monitor

# Update all nodes to latest code (NEW - Dec 2025)
cd ai-service && python scripts/update_all_nodes.py --restart-p2p
```

**Update Script** - Update all nodes in parallel:

```bash
# Update all nodes
python scripts/update_all_nodes.py

# With P2P restart
python scripts/update_all_nodes.py --restart-p2p

# Dry run preview
python scripts/update_all_nodes.py --dry-run
```

**P2P Stability Fixes** (commits 1270b64, dade90f, 6649601):

- Pre-flight dependency validation (aiohttp, psutil, yaml)
- Gzip magic byte detection in gossip handler
- 120s startup grace period for slow state loading
- SystemExit handling in task wrapper
- /dev/shm fallback for macOS compatibility
- Clear port binding error messages

See `ai-service/config/distributed_hosts.yaml` for full cluster configuration.

## Key Features

- **GPU Selfplay**: Vectorized game simulation on CUDA (`app/ai/gpu_parallel_games.py`)
- **Gumbel MCTS**: Quality-focused tree search for training data
- **Transfer Learning**: Train 4-player models from 2-player checkpoints
- **Parity Testing**: Verify Python engine matches TypeScript rules

## See Also

- `ai-service/CLAUDE.md` - Detailed AI service context
- `ai-service/AGENTS.md` - Coding guidelines for AI service
- `AGENTS.md` - Root-level coding guidelines
