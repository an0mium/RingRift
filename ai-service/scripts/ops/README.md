# Operational Scripts

Shell scripts for managing RingRift cluster nodes.

## Scripts

### `p2p_keepalive.sh`

Cron-compatible script that ensures P2P orchestrator is running. Add to crontab:

```bash
# Check every minute
* * * * * /path/to/ringrift/ai-service/scripts/ops/p2p_keepalive.sh
```

Environment variables:

- `RINGRIFT_NODE_ID` - Node identifier (default: hostname)
- `RINGRIFT_P2P_PORT` - P2P port (default: 8770)
- `RINGRIFT_AI_SERVICE` - Path to ai-service directory
- `RINGRIFT_P2P_SEEDS` - Comma-separated peer URLs

### `p2p_supervisor.sh`

Foreground supervisor that automatically restarts P2P on crash with exponential backoff:

```bash
# Run in background
nohup ./p2p_supervisor.sh mynode /path/to/ai-service 8770 &

# Or with systemd
[Service]
ExecStart=/path/to/ai-service/scripts/ops/p2p_supervisor.sh
Restart=always
```

Arguments:

1. `node-id` - Node identifier
2. `ai-service-path` - Path to ai-service directory
3. `port` - P2P port (default: 8770)

### `maintain_selfplay_load.sh`

Automatically maintains a target number of selfplay jobs:

```bash
# Default: 10 jobs, hex8, 2 players
./maintain_selfplay_load.sh

# Custom: 25 jobs, square19, 2 players
./maintain_selfplay_load.sh 25 square19 2

# Or use environment variables
RINGRIFT_TARGET_SELFPLAY_JOBS=20 \
RINGRIFT_SELFPLAY_BOARD=hex8 \
RINGRIFT_SELFPLAY_ENGINE=gumbel \
./maintain_selfplay_load.sh
```

Environment variables:

- `RINGRIFT_TARGET_SELFPLAY_JOBS` - Target concurrent jobs (default: 10)
- `RINGRIFT_SELFPLAY_BOARD` - Board type (default: hex8)
- `RINGRIFT_SELFPLAY_PLAYERS` - Player count (default: 2)
- `RINGRIFT_SELFPLAY_ENGINE` - Engine mode (default: gumbel)
- `RINGRIFT_SELFPLAY_BUDGET` - Simulation budget (default: 200)
- `RINGRIFT_GAMES_PER_JOB` - Games per job (default: 30)

## Deployment

These scripts are deployed to cluster nodes via `update_all_nodes.py`. They can also be copied manually:

```bash
scp scripts/ops/*.sh user@node:/path/to/ai-service/scripts/ops/
```

## Crontab Setup

Recommended crontab entries for cluster nodes:

```bash
# P2P keepalive - check every minute
* * * * * /path/to/ai-service/scripts/ops/p2p_keepalive.sh

# Selfplay load maintenance - optional, for dedicated selfplay nodes
# @reboot /path/to/ai-service/scripts/ops/maintain_selfplay_load.sh 20 hex8 2
```
