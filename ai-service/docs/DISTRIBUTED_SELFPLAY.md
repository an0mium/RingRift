# Distributed Self-Play Setup

This guide explains how to set up distributed self-play across multiple local Macs.

## Quick Start

```bash
# From ai-service/
./scripts/cluster_setup.sh discover     # Find Macs on network
./scripts/cluster_setup.sh test         # Verify SSH connectivity
./scripts/cluster_setup.sh setup-all    # Install dependencies on workers
./scripts/cluster_setup.sh status       # Check worker status
./scripts/run_distributed_selfplay_matrix.sh  # Run distributed jobs
```

## Architecture

```
┌─────────────────┐     SSH     ┌─────────────────┐
│  Main Machine   │────────────>│  Worker Mac 1   │
│  (Coordinator)  │             │  (10.0.0.170)   │
│                 │             └─────────────────┘
│  Runs:          │     SSH     ┌─────────────────┐
│  - Job dist.    │────────────>│  Worker Mac 2   │
│  - Result merge │             │  (10.0.0.229)   │
└─────────────────┘             └─────────────────┘
```

Jobs are distributed round-robin to workers and run in parallel.

## Setup Steps

### 1. Enable SSH on Worker Macs

On each Mac you want to use as a worker:

1. Open **System Preferences** (or System Settings on macOS 13+)
2. Go to **Sharing**
3. Enable **Remote Login**
4. Note the IP address shown

### 2. Configure SSH Key Authentication

From your main machine, copy your SSH key to each worker:

```bash
# Generate key if you don't have one
ssh-keygen -t ed25519

# Copy to each worker
ssh-copy-id 10.0.0.170
ssh-copy-id 10.0.0.229
```

### 3. Add Workers to Configuration

Edit `scripts/cluster_workers.txt`:

```
# List of worker hosts
10.0.0.170
10.0.0.229
```

### 4. Set Up Workers

Run the automated setup for all workers:

```bash
./scripts/cluster_setup.sh setup-all
```

This will:

- Clone or update the RingRift repository
- Create a Python virtual environment
- Install all dependencies

### 5. Verify Setup

```bash
./scripts/cluster_setup.sh test    # Test SSH connectivity
./scripts/cluster_setup.sh status  # Check worker health
```

## Running Distributed Self-Play

```bash
# Run the full matrix across all workers
./scripts/run_distributed_selfplay_matrix.sh

# Run locally only (skip remote workers)
LOCAL_ONLY=1 ./scripts/run_distributed_selfplay_matrix.sh
```

### Environment Variables

| Variable                             | Default                     | Description                     |
| ------------------------------------ | --------------------------- | ------------------------------- |
| `LOCAL_ONLY`                         | 0                           | Set to 1 to skip remote workers |
| `CLUSTER_WORKERS_FILE`               | scripts/cluster_workers.txt | Path to workers file            |
| `REMOTE_PROJECT_DIR`                 | ~/Development/RingRift      | Project dir on workers          |
| `GAMES_2P` / `GAMES_3P` / `GAMES_4P` | 5/3/2                       | Games per player count          |
| `SQUARE8_MAX_MOVES_*P`               | 150/200/250                 | Max moves for square8           |
| `SQUARE19_MAX_MOVES_*P`              | 350/450/550                 | Max moves for square19          |

## Cluster Setup Commands

```bash
./scripts/cluster_setup.sh discover     # Scan network for Macs
./scripts/cluster_setup.sh test         # Test SSH connectivity
./scripts/cluster_setup.sh setup HOST   # Set up single worker
./scripts/cluster_setup.sh setup-all    # Set up all workers
./scripts/cluster_setup.sh start HOST   # Start worker service
./scripts/cluster_setup.sh status       # Check all worker status
```

## HTTP Worker Service (Optional)

For more sophisticated job distribution, workers can run an HTTP service:

```bash
# Start worker service on a remote host
./scripts/cluster_setup.sh start 10.0.0.170

# Check health
curl http://10.0.0.170:8765/health
```

The HTTP service supports:

- Health checks
- Task submission via REST API
- Bonjour/mDNS discovery (when network allows)

## Troubleshooting

### SSH Connection Refused

1. Verify Remote Login is enabled in Sharing preferences
2. Check the Mac's firewall settings
3. Test with: `nc -zv <host> 22`

### SSH Auth Failed

1. Copy your SSH key: `ssh-copy-id <host>`
2. Test: `ssh -o BatchMode=yes <host> echo ok`

### Worker Not Starting

Check the worker log:

```bash
ssh <host> "cat /tmp/cluster_worker.log"
```

### Python Version Too Old

The codebase requires Python 3.10+. If the system Python is older:

```bash
# Install via Homebrew
brew install python@3.11

# The setup script will auto-detect newer Python versions
```

## File Locations

- **Worker config**: `scripts/cluster_workers.txt`
- **Setup script**: `scripts/cluster_setup.sh`
- **Distributed runner**: `scripts/run_distributed_selfplay_matrix.sh`
- **Worker service**: `scripts/cluster_worker.py`
- **Results**: `logs/selfplay_matrix/`, `data/games/`
