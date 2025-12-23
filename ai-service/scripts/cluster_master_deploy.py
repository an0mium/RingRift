#!/usr/bin/env python3
"""Master cluster deployment and monitoring script.

Orchestrates all cluster operations:
- Sync code to all nodes
- Deploy matchup-diverse selfplay
- Deploy persona tournaments
- Deploy CMA-ES optimization
- Harvest and consolidate results

Usage:
    # Full deployment: sync + all jobs
    python scripts/cluster_master_deploy.py --deploy-all

    # Just sync code
    python scripts/cluster_master_deploy.py --sync

    # Check all cluster status
    python scripts/cluster_master_deploy.py --status

    # Harvest all data
    python scripts/cluster_master_deploy.py --harvest

    # Stop everything
    python scripts/cluster_master_deploy.py --stop-all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# =============================================================================
# Cluster Configuration
# =============================================================================

GH200_NODES = [
    "lambda-gh200-a", "lambda-gh200-b", "lambda-gh200-c", "lambda-gh200-d",
    "lambda-gh200-e", "lambda-gh200-f", "lambda-gh200-g", "lambda-gh200-h",
    "lambda-gh200-i", "lambda-gh200-j", "lambda-gh200-k", "lambda-gh200-l",
]

H100_NODES = ["lambda-h100", "lambda-2xh100"]
A10_NODES = ["lambda-a10"]

ALL_GPU_NODES = GH200_NODES + H100_NODES + A10_NODES

# Job allocation
SELFPLAY_NODES = GH200_NODES[:8]  # 8 nodes for selfplay
TOURNAMENT_NODES = GH200_NODES[8:10]  # 2 nodes for tournaments
CMAES_NODES = GH200_NODES[10:]  # Remaining for CMA-ES


# =============================================================================
# Utilities
# =============================================================================

def run_ssh(node: str, cmd: str, timeout: int = 30) -> tuple[bool, str]:
    """Run SSH command."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes", node, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def rsync_to_node(node: str, timeout: int = 120) -> bool:
    """Sync code to node."""
    try:
        result = subprocess.run(
            ["rsync", "-az", "--delete",
             "--exclude=*.pyc", "--exclude=__pycache__",
             "--exclude=.git", "--exclude=data", "--exclude=*.db",
             "--exclude=logs", "--exclude=*.log",
             ".", f"{node}:~/ringrift/ai-service/"],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_node_health(node: str) -> dict:
    """Check node health and running jobs."""
    success, _ = run_ssh(node, "echo ok", timeout=5)
    if not success:
        return {"node": node, "reachable": False}

    # Check GPU
    _, gpu_out = run_ssh(node, "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1")

    # Check running jobs
    _, jobs_out = run_ssh(node, "ps aux | grep -E 'selfplay|tournament|cmaes' | grep python | grep -v grep | wc -l")

    # Check disk space
    _, disk_out = run_ssh(node, "df -h ~ | tail -1 | awk '{print $4}'")

    return {
        "node": node,
        "reachable": True,
        "gpu": gpu_out.strip() if gpu_out else "unknown",
        "jobs": int(jobs_out.strip()) if jobs_out.strip().isdigit() else 0,
        "disk_free": disk_out.strip() if disk_out else "unknown",
    }


# =============================================================================
# Deployment Functions
# =============================================================================

def sync_all_nodes(nodes: list[str]) -> dict[str, bool]:
    """Sync code to all nodes in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(rsync_to_node, node): node for node in nodes}
        for future in as_completed(futures):
            node = futures[future]
            try:
                results[node] = future.result()
            except Exception:
                results[node] = False

    return results


def deploy_selfplay(nodes: list[str]) -> dict[str, bool]:
    """Deploy matchup selfplay to nodes."""
    results = {}
    matchups = [
        "aggressive_vs_defensive", "territorial_vs_aggressive",
        "balanced_vs_aggressive", "balanced_vs_defensive",
        "balanced_vs_territorial", "defensive_vs_territorial",
        "aggressive_mirror", "defensive_mirror",
    ]

    for i, node in enumerate(nodes):
        matchup = matchups[i % len(matchups)]
        cmd = f"""
cd ~/ringrift/ai-service && \\
mkdir -p ~/training_data/matchup_selfplay && \\
nohup python3 scripts/run_gpu_selfplay.py \\
    --board square8 --num-players 2 \\
    --matchup {matchup} \\
    --num-games 50000 --batch-size 512 \\
    --output-dir ~/training_data/matchup_selfplay/{matchup} \\
    --use-heuristic --continuous \\
    > /tmp/selfplay_{matchup}.log 2>&1 &
"""
        success, _ = run_ssh(node, cmd, timeout=30)
        results[node] = success

    return results


def deploy_tournaments(nodes: list[str]) -> dict[str, bool]:
    """Deploy persona tournaments."""
    results = {}
    boards = ["square8", "hex8"]

    for i, node in enumerate(nodes):
        board = boards[i % len(boards)]
        cmd = f"""
cd ~/ringrift/ai-service && \\
mkdir -p ~/tournament_results && \\
nohup python3 -c '
import sys
sys.path.insert(0, ".")
from app.ai.gpu_parallel_games import ParallelGameRunner
import json

matchups = ParallelGameRunner.get_all_training_matchups()
results = {{}}

for matchup in matchups:
    print(f"Running {{matchup}}...")
    r = ParallelGameRunner.run_matchup_tournament(
        matchups=[matchup],
        games_per_matchup=200,
        max_moves=400
    )
    results[matchup] = r[matchup]
    with open(f"~/tournament_results/{board}_results.json", "w") as f:
        json.dump(results, f, indent=2)

print("Tournament complete!")
' > /tmp/tournament_{board}.log 2>&1 &
"""
        success, _ = run_ssh(node, cmd, timeout=30)
        results[node] = success

    return results


def stop_all_jobs(nodes: list[str]) -> dict[str, bool]:
    """Stop all running jobs on nodes."""
    results = {}
    stop_cmd = "pkill -f 'run_gpu_selfplay\\|tournament\\|cmaes' || true"

    for node in nodes:
        success, _ = run_ssh(node, stop_cmd, timeout=10)
        results[node] = success

    return results


def harvest_data(nodes: list[str], local_dir: str) -> dict[str, int]:
    """Harvest training data from cluster nodes."""
    os.makedirs(local_dir, exist_ok=True)
    results = {}

    for node in nodes:
        # Get list of data files
        _, files_out = run_ssh(node, "find ~/training_data -name '*.jsonl' -o -name '*.db' 2>/dev/null | head -100")

        if not files_out.strip():
            results[node] = 0
            continue

        files = files_out.strip().split("\n")
        results[node] = len(files)

        # Rsync data
        try:
            subprocess.run(
                ["rsync", "-avz", "--progress",
                 f"{node}:~/training_data/", f"{local_dir}/{node}/"],
                timeout=600,
            )
        except Exception:
            pass

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Master cluster deployment")
    parser.add_argument("--sync", action="store_true", help="Sync code to all nodes")
    parser.add_argument("--status", action="store_true", help="Check cluster status")
    parser.add_argument("--deploy-all", action="store_true", help="Deploy all jobs")
    parser.add_argument("--deploy-selfplay", action="store_true", help="Deploy selfplay only")
    parser.add_argument("--deploy-tournament", action="store_true", help="Deploy tournaments only")
    parser.add_argument("--stop-all", action="store_true", help="Stop all jobs")
    parser.add_argument("--harvest", action="store_true", help="Harvest data from cluster")
    parser.add_argument("--harvest-dir", type=str, default="./harvested_data",
                        help="Local directory for harvested data")
    parser.add_argument("--nodes", type=str, help="Comma-separated node list")
    args = parser.parse_args()

    nodes = [n.strip() for n in args.nodes.split(",")] if args.nodes else ALL_GPU_NODES

    if args.status:
        print("=" * 70)
        print("CLUSTER STATUS")
        print("=" * 70)
        print(f"Checking {len(nodes)} nodes...\n")

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(check_node_health, n): n for n in nodes}
            for future in as_completed(futures):
                status = future.result()
                if status["reachable"]:
                    icon = "✓" if status["jobs"] > 0 else "○"
                    print(f"{icon} {status['node']}: {status['jobs']} jobs, {status['disk_free']} free")
                    if status.get("gpu"):
                        print(f"    GPU: {status['gpu'][:60]}")
                else:
                    print(f"✗ {status['node']}: UNREACHABLE")
        return

    if args.sync:
        print("=" * 70)
        print("SYNCING CODE TO CLUSTER")
        print("=" * 70)
        print(f"Syncing to {len(nodes)} nodes...\n")

        results = sync_all_nodes(nodes)
        success = sum(1 for v in results.values() if v)
        print(f"\nSynced: {success}/{len(nodes)} nodes")
        for node, ok in results.items():
            print(f"  {'✓' if ok else '✗'} {node}")
        return

    if args.stop_all:
        print("=" * 70)
        print("STOPPING ALL JOBS")
        print("=" * 70)

        results = stop_all_jobs(nodes)
        for node, ok in results.items():
            print(f"  {'✓' if ok else '✗'} {node}")
        return

    if args.deploy_selfplay or args.deploy_all:
        print("=" * 70)
        print("DEPLOYING MATCHUP SELFPLAY")
        print("=" * 70)

        selfplay_nodes = SELFPLAY_NODES if not args.nodes else nodes
        results = deploy_selfplay(selfplay_nodes)
        success = sum(1 for v in results.values() if v)
        print(f"Deployed: {success}/{len(selfplay_nodes)} nodes")
        for node, ok in results.items():
            print(f"  {'✓' if ok else '✗'} {node}")

    if args.deploy_tournament or args.deploy_all:
        print("\n" + "=" * 70)
        print("DEPLOYING TOURNAMENTS")
        print("=" * 70)

        tournament_nodes = TOURNAMENT_NODES if not args.nodes else nodes[:2]
        results = deploy_tournaments(tournament_nodes)
        for node, ok in results.items():
            print(f"  {'✓' if ok else '✗'} {node}")

    if args.harvest:
        print("=" * 70)
        print("HARVESTING DATA FROM CLUSTER")
        print("=" * 70)

        results = harvest_data(nodes, args.harvest_dir)
        total_files = sum(results.values())
        print(f"\nHarvested {total_files} files from {len(results)} nodes")
        print(f"Data saved to: {args.harvest_dir}")
        return

    if not any([args.sync, args.status, args.deploy_all, args.deploy_selfplay,
                args.deploy_tournament, args.stop_all, args.harvest]):
        print("=" * 70)
        print("CLUSTER MASTER DEPLOYMENT")
        print("=" * 70)
        print(f"\nConfigured nodes: {len(ALL_GPU_NODES)}")
        print(f"  Selfplay nodes: {len(SELFPLAY_NODES)}")
        print(f"  Tournament nodes: {len(TOURNAMENT_NODES)}")
        print(f"  CMA-ES nodes: {len(CMAES_NODES)}")
        print("\nCommands:")
        print("  --sync          Sync code to all nodes")
        print("  --status        Check cluster status")
        print("  --deploy-all    Deploy all jobs (selfplay + tournament)")
        print("  --stop-all      Stop all running jobs")
        print("  --harvest       Harvest data from cluster")


if __name__ == "__main__":
    main()
