#!/usr/bin/env python3
"""Deploy matchup-diverse selfplay data generation across cluster nodes.

This script distributes GPU selfplay with different persona matchups across
all available cluster nodes for maximum training data diversity.

Usage:
    # Deploy to all GH200 nodes
    python scripts/deploy_matchup_selfplay.py --deploy

    # Check status
    python scripts/deploy_matchup_selfplay.py --status

    # Stop all jobs
    python scripts/deploy_matchup_selfplay.py --stop

    # Deploy specific matchups to specific nodes
    python scripts/deploy_matchup_selfplay.py --deploy --nodes lambda-gh200-a,lambda-gh200-b
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Available cluster nodes
GH200_NODES = [
    "lambda-gh200-a", "lambda-gh200-b", "lambda-gh200-c", "lambda-gh200-d",
    "lambda-gh200-e", "lambda-gh200-f", "lambda-gh200-g", "lambda-gh200-h",
    "lambda-gh200-i", "lambda-gh200-j", "lambda-gh200-k", "lambda-gh200-l",
]

H100_NODES = ["lambda-h100", "lambda-2xh100"]

A10_NODES = ["lambda-a10"]

ALL_GPU_NODES = GH200_NODES + H100_NODES + A10_NODES

# All available matchups from ParallelGameRunner.TRAINING_MATCHUPS
MATCHUPS = [
    "aggressive_vs_defensive",
    "territorial_vs_aggressive",
    "balanced_vs_aggressive",
    "balanced_vs_defensive",
    "balanced_vs_territorial",
    "defensive_vs_territorial",
    "aggressive_mirror",
    "defensive_mirror",
    "balanced_mirror",
    "territorial_mirror",
]

# Board configurations to generate data for
BOARD_CONFIGS = [
    {"board": "square8", "players": 2, "batch_size": 512, "games": 10000},
    {"board": "square8", "players": 3, "batch_size": 256, "games": 5000},
    {"board": "hex8", "players": 2, "batch_size": 512, "games": 10000},
]


@dataclass
class DeploymentConfig:
    """Configuration for a single node deployment."""
    node: str
    board: str
    players: int
    matchup: str
    batch_size: int
    games: int
    output_dir: str


def get_deployment_plan(nodes: list[str] | None = None) -> list[DeploymentConfig]:
    """Create deployment plan distributing matchups across nodes."""
    nodes = nodes or GH200_NODES
    deployments = []

    node_idx = 0
    for config in BOARD_CONFIGS:
        for matchup in MATCHUPS:
            if node_idx >= len(nodes):
                node_idx = 0  # Wrap around

            node = nodes[node_idx]
            output_dir = f"~/training_data/matchup_{config['board']}_{config['players']}p/{matchup}"

            deployments.append(DeploymentConfig(
                node=node,
                board=config["board"],
                players=config["players"],
                matchup=matchup,
                batch_size=config["batch_size"],
                games=config["games"],
                output_dir=output_dir,
            ))
            node_idx += 1

    return deployments


def run_ssh_command(node: str, command: str, timeout: int = 10) -> tuple[bool, str]:
    """Run SSH command on remote node."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no", node, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def check_node_available(node: str) -> bool:
    """Check if node is reachable."""
    success, _ = run_ssh_command(node, "echo ok", timeout=5)
    return success


def deploy_to_node(config: DeploymentConfig, dry_run: bool = False) -> bool:
    """Deploy selfplay job to a single node."""
    # Build the command
    cmd = f"""
cd ~/ringrift/ai-service && \\
mkdir -p {config.output_dir} && \\
nohup python3 scripts/run_gpu_selfplay.py \\
  --board {config.board} \\
  --num-players {config.players} \\
  --matchup {config.matchup} \\
  --num-games {config.games} \\
  --batch-size {config.batch_size} \\
  --output-dir {config.output_dir} \\
  --use-heuristic \\
  > /tmp/matchup_selfplay_{config.matchup}.log 2>&1 &
"""

    if dry_run:
        print(f"  [DRY RUN] Would deploy to {config.node}:")
        print(f"    Board: {config.board}, Players: {config.players}")
        print(f"    Matchup: {config.matchup}")
        print(f"    Games: {config.games}, Batch: {config.batch_size}")
        return True

    success, output = run_ssh_command(config.node, cmd, timeout=30)
    return success


def check_job_status(node: str) -> dict:
    """Check if selfplay job is running on node."""
    cmd = "ps aux | grep 'run_gpu_selfplay.py.*matchup' | grep -v grep | wc -l"
    success, output = run_ssh_command(node, cmd, timeout=10)

    running = False
    if success:
        try:
            running = int(output.strip()) > 0
        except ValueError:
            pass

    # Get log tail if running
    log_tail = ""
    if running:
        cmd2 = "tail -3 /tmp/matchup_selfplay_*.log 2>/dev/null | head -10"
        _, log_tail = run_ssh_command(node, cmd2, timeout=10)

    return {
        "node": node,
        "reachable": success,
        "running": running,
        "log_tail": log_tail.strip(),
    }


def stop_jobs(node: str) -> bool:
    """Stop selfplay jobs on node."""
    cmd = "pkill -f 'run_gpu_selfplay.py.*matchup' || true"
    success, _ = run_ssh_command(node, cmd, timeout=10)
    return success


def sync_code_to_node(node: str) -> bool:
    """Sync latest code to node."""
    try:
        result = subprocess.run(
            ["rsync", "-az", "--delete",
             "--exclude", "*.pyc", "--exclude", "__pycache__",
             "--exclude", ".git", "--exclude", "data",
             ".", f"{node}:~/ringrift/ai-service/"],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy matchup-diverse selfplay to cluster")
    parser.add_argument("--deploy", action="store_true", help="Deploy jobs to cluster")
    parser.add_argument("--status", action="store_true", help="Check job status")
    parser.add_argument("--stop", action="store_true", help="Stop all jobs")
    parser.add_argument("--sync", action="store_true", help="Sync code before deploy")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--nodes", type=str, help="Comma-separated list of nodes")
    parser.add_argument("--plan", action="store_true", help="Show deployment plan")
    args = parser.parse_args()

    # Parse nodes
    nodes = None
    if args.nodes:
        nodes = [n.strip() for n in args.nodes.split(",")]

    if args.plan or (not args.deploy and not args.status and not args.stop):
        print("=" * 70)
        print("MATCHUP-DIVERSE SELFPLAY DEPLOYMENT PLAN")
        print("=" * 70)

        plan = get_deployment_plan(nodes)

        # Group by node
        by_node = {}
        for config in plan:
            if config.node not in by_node:
                by_node[config.node] = []
            by_node[config.node].append(config)

        for node, configs in sorted(by_node.items()):
            print(f"\n{node}:")
            for c in configs:
                print(f"  - {c.board} {c.players}p: {c.matchup} ({c.games} games)")

        print(f"\nTotal: {len(plan)} jobs across {len(by_node)} nodes")
        print("\nUse --deploy to start, --status to check, --stop to halt")
        return

    if args.status:
        print("=" * 70)
        print("CLUSTER JOB STATUS")
        print("=" * 70)

        target_nodes = nodes or GH200_NODES
        for node in target_nodes:
            status = check_job_status(node)
            icon = "✓" if status["running"] else "○" if status["reachable"] else "✗"
            state = "RUNNING" if status["running"] else "IDLE" if status["reachable"] else "UNREACHABLE"
            print(f"{icon} {node}: {state}")
            if status["log_tail"]:
                for line in status["log_tail"].split("\n")[:2]:
                    print(f"    {line[:70]}")
        return

    if args.stop:
        print("Stopping all matchup selfplay jobs...")
        target_nodes = nodes or GH200_NODES
        for node in target_nodes:
            success = stop_jobs(node)
            icon = "✓" if success else "✗"
            print(f"{icon} {node}")
        return

    if args.deploy:
        print("=" * 70)
        print("DEPLOYING MATCHUP-DIVERSE SELFPLAY")
        print("=" * 70)

        plan = get_deployment_plan(nodes)

        if args.sync:
            print("\nSyncing code to nodes...")
            target_nodes = list(set(c.node for c in plan))
            for node in target_nodes:
                success = sync_code_to_node(node)
                icon = "✓" if success else "✗"
                print(f"  {icon} {node}")

        print("\nDeploying jobs...")
        success_count = 0
        fail_count = 0

        for config in plan:
            success = deploy_to_node(config, dry_run=args.dry_run)
            if success:
                success_count += 1
                print(f"  ✓ {config.node}: {config.matchup} ({config.board} {config.players}p)")
            else:
                fail_count += 1
                print(f"  ✗ {config.node}: {config.matchup} - FAILED")

        print(f"\nDeployed: {success_count}/{len(plan)} jobs")
        if fail_count > 0:
            print(f"Failed: {fail_count} jobs")


if __name__ == "__main__":
    main()
