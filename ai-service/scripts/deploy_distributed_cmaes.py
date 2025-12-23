#!/usr/bin/env python3
"""Deploy distributed CMA-ES weight optimization across cluster.

This script launches distributed CMA-ES optimization using the cluster
for high-throughput fitness evaluation.

Usage:
    # Launch coordinator + workers
    python scripts/deploy_distributed_cmaes.py --deploy

    # Check status
    python scripts/deploy_distributed_cmaes.py --status

    # Stop all jobs
    python scripts/deploy_distributed_cmaes.py --stop
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

# Cluster nodes
GH200_NODES = [
    "lambda-gh200-a", "lambda-gh200-b", "lambda-gh200-c", "lambda-gh200-d",
    "lambda-gh200-e", "lambda-gh200-f", "lambda-gh200-g", "lambda-gh200-h",
    "lambda-gh200-i", "lambda-gh200-j", "lambda-gh200-k", "lambda-gh200-l",
]

# CMA-ES configuration
CMAES_CONFIG = {
    "board": "square8",
    "players": 2,
    "generations": 200,
    "population_size": 48,  # Divisible by num workers
    "games_per_eval": 100,
    "worker_port": 8766,
}


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


def deploy_worker(node: str, port: int, board: str) -> bool:
    """Deploy CMA-ES worker to a node."""
    cmd = f"""
cd ~/ringrift/ai-service && \\
nohup python3 scripts/run_distributed_gpu_cmaes.py \\
    --mode worker \\
    --port {port} \\
    --board {board} \\
    > /tmp/cmaes_worker.log 2>&1 &
"""
    success, _ = run_ssh_command(node, cmd, timeout=30)
    return success


def deploy_coordinator(node: str, workers: list[str], config: dict) -> bool:
    """Deploy CMA-ES coordinator to a node."""
    worker_list = ",".join(f"{w}:{config['worker_port']}" for w in workers)

    cmd = f"""
cd ~/ringrift/ai-service && \\
nohup python3 scripts/run_distributed_gpu_cmaes.py \\
    --mode coordinator \\
    --board {config['board']} \\
    --num-players {config['players']} \\
    --generations {config['generations']} \\
    --population-size {config['population_size']} \\
    --games-per-eval {config['games_per_eval']} \\
    --workers {worker_list} \\
    --output-dir ~/cmaes_results \\
    > /tmp/cmaes_coordinator.log 2>&1 &
"""
    success, _ = run_ssh_command(node, cmd, timeout=30)
    return success


def check_cmaes_status(node: str) -> dict:
    """Check CMA-ES job status on node."""
    cmd = "ps aux | grep 'run_distributed_gpu_cmaes' | grep -v grep"
    success, output = run_ssh_command(node, cmd, timeout=10)

    is_worker = "worker" in output.lower()
    is_coordinator = "coordinator" in output.lower()
    running = is_worker or is_coordinator

    role = "coordinator" if is_coordinator else "worker" if is_worker else "none"

    return {
        "node": node,
        "reachable": success or "Timeout" not in output,
        "running": running,
        "role": role,
    }


def stop_cmaes(node: str) -> bool:
    """Stop CMA-ES jobs on node."""
    cmd = "pkill -f 'run_distributed_gpu_cmaes' || true"
    success, _ = run_ssh_command(node, cmd, timeout=10)
    return success


def sync_code(node: str) -> bool:
    """Sync code to node."""
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
    parser = argparse.ArgumentParser(description="Deploy distributed CMA-ES")
    parser.add_argument("--deploy", action="store_true", help="Deploy CMA-ES cluster")
    parser.add_argument("--status", action="store_true", help="Check status")
    parser.add_argument("--stop", action="store_true", help="Stop all jobs")
    parser.add_argument("--sync", action="store_true", help="Sync code first")
    parser.add_argument("--coordinator", type=str, default="lambda-gh200-a",
                        help="Coordinator node")
    parser.add_argument("--workers", type=str, help="Comma-separated worker nodes")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    args = parser.parse_args()

    # Parse workers
    if args.workers:
        workers = [w.strip() for w in args.workers.split(",")]
    else:
        workers = [n for n in GH200_NODES if n != args.coordinator]

    if args.status:
        print("=" * 70)
        print("DISTRIBUTED CMA-ES STATUS")
        print("=" * 70)

        all_nodes = [args.coordinator] + workers
        for node in all_nodes:
            status = check_cmaes_status(node)
            if status["running"]:
                icon = "▶"
                state = f"RUNNING ({status['role']})"
            elif status["reachable"]:
                icon = "○"
                state = "IDLE"
            else:
                icon = "✗"
                state = "UNREACHABLE"
            print(f"{icon} {node}: {state}")
        return

    if args.stop:
        print("Stopping CMA-ES jobs...")
        all_nodes = [args.coordinator] + workers
        for node in all_nodes:
            success = stop_cmaes(node)
            icon = "✓" if success else "✗"
            print(f"  {icon} {node}")
        return

    if args.deploy:
        print("=" * 70)
        print("DEPLOYING DISTRIBUTED CMA-ES")
        print("=" * 70)

        config = CMAES_CONFIG.copy()
        config["board"] = args.board

        all_nodes = [args.coordinator] + workers

        if args.sync:
            print("\nSyncing code...")
            for node in all_nodes:
                success = sync_code(node)
                icon = "✓" if success else "✗"
                print(f"  {icon} {node}")

        # Stop existing jobs first
        print("\nStopping existing jobs...")
        for node in all_nodes:
            stop_cmaes(node)

        time.sleep(2)

        # Deploy workers first
        print("\nDeploying workers...")
        for node in workers:
            success = deploy_worker(node, config["worker_port"], config["board"])
            icon = "✓" if success else "✗"
            print(f"  {icon} {node}")

        # Wait for workers to start
        print("\nWaiting for workers to initialize...")
        time.sleep(5)

        # Deploy coordinator
        print("\nDeploying coordinator...")
        success = deploy_coordinator(args.coordinator, workers, config)
        icon = "✓" if success else "✗"
        print(f"  {icon} {args.coordinator}")

        print(f"\nDeployment complete!")
        print(f"  Coordinator: {args.coordinator}")
        print(f"  Workers: {len(workers)}")
        print(f"  Board: {config['board']}")
        print(f"  Generations: {config['generations']}")
        print(f"\nMonitor with: ssh {args.coordinator} 'tail -f /tmp/cmaes_coordinator.log'")
        return

    # Default: show plan
    print("=" * 70)
    print("DISTRIBUTED CMA-ES DEPLOYMENT PLAN")
    print("=" * 70)
    print(f"\nCoordinator: {args.coordinator}")
    print(f"Workers ({len(workers)}):")
    for w in workers:
        print(f"  - {w}")
    print(f"\nConfiguration:")
    print(f"  Board: {CMAES_CONFIG['board']}")
    print(f"  Generations: {CMAES_CONFIG['generations']}")
    print(f"  Population: {CMAES_CONFIG['population_size']}")
    print(f"  Games/eval: {CMAES_CONFIG['games_per_eval']}")
    print(f"\nUse --deploy to start, --status to check, --stop to halt")


if __name__ == "__main__":
    main()
