#!/usr/bin/env python
"""
Cluster management utilities for local Mac cluster CMA-ES training.

This script provides commands for managing a cluster of worker machines:
- Discover workers on the network
- Check worker health
- Preload state pools on workers
- Run a test evaluation

Usage:
------
    # Discover workers on the network
    python scripts/cluster_manager.py discover

    # Check health of specific workers
    python scripts/cluster_manager.py health --workers 192.168.1.10:8765,192.168.1.11:8765

    # Check health with auto-discovery
    python scripts/cluster_manager.py health --discover

    # Preload state pools on workers
    python scripts/cluster_manager.py preload --discover --board square8 --num-players 2 --pool-id v1

    # Run a test evaluation across workers
    python scripts/cluster_manager.py test --discover --games 4

    # Show cluster stats
    python scripts/cluster_manager.py stats --workers 192.168.1.10:8765
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Allow imports from app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.distributed.discovery import (
    WorkerInfo,
    WorkerDiscovery,
    discover_workers,
    wait_for_workers,
    parse_manual_workers,
    verify_worker_health,
    filter_healthy_workers,
)
from app.distributed.client import (
    WorkerClient,
    DistributedEvaluator,
)
from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS


def get_workers(args) -> List[WorkerInfo]:
    """Get worker list from args (manual or discovery)."""
    if args.workers:
        return parse_manual_workers(args.workers)
    elif getattr(args, "discover", False):
        print("Discovering workers on network...")
        workers = wait_for_workers(
            min_workers=getattr(args, "min_workers", 1),
            timeout=getattr(args, "timeout", 10.0),
        )
        if not workers:
            print("No workers found on network.")
            print("Ensure workers are running: python scripts/cluster_worker.py --register-bonjour")
        return workers
    else:
        print("Error: Specify --workers or --discover")
        return []


def cmd_discover(args) -> None:
    """Discover workers on the network."""
    print(f"Scanning network for {args.timeout}s...")
    workers = wait_for_workers(
        min_workers=args.min_workers,
        timeout=args.timeout,
    )

    if not workers:
        print("No workers found.")
        print("\nTo register a worker, run on each Mac:")
        print("  python scripts/cluster_worker.py --register-bonjour")
        return

    print(f"\nFound {len(workers)} worker(s):")
    print("-" * 60)
    for w in workers:
        print(f"  {w.worker_id:20s}  {w.url:20s}  ({w.hostname})")
    print("-" * 60)

    # Verify health
    print("\nVerifying worker health...")
    healthy = filter_healthy_workers(workers)
    print(f"  Healthy: {len(healthy)}/{len(workers)}")


def cmd_health(args) -> None:
    """Check worker health."""
    workers = get_workers(args)
    if not workers:
        return

    print(f"\nChecking health of {len(workers)} worker(s)...")
    print("-" * 70)

    healthy_count = 0
    for worker in workers:
        client = WorkerClient(worker.url)
        result = client.health_check()

        if result.get("status") == "healthy":
            healthy_count += 1
            tasks = result.get("tasks_completed", 0)
            print(f"  [OK]    {worker.url:20s}  tasks_completed={tasks}")
        else:
            error = result.get("error", "unknown error")
            print(f"  [FAIL]  {worker.url:20s}  {error}")

    print("-" * 70)
    print(f"Healthy: {healthy_count}/{len(workers)}")


def cmd_stats(args) -> None:
    """Show detailed worker statistics."""
    workers = get_workers(args)
    if not workers:
        return

    print(f"\nGathering stats from {len(workers)} worker(s)...")

    for worker in workers:
        client = WorkerClient(worker.url)
        stats = client.get_stats()

        print(f"\n{'='*60}")
        print(f"Worker: {worker.url}")
        print(f"{'='*60}")

        if "error" in stats:
            print(f"  Error: {stats['error']}")
            continue

        print(f"  Worker ID:         {stats.get('worker_id', 'unknown')}")
        print(f"  Tasks completed:   {stats.get('tasks_completed', 0)}")
        print(f"  Tasks failed:      {stats.get('tasks_failed', 0)}")
        print(f"  Total games:       {stats.get('total_games_played', 0)}")
        print(f"  Total eval time:   {stats.get('total_evaluation_time_sec', 0):.1f}s")
        print(f"  Uptime:            {stats.get('uptime_sec', 0):.0f}s")
        print(f"  Cached pools:      {stats.get('cached_pools', [])}")


def cmd_preload(args) -> None:
    """Preload state pools on workers."""
    workers = get_workers(args)
    if not workers:
        return

    print(f"\nPreloading pool on {len(workers)} worker(s)...")
    print(f"  Board: {args.board}")
    print(f"  Players: {args.num_players}")
    print(f"  Pool ID: {args.pool_id}")
    print()

    for worker in workers:
        client = WorkerClient(worker.url)
        result = client.preload_pool(args.board, args.num_players, args.pool_id)

        if result.get("status") == "success":
            size = result.get("pool_size", 0)
            print(f"  [OK]    {worker.url:20s}  loaded {size} states")
        else:
            error = result.get("error", "unknown error")
            print(f"  [FAIL]  {worker.url:20s}  {error}")


def cmd_test(args) -> None:
    """Run a test evaluation across workers."""
    workers = get_workers(args)
    if not workers:
        return

    healthy = filter_healthy_workers(workers)
    if not healthy:
        print("No healthy workers available for testing")
        return

    print(f"\nRunning test evaluation on {len(healthy)} worker(s)...")
    print(f"  Board: {args.board}")
    print(f"  Players: {args.num_players}")
    print(f"  Games per worker: {args.games}")
    print()

    evaluator = DistributedEvaluator(
        workers=[w.url for w in healthy],
        board_type=args.board,
        num_players=args.num_players,
        games_per_eval=args.games,
        eval_mode="multi-start",
        state_pool_id=args.pool_id,
        max_moves=200,
        eval_randomness=0.02,
    )

    # Preload pools
    print("Preloading state pools...")
    evaluator.preload_pools()

    # Create test population (baseline weights)
    population = [BASE_V1_BALANCED_WEIGHTS for _ in range(len(healthy))]

    print(f"\nEvaluating {len(population)} candidates...")
    start_time = time.time()

    def progress_callback(completed: int, total: int) -> None:
        print(f"  Progress: {completed}/{total}")

    fitness_scores, stats = evaluator.evaluate_population(
        population=population,
        progress_callback=progress_callback,
    )

    elapsed = time.time() - start_time

    print(f"\nResults:")
    print(f"  Candidates evaluated: {stats.total_candidates}")
    print(f"  Successful: {stats.successful_evaluations}")
    print(f"  Failed: {stats.failed_evaluations}")
    print(f"  Total games: {stats.total_games}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Fitness scores: {fitness_scores}")

    print("\nWorker task distribution:")
    for worker, count in stats.worker_task_counts.items():
        print(f"  {worker}: {count} tasks")


def cmd_kill_workers(args) -> None:
    """Show command to stop workers (workers must be stopped manually)."""
    print("To stop workers, press Ctrl+C in each worker terminal.")
    print("\nAlternatively, use SSH to stop workers remotely:")
    print("  ssh user@worker-ip 'pkill -f cluster_worker.py'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster management utilities for local Mac cluster"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # discover command
    p_discover = subparsers.add_parser("discover", help="Discover workers on network")
    p_discover.add_argument(
        "--timeout", type=float, default=10.0,
        help="Discovery timeout in seconds (default: 10)"
    )
    p_discover.add_argument(
        "--min-workers", type=int, default=0,
        help="Minimum workers to wait for (default: 0)"
    )

    # health command
    p_health = subparsers.add_parser("health", help="Check worker health")
    p_health.add_argument("--workers", type=str, help="Comma-separated worker URLs")
    p_health.add_argument("--discover", action="store_true", help="Auto-discover workers")
    p_health.add_argument("--timeout", type=float, default=10.0)
    p_health.add_argument("--min-workers", type=int, default=1)

    # stats command
    p_stats = subparsers.add_parser("stats", help="Show worker statistics")
    p_stats.add_argument("--workers", type=str, help="Comma-separated worker URLs")
    p_stats.add_argument("--discover", action="store_true", help="Auto-discover workers")
    p_stats.add_argument("--timeout", type=float, default=10.0)
    p_stats.add_argument("--min-workers", type=int, default=1)

    # preload command
    p_preload = subparsers.add_parser("preload", help="Preload state pools on workers")
    p_preload.add_argument("--workers", type=str, help="Comma-separated worker URLs")
    p_preload.add_argument("--discover", action="store_true", help="Auto-discover workers")
    p_preload.add_argument("--board", type=str, default="square8", help="Board type")
    p_preload.add_argument("--num-players", type=int, default=2, help="Number of players")
    p_preload.add_argument("--pool-id", type=str, default="v1", help="State pool ID")
    p_preload.add_argument("--timeout", type=float, default=10.0)
    p_preload.add_argument("--min-workers", type=int, default=1)

    # test command
    p_test = subparsers.add_parser("test", help="Run test evaluation")
    p_test.add_argument("--workers", type=str, help="Comma-separated worker URLs")
    p_test.add_argument("--discover", action="store_true", help="Auto-discover workers")
    p_test.add_argument("--board", type=str, default="square8", help="Board type")
    p_test.add_argument("--num-players", type=int, default=2, help="Number of players")
    p_test.add_argument("--pool-id", type=str, default="v1", help="State pool ID")
    p_test.add_argument("--games", type=int, default=4, help="Games per candidate")
    p_test.add_argument("--timeout", type=float, default=10.0)
    p_test.add_argument("--min-workers", type=int, default=1)

    args = parser.parse_args()

    if args.command == "discover":
        cmd_discover(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "preload":
        cmd_preload(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
