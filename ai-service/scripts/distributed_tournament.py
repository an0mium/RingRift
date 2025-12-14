#!/usr/bin/env python3
"""Distributed tournament execution across cluster hosts.

Runs Elo tournaments in parallel across multiple cluster nodes to speed up
model evaluation. Each host runs a subset of matchups, and results are
aggregated to the unified Elo database.

Usage:
    # Run distributed tournament on all available hosts
    python scripts/distributed_tournament.py --board square8 --players 2 --games 8

    # Run on specific hosts
    python scripts/distributed_tournament.py --hosts lambda-2xh100,lambda-h100 --games 4

    # Dry run - show matchup distribution
    python scripts/distributed_tournament.py --dry-run

    # Use only local execution (for testing)
    python scripts/distributed_tournament.py --local-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.distributed.hosts import load_remote_hosts, HostConfig

# Configuration
SSH_TIMEOUT = 30
TOURNAMENT_TIMEOUT = 1800  # 30 minutes per host batch
MAX_MATCHUPS_PER_HOST = 50  # Limit matchups per host to balance load


@dataclass
class MatchupAssignment:
    """A matchup assigned to a specific host."""
    model_a: str
    model_b: str
    games: int
    host: str
    board_type: str
    num_players: int


@dataclass
class TournamentResult:
    """Result from a distributed tournament batch."""
    host: str
    matchups_completed: int
    games_played: int
    success: bool
    error: Optional[str] = None
    duration_seconds: float = 0.0


def _ssh_base_opts(host: HostConfig) -> List[str]:
    """Get SSH options for a host."""
    opts = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
    try:
        key_path = host.ssh_key_path
    except Exception:
        key_path = ""
    if key_path:
        opts.extend(["-i", key_path])
    if int(getattr(host, "ssh_port", 22) or 22) != 22:
        opts.extend(["-p", str(int(host.ssh_port))])
    return opts


def _pick_reachable_ssh_target(host: HostConfig) -> Tuple[Optional[str], Optional[str]]:
    """Pick the first reachable SSH target for a host."""
    opts = _ssh_base_opts(host)
    for target in getattr(host, "ssh_targets", []) or [host.ssh_target]:
        try:
            result = subprocess.run(
                ["ssh", *opts, target, "echo ok"],
                capture_output=True,
                timeout=SSH_TIMEOUT,
                text=True,
            )
            if result.returncode == 0:
                return target, None
        except Exception as e:
            continue
    return None, "unreachable"


def get_available_hosts(hosts: Dict[str, HostConfig], specific_hosts: Optional[List[str]] = None) -> List[Tuple[str, HostConfig, str]]:
    """Get list of available hosts with their SSH targets.

    Returns list of (name, config, ssh_target).
    """
    available = []

    target_hosts = specific_hosts or list(hosts.keys())

    for name in target_hosts:
        if name not in hosts:
            print(f"Warning: Unknown host {name}")
            continue

        host = hosts[name]
        ssh_target, err = _pick_reachable_ssh_target(host)

        if ssh_target:
            available.append((name, host, ssh_target))
            print(f"  {name}: available")
        else:
            print(f"  {name}: unavailable ({err})")

    return available


def get_models_for_tournament(
    board_type: str,
    num_players: int,
    top_n: int = 50,
) -> List[str]:
    """Get top models for tournament from unified Elo database."""
    import sqlite3

    db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
    if not db_path.exists():
        # Fall back to listing model files
        models = []
        for pth in (AI_SERVICE_ROOT / "models").glob(f"*{board_type}*{num_players}p*.pth"):
            models.append(pth.name)
        return models[:top_n]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT participant_id, elo, games_played
            FROM participants
            WHERE participant_type = 'nn'
            AND board_type = ?
            AND num_players = ?
            ORDER BY elo DESC
            LIMIT ?
        """, (board_type, num_players, top_n))

        models = []
        for row in cursor.fetchall():
            participant_id = row[0]
            if participant_id.startswith("nn:"):
                model_path = participant_id[3:]
                models.append(Path(model_path).name)
            else:
                models.append(participant_id)

        return models

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        conn.close()


def generate_matchups(
    models: List[str],
    games_per_matchup: int,
    board_type: str,
    num_players: int,
) -> List[Tuple[str, str, int]]:
    """Generate all matchups for round-robin tournament.

    Returns list of (model_a, model_b, games).
    """
    matchups = []

    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            matchups.append((model_a, model_b, games_per_matchup))

    # Shuffle for better load distribution
    random.shuffle(matchups)

    return matchups


def distribute_matchups(
    matchups: List[Tuple[str, str, int]],
    hosts: List[Tuple[str, HostConfig, str]],
    board_type: str,
    num_players: int,
) -> Dict[str, List[MatchupAssignment]]:
    """Distribute matchups across available hosts.

    Returns dict of host_name -> list of matchup assignments.
    """
    assignments: Dict[str, List[MatchupAssignment]] = {name: [] for name, _, _ in hosts}

    # Round-robin distribution
    host_idx = 0
    for model_a, model_b, games in matchups:
        host_name = hosts[host_idx % len(hosts)][0]

        # Respect per-host limit
        if len(assignments[host_name]) >= MAX_MATCHUPS_PER_HOST:
            # Find host with fewest assignments
            host_name = min(assignments.keys(), key=lambda h: len(assignments[h]))

        assignments[host_name].append(MatchupAssignment(
            model_a=model_a,
            model_b=model_b,
            games=games,
            host=host_name,
            board_type=board_type,
            num_players=num_players,
        ))

        host_idx += 1

    return assignments


def run_tournament_on_host(
    host_name: str,
    host: HostConfig,
    ssh_target: str,
    matchups: List[MatchupAssignment],
    both_ai_types: bool = True,
    dry_run: bool = False,
) -> TournamentResult:
    """Run tournament matchups on a remote host."""
    if not matchups:
        return TournamentResult(host=host_name, matchups_completed=0, games_played=0, success=True)

    start_time = time.time()

    if dry_run:
        return TournamentResult(
            host=host_name,
            matchups_completed=len(matchups),
            games_played=sum(m.games for m in matchups),
            success=True,
            duration_seconds=0,
        )

    # Build matchup list for remote execution
    matchup_data = [
        {"model_a": m.model_a, "model_b": m.model_b, "games": m.games}
        for m in matchups
    ]

    board_type = matchups[0].board_type
    num_players = matchups[0].num_players

    work_dir = (getattr(host, "work_directory", host.work_dir or "") or "").strip() or "~/Development/RingRift/ai-service"

    # Create remote script to run matchups
    matchup_json = json.dumps(matchup_data)
    ai_types_flag = "--both-ai-types" if both_ai_types else ""

    remote_cmd = f"""
cd {work_dir}
export PYTHONPATH={work_dir}

# Run tournament for assigned matchups
python3 scripts/run_model_elo_tournament.py \\
    --board {board_type} \\
    --players {num_players} \\
    --games {matchups[0].games} \\
    --top-n {len(set(m.model_a for m in matchups) | set(m.model_b for m in matchups))} \\
    {ai_types_flag} \\
    --run 2>&1

echo "DISTRIBUTED_TOURNAMENT_COMPLETE"
"""

    ssh_opts = _ssh_base_opts(host)

    try:
        result = subprocess.run(
            ["ssh", *ssh_opts, ssh_target, remote_cmd],
            capture_output=True,
            timeout=TOURNAMENT_TIMEOUT,
            text=True,
        )

        duration = time.time() - start_time

        if result.returncode == 0 and "DISTRIBUTED_TOURNAMENT_COMPLETE" in result.stdout:
            return TournamentResult(
                host=host_name,
                matchups_completed=len(matchups),
                games_played=sum(m.games for m in matchups) * (4 if both_ai_types else 1),
                success=True,
                duration_seconds=duration,
            )
        else:
            return TournamentResult(
                host=host_name,
                matchups_completed=0,
                games_played=0,
                success=False,
                error=result.stderr[:500] if result.stderr else result.stdout[:500],
                duration_seconds=duration,
            )

    except subprocess.TimeoutExpired:
        return TournamentResult(
            host=host_name,
            matchups_completed=0,
            games_played=0,
            success=False,
            error="Timeout",
            duration_seconds=TOURNAMENT_TIMEOUT,
        )
    except Exception as e:
        return TournamentResult(
            host=host_name,
            matchups_completed=0,
            games_played=0,
            success=False,
            error=str(e),
            duration_seconds=time.time() - start_time,
        )


def run_distributed_tournament(
    board_type: str,
    num_players: int,
    games_per_matchup: int,
    top_n: int,
    hosts: List[Tuple[str, HostConfig, str]],
    both_ai_types: bool = True,
    dry_run: bool = False,
    max_workers: int = 8,
) -> List[TournamentResult]:
    """Run distributed tournament across multiple hosts."""

    # Get models
    print(f"\nGetting top {top_n} models for {board_type}_{num_players}p...")
    models = get_models_for_tournament(board_type, num_players, top_n)
    print(f"Found {len(models)} models")

    if len(models) < 2:
        print("Not enough models for tournament")
        return []

    # Generate matchups
    matchups = generate_matchups(models, games_per_matchup, board_type, num_players)
    total_games = sum(m[2] for m in matchups) * (4 if both_ai_types else 1)
    print(f"Generated {len(matchups)} matchups ({total_games} total games with cross-inference)")

    # Distribute matchups
    assignments = distribute_matchups(matchups, hosts, board_type, num_players)

    print(f"\nMatchup distribution:")
    for host_name, host_matchups in assignments.items():
        games = sum(m.games for m in host_matchups) * (4 if both_ai_types else 1)
        print(f"  {host_name}: {len(host_matchups)} matchups ({games} games)")

    if dry_run:
        print("\n[DRY RUN] Would execute on hosts")
        return [
            TournamentResult(host=name, matchups_completed=len(m), games_played=sum(x.games for x in m), success=True)
            for name, m in assignments.items()
        ]

    # Execute in parallel
    print(f"\nExecuting tournaments on {len(hosts)} hosts...")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for host_name, host, ssh_target in hosts:
            host_matchups = assignments[host_name]
            if not host_matchups:
                continue

            future = executor.submit(
                run_tournament_on_host,
                host_name, host, ssh_target, host_matchups, both_ai_types, dry_run
            )
            futures[future] = host_name

        for future in as_completed(futures):
            host_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result.success:
                    print(f"  {host_name}: completed {result.matchups_completed} matchups in {result.duration_seconds:.1f}s")
                else:
                    print(f"  {host_name}: FAILED - {result.error}")
            except Exception as e:
                print(f"  {host_name}: EXCEPTION - {e}")
                results.append(TournamentResult(host=host_name, matchups_completed=0, games_played=0, success=False, error=str(e)))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run distributed Elo tournament")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--games", type=int, default=4, help="Games per matchup")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top models to include")
    parser.add_argument("--hosts", type=str, help="Comma-separated list of hosts")
    parser.add_argument("--local-only", action="store_true", help="Run locally only")
    parser.add_argument("--dry-run", action="store_true", help="Show distribution without executing")
    parser.add_argument("--no-cross-inference", action="store_true", help="Disable cross-inference (4 AI type combos)")

    args = parser.parse_args()

    print("=" * 60)
    print("DISTRIBUTED ELO TOURNAMENT")
    print("=" * 60)
    print(f"Board: {args.board}")
    print(f"Players: {args.players}")
    print(f"Games per matchup: {args.games}")
    print(f"Top N models: {args.top_n}")
    print(f"Cross-inference: {not args.no_cross_inference}")

    if args.local_only:
        # Run locally using standard tournament script
        print("\nRunning locally...")
        cmd = [
            sys.executable, "scripts/run_model_elo_tournament.py",
            "--board", args.board,
            "--players", str(args.players),
            "--games", str(args.games),
            "--top-n", str(args.top_n),
            "--run",
        ]
        if not args.no_cross_inference:
            cmd.append("--both-ai-types")

        os.execv(sys.executable, cmd)
        return

    # Load remote hosts
    print("\nChecking available hosts...")
    all_hosts = load_remote_hosts()
    specific_hosts = args.hosts.split(",") if args.hosts else None

    available = get_available_hosts(all_hosts, specific_hosts)

    if not available:
        print("No hosts available. Falling back to local execution.")
        args.local_only = True
        main()
        return

    print(f"\n{len(available)} hosts available")

    # Run distributed tournament
    results = run_distributed_tournament(
        board_type=args.board,
        num_players=args.players,
        games_per_matchup=args.games,
        top_n=args.top_n,
        hosts=available,
        both_ai_types=not args.no_cross_inference,
        dry_run=args.dry_run,
    )

    # Summary
    print("\n" + "=" * 60)
    print("TOURNAMENT SUMMARY")
    print("=" * 60)

    total_matchups = sum(r.matchups_completed for r in results)
    total_games = sum(r.games_played for r in results)
    successful = sum(1 for r in results if r.success)

    print(f"Hosts successful: {successful}/{len(results)}")
    print(f"Total matchups: {total_matchups}")
    print(f"Total games: {total_games}")

    if results:
        avg_duration = sum(r.duration_seconds for r in results) / len(results)
        print(f"Average duration: {avg_duration:.1f}s per host")


if __name__ == "__main__":
    main()
