#!/usr/bin/env python3
"""Deploy distributed persona tournament across cluster nodes.

This script runs a large-scale tournament between all persona matchups
to evaluate their relative strengths with statistical significance.

Usage:
    # Deploy tournament to all nodes
    python scripts/deploy_persona_tournament.py --deploy

    # Check progress
    python scripts/deploy_persona_tournament.py --status

    # Collect and aggregate results
    python scripts/deploy_persona_tournament.py --collect

    # Generate Elo ratings from results
    python scripts/deploy_persona_tournament.py --analyze
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Cluster configuration
GH200_NODES = [
    "lambda-gh200-a", "lambda-gh200-b", "lambda-gh200-c", "lambda-gh200-d",
    "lambda-gh200-e", "lambda-gh200-f", "lambda-gh200-g", "lambda-gh200-h",
    "lambda-gh200-i", "lambda-gh200-j", "lambda-gh200-k", "lambda-gh200-l",
]

# All matchups for round-robin tournament
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

# Tournament configuration
GAMES_PER_MATCHUP = 500  # For statistical significance
MAX_MOVES_PER_GAME = 500
BATCH_SIZE = 256


@dataclass
class TournamentJob:
    """A single tournament job configuration."""
    node: str
    matchup: str
    board: str
    players: int
    games: int
    output_file: str


def get_tournament_plan(
    nodes: list[str] | None = None,
    boards: list[str] | None = None,
) -> list[TournamentJob]:
    """Create tournament plan distributing matchups across nodes."""
    nodes = nodes or GH200_NODES
    boards = boards or ["square8"]

    jobs = []
    node_idx = 0

    for board in boards:
        for matchup in MATCHUPS:
            node = nodes[node_idx % len(nodes)]
            output_file = f"~/tournament_results/{board}_{matchup}.json"

            jobs.append(TournamentJob(
                node=node,
                matchup=matchup,
                board=board,
                players=2,
                games=GAMES_PER_MATCHUP,
                output_file=output_file,
            ))
            node_idx += 1

    return jobs


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


def deploy_tournament_job(job: TournamentJob, dry_run: bool = False) -> bool:
    """Deploy a single tournament job to a node."""
    # Python script to run tournament and save results
    tournament_script = f'''
import json
import sys
sys.path.insert(0, "/root/ringrift/ai-service")

from app.ai.gpu_parallel_games import ParallelGameRunner

print("Running tournament: {job.matchup}")
runner = ParallelGameRunner.create_with_matchup("{job.matchup}", batch_size={BATCH_SIZE})

results = {{
    "matchup": "{job.matchup}",
    "board": "{job.board}",
    "players": {job.players},
    "games": {job.games},
    "p1_wins": 0,
    "p2_wins": 0,
    "draws": 0,
    "total_moves": 0,
}}

games_completed = 0
while games_completed < {job.games}:
    batch_games = min({BATCH_SIZE}, {job.games} - games_completed)
    runner.batch_size = batch_games
    stats = runner.run_games(max_moves={MAX_MOVES_PER_GAME})

    # Extract results from runner state
    for g in range(batch_games):
        winner = runner.state.winner[g].item()
        if winner == 1:
            results["p1_wins"] += 1
        elif winner == 2:
            results["p2_wins"] += 1
        else:
            results["draws"] += 1

    games_completed += batch_games
    print(f"  Progress: {{games_completed}}/{job.games} games")

    runner.reset_games()

results["total_games"] = results["p1_wins"] + results["p2_wins"] + results["draws"]
print(f"Results: P1={{results['p1_wins']}} P2={{results['p2_wins']}} Draw={{results['draws']}}")

with open("{job.output_file.replace('~', '/root')}", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {job.output_file}")
'''

    # Escape for shell
    escaped_script = tournament_script.replace("'", "'\\''")

    cmd = f"""
mkdir -p ~/tournament_results && \\
cd ~/ringrift/ai-service && \\
nohup python3 -c '{escaped_script}' > /tmp/tournament_{job.matchup}.log 2>&1 &
"""

    if dry_run:
        print(f"  [DRY RUN] Would deploy to {job.node}: {job.matchup}")
        return True

    success, output = run_ssh_command(job.node, cmd, timeout=30)
    return success


def check_tournament_status(node: str) -> dict:
    """Check tournament job status on node."""
    cmd = "ps aux | grep 'tournament' | grep python | grep -v grep | wc -l"
    success, output = run_ssh_command(node, cmd, timeout=10)

    running = False
    if success:
        try:
            running = int(output.strip()) > 0
        except ValueError:
            pass

    # Check for completed result files
    cmd2 = "ls ~/tournament_results/*.json 2>/dev/null | wc -l"
    _, result_count = run_ssh_command(node, cmd2, timeout=10)

    return {
        "node": node,
        "reachable": success,
        "running": running,
        "result_files": int(result_count.strip()) if result_count.strip().isdigit() else 0,
    }


def collect_results(nodes: list[str] | None = None) -> list[dict]:
    """Collect tournament results from all nodes."""
    nodes = nodes or GH200_NODES
    results = []

    for node in nodes:
        # List result files
        cmd = "cat ~/tournament_results/*.json 2>/dev/null"
        success, output = run_ssh_command(node, cmd, timeout=30)

        if success and output.strip():
            # Parse each JSON object (they might be concatenated)
            for line in output.strip().split("}{"):
                try:
                    if not line.startswith("{"):
                        line = "{" + line
                    if not line.endswith("}"):
                        line = line + "}"
                    data = json.loads(line)
                    data["source_node"] = node
                    results.append(data)
                except json.JSONDecodeError:
                    continue

    return results


def analyze_results(results: list[dict]) -> dict:
    """Analyze tournament results and compute Elo ratings."""
    if not results:
        return {"error": "No results to analyze"}

    # Aggregate by matchup
    matchup_stats = {}
    for r in results:
        matchup = r.get("matchup", "unknown")
        if matchup not in matchup_stats:
            matchup_stats[matchup] = {
                "p1_wins": 0, "p2_wins": 0, "draws": 0, "total": 0
            }
        matchup_stats[matchup]["p1_wins"] += r.get("p1_wins", 0)
        matchup_stats[matchup]["p2_wins"] += r.get("p2_wins", 0)
        matchup_stats[matchup]["draws"] += r.get("draws", 0)
        matchup_stats[matchup]["total"] += r.get("total_games", 0)

    # Compute win rates
    for matchup, stats in matchup_stats.items():
        total = stats["total"] or 1
        stats["p1_win_rate"] = stats["p1_wins"] / total
        stats["p2_win_rate"] = stats["p2_wins"] / total
        stats["draw_rate"] = stats["draws"] / total

    # Simple Elo estimation (Bradley-Terry model approximation)
    personas = ["aggressive", "defensive", "balanced", "territorial"]
    elo = {p: 1500.0 for p in personas}  # Start at 1500

    # Update Elo based on matchup results
    K = 32  # Elo K-factor
    for matchup, stats in matchup_stats.items():
        if "_vs_" in matchup:
            p1_name, p2_name = matchup.split("_vs_")
            if p1_name in elo and p2_name in elo:
                # Expected scores
                e1 = 1 / (1 + 10 ** ((elo[p2_name] - elo[p1_name]) / 400))
                e2 = 1 - e1

                # Actual scores
                total = stats["total"] or 1
                s1 = (stats["p1_wins"] + 0.5 * stats["draws"]) / total
                s2 = (stats["p2_wins"] + 0.5 * stats["draws"]) / total

                # Update
                elo[p1_name] += K * (s1 - e1)
                elo[p2_name] += K * (s2 - e2)

    return {
        "matchup_stats": matchup_stats,
        "elo_ratings": dict(sorted(elo.items(), key=lambda x: -x[1])),
        "total_games": sum(s["total"] for s in matchup_stats.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Deploy distributed persona tournament")
    parser.add_argument("--deploy", action="store_true", help="Deploy tournament jobs")
    parser.add_argument("--status", action="store_true", help="Check job status")
    parser.add_argument("--collect", action="store_true", help="Collect results")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    parser.add_argument("--plan", action="store_true", help="Show deployment plan")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--nodes", type=str, help="Comma-separated node list")
    parser.add_argument("--boards", type=str, default="square8", help="Board types")
    args = parser.parse_args()

    nodes = [n.strip() for n in args.nodes.split(",")] if args.nodes else None
    boards = [b.strip() for b in args.boards.split(",")]

    if args.plan or (not args.deploy and not args.status and not args.collect and not args.analyze):
        print("=" * 70)
        print("PERSONA TOURNAMENT DEPLOYMENT PLAN")
        print("=" * 70)

        plan = get_tournament_plan(nodes, boards)

        for job in plan:
            print(f"  {job.node}: {job.matchup} ({job.board}, {job.games} games)")

        print(f"\nTotal: {len(plan)} tournament jobs")
        print(f"Games per matchup: {GAMES_PER_MATCHUP}")
        print(f"Total games: {len(plan) * GAMES_PER_MATCHUP}")
        return

    if args.status:
        print("=" * 70)
        print("TOURNAMENT STATUS")
        print("=" * 70)

        target_nodes = nodes or GH200_NODES
        for node in target_nodes:
            status = check_tournament_status(node)
            icon = "▶" if status["running"] else "○" if status["reachable"] else "✗"
            state = "RUNNING" if status["running"] else f"IDLE ({status['result_files']} results)"
            print(f"{icon} {node}: {state}")
        return

    if args.collect or args.analyze:
        print("Collecting results from cluster...")
        results = collect_results(nodes)
        print(f"Collected {len(results)} result files")

        if args.analyze:
            print("\n" + "=" * 70)
            print("TOURNAMENT ANALYSIS")
            print("=" * 70)

            analysis = analyze_results(results)

            if "error" in analysis:
                print(f"Error: {analysis['error']}")
                return

            print(f"\nTotal games: {analysis['total_games']}")

            print("\n--- Elo Ratings ---")
            for persona, rating in analysis["elo_ratings"].items():
                print(f"  {persona}: {rating:.0f}")

            print("\n--- Matchup Results ---")
            for matchup, stats in sorted(analysis["matchup_stats"].items()):
                if stats["total"] > 0:
                    print(f"  {matchup}:")
                    print(f"    P1: {stats['p1_win_rate']*100:.1f}%  P2: {stats['p2_win_rate']*100:.1f}%  Draw: {stats['draw_rate']*100:.1f}%")
        return

    if args.deploy:
        print("=" * 70)
        print("DEPLOYING PERSONA TOURNAMENT")
        print("=" * 70)

        plan = get_tournament_plan(nodes, boards)

        success_count = 0
        for job in plan:
            success = deploy_tournament_job(job, dry_run=args.dry_run)
            icon = "✓" if success else "✗"
            print(f"  {icon} {job.node}: {job.matchup}")
            if success:
                success_count += 1

        print(f"\nDeployed: {success_count}/{len(plan)} jobs")


if __name__ == "__main__":
    main()
