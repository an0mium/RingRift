#!/usr/bin/env python
"""Distributed self-play soak runner for RingRift.

Runs mixed-AI self-play across multiple machines (local + remote SSH hosts)
for all board types and player counts. Records games to SQLite databases
for subsequent parity validation.

Example usage (from ai-service/):

    # Run 100 games per configuration across local + m1-pro
    python scripts/run_distributed_selfplay_soak.py \
        --games-per-config 100 \
        --hosts local,m1-pro \
        --output-dir data/games/distributed_soak

    # Dry run to see what would be executed
    python scripts/run_distributed_selfplay_soak.py \
        --games-per-config 50 \
        --dry-run

After completion, run parity checks on all generated databases:

    python scripts/check_ts_python_replay_parity.py \
        --db data/games/distributed_soak/*.db
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Board configurations with appropriate max moves
BOARD_CONFIGS: Dict[str, Dict[int, int]] = {
    # board_type: {num_players: max_moves}
    "square8": {2: 200, 3: 250, 4: 300},
    "square19": {2: 1200, 3: 1400, 4: 1600},
    "hexagonal": {2: 1200, 3: 1400, 4: 1600},
}

# Default config file paths (relative to ai-service/)
CONFIG_FILE_PATH = "config/distributed_hosts.yaml"
TEMPLATE_CONFIG_PATH = "config/distributed_hosts.template.yaml"


def load_remote_hosts(config_path: Optional[str] = None) -> Dict[str, Dict]:
    """Load remote host configuration from YAML file.

    Looks for config in:
    1. Explicitly provided path (--config flag)
    2. config/distributed_hosts.yaml (gitignored, local config)
    3. Falls back to empty dict if neither exists

    Copy config/distributed_hosts.template.yaml to config/distributed_hosts.yaml
    and fill in your actual host details.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ai_service_dir = os.path.dirname(script_dir)

    # Try explicit path first
    if config_path:
        if not os.path.isabs(config_path):
            config_path = os.path.join(ai_service_dir, config_path)
        paths_to_try = [config_path]
    else:
        paths_to_try = [
            os.path.join(ai_service_dir, CONFIG_FILE_PATH),
        ]

    for path in paths_to_try:
        if os.path.exists(path):
            if not HAS_YAML:
                print(f"Warning: PyYAML not installed. Install with: pip install pyyaml")
                print(f"         Cannot load config from {path}")
                return {}

            with open(path, "r") as f:
                config = yaml.safe_load(f)

            hosts = config.get("hosts", {})
            if hosts:
                print(f"Loaded {len(hosts)} remote host(s) from {path}")
            return hosts

    # No config found - print helpful message
    template_path = os.path.join(ai_service_dir, TEMPLATE_CONFIG_PATH)
    if os.path.exists(template_path):
        print(f"No host configuration found.")
        print(f"Copy {template_path}")
        print(f"  to {os.path.join(ai_service_dir, CONFIG_FILE_PATH)}")
        print(f"and fill in your host details.")

    return {}


# Remote hosts loaded at module init (can be overridden by load_remote_hosts)
REMOTE_HOSTS: Dict[str, Dict] = {}


@dataclass
class JobConfig:
    """Configuration for a single self-play job."""
    job_id: str
    host: str
    board_type: str
    num_players: int
    num_games: int
    max_moves: int
    output_db: str
    log_jsonl: str
    seed: int


def generate_job_configs(
    games_per_config: int,
    hosts: List[str],
    output_dir: str,
    base_seed: int = 42,
) -> List[JobConfig]:
    """Generate job configurations distributed across hosts."""
    jobs = []
    job_idx = 0

    # Calculate games per host per config
    num_hosts = len(hosts)
    games_per_host = games_per_config // num_hosts
    remainder = games_per_config % num_hosts

    for board_type, player_configs in BOARD_CONFIGS.items():
        for num_players, max_moves in player_configs.items():
            for host_idx, host in enumerate(hosts):
                # Distribute remainder across first hosts
                host_games = games_per_host + (1 if host_idx < remainder else 0)
                if host_games == 0:
                    continue

                config_id = f"{board_type}_{num_players}p"
                job_id = f"{config_id}_{host}_{job_idx}"

                # Different seed per job for variety
                job_seed = base_seed + job_idx * 1000

                jobs.append(JobConfig(
                    job_id=job_id,
                    host=host,
                    board_type=board_type,
                    num_players=num_players,
                    num_games=host_games,
                    max_moves=max_moves,
                    output_db=os.path.join(output_dir, f"selfplay_{config_id}_{host}.db"),
                    log_jsonl=os.path.join(output_dir, f"selfplay_{config_id}_{host}.jsonl"),
                    seed=job_seed,
                ))
                job_idx += 1

    return jobs


def build_soak_command(job: JobConfig, is_remote: bool = False) -> str:
    """Build the self-play soak command for a job."""
    cmd_parts = [
        "PYTHONPATH=.",
        "RINGRIFT_SKIP_SHADOW_CONTRACTS=true",
        "python",
        "scripts/run_self_play_soak.py",
        f"--num-games {job.num_games}",
        f"--board-type {job.board_type}",
        f"--num-players {job.num_players}",
        f"--max-moves {job.max_moves}",
        f"--seed {job.seed}",
        "--engine-mode mixed",
        "--difficulty-band canonical",
        f"--log-jsonl {job.log_jsonl}",
        f"--record-db {job.output_db}",
        "--verbose 10",
        "--gc-interval 5",
    ]

    # Add memory constraints for large boards
    if job.board_type in ("square19", "hexagonal"):
        cmd_parts.append("--memory-constrained")

    return " ".join(cmd_parts)


def run_local_job(job: JobConfig, ringrift_ai_dir: str) -> Tuple[str, bool, str]:
    """Run a self-play job on the local machine."""
    cmd = build_soak_command(job)

    print(f"[LOCAL] Starting job {job.job_id}: {job.num_games} games of "
          f"{job.board_type} {job.num_players}p")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=ringrift_ai_dir,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print(f"[LOCAL] Job {job.job_id} completed successfully")
        else:
            print(f"[LOCAL] Job {job.job_id} failed with code {result.returncode}")

        return job.job_id, success, output

    except subprocess.TimeoutExpired:
        print(f"[LOCAL] Job {job.job_id} timed out")
        return job.job_id, False, "Job timed out after 2 hours"
    except Exception as e:
        print(f"[LOCAL] Job {job.job_id} error: {e}")
        return job.job_id, False, str(e)


def run_remote_job(job: JobConfig, host_config: Dict) -> Tuple[str, bool, str]:
    """Run a self-play job on a remote machine via SSH."""
    ssh_host = host_config["ssh_host"]
    ringrift_path = host_config["ringrift_path"]
    venv_activate = host_config["venv_activate"]
    ssh_key = host_config.get("ssh_key")

    # Build remote command
    soak_cmd = build_soak_command(job, is_remote=True)
    remote_cmd = f"cd {ringrift_path}/ai-service && {venv_activate} && {soak_cmd}"

    ssh_cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=60",
    ]
    if ssh_key:
        ssh_cmd.extend(["-i", os.path.expanduser(ssh_key)])
    ssh_cmd.extend([ssh_host, remote_cmd])

    print(f"[{ssh_host.upper()}] Starting job {job.job_id}: {job.num_games} games of "
          f"{job.board_type} {job.num_players}p")

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print(f"[{ssh_host.upper()}] Job {job.job_id} completed successfully")
        else:
            print(f"[{ssh_host.upper()}] Job {job.job_id} failed with code {result.returncode}")

        return job.job_id, success, output

    except subprocess.TimeoutExpired:
        print(f"[{ssh_host.upper()}] Job {job.job_id} timed out")
        return job.job_id, False, "Job timed out after 2 hours"
    except Exception as e:
        print(f"[{ssh_host.upper()}] Job {job.job_id} error: {e}")
        return job.job_id, False, str(e)


def run_job(job: JobConfig, ringrift_ai_dir: str) -> Tuple[str, bool, str]:
    """Dispatch job to appropriate host."""
    if job.host == "local":
        return run_local_job(job, ringrift_ai_dir)
    elif job.host in REMOTE_HOSTS:
        return run_remote_job(job, REMOTE_HOSTS[job.host])
    else:
        return job.job_id, False, f"Unknown host: {job.host}"


def fetch_remote_results(jobs: List[JobConfig], output_dir: str) -> None:
    """Fetch database files from remote hosts."""
    for job in jobs:
        if job.host != "local" and job.host in REMOTE_HOSTS:
            host_config = REMOTE_HOSTS[job.host]
            ssh_host = host_config["ssh_host"]
            ringrift_path = host_config["ringrift_path"]

            remote_db = f"{ringrift_path}/ai-service/{job.output_db}"
            local_db = os.path.join(output_dir, os.path.basename(job.output_db))

            print(f"Fetching {remote_db} from {ssh_host}...")

            try:
                subprocess.run(
                    ["scp", f"{ssh_host}:{remote_db}", local_db],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
                print(f"  -> Saved to {local_db}")
            except subprocess.CalledProcessError as e:
                print(f"  -> Failed to fetch: {e}")
            except subprocess.TimeoutExpired:
                print(f"  -> Fetch timed out")


def main():
    parser = argparse.ArgumentParser(
        description="Run distributed self-play soaks across multiple machines"
    )
    parser.add_argument(
        "--games-per-config",
        type=int,
        default=100,
        help="Number of games per (board_type, num_players) configuration",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default="local",
        help="Comma-separated list of hosts to use (local, m1-pro). Default: local",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/games/distributed_soak",
        help="Directory for output databases and logs",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--max-parallel-per-host",
        type=int,
        default=2,
        help="Maximum parallel jobs per host (default: 2 to avoid memory exhaustion)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job configurations without running",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching results from remote hosts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to host configuration YAML file (default: {CONFIG_FILE_PATH})",
    )

    args = parser.parse_args()

    # Load remote host configuration
    global REMOTE_HOSTS
    REMOTE_HOSTS = load_remote_hosts(args.config)

    # Parse hosts
    hosts = [h.strip() for h in args.hosts.split(",")]

    # Validate hosts
    for host in hosts:
        if host != "local" and host not in REMOTE_HOSTS:
            available = ["local"] + list(REMOTE_HOSTS.keys())
            print(f"Error: Unknown host '{host}'.")
            print(f"Available hosts: {', '.join(available)}")
            if not REMOTE_HOSTS:
                print(f"\nNo remote hosts configured. See: config/distributed_hosts.template.yaml")
            sys.exit(1)

    # Determine ai-service directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ringrift_ai_dir = os.path.dirname(script_dir)

    # Create output directory
    output_dir = os.path.join(ringrift_ai_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Generate job configurations
    jobs = generate_job_configs(
        games_per_config=args.games_per_config,
        hosts=hosts,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
    )

    print(f"\n{'='*60}")
    print(f"Distributed Self-Play Soak Configuration")
    print(f"{'='*60}")
    print(f"Games per config: {args.games_per_config}")
    print(f"Hosts: {', '.join(hosts)}")
    print(f"Output directory: {output_dir}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Total games: {sum(j.num_games for j in jobs)}")
    print()

    # Print job summary
    print("Job Summary:")
    print("-" * 60)
    for job in jobs:
        print(f"  {job.job_id}: {job.board_type} {job.num_players}p x{job.num_games} "
              f"-> {job.output_db}")
    print()

    if args.dry_run:
        print("DRY RUN - commands that would be executed:")
        print("-" * 60)
        for job in jobs:
            cmd = build_soak_command(job)
            if job.host == "local":
                print(f"[LOCAL] {cmd}")
            else:
                print(f"[{job.host.upper()}] ssh {job.host} 'cd ... && {cmd}'")
        return

    # Run jobs
    print(f"\nStarting distributed self-play at {datetime.now().isoformat()}")
    print("=" * 60)

    start_time = time.time()
    results = []

    # Group jobs by host for parallel execution
    jobs_by_host: Dict[str, List[JobConfig]] = {}
    for job in jobs:
        if job.host not in jobs_by_host:
            jobs_by_host[job.host] = []
        jobs_by_host[job.host].append(job)

    # Run jobs with limited parallelism per host
    with ThreadPoolExecutor(max_workers=len(hosts) * args.max_parallel_per_host) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(run_job, job, ringrift_ai_dir)
            futures[future] = job

        for future in as_completed(futures):
            job_id, success, output = future.result()
            results.append((job_id, success, output))

    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print(f"Distributed Self-Play Complete")
    print("=" * 60)
    print(f"Elapsed time: {elapsed/60:.1f} minutes")

    successful = sum(1 for _, s, _ in results if s)
    failed = len(results) - successful

    print(f"Successful jobs: {successful}/{len(results)}")
    if failed > 0:
        print(f"Failed jobs: {failed}")
        for job_id, success, output in results:
            if not success:
                print(f"  - {job_id}")

    # Fetch results from remote hosts
    if not args.skip_fetch and any(j.host != "local" for j in jobs):
        print()
        print("Fetching results from remote hosts...")
        fetch_remote_results(jobs, output_dir)

    # Write summary
    summary_path = os.path.join(output_dir, "distributed_soak_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "games_per_config": args.games_per_config,
        "hosts": hosts,
        "total_jobs": len(jobs),
        "successful_jobs": successful,
        "failed_jobs": failed,
        "job_results": [
            {"job_id": jid, "success": s}
            for jid, s, _ in results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")

    print()
    print("Next steps:")
    print("  1. Run parity checks on generated databases:")
    print(f"     cd ai-service && python scripts/check_ts_python_replay_parity.py \\")
    print(f"         --db {args.output_dir}/*.db")
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
