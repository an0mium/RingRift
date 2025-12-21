#!/usr/bin/env python3
"""Orchestrate diverse tournaments across all board/player combinations.

This script schedules and runs diverse selfplay tournaments for Elo calibration
across all supported configurations:
- Board types: square8, square19, hexagonal
- Player counts: 2, 3, 4

Supports both local and distributed (parallel) execution across cluster hosts.

Usage:
    # Run locally (sequential)
    python scripts/run_diverse_tournaments.py --games-per-config 100

    # Run distributed across cluster (parallel)
    python scripts/run_diverse_tournaments.py --distributed --games-per-config 100

    # Run continuously every 4 hours
    python scripts/run_diverse_tournaments.py --distributed --interval-hours 4

    # Specific configurations only
    python scripts/run_diverse_tournaments.py --board-types square8 --player-counts 2 3

    # Dry run to see what would be scheduled
    python scripts/run_diverse_tournaments.py --distributed --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import resilience utilities
from scripts.lib.resilience import (
    HostHealthTracker,
    exponential_backoff_delay,
    is_network_error,
)

# Unified logging setup
import contextlib

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_diverse_tournaments")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

# Minimum memory requirement for task assignment
MIN_MEMORY_GB_FOR_TASKS = 64

@dataclass
class ClusterHost:
    """A host in the distributed cluster."""
    name: str
    ssh_host: str
    ssh_user: str = "root"
    ssh_key: str | None = None
    ssh_port: int = 22
    ringrift_path: str = "~/ringrift"
    venv_activate: str | None = None
    status: str = "ready"
    memory_gb: int = 0  # System memory in GB (0 = unknown)

    def ssh_cmd_prefix(self) -> list[str]:
        """Build SSH command prefix for this host."""
        cmd = ["ssh", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
        if self.ssh_key:
            key_path = os.path.expanduser(self.ssh_key)
            cmd.extend(["-i", key_path])
        if self.ssh_port != 22:
            cmd.extend(["-p", str(self.ssh_port)])
        cmd.append(f"{self.ssh_user}@{self.ssh_host}")
        return cmd


@dataclass
class TournamentConfig:
    """Configuration for a single tournament."""
    board_type: str
    num_players: int
    num_games: int
    output_path: str
    seed: int | None = None


@dataclass
class TournamentResult:
    """Result of running a tournament."""
    config: TournamentConfig
    host: str  # "local" or host name
    success: bool
    games_completed: int
    samples_generated: int
    duration_sec: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_BOARD_TYPES = ["square8", "square19", "hexagonal"]
ALL_PLAYER_COUNTS = [2, 3, 4]

# Recommended games per config for meaningful Elo estimates
DEFAULT_GAMES_PER_CONFIG = {
    ("square8", 2): 200,
    ("square8", 3): 100,
    ("square8", 4): 100,
    ("square19", 2): 50,
    ("square19", 3): 30,
    ("square19", 4): 30,
    ("hexagonal", 2): 100,
    ("hexagonal", 3): 50,
    ("hexagonal", 4): 50,
}


# ---------------------------------------------------------------------------
# Host Management
# ---------------------------------------------------------------------------

def load_cluster_hosts(config_path: str | None = None) -> list[ClusterHost]:
    """Load cluster hosts from distributed_hosts.yaml."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, cannot load cluster hosts")
        return []

    if config_path is None:
        config_path = ROOT / "config" / "distributed_hosts.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Cluster config not found: {config_path}")
        return []

    try:
        data = yaml.safe_load(config_path.read_text()) or {}
        hosts_data = data.get("hosts", {})
    except Exception as e:
        logger.error(f"Failed to parse cluster config: {e}")
        return []

    hosts = []
    for name, cfg in hosts_data.items():
        if cfg.get("status") == "disabled":
            continue

        ssh_host = cfg.get("ssh_host") or cfg.get("tailscale_ip")
        if not ssh_host:
            continue

        memory_gb = int(cfg.get("memory_gb", 0) or 0)
        # Skip low-memory hosts
        if memory_gb > 0 and memory_gb < MIN_MEMORY_GB_FOR_TASKS:
            logger.info(f"Skipping {name}: {memory_gb}GB RAM < {MIN_MEMORY_GB_FOR_TASKS}GB minimum")
            continue

        host = ClusterHost(
            name=name,
            ssh_host=ssh_host,
            ssh_user=cfg.get("ssh_user", "root"),
            ssh_key=cfg.get("ssh_key"),
            ssh_port=cfg.get("ssh_port", 22),
            ringrift_path=cfg.get("ringrift_path", "~/ringrift"),
            venv_activate=cfg.get("venv_activate"),
            status=cfg.get("status", "ready"),
            memory_gb=memory_gb,
        )
        hosts.append(host)

    return hosts


async def check_host_available(host: ClusterHost, timeout: int = 10, retries: int = 2) -> bool:
    """Check if a host is reachable and ready with retries.

    Args:
        host: Cluster host to check
        timeout: Timeout per attempt in seconds
        retries: Number of retry attempts on failure

    Returns:
        True if host is reachable, False otherwise
    """
    cmd = [*host.ssh_cmd_prefix(), "echo ok"]

    for attempt in range(retries):
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode == 0 and b"ok" in stdout:
                return True
        except Exception as e:
            if attempt < retries - 1:
                delay = exponential_backoff_delay(attempt, base_delay=1.0, max_delay=5.0)
                logger.debug(f"Host check failed for {host.name}, retrying in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
            continue

    return False


async def filter_available_hosts(hosts: list[ClusterHost]) -> list[ClusterHost]:
    """Filter to only available hosts."""
    logger.info(f"Checking {len(hosts)} hosts for availability...")

    tasks = [check_host_available(h) for h in hosts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    available = []
    for host, result in zip(hosts, results, strict=False):
        if result is True:
            available.append(host)
            logger.info(f"  {host.name}: available")
        else:
            logger.warning(f"  {host.name}: unavailable")

    return available


# ---------------------------------------------------------------------------
# Tournament Execution
# ---------------------------------------------------------------------------

def get_output_path(base_dir: str, board_type: str, num_players: int) -> str:
    """Generate output path for a tournament."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"diverse_{board_type}_{num_players}p_{timestamp}.jsonl"
    return os.path.join(base_dir, filename)


def run_tournament_local(config: TournamentConfig) -> TournamentResult:
    """Run a tournament locally."""
    start_time = time.time()

    logger.info(f"[local] Starting: {config.board_type} {config.num_players}p, {config.num_games} games")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_distributed_selfplay.py"),
        "--board-type", config.board_type,
        "--num-players", str(config.num_players),
        "--num-games", str(config.num_games),
        "--engine-mode", "diverse",
        "--output", f"file://{config.output_path}",
    ]

    if config.seed is not None:
        cmd.extend(["--seed", str(config.seed)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max
        )

        duration = time.time() - start_time
        games_completed, samples_generated = parse_selfplay_output(result.stdout)

        return TournamentResult(
            config=config,
            host="local",
            success=result.returncode == 0,
            games_completed=games_completed,
            samples_generated=samples_generated,
            duration_sec=duration,
            error=result.stderr[:500] if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return TournamentResult(
            config=config,
            host="local",
            success=False,
            games_completed=0,
            samples_generated=0,
            duration_sec=time.time() - start_time,
            error="Timeout",
        )
    except Exception as e:
        return TournamentResult(
            config=config,
            host="local",
            success=False,
            games_completed=0,
            samples_generated=0,
            duration_sec=time.time() - start_time,
            error=str(e),
        )


async def run_tournament_remote(
    host: ClusterHost,
    config: TournamentConfig,
    health_tracker: HostHealthTracker | None = None,
    ssh_retries: int = 3,
) -> TournamentResult:
    """Run a tournament on a remote host via SSH with retry logic.

    Args:
        host: Remote host to run on
        config: Tournament configuration
        health_tracker: Optional tracker for host health
        ssh_retries: Number of retry attempts for transient SSH failures

    Returns:
        TournamentResult with success/failure status
    """
    start_time = time.time()

    # Check if host is degraded
    if health_tracker and health_tracker.is_degraded(host.name):
        logger.warning(f"[{host.name}] Skipping degraded host")
        return TournamentResult(
            config=config,
            host=host.name,
            success=False,
            games_completed=0,
            samples_generated=0,
            duration_sec=0,
            error="Host is degraded",
        )

    logger.info(f"[{host.name}] Starting: {config.board_type} {config.num_players}p, {config.num_games} games")

    # Build remote command
    # Note: ringrift_path in config points to ai-service directory
    # Expand ~ to $HOME for proper path expansion in file URIs
    ai_service_path = host.ringrift_path
    if ai_service_path.startswith("~"):
        expanded_path = f"$HOME{ai_service_path[1:]}"
    else:
        expanded_path = ai_service_path
    remote_output = f"{expanded_path}/data/tournaments/{os.path.basename(config.output_path)}"

    # Build venv activation if configured
    venv_cmd = ""
    if host.venv_activate:
        venv_cmd = f"{host.venv_activate} && "

    # Build the selfplay command as a single line
    seed_arg = f" --seed {config.seed}" if config.seed is not None else ""
    selfplay_cmd = (
        f"PYTHONPATH=. RINGRIFT_SKIP_SHADOW_CONTRACTS=true "
        f"python3 scripts/run_distributed_selfplay.py "
        f"--board-type {config.board_type} "
        f"--num-players {config.num_players} "
        f"--num-games {config.num_games} "
        f"--engine-mode diverse "
        f"--output 'file://{remote_output}'"
        f"{seed_arg}"
    )

    remote_cmd = f"cd {ai_service_path} && {venv_cmd}mkdir -p data/tournaments && {selfplay_cmd}"
    ssh_cmd = [*host.ssh_cmd_prefix(), remote_cmd]

    # Retry loop for transient SSH failures
    last_error: str | None = None
    for attempt in range(ssh_retries):
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=7200)
            output = stdout.decode("utf-8", errors="replace")

            duration = time.time() - start_time
            games_completed, samples_generated = parse_selfplay_output(output)

            success = proc.returncode == 0 and games_completed > 0

            if success:
                if health_tracker:
                    health_tracker.record_success(host.name)
                logger.info(
                    f"[{host.name}] Complete: {config.board_type} {config.num_players}p, "
                    f"{games_completed} games, {duration:.0f}s"
                )
            else:
                last_error = output[-500:]
                if health_tracker and is_network_error(output):
                    became_degraded = health_tracker.record_failure(host.name, last_error)
                    if became_degraded:
                        logger.error(f"[{host.name}] marked as degraded")

            return TournamentResult(
                config=config,
                host=host.name,
                success=success,
                games_completed=games_completed,
                samples_generated=samples_generated,
                duration_sec=duration,
                error=last_error if not success else None,
            )

        except asyncio.TimeoutError:
            last_error = "SSH timeout after 2 hours"
            logger.error(f"[{host.name}] Timeout: {config.board_type} {config.num_players}p")
            if health_tracker:
                health_tracker.record_failure(host.name, last_error)
            # Don't retry timeouts - they indicate the command ran but took too long
            break

        except Exception as e:
            last_error = str(e)
            if is_network_error(e):
                if health_tracker:
                    became_degraded = health_tracker.record_failure(host.name, last_error)
                    if became_degraded:
                        logger.error(f"[{host.name}] marked as degraded after network error")
                        break

                if attempt < ssh_retries - 1:
                    delay = exponential_backoff_delay(attempt, base_delay=2.0, max_delay=30.0)
                    logger.warning(
                        f"[{host.name}] SSH error (attempt {attempt + 1}/{ssh_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
            else:
                # Non-network error, don't retry
                logger.error(f"[{host.name}] Error: {config.board_type} {config.num_players}p: {e}")
                break

    # All retries failed
    return TournamentResult(
        config=config,
        host=host.name,
        success=False,
        games_completed=0,
        samples_generated=0,
        duration_sec=time.time() - start_time,
        error=last_error,
    )


def parse_selfplay_output(output: str) -> tuple[int, int]:
    """Parse games_completed and samples_generated from selfplay output."""
    games_completed = 0
    samples_generated = 0

    for line in output.split("\n"):
        if "Games completed:" in line:
            with contextlib.suppress(ValueError):
                games_completed = int(line.split(":")[-1].strip())
        elif "Samples generated:" in line:
            with contextlib.suppress(ValueError):
                samples_generated = int(line.split(":")[-1].strip())

    return games_completed, samples_generated


# ---------------------------------------------------------------------------
# Distributed Orchestration
# ---------------------------------------------------------------------------

async def run_tournament_round_distributed(
    configs: list[TournamentConfig],
    hosts: list[ClusterHost],
    host_failure_threshold: int = 3,
    ssh_retries: int = 3,
    reassign_failed_jobs: bool = True,
) -> list[TournamentResult]:
    """Run tournaments distributed across hosts in parallel with resilience.

    Args:
        configs: Tournament configurations to run
        hosts: Available cluster hosts
        host_failure_threshold: Failures before marking host as degraded
        ssh_retries: Number of SSH retry attempts per tournament
        reassign_failed_jobs: Whether to reassign jobs from degraded hosts

    Returns:
        List of tournament results
    """
    if not hosts:
        logger.error("No available hosts for distributed execution")
        return []

    # Check host availability
    available_hosts = await filter_available_hosts(hosts)
    if not available_hosts:
        logger.error("No hosts available")
        return []

    logger.info(f"Running {len(configs)} tournaments across {len(available_hosts)} hosts")

    # Create host health tracker
    health_tracker = HostHealthTracker(failure_threshold=host_failure_threshold)

    # Create task queue
    results: list[TournamentResult] = []
    pending_configs = list(configs)
    failed_configs: list[TournamentConfig] = []  # Configs to reassign
    running_tasks: dict[asyncio.Task, tuple[ClusterHost, TournamentConfig]] = {}
    host_busy: dict[str, bool] = {h.name: False for h in available_hosts}

    while pending_configs or running_tasks or failed_configs:
        # Get healthy hosts (filter out degraded ones)
        healthy_host_names = health_tracker.get_healthy_hosts([h.name for h in available_hosts])

        # Reassign failed configs to healthy hosts
        if reassign_failed_jobs and failed_configs and healthy_host_names:
            pending_configs.extend(failed_configs)
            logger.info(f"Reassigning {len(failed_configs)} failed jobs to healthy hosts")
            failed_configs = []

        # Start new tasks on idle healthy hosts
        for host in available_hosts:
            if host.name not in healthy_host_names:
                continue  # Skip degraded hosts
            if not host_busy[host.name] and pending_configs:
                config = pending_configs.pop(0)
                task = asyncio.create_task(
                    run_tournament_remote(
                        host,
                        config,
                        health_tracker=health_tracker,
                        ssh_retries=ssh_retries,
                    )
                )
                running_tasks[task] = (host, config)
                host_busy[host.name] = True

        if not running_tasks:
            # No running tasks - check if we have pending work but no healthy hosts
            if pending_configs or failed_configs:
                if not healthy_host_names:
                    logger.error("All hosts are degraded, cannot complete remaining tasks")
                    # Record remaining as failed
                    for cfg in pending_configs + failed_configs:
                        results.append(TournamentResult(
                            config=cfg,
                            host="none",
                            success=False,
                            games_completed=0,
                            samples_generated=0,
                            duration_sec=0,
                            error="All hosts degraded",
                        ))
                    pending_configs = []
                    failed_configs = []
            break

        # Wait for at least one task to complete
        done, _ = await asyncio.wait(
            running_tasks.keys(),
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Process completed tasks
        for task in done:
            host, config = running_tasks.pop(task)
            host_busy[host.name] = False

            try:
                result = task.result()
                results.append(result)

                # Check if job should be reassigned
                if not result.success and reassign_failed_jobs:
                    if health_tracker.is_degraded(host.name) and is_network_error(result.error or ""):
                        # Network failure on degraded host - reassign
                        failed_configs.append(config)
                        logger.info(
                            f"Job {config.board_type} {config.num_players}p will be reassigned "
                            f"(host {host.name} is degraded)"
                        )

            except Exception as e:
                logger.error(f"Task failed for {config.board_type} {config.num_players}p: {e}")
                results.append(TournamentResult(
                    config=config,
                    host=host.name,
                    success=False,
                    games_completed=0,
                    samples_generated=0,
                    duration_sec=0,
                    error=str(e),
                ))

    # Log health summary
    summary = health_tracker.get_failure_summary()
    if any(info["degraded"] for info in summary.values()):
        logger.warning(f"Host health summary: {summary}")

    return results


def run_tournament_round_local(configs: list[TournamentConfig]) -> list[TournamentResult]:
    """Run tournaments locally (sequential)."""
    results = []
    for config in configs:
        result = run_tournament_local(config)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_tournament_configs(
    board_types: list[str],
    player_counts: list[int],
    games_per_config: int | None,
    output_base: str,
    seed: int | None = None,
) -> list[TournamentConfig]:
    """Build list of tournament configurations."""
    configs = []

    os.makedirs(output_base, exist_ok=True)

    for board_type in board_types:
        for num_players in player_counts:
            if games_per_config:
                num_games = games_per_config
            else:
                num_games = DEFAULT_GAMES_PER_CONFIG.get((board_type, num_players), 50)

            config = TournamentConfig(
                board_type=board_type,
                num_players=num_players,
                num_games=num_games,
                output_path=get_output_path(output_base, board_type, num_players),
                seed=seed,
            )
            configs.append(config)

    return configs


def print_summary(results: list[TournamentResult]) -> None:
    """Print summary of tournament results."""
    print("\n" + "=" * 80)
    print("TOURNAMENT ROUND SUMMARY")
    print("=" * 80)

    total_games = sum(r.games_completed for r in results)
    total_samples = sum(r.samples_generated for r in results)
    total_duration = max((r.duration_sec for r in results), default=0)  # Wall time = max
    successful = sum(1 for r in results if r.success)

    print(f"\nConfigurations: {len(results)} ({successful} successful)")
    print(f"Total games:    {total_games}")
    print(f"Total samples:  {total_samples}")
    print(f"Wall time:      {total_duration:.1f}s ({total_duration/60:.1f} min)")

    # Group by host
    hosts_used = {r.host for r in results}
    print(f"Hosts used:     {len(hosts_used)}")

    print("\nPer-configuration breakdown:")
    print("-" * 80)
    print(f"{'Config':<20} {'Host':<20} {'Games':>8} {'Samples':>10} {'Time':>8} {'Status':>8}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: (x.config.board_type, x.config.num_players)):
        config_name = f"{r.config.board_type} {r.config.num_players}p"
        status = "OK" if r.success else "FAIL"
        print(
            f"{config_name:<20} {r.host:<20} {r.games_completed:>8} "
            f"{r.samples_generated:>10} {r.duration_sec:>7.0f}s {status:>8}"
        )

    print("-" * 80)

    # Show failures
    failures = [r for r in results if not r.success]
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  {r.config.board_type} {r.config.num_players}p on {r.host}: {r.error[:100] if r.error else 'Unknown'}")

    print()


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""

    # Build configurations
    configs = build_tournament_configs(
        board_types=args.board_types,
        player_counts=args.player_counts,
        games_per_config=args.games_per_config,
        output_base=args.output_dir,
        seed=args.seed,
    )

    logger.info(f"Tournament configurations: {len(configs)}")
    for cfg in configs:
        logger.info(f"  {cfg.board_type} {cfg.num_players}p: {cfg.num_games} games")

    if args.dry_run:
        if args.distributed:
            hosts = load_cluster_hosts()
            available = await filter_available_hosts(hosts)
            logger.info(f"Would distribute across {len(available)} hosts")
        logger.info("Dry run - not executing")
        return 0

    # Load hosts for distributed mode
    hosts = []
    if args.distributed:
        hosts = load_cluster_hosts()
        if not hosts:
            logger.error("No cluster hosts configured. Add hosts to config/distributed_hosts.yaml")
            return 1

    # Run tournament rounds
    round_num = 0
    while True:
        round_num += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"TOURNAMENT ROUND {round_num}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()

        if args.distributed:
            results = await run_tournament_round_distributed(configs, hosts)
        else:
            results = run_tournament_round_local(configs)

        print_summary(results)

        # Save results
        results_path = os.path.join(args.output_dir, f"round_{round_num}_results.json")
        with open(results_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        logger.info(f"Results saved to: {results_path}")

        if args.interval_hours is None:
            break

        # Wait for next round
        elapsed = time.time() - start_time
        wait_sec = max(0, args.interval_hours * 3600 - elapsed)
        if wait_sec > 0:
            logger.info(f"Waiting {wait_sec/3600:.1f} hours until next round...")
            await asyncio.sleep(wait_sec)

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Orchestrate diverse tournaments across board/player combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--board-types",
        nargs="+",
        choices=ALL_BOARD_TYPES,
        default=ALL_BOARD_TYPES,
        help="Board types to include (default: all)",
    )
    parser.add_argument(
        "--player-counts",
        nargs="+",
        type=int,
        choices=ALL_PLAYER_COUNTS,
        default=ALL_PLAYER_COUNTS,
        help="Player counts to include (default: all)",
    )
    parser.add_argument(
        "--games-per-config",
        type=int,
        default=None,
        help="Games per configuration (default: varies by config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "data" / "tournaments"),
        help="Output directory for tournament results",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=None,
        help="Run continuously with this interval (hours). Omit for single run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run distributed across cluster hosts (parallel)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
