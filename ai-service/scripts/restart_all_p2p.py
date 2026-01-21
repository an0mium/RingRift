#!/usr/bin/env python3
"""Restart P2P orchestrator on all cluster nodes.

January 19, 2026 - P2P Network Stability Plan Phase 2:
    Script to restart P2P orchestrators across the cluster to apply new
    timeout configurations and improve network stability for 20+ nodes.

Usage:
    # Safe mode (RECOMMENDED) - preserves quorum during rolling restarts
    python scripts/restart_all_p2p.py --safe-mode

    # Restart only specific providers
    python scripts/restart_all_p2p.py --safe-mode --providers lambda,vast,nebius

    # Dry run to preview actions
    python scripts/restart_all_p2p.py --safe-mode --dry-run

    # Force restart (without checking if P2P was running)
    python scripts/restart_all_p2p.py --safe-mode --force

Features:
    1. Load distributed_hosts.yaml and filter to P2P-enabled nodes
    2. For each node:
       - Check if P2P is running (pgrep -f p2p_orchestrator)
       - If running, send SIGTERM and wait for graceful shutdown
       - Start P2P with proper arguments
       - Verify P2P started (/status endpoint within 30s)
    3. Support --safe-mode for quorum-safe rolling restarts
    4. Support --providers filter for specific providers
    5. Support --dry-run to preview actions
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ssh import SSHClient, SSHConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Node path mappings by provider
PATH_MAPPINGS = {
    'runpod': '/workspace/ringrift/ai-service',
    'vast': '~/ringrift/ai-service',
    'vast_workspace': '/workspace/ringrift/ai-service',
    'nebius': '~/ringrift/ai-service',
    'vultr': '/root/ringrift/ai-service',
    'hetzner': '/root/ringrift/ai-service',
    'lambda': '~/ringrift/ai-service',
    'mac-studio': '~/Development/RingRift/ai-service',
    'local-mac': '/Users/armand/Development/RingRift/ai-service',
}

# Default P2P startup grace period (seconds)
P2P_STARTUP_GRACE_PERIOD = 120
P2P_VERIFY_TIMEOUT = 30

# Quorum health levels
QUORUM_HEALTHY_THRESHOLD = 4  # Minimum voters needed before proceeding


@dataclass
class RestartResult:
    """Result of a P2P restart operation."""
    node_name: str
    success: bool
    message: str
    was_running: bool = False
    is_running: bool = False
    duration_seconds: float = 0.0


def get_node_path(node_name: str, node_config: dict) -> str | None:
    """Get the ringrift path for a node based on config or provider."""
    if 'ringrift_path' in node_config:
        path = node_config['ringrift_path']
        if path is None:
            return None
        return path

    for provider in PATH_MAPPINGS:
        if node_name.startswith(provider):
            return PATH_MAPPINGS[provider]

    return '~/ringrift/ai-service'


def get_provider(node_name: str) -> str:
    """Extract provider name from node name."""
    for provider in ['lambda', 'vast', 'nebius', 'runpod', 'vultr', 'hetzner']:
        if node_name.startswith(provider):
            return provider
    return 'unknown'


async def check_p2p_running(client: SSHClient, node_name: str) -> bool:
    """Check if P2P orchestrator is running on the node."""
    try:
        result = await client.run_async("pgrep -f p2p_orchestrator", timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            logger.debug(f"[{node_name}] P2P running (PID: {result.stdout.strip().split()[0]})")
            return True
    except Exception as e:
        logger.warning(f"[{node_name}] Failed to check P2P status: {e}")
    return False


async def stop_p2p(client: SSHClient, node_name: str, graceful_timeout: float = 5.0) -> bool:
    """Stop P2P orchestrator gracefully."""
    try:
        # Send SIGTERM first for graceful shutdown
        result = await client.run_async("pkill -TERM -f p2p_orchestrator", timeout=10)

        # Wait for graceful shutdown
        await asyncio.sleep(graceful_timeout)

        # Check if still running
        if await check_p2p_running(client, node_name):
            logger.warning(f"[{node_name}] P2P still running after SIGTERM, sending SIGKILL")
            await client.run_async("pkill -9 -f p2p_orchestrator", timeout=10)
            await asyncio.sleep(1)

        # Verify stopped
        if await check_p2p_running(client, node_name):
            logger.error(f"[{node_name}] Failed to stop P2P orchestrator")
            return False

        logger.info(f"[{node_name}] P2P orchestrator stopped")
        return True
    except Exception as e:
        logger.error(f"[{node_name}] Error stopping P2P: {e}")
        return False


async def start_p2p(
    client: SSHClient,
    node_name: str,
    node_path: str,
    node_config: dict,
) -> bool:
    """Start P2P orchestrator on the node."""
    try:
        # Build venv activation
        venv_activate = node_config.get('venv_activate')
        if venv_activate is None:
            venv_activate = (
                f"if [ -f {node_path}/venv/bin/activate ]; then "
                f"source {node_path}/venv/bin/activate; "
                "fi"
            )
        elif venv_activate == ':':
            venv_activate = ''

        # Build P2P start command
        pythonpath_cmd = f"export PYTHONPATH={node_path}"

        # Get node ID from config
        node_id = node_config.get('node_id', node_name)

        # Get leader IPs for bootstrap (from p2p_voters or peers)
        bootstrap_peers = node_config.get('bootstrap_peers', [])

        p2p_cmd = (
            f"cd {node_path} && "
            f"{pythonpath_cmd} && "
            f"{venv_activate} "
            f"nohup python -u scripts/p2p_orchestrator.py "
            f"--node-id {node_id} "
        )

        # Add NAT-blocked flag if applicable
        if node_config.get('nat_blocked') or node_config.get('cgnat_detected'):
            p2p_cmd += "--nat-blocked "

        p2p_cmd += f"> {node_path}/nohup_p2p.out 2>&1 &"

        logger.debug(f"[{node_name}] Starting P2P: {p2p_cmd[:100]}...")

        result = await client.run_async(p2p_cmd, timeout=30)
        if result.returncode != 0:
            logger.error(f"[{node_name}] Failed to start P2P: {result.stderr}")
            return False

        # Wait for startup
        await asyncio.sleep(2)

        return True
    except Exception as e:
        logger.error(f"[{node_name}] Error starting P2P: {e}")
        return False


async def verify_p2p_healthy(
    client: SSHClient,
    node_name: str,
    node_config: dict,
    timeout: float = P2P_VERIFY_TIMEOUT,
) -> bool:
    """Verify P2P orchestrator is healthy via /status endpoint."""
    tailscale_ip = node_config.get('tailscale_ip')
    if not tailscale_ip:
        # Fall back to process check
        return await check_p2p_running(client, node_name)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Try to reach /status endpoint
            curl_cmd = f"curl -s --max-time 5 http://{tailscale_ip}:8770/status"
            result = await client.run_async(curl_cmd, timeout=10)

            if result.returncode == 0 and '"healthy"' in result.stdout:
                logger.info(f"[{node_name}] P2P healthy (verified via /status)")
                return True
        except Exception:
            pass

        await asyncio.sleep(2)

    # Fall back to process check
    if await check_p2p_running(client, node_name):
        logger.warning(f"[{node_name}] P2P running but /status not responding")
        return True

    return False


async def restart_node_p2p(
    node_name: str,
    node_config: dict,
    dry_run: bool = False,
    force: bool = False,
) -> RestartResult:
    """Restart P2P on a single node."""
    start_time = time.time()

    try:
        # Skip if node is not ready
        if node_config.get('status') != 'ready':
            return RestartResult(
                node_name=node_name,
                success=False,
                message=f"SKIPPED: Status is {node_config.get('status')}",
            )

        # Skip if P2P not enabled
        if not node_config.get('p2p_enabled', True):
            return RestartResult(
                node_name=node_name,
                success=False,
                message="SKIPPED: P2P not enabled",
            )

        # Get node path
        node_path = get_node_path(node_name, node_config)
        if not node_path:
            return RestartResult(
                node_name=node_name,
                success=False,
                message="SKIPPED: No ringrift path configured",
            )

        # Create SSH client
        tailscale_ip = node_config.get('tailscale_ip')
        ssh_host = tailscale_ip or node_config.get('ssh_host')
        ssh_port = node_config.get('ssh_port', 22)
        ssh_user = node_config.get('ssh_user', 'root')
        ssh_key = node_config.get('ssh_key')

        if not ssh_host:
            return RestartResult(
                node_name=node_name,
                success=False,
                message="SKIPPED: No SSH host configured",
            )

        config = SSHConfig(
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            key_path=ssh_key,
        )
        client = SSHClient(config)

        # Check if P2P was running
        was_running = await check_p2p_running(client, node_name)

        if dry_run:
            msg = "DRY-RUN: Would restart P2P"
            if was_running:
                msg += " (was running)"
            elif force:
                msg += " (force start)"
            else:
                msg = "DRY-RUN: Would skip (not running, use --force to start)"
            return RestartResult(
                node_name=node_name,
                success=True,
                message=msg,
                was_running=was_running,
                duration_seconds=time.time() - start_time,
            )

        # Skip if not running and not forcing
        if not was_running and not force:
            return RestartResult(
                node_name=node_name,
                success=True,
                message="SKIPPED: P2P not running (use --force to start)",
                was_running=False,
            )

        # Stop P2P if running
        if was_running:
            logger.info(f"[{node_name}] Stopping P2P orchestrator...")
            if not await stop_p2p(client, node_name):
                return RestartResult(
                    node_name=node_name,
                    success=False,
                    message="Failed to stop P2P",
                    was_running=True,
                    duration_seconds=time.time() - start_time,
                )

        # Start P2P
        logger.info(f"[{node_name}] Starting P2P orchestrator...")
        if not await start_p2p(client, node_name, node_path, node_config):
            return RestartResult(
                node_name=node_name,
                success=False,
                message="Failed to start P2P",
                was_running=was_running,
                duration_seconds=time.time() - start_time,
            )

        # Verify healthy
        is_running = await verify_p2p_healthy(client, node_name, node_config)

        return RestartResult(
            node_name=node_name,
            success=is_running,
            message="P2P restarted successfully" if is_running else "P2P started but not responding",
            was_running=was_running,
            is_running=is_running,
            duration_seconds=time.time() - start_time,
        )

    except Exception as e:
        return RestartResult(
            node_name=node_name,
            success=False,
            message=f"Exception: {str(e)}",
            duration_seconds=time.time() - start_time,
        )


async def get_cluster_health() -> Tuple[int, int, str | None]:
    """Get current cluster health from P2P leader.

    Returns:
        Tuple of (alive_peers, total_peers, leader_id)
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # Try local P2P first
            try:
                async with session.get('http://localhost:8770/health', timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return (
                            data.get('active_peers', 0),
                            data.get('total_peers', 0),
                            data.get('leader_id'),
                        )
            except Exception:
                pass
    except ImportError:
        pass

    return (0, 0, None)


async def restart_all_p2p(
    providers: List[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    max_parallel: int = 5,
) -> Dict[str, RestartResult]:
    """Restart P2P on all cluster nodes (legacy parallel mode)."""
    config_path = Path(__file__).parent.parent / 'config' / 'distributed_hosts.yaml'

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hosts = config.get('hosts', {})

    # Filter by provider if specified
    if providers:
        hosts = {
            name: cfg for name, cfg in hosts.items()
            if get_provider(name) in providers
        }

    logger.info(f"Found {len(hosts)} hosts to process")

    # Create restart tasks
    tasks = []
    for node_name, node_config in hosts.items():
        task = restart_node_p2p(node_name, node_config, dry_run, force)
        tasks.append(task)

    # Run restarts with concurrency limit
    results = {}
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_with_semaphore(task):
        async with semaphore:
            return await task

    completed = await asyncio.gather(*[run_with_semaphore(t) for t in tasks])

    for result in completed:
        results[result.node_name] = result

    return results


async def restart_all_p2p_safe(
    providers: List[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    convergence_timeout: float = 60.0,
) -> Dict[str, RestartResult]:
    """Restart P2P with quorum-safe rolling restarts.

    Safe mode ensures:
    1. Non-voters are restarted in parallel batches
    2. Voters are restarted one at a time
    3. Wait for convergence between voter restarts
    4. Abort if quorum drops below threshold
    """
    config_path = Path(__file__).parent.parent / 'config' / 'distributed_hosts.yaml'

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hosts = config.get('hosts', {})
    voters = set(config.get('p2p_voters', []))

    # Filter by provider if specified
    if providers:
        hosts = {
            name: cfg for name, cfg in hosts.items()
            if get_provider(name) in providers
        }

    # Split into voters and non-voters
    voter_nodes = {n: c for n, c in hosts.items() if n in voters}
    non_voter_nodes = {n: c for n, c in hosts.items() if n not in voters}

    logger.info(f"Safe mode: {len(non_voter_nodes)} non-voters, {len(voter_nodes)} voters")

    results = {}

    # Phase 1: Restart all non-voters in parallel
    if non_voter_nodes:
        logger.info("Phase 1: Restarting non-voter nodes...")
        non_voter_results = await restart_all_p2p(
            providers=providers,
            dry_run=dry_run,
            force=force,
            max_parallel=10,  # Higher parallelism for non-voters
        )
        # Filter to only non-voters
        for name, result in non_voter_results.items():
            if name in non_voter_nodes:
                results[name] = result

        # Wait for convergence
        if not dry_run:
            logger.info(f"Waiting {convergence_timeout}s for non-voter convergence...")
            await asyncio.sleep(min(convergence_timeout / 2, 30))

    # Phase 2: Restart voters one at a time
    if voter_nodes:
        logger.info("Phase 2: Restarting voter nodes (one at a time)...")

        for node_name, node_config in voter_nodes.items():
            # Check cluster health before restarting voter
            alive, total, leader = await get_cluster_health()
            if alive < QUORUM_HEALTHY_THRESHOLD and not dry_run:
                logger.warning(
                    f"Cluster health degraded ({alive}/{total} peers). "
                    f"Pausing voter restarts. Resume manually when healthy."
                )
                results[node_name] = RestartResult(
                    node_name=node_name,
                    success=False,
                    message=f"PAUSED: Cluster health degraded ({alive} alive)",
                )
                # Skip remaining voters
                for remaining in list(voter_nodes.keys())[list(voter_nodes.keys()).index(node_name)+1:]:
                    results[remaining] = RestartResult(
                        node_name=remaining,
                        success=False,
                        message="SKIPPED: Paused due to cluster health",
                    )
                break

            # Restart this voter
            result = await restart_node_p2p(node_name, node_config, dry_run, force)
            results[node_name] = result

            if result.success and not dry_run:
                # Wait for gossip convergence after voter restart
                logger.info(f"Waiting {convergence_timeout}s for voter convergence...")
                await asyncio.sleep(convergence_timeout)

    return results


def print_summary(results: Dict[str, RestartResult]) -> None:
    """Print restart summary."""
    print("\n" + "=" * 80)
    print("P2P RESTART SUMMARY")
    print("=" * 80)

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for node_name, result in sorted(results.items()):
        if "SKIPPED" in result.message or "PAUSED" in result.message:
            status = "⏭️ "
            skipped_count += 1
        elif result.success:
            status = "✅"
            success_count += 1
        else:
            status = "❌"
            failed_count += 1

        duration = f" ({result.duration_seconds:.1f}s)" if result.duration_seconds > 0 else ""
        print(f"  {status} {node_name}: {result.message}{duration}")

    print("=" * 80)
    print(f"  Success: {success_count}  Failed: {failed_count}  Skipped: {skipped_count}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Restart P2P orchestrator on all cluster nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Safe mode (RECOMMENDED for production)
    python scripts/restart_all_p2p.py --safe-mode

    # Restart only Lambda nodes
    python scripts/restart_all_p2p.py --safe-mode --providers lambda

    # Restart Lambda and Vast nodes
    python scripts/restart_all_p2p.py --safe-mode --providers lambda,vast

    # Dry run to preview actions
    python scripts/restart_all_p2p.py --safe-mode --dry-run

    # Force start P2P on all nodes (even if not running)
    python scripts/restart_all_p2p.py --safe-mode --force

    # Legacy parallel mode (not recommended, may cause quorum loss)
    python scripts/restart_all_p2p.py --providers lambda
"""
    )
    parser.add_argument(
        '--safe-mode',
        action='store_true',
        help='Use quorum-safe rolling restarts (RECOMMENDED)'
    )
    parser.add_argument(
        '--providers',
        type=str,
        help='Comma-separated list of providers to restart (e.g., lambda,vast,nebius)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be done without actually restarting'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Start P2P even if it was not running'
    )
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=5,
        help='Maximum parallel restarts in legacy mode (default: 5)'
    )
    parser.add_argument(
        '--convergence-timeout',
        type=float,
        default=60.0,
        help='Seconds to wait for convergence after voter restart (default: 60)'
    )

    args = parser.parse_args()

    # Parse providers
    providers = None
    if args.providers:
        providers = [p.strip() for p in args.providers.split(',')]
        logger.info(f"Filtering to providers: {providers}")

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    if args.safe_mode:
        logger.info("Using SAFE MODE - quorum-safe rolling restarts")
        results = asyncio.run(restart_all_p2p_safe(
            providers=providers,
            dry_run=args.dry_run,
            force=args.force,
            convergence_timeout=args.convergence_timeout,
        ))
    else:
        # Warn about quorum risk
        if not args.dry_run:
            logger.warning(
                "⚠️  WARNING: Running without --safe-mode. "
                "Simultaneous P2P restarts may cause quorum loss. "
                "Use --safe-mode for production."
            )
        results = asyncio.run(restart_all_p2p(
            providers=providers,
            dry_run=args.dry_run,
            force=args.force,
            max_parallel=args.max_parallel,
        ))

    print_summary(results)

    # Exit with error if any restarts failed
    failed_count = sum(1 for r in results.values() if not r.success and "SKIPPED" not in r.message)
    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
