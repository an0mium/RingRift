#!/usr/bin/env python3
"""Batch recover Tailscale on offline cluster nodes via SSH gateway.

This script SSHes into offline nodes (via their SSH gateway if applicable) and
restarts Tailscale with proper authentication. Useful for recovering Vast.ai
and RunPod nodes whose Tailscale has died but SSH gateway still works.

Usage:
    # Set your Tailscale auth key
    export TAILSCALE_AUTH_KEY=tskey-auth-xxxxx

    # Recover all offline nodes
    PYTHONPATH=. python scripts/recover_tailscale_nodes.py

    # Dry run (show what would be done)
    PYTHONPATH=. python scripts/recover_tailscale_nodes.py --dry-run

    # Recover specific nodes
    PYTHONPATH=. python scripts/recover_tailscale_nodes.py --nodes vast-29129529,vast-29118471

    # Only target Vast.ai nodes
    PYTHONPATH=. python scripts/recover_tailscale_nodes.py --provider vast

December 2025
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    node_name: str
    success: bool
    tailscale_ip: Optional[str] = None
    error: Optional[str] = None


# Recovery command for container nodes (Vast.ai, RunPod)
CONTAINER_RECOVERY_CMD = """
pkill -9 tailscaled 2>/dev/null || true
sleep 2
mkdir -p /var/lib/tailscale /var/run/tailscale
nohup tailscaled --tun=userspace-networking --statedir=/var/lib/tailscale > /tmp/tailscaled.log 2>&1 &
sleep 5
tailscale up {authkey_arg} --accept-routes --hostname='{hostname}'
tailscale ip -4
"""

# Recovery command for regular hosts (Lambda, Nebius)
HOST_RECOVERY_CMD = """
systemctl restart tailscaled 2>/dev/null || {{
    pkill -9 tailscaled
    sleep 2
    tailscaled --state=/var/lib/tailscale/tailscaled.state &
    sleep 5
}}
tailscale up {authkey_arg} --accept-routes --hostname='{hostname}'
tailscale ip -4
"""


async def ssh_recover_node(
    node_name: str,
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_key: Optional[str],
    authkey: str,
    is_container: bool = True,
    timeout: float = 60.0,
) -> RecoveryResult:
    """Attempt to recover Tailscale on a node via SSH.

    Args:
        node_name: Name of the node
        ssh_host: SSH hostname or IP
        ssh_port: SSH port
        ssh_user: SSH username
        ssh_key: Path to SSH key file (optional)
        authkey: Tailscale auth key
        is_container: Whether the node is a container (uses userspace networking)
        timeout: SSH timeout in seconds

    Returns:
        RecoveryResult with success status and tailscale IP if successful.
    """
    # Build auth key argument
    authkey_arg = f"--authkey={authkey}" if authkey else ""

    # Select recovery command template
    if is_container:
        cmd_template = CONTAINER_RECOVERY_CMD
    else:
        cmd_template = HOST_RECOVERY_CMD

    # Format recovery command
    recovery_cmd = cmd_template.format(
        authkey_arg=authkey_arg,
        hostname=node_name,
    )

    # Build SSH command
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
    ]
    if ssh_port != 22:
        ssh_cmd.extend(["-p", str(ssh_port)])
    if ssh_key:
        ssh_cmd.extend(["-i", os.path.expanduser(ssh_key)])
    ssh_cmd.extend([f"{ssh_user}@{ssh_host}", recovery_cmd])

    logger.info(f"[{node_name}] Connecting to {ssh_host}:{ssh_port}...")

    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        stdout_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()

        if proc.returncode == 0:
            # Look for Tailscale IP (100.x.x.x) in output
            tailscale_ip = None
            for line in stdout_str.split("\n"):
                line = line.strip()
                if line.startswith("100."):
                    tailscale_ip = line
                    break

            if tailscale_ip:
                logger.info(f"[{node_name}] ✓ Recovery succeeded, IP: {tailscale_ip}")
                return RecoveryResult(
                    node_name=node_name,
                    success=True,
                    tailscale_ip=tailscale_ip,
                )
            else:
                logger.warning(f"[{node_name}] SSH succeeded but no Tailscale IP found")
                return RecoveryResult(
                    node_name=node_name,
                    success=False,
                    error="No Tailscale IP in output",
                )
        else:
            error_msg = stderr_str or f"SSH exit code {proc.returncode}"
            logger.error(f"[{node_name}] ✗ Recovery failed: {error_msg}")
            return RecoveryResult(
                node_name=node_name,
                success=False,
                error=error_msg,
            )

    except asyncio.TimeoutError:
        logger.error(f"[{node_name}] ✗ SSH timed out after {timeout}s")
        return RecoveryResult(
            node_name=node_name,
            success=False,
            error=f"Timeout after {timeout}s",
        )
    except Exception as e:
        logger.error(f"[{node_name}] ✗ SSH error: {e}")
        return RecoveryResult(
            node_name=node_name,
            success=False,
            error=str(e),
        )


def is_container_node(node_name: str) -> bool:
    """Determine if a node is a container (Vast.ai, RunPod) based on name."""
    container_prefixes = ("vast-", "runpod-")
    return any(node_name.startswith(prefix) for prefix in container_prefixes)


async def recover_nodes(
    node_filter: Optional[list[str]] = None,
    provider_filter: Optional[str] = None,
    dry_run: bool = False,
    max_parallel: int = 5,
) -> list[RecoveryResult]:
    """Recover Tailscale on multiple nodes.

    Args:
        node_filter: List of specific node names to recover (None = all offline)
        provider_filter: Only recover nodes from this provider (vast, runpod, etc.)
        dry_run: If True, just show what would be done
        max_parallel: Maximum concurrent SSH connections

    Returns:
        List of RecoveryResult objects.
    """
    # Get Tailscale auth key
    authkey = os.environ.get("TAILSCALE_AUTH_KEY", "")
    if not authkey and not dry_run:
        logger.error("TAILSCALE_AUTH_KEY environment variable not set")
        logger.error("Get a key from https://login.tailscale.com/admin/settings/keys")
        return []

    # Load cluster config
    try:
        from app.config.cluster_config import get_cluster_nodes
        nodes = get_cluster_nodes()
    except ImportError as e:
        logger.error(f"Failed to import cluster config: {e}")
        logger.error("Run with PYTHONPATH=. from ai-service directory")
        return []

    # Filter nodes
    target_nodes = []
    for name, node in nodes.items():
        # Skip if specific nodes requested and not in list
        if node_filter and name not in node_filter:
            continue

        # Skip if provider filter doesn't match
        if provider_filter:
            if not name.startswith(f"{provider_filter}-"):
                continue

        # Skip if node doesn't have SSH access
        if not node.ssh_host:
            continue

        # Skip coordinator and setup nodes
        if node.status in ("coordinator", "setup", "proxy_only"):
            continue

        # For non-specific requests, only target offline-looking nodes
        if not node_filter and node.status not in ("offline", "unknown", "needs_tailscale_auth"):
            continue

        target_nodes.append(node)

    if not target_nodes:
        logger.info("No nodes to recover")
        return []

    logger.info(f"Found {len(target_nodes)} nodes to recover:")
    for node in target_nodes:
        logger.info(f"  - {node.name} ({node.ssh_host}:{node.ssh_port})")

    if dry_run:
        logger.info("Dry run mode - no changes will be made")
        return [
            RecoveryResult(node_name=n.name, success=False, error="dry_run")
            for n in target_nodes
        ]

    # Run recovery in parallel with semaphore
    semaphore = asyncio.Semaphore(max_parallel)

    async def recover_with_limit(node):
        async with semaphore:
            return await ssh_recover_node(
                node_name=node.name,
                ssh_host=node.ssh_host,
                ssh_port=node.ssh_port,
                ssh_user=node.ssh_user or "root",
                ssh_key=node.ssh_key,
                authkey=authkey,
                is_container=is_container_node(node.name),
            )

    tasks = [recover_with_limit(node) for node in target_nodes]
    results = await asyncio.gather(*tasks)

    # Summary
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded

    logger.info("")
    logger.info("=" * 60)
    logger.info("RECOVERY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✓ Succeeded: {succeeded}")
    logger.info(f"✗ Failed: {failed}")

    if succeeded > 0:
        logger.info("")
        logger.info("Recovered nodes:")
        for r in results:
            if r.success:
                logger.info(f"  - {r.node_name}: {r.tailscale_ip}")

    if failed > 0:
        logger.info("")
        logger.info("Failed nodes:")
        for r in results:
            if not r.success:
                logger.info(f"  - {r.node_name}: {r.error}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Recover Tailscale on offline cluster nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Recover all offline nodes
    TAILSCALE_AUTH_KEY=tskey-xxx python scripts/recover_tailscale_nodes.py

    # Dry run
    python scripts/recover_tailscale_nodes.py --dry-run

    # Specific nodes
    python scripts/recover_tailscale_nodes.py --nodes vast-29129529,vast-29118471

    # Only Vast.ai nodes
    python scripts/recover_tailscale_nodes.py --provider vast
""",
    )
    parser.add_argument(
        "--nodes",
        help="Comma-separated list of specific node names to recover",
    )
    parser.add_argument(
        "--provider",
        choices=["vast", "runpod", "lambda", "nebius", "vultr", "hetzner"],
        help="Only recover nodes from this provider",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Maximum parallel SSH connections (default: 5)",
    )

    args = parser.parse_args()

    node_filter = None
    if args.nodes:
        node_filter = [n.strip() for n in args.nodes.split(",")]

    results = asyncio.run(recover_nodes(
        node_filter=node_filter,
        provider_filter=args.provider,
        dry_run=args.dry_run,
        max_parallel=args.parallel,
    ))

    # Exit with error if any recovery failed
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
