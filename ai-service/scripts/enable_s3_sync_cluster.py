#!/usr/bin/env python3
"""Enable S3 sync on all cluster nodes.

This script deploys the S3NodeSyncDaemon across the cluster and ensures
all nodes are pushing their data to S3.

Usage:
    # Enable S3 sync on all nodes
    python scripts/enable_s3_sync_cluster.py

    # Enable on specific nodes
    python scripts/enable_s3_sync_cluster.py --nodes nebius-h100-3,runpod-a100-1

    # Dry run (show what would be done)
    python scripts/enable_s3_sync_cluster.py --dry-run

    # Force initial sync on all nodes
    python scripts/enable_s3_sync_cluster.py --force-sync

December 2025: Created for cluster-wide S3 backup infrastructure.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class NodeResult:
    """Result of enabling S3 sync on a node."""
    node_id: str
    success: bool
    message: str
    files_uploaded: int = 0


def get_cluster_nodes() -> list[dict[str, Any]]:
    """Get list of cluster nodes from config."""
    try:
        from app.config.cluster_config import get_cluster_nodes
        nodes = get_cluster_nodes()
        return [
            {
                "name": name,
                "ssh_host": node.ssh_host,
                "ssh_port": node.ssh_port or 22,
                "ssh_user": node.ssh_user or "root",
                "ssh_key": node.ssh_key,
                "status": node.status,
                "provider": node.provider,
            }
            for name, node in nodes.items()
            if node.status in ("ready", "active")
        ]
    except Exception as e:
        logger.warning(f"Failed to load cluster config: {e}")
        return []


def run_ssh_command(
    host: str,
    port: int,
    user: str,
    key: str | None,
    command: str,
    timeout: int = 120,
) -> tuple[int, str, str]:
    """Run SSH command on remote host."""
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30"]

    if key and os.path.exists(os.path.expanduser(key)):
        ssh_cmd.extend(["-i", os.path.expanduser(key)])

    ssh_cmd.extend(["-p", str(port), f"{user}@{host}", command])

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def enable_s3_sync_on_node(
    node: dict[str, Any],
    force_sync: bool = False,
    dry_run: bool = False,
) -> NodeResult:
    """Enable S3 sync on a single node."""
    node_id = node["name"]
    host = node["ssh_host"]
    port = node.get("ssh_port", 22)
    user = node.get("ssh_user", "root")
    key = node.get("ssh_key")

    logger.info(f"Enabling S3 sync on {node_id}...")

    if dry_run:
        return NodeResult(
            node_id=node_id,
            success=True,
            message="Would enable S3 sync (dry run)",
        )

    # Check AWS credentials
    check_aws = "aws sts get-caller-identity --query Account --output text 2>/dev/null"
    rc, stdout, stderr = run_ssh_command(host, port, user, key, check_aws)

    if rc != 0:
        return NodeResult(
            node_id=node_id,
            success=False,
            message="AWS credentials not configured",
        )

    # Find the ringrift directory
    find_dir = """
    for dir in ~/ringrift/ai-service /workspace/ringrift/ai-service /root/ringrift/ai-service; do
        if [ -d "$dir" ]; then
            echo "$dir"
            exit 0
        fi
    done
    echo "NOT_FOUND"
    """
    rc, stdout, stderr = run_ssh_command(host, port, user, key, find_dir)

    if rc != 0 or "NOT_FOUND" in stdout:
        return NodeResult(
            node_id=node_id,
            success=False,
            message="RingRift directory not found",
        )

    ringrift_dir = stdout.strip()

    # Set environment variables and run sync daemon once
    sync_cmd = f"""
    cd {ringrift_dir}
    export PYTHONPATH=.
    export RINGRIFT_S3_BUCKET=ringrift-models-20251214
    export RINGRIFT_NODE_ID={node_id}

    # Run sync once
    python -m app.coordination.s3_node_sync_daemon --once 2>&1
    """

    if force_sync:
        rc, stdout, stderr = run_ssh_command(host, port, user, key, sync_cmd, timeout=600)

        if rc != 0:
            return NodeResult(
                node_id=node_id,
                success=False,
                message=f"Sync failed: {stderr[:200]}",
            )

        # Parse output for uploaded files count
        uploaded = stdout.count("Uploaded") + stdout.count("uploaded")

        return NodeResult(
            node_id=node_id,
            success=True,
            message=f"Initial sync complete",
            files_uploaded=uploaded,
        )

    # Add to systemd/cron for persistent sync
    enable_persistent = f"""
    cd {ringrift_dir}

    # Create sync script
    cat > /tmp/s3_sync.sh << 'EOF'
#!/bin/bash
cd {ringrift_dir}
export PYTHONPATH=.
export RINGRIFT_S3_BUCKET=ringrift-models-20251214
export RINGRIFT_NODE_ID={node_id}
python -m app.coordination.s3_node_sync_daemon --once >> /tmp/s3_sync.log 2>&1
EOF
    chmod +x /tmp/s3_sync.sh

    # Add cron job (every hour)
    (crontab -l 2>/dev/null | grep -v s3_sync; echo "0 * * * * /tmp/s3_sync.sh") | crontab -

    # Run initial sync
    /tmp/s3_sync.sh &
    echo "S3 sync enabled via cron"
    """

    rc, stdout, stderr = run_ssh_command(host, port, user, key, enable_persistent, timeout=120)

    if rc != 0:
        return NodeResult(
            node_id=node_id,
            success=False,
            message=f"Failed to enable: {stderr[:200]}",
        )

    return NodeResult(
        node_id=node_id,
        success=True,
        message="S3 sync enabled via cron",
    )


def enable_s3_sync_cluster(
    nodes: list[dict[str, Any]],
    force_sync: bool = False,
    dry_run: bool = False,
    max_parallel: int = 10,
) -> list[NodeResult]:
    """Enable S3 sync on all cluster nodes."""
    results = []

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(enable_s3_sync_on_node, node, force_sync, dry_run): node
            for node in nodes
        }

        for future in as_completed(futures):
            node = futures[future]
            try:
                result = future.result()
                results.append(result)

                status = "✓" if result.success else "✗"
                logger.info(f"  {status} {result.node_id}: {result.message}")

            except Exception as e:
                results.append(NodeResult(
                    node_id=node["name"],
                    success=False,
                    message=str(e),
                ))
                logger.error(f"  ✗ {node['name']}: {e}")

    return results


async def run_consolidation() -> None:
    """Run S3 consolidation on coordinator."""
    logger.info("Running S3 consolidation...")

    from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

    daemon = S3ConsolidationDaemon()
    await daemon._run_consolidation()

    logger.info("Consolidation complete")


def main():
    parser = argparse.ArgumentParser(description="Enable S3 sync on cluster nodes")
    parser.add_argument(
        "--nodes",
        type=str,
        help="Comma-separated list of node names (default: all active nodes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--force-sync",
        action="store_true",
        help="Force immediate sync on all nodes",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Run consolidation after enabling sync",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum parallel SSH connections",
    )

    args = parser.parse_args()

    # Get nodes
    all_nodes = get_cluster_nodes()

    if args.nodes:
        node_names = [n.strip() for n in args.nodes.split(",")]
        nodes = [n for n in all_nodes if n["name"] in node_names]

        if not nodes:
            logger.error(f"No matching nodes found. Available: {[n['name'] for n in all_nodes]}")
            sys.exit(1)
    else:
        nodes = all_nodes

    if not nodes:
        logger.error("No cluster nodes configured")
        sys.exit(1)

    logger.info(f"Enabling S3 sync on {len(nodes)} nodes...")
    if args.dry_run:
        logger.info("(dry run mode)")

    # Enable on all nodes
    results = enable_s3_sync_cluster(
        nodes,
        force_sync=args.force_sync,
        dry_run=args.dry_run,
        max_parallel=args.max_parallel,
    )

    # Summary
    success = sum(1 for r in results if r.success)
    failed = len(results) - success
    total_files = sum(r.files_uploaded for r in results)

    print("\n" + "=" * 50)
    print(f"S3 Sync Enablement Summary")
    print("=" * 50)
    print(f"Total nodes:      {len(results)}")
    print(f"Successful:       {success}")
    print(f"Failed:           {failed}")
    if args.force_sync:
        print(f"Files uploaded:   {total_files}")

    if failed > 0:
        print("\nFailed nodes:")
        for r in results:
            if not r.success:
                print(f"  - {r.node_id}: {r.message}")

    # Run consolidation if requested
    if args.consolidate and not args.dry_run:
        asyncio.run(run_consolidation())

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
