#!/usr/bin/env python3
"""Sync all NN and NNUE models to all cluster hosts.

This script ensures model parity across the distributed cluster by:
1. Discovering all .pth (NN) and .nnue models locally and on remote hosts
2. Building a union of all unique models
3. Syncing missing models to each host via rsync over SSH
4. Optionally integrating with the cluster orchestrator for automatic sync

Usage:
    # Dry run - show what would be synced
    python scripts/sync_models_to_cluster.py --dry-run

    # Sync all models to all available hosts
    python scripts/sync_models_to_cluster.py --sync

    # Sync to specific hosts only
    python scripts/sync_models_to_cluster.py --sync --hosts lambda-gh200-a,lambda-gh200-b

    # Check model inventory across cluster
    python scripts/sync_models_to_cluster.py --inventory

    # Enable daemon mode (sync every N minutes)
    python scripts/sync_models_to_cluster.py --daemon --interval 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.distributed.hosts import load_remote_hosts, HostConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
LOCAL_MODELS_DIR = AI_SERVICE_ROOT / "models"
SYNC_STATE_FILE = AI_SERVICE_ROOT / "logs" / "model_sync_state.json"
SSH_TIMEOUT = 30
RSYNC_TIMEOUT = 600  # 10 minutes for large model files

# Model file patterns
MODEL_PATTERNS = {
    "nn": ["*.pth"],
    "nnue": ["*.nnue", "nnue/*.pt"],
}


def _ssh_base_opts(host: HostConfig) -> List[str]:
    opts = ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
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
    """Pick the first reachable SSH target for a host.

    Returns (target, error).
    """
    opts = _ssh_base_opts(host)
    last_err: Optional[str] = None
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
            last_err = (result.stderr or result.stdout or "unreachable").strip()[:200]
        except subprocess.TimeoutExpired:
            last_err = "SSH timeout"
        except Exception as e:
            last_err = str(e)[:200]
    return None, last_err


@dataclass
class HostModelInventory:
    """Model inventory for a single host."""
    host_name: str
    nn_models: Set[str] = field(default_factory=set)
    nnue_models: Set[str] = field(default_factory=set)
    work_dir: str = ""
    reachable: bool = False
    error: Optional[str] = None
    last_updated: Optional[str] = None

    def total_models(self) -> int:
        return len(self.nn_models) + len(self.nnue_models)


@dataclass
class ClusterModelState:
    """Aggregated model state across the cluster."""
    all_nn_models: Set[str] = field(default_factory=set)
    all_nnue_models: Set[str] = field(default_factory=set)
    host_inventories: Dict[str, HostModelInventory] = field(default_factory=dict)
    last_sync: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "all_nn_models": sorted(self.all_nn_models),
            "all_nnue_models": sorted(self.all_nnue_models),
            "host_inventories": {
                name: {
                    "host_name": inv.host_name,
                    "nn_models": sorted(inv.nn_models),
                    "nnue_models": sorted(inv.nnue_models),
                    "work_dir": inv.work_dir,
                    "reachable": inv.reachable,
                    "error": inv.error,
                    "last_updated": inv.last_updated,
                }
                for name, inv in self.host_inventories.items()
            },
            "last_sync": self.last_sync,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ClusterModelState":
        state = cls(
            all_nn_models=set(data.get("all_nn_models", [])),
            all_nnue_models=set(data.get("all_nnue_models", [])),
            last_sync=data.get("last_sync"),
        )
        for name, inv_data in data.get("host_inventories", {}).items():
            state.host_inventories[name] = HostModelInventory(
                host_name=inv_data["host_name"],
                nn_models=set(inv_data.get("nn_models", [])),
                nnue_models=set(inv_data.get("nnue_models", [])),
                work_dir=inv_data.get("work_dir", ""),
                reachable=inv_data.get("reachable", False),
                error=inv_data.get("error"),
                last_updated=inv_data.get("last_updated"),
            )
        return state


def get_local_models() -> Tuple[Set[str], Set[str]]:
    """Get all NN and NNUE models from local machine."""
    nn_models = set()
    nnue_models = set()

    if LOCAL_MODELS_DIR.exists():
        # NN models (.pth files)
        for pth in LOCAL_MODELS_DIR.glob("*.pth"):
            nn_models.add(pth.name)

        # NNUE models
        for nnue in LOCAL_MODELS_DIR.glob("*.nnue"):
            nnue_models.add(nnue.name)
        for nnue in LOCAL_MODELS_DIR.glob("nnue/*.pt"):
            nnue_models.add(f"nnue/{nnue.name}")

    return nn_models, nnue_models


def get_remote_models(host: HostConfig) -> HostModelInventory:
    """Get model inventory from a remote host."""
    inventory = HostModelInventory(
        host_name=host.name,
        work_dir=getattr(host, "work_directory", host.work_dir or "").strip() or "~/Development/RingRift/ai-service",
    )

    ssh_target, err = _pick_reachable_ssh_target(host)
    if not ssh_target:
        inventory.error = err or "unreachable"
        return inventory

    ssh_opts = _ssh_base_opts(host)

    # Command to list all models
    cmd_script = f"""
cd {inventory.work_dir}/models 2>/dev/null || exit 1
echo "NN_MODELS:"
ls -1 *.pth 2>/dev/null || true
echo "NNUE_MODELS:"
ls -1 *.nnue 2>/dev/null || true
ls -1 nnue/*.pt 2>/dev/null | sed 's|^|nnue/|' || true
"""

    try:
        result = subprocess.run(
            ["ssh"] + ssh_opts + [ssh_target, cmd_script],
            capture_output=True,
            timeout=SSH_TIMEOUT,
            text=True,
        )

        if result.returncode == 0:
            inventory.reachable = True
            lines = result.stdout.strip().split("\n")
            section = None
            for line in lines:
                if line == "NN_MODELS:":
                    section = "nn"
                elif line == "NNUE_MODELS:":
                    section = "nnue"
                elif line and section:
                    if section == "nn" and line.endswith(".pth"):
                        inventory.nn_models.add(line)
                    elif section == "nnue":
                        inventory.nnue_models.add(line)
            inventory.last_updated = datetime.now().isoformat()
        else:
            inventory.error = result.stderr[:200] if result.stderr else "Unknown error"
    except subprocess.TimeoutExpired:
        inventory.error = "SSH timeout"
    except Exception as e:
        inventory.error = str(e)[:200]

    return inventory


def scan_cluster(hosts: Dict[str, HostConfig], max_workers: int = 8) -> ClusterModelState:
    """Scan all hosts in parallel to build cluster model state."""
    state = ClusterModelState()

    # Get local models first
    local_nn, local_nnue = get_local_models()
    state.all_nn_models.update(local_nn)
    state.all_nnue_models.update(local_nnue)
    state.host_inventories["local"] = HostModelInventory(
        host_name="local",
        nn_models=local_nn,
        nnue_models=local_nnue,
        work_dir=str(AI_SERVICE_ROOT),
        reachable=True,
        last_updated=datetime.now().isoformat(),
    )

    logger.info(f"Local models: {len(local_nn)} NN, {len(local_nnue)} NNUE")

    # Scan remote hosts in parallel
    logger.info(f"Scanning {len(hosts)} remote hosts...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_host = {
            executor.submit(get_remote_models, host): name
            for name, host in hosts.items()
        }

        for future in as_completed(future_to_host):
            name = future_to_host[future]
            try:
                inventory = future.result()
                state.host_inventories[name] = inventory
                if inventory.reachable:
                    state.all_nn_models.update(inventory.nn_models)
                    state.all_nnue_models.update(inventory.nnue_models)
                    logger.info(
                        f"  {name}: {len(inventory.nn_models)} NN, "
                        f"{len(inventory.nnue_models)} NNUE"
                    )
                else:
                    logger.warning(f"  {name}: UNREACHABLE - {inventory.error}")
            except Exception as e:
                logger.error(f"  {name}: ERROR - {e}")

    state.last_sync = datetime.now().isoformat()
    return state


def sync_models_to_host(
    host: HostConfig,
    missing_nn: Set[str],
    missing_nnue: Set[str],
    dry_run: bool = False,
) -> Tuple[int, int, Optional[str]]:
    """Sync missing models to a single host using rsync."""
    nn_synced = 0
    nnue_synced = 0
    error = None

    if not missing_nn and not missing_nnue:
        return 0, 0, None

    work_dir = (getattr(host, "work_directory", host.work_dir or "") or "").strip() or "~/Development/RingRift/ai-service"
    ssh_target, err = _pick_reachable_ssh_target(host)
    if not ssh_target:
        return 0, 0, (err or "unreachable")

    # Build rsync command base
    ssh_opts = _ssh_base_opts(host)
    ssh_rsync = "ssh " + " ".join(ssh_opts)
    rsync_base = [
        "rsync", "-avz", "--progress",
        "-e", ssh_rsync,
    ]

    if dry_run:
        rsync_base.append("--dry-run")

    # Sync NN models
    if missing_nn:
        logger.info(f"  Syncing {len(missing_nn)} NN models to {host.name}...")
        for model_name in missing_nn:
            local_path = LOCAL_MODELS_DIR / model_name
            if not local_path.exists():
                logger.warning(f"    Local model not found: {model_name}")
                continue

            remote_path = f"{ssh_target}:{work_dir}/models/"
            try:
                result = subprocess.run(
                    rsync_base + [str(local_path), remote_path],
                    capture_output=True,
                    timeout=RSYNC_TIMEOUT,
                    text=True,
                )
                if result.returncode == 0:
                    nn_synced += 1
                else:
                    logger.warning(f"    Failed to sync {model_name}: {result.stderr[:100]}")
            except subprocess.TimeoutExpired:
                logger.warning(f"    Timeout syncing {model_name}")
            except Exception as e:
                logger.warning(f"    Error syncing {model_name}: {e}")

    # Sync NNUE models
    if missing_nnue:
        logger.info(f"  Syncing {len(missing_nnue)} NNUE models to {host.name}...")
        for model_name in missing_nnue:
            local_path = LOCAL_MODELS_DIR / model_name
            if not local_path.exists():
                logger.warning(f"    Local NNUE model not found: {model_name}")
                continue

            # Ensure nnue subdirectory exists
            if model_name.startswith("nnue/"):
                mkdir_cmd = f"mkdir -p {work_dir}/models/nnue"
                subprocess.run(
                    ["ssh", *_ssh_base_opts(host), ssh_target, mkdir_cmd],
                    capture_output=True, timeout=10
                )

            remote_path = f"{ssh_target}:{work_dir}/models/{model_name}"
            try:
                result = subprocess.run(
                    rsync_base + [str(local_path), remote_path],
                    capture_output=True,
                    timeout=RSYNC_TIMEOUT,
                    text=True,
                )
                if result.returncode == 0:
                    nnue_synced += 1
                else:
                    logger.warning(f"    Failed to sync {model_name}: {result.stderr[:100]}")
            except Exception as e:
                logger.warning(f"    Error syncing {model_name}: {e}")

    return nn_synced, nnue_synced, error


def sync_all_hosts(
    hosts: Dict[str, HostConfig],
    state: ClusterModelState,
    dry_run: bool = False,
    specific_hosts: Optional[List[str]] = None,
) -> Dict[str, Tuple[int, int]]:
    """Sync models to all hosts (or specific hosts if provided)."""
    results = {}

    target_hosts = specific_hosts or list(hosts.keys())

    for name in target_hosts:
        if name not in hosts:
            logger.warning(f"Unknown host: {name}")
            continue

        host = hosts[name]
        inventory = state.host_inventories.get(name)

        if not inventory or not inventory.reachable:
            logger.warning(f"Skipping {name}: not reachable")
            continue

        # Calculate missing models
        missing_nn = state.all_nn_models - inventory.nn_models
        missing_nnue = state.all_nnue_models - inventory.nnue_models

        if not missing_nn and not missing_nnue:
            logger.info(f"{name}: Already in sync")
            results[name] = (0, 0)
            continue

        logger.info(f"{name}: Missing {len(missing_nn)} NN, {len(missing_nnue)} NNUE")

        nn_synced, nnue_synced, error = sync_models_to_host(
            host, missing_nn, missing_nnue, dry_run
        )
        results[name] = (nn_synced, nnue_synced)

        if error:
            logger.error(f"{name}: Sync error - {error}")

    return results


def save_state(state: ClusterModelState):
    """Save cluster model state to file."""
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
    logger.info(f"Saved state to {SYNC_STATE_FILE}")


def load_state() -> Optional[ClusterModelState]:
    """Load cluster model state from file."""
    if SYNC_STATE_FILE.exists():
        try:
            with open(SYNC_STATE_FILE) as f:
                return ClusterModelState.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    return None


def print_inventory(state: ClusterModelState):
    """Print formatted inventory report."""
    print("\n" + "=" * 70)
    print("CLUSTER MODEL INVENTORY")
    print("=" * 70)
    print(f"\nTotal unique models: {len(state.all_nn_models)} NN, {len(state.all_nnue_models)} NNUE")
    print(f"Last scan: {state.last_sync or 'Never'}")

    print("\nPer-host breakdown:")
    print("-" * 70)
    print(f"{'Host':<25} {'NN':>6} {'NNUE':>6} {'Status':<20}")
    print("-" * 70)

    for name, inv in sorted(state.host_inventories.items()):
        status = "OK" if inv.reachable else f"ERROR: {inv.error[:15]}"
        print(f"{name:<25} {len(inv.nn_models):>6} {len(inv.nnue_models):>6} {status:<20}")

    print("-" * 70)

    # Show models missing from most hosts
    print("\nModels missing from hosts:")
    for name, inv in sorted(state.host_inventories.items()):
        if not inv.reachable:
            continue
        missing_nn = state.all_nn_models - inv.nn_models
        missing_nnue = state.all_nnue_models - inv.nnue_models
        if missing_nn or missing_nnue:
            print(f"  {name}: missing {len(missing_nn)} NN, {len(missing_nnue)} NNUE")


def run_daemon(hosts: Dict[str, HostConfig], interval_minutes: int = 30):
    """Run continuous sync daemon."""
    logger.info(f"Starting model sync daemon (interval: {interval_minutes} min)")

    while True:
        try:
            logger.info("=" * 50)
            logger.info(f"Daemon sync cycle starting at {datetime.now().isoformat()}")

            # Scan cluster
            state = scan_cluster(hosts)
            save_state(state)

            # Sync to all reachable hosts
            results = sync_all_hosts(hosts, state, dry_run=False)

            total_nn = sum(r[0] for r in results.values())
            total_nnue = sum(r[1] for r in results.values())
            logger.info(f"Sync complete: {total_nn} NN, {total_nnue} NNUE models synced")

            logger.info(f"Sleeping for {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
            break
        except Exception as e:
            logger.error(f"Daemon error: {e}")
            time.sleep(60)  # Wait 1 minute on error


def main():
    parser = argparse.ArgumentParser(description="Sync NN/NNUE models across cluster")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--sync", action="store_true", help="Perform actual sync")
    parser.add_argument("--inventory", action="store_true", help="Show cluster inventory")
    parser.add_argument("--daemon", action="store_true", help="Run as continuous daemon")
    parser.add_argument("--interval", type=int, default=30, help="Daemon interval in minutes")
    parser.add_argument("--hosts", type=str, help="Comma-separated list of specific hosts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load remote hosts configuration
    hosts = load_remote_hosts()
    logger.info(f"Loaded {len(hosts)} host configurations")

    specific_hosts = args.hosts.split(",") if args.hosts else None

    if args.daemon:
        run_daemon(hosts, args.interval)
        return

    # Scan cluster for current state
    logger.info("Scanning cluster for model inventory...")
    state = scan_cluster(hosts)
    save_state(state)

    if args.inventory:
        print_inventory(state)
        return

    if args.sync or args.dry_run:
        logger.info("\n" + "=" * 50)
        if args.dry_run:
            logger.info("DRY RUN - No changes will be made")
        logger.info("=" * 50)

        results = sync_all_hosts(hosts, state, dry_run=args.dry_run, specific_hosts=specific_hosts)

        # Summary
        print("\n" + "=" * 50)
        print("SYNC SUMMARY")
        print("=" * 50)
        total_nn = sum(r[0] for r in results.values())
        total_nnue = sum(r[1] for r in results.values())
        print(f"Total synced: {total_nn} NN models, {total_nnue} NNUE models")
        for name, (nn, nnue) in results.items():
            print(f"  {name}: {nn} NN, {nnue} NNUE")
    else:
        # Default: show inventory
        print_inventory(state)
        print("\nUse --sync to synchronize models, or --dry-run to preview")


if __name__ == "__main__":
    main()
