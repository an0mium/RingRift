#!/usr/bin/env python3
"""Verify architecture overhaul changes are correctly applied.

Checks:
1. Raft members = 2 coordinators only (mac-studio, local-mac)
2. local-mac blocks heavy work (export, training, selfplay, gauntlet, consolidation)
3. S3 bucket is accessible
4. Disk proactive threshold = 70%
5. get_raft_members() and get_coordinator_nodes() return expected values

Usage:
    PYTHONPATH=. python3 scripts/verify_architecture_changes.py
"""

from __future__ import annotations

import subprocess
import sys


def check_mark(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def verify_raft_members() -> bool:
    """Check Raft members = 2 coordinators only."""
    try:
        from app.config.cluster_config import get_raft_members

        members = get_raft_members()
        expected = {"mac-studio", "local-mac"}
        actual = set(members)
        passed = actual == expected
        print(f"  [{check_mark(passed)}] Raft members: {members}")
        if not passed:
            print(f"         Expected: {sorted(expected)}, Got: {sorted(actual)}")
        return passed
    except Exception as e:
        print(f"  [FAIL] Raft members check failed: {e}")
        return False


def verify_coordinator_nodes() -> bool:
    """Check coordinator nodes list."""
    try:
        from app.config.cluster_config import get_coordinator_nodes

        nodes = get_coordinator_nodes()
        expected = {"mac-studio", "local-mac"}
        actual = set(nodes)
        passed = actual == expected
        print(f"  [{check_mark(passed)}] Coordinator nodes: {nodes}")
        if not passed:
            print(f"         Expected: {sorted(expected)}, Got: {sorted(actual)}")
        return passed
    except Exception as e:
        print(f"  [FAIL] Coordinator nodes check failed: {e}")
        return False


def verify_local_mac_blocked() -> bool:
    """Check local-mac blocks heavy work."""
    try:
        from app.config.env import RingRiftEnv
        import os

        # Simulate local-mac node
        old_node_id = os.environ.get("RINGRIFT_NODE_ID")
        os.environ["RINGRIFT_NODE_ID"] = "local-mac"

        # Create fresh env instance (bypass cached_property)
        test_env = RingRiftEnv()

        passed = test_env.is_heavy_work_blocked is True
        print(f"  [{check_mark(passed)}] is_heavy_work_blocked for local-mac: {test_env.is_heavy_work_blocked}")

        # Check all *_enabled properties return False
        checks = {
            "selfplay_enabled": test_env.selfplay_enabled,
            "training_enabled": test_env.training_enabled,
            "gauntlet_enabled": test_env.gauntlet_enabled,
            "export_enabled": test_env.export_enabled,
            "consolidation_enabled": test_env.consolidation_enabled,
        }
        all_disabled = all(not v for v in checks.values())
        print(f"  [{check_mark(all_disabled)}] All heavy work disabled: {checks}")

        # Restore
        if old_node_id is not None:
            os.environ["RINGRIFT_NODE_ID"] = old_node_id
        else:
            os.environ.pop("RINGRIFT_NODE_ID", None)

        return passed and all_disabled
    except Exception as e:
        print(f"  [FAIL] local-mac block check failed: {e}")
        return False


def verify_s3_accessible() -> bool:
    """Check S3 bucket is accessible."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", "s3://ringrift-models-20251214/consolidated/"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        passed = result.returncode == 0
        print(f"  [{check_mark(passed)}] S3 bucket accessible (ringrift-models-20251214)")
        if not passed:
            print(f"         stderr: {result.stderr.strip()[:200]}")
        return passed
    except FileNotFoundError:
        print("  [SKIP] aws CLI not installed")
        return True  # Don't fail on nodes without aws CLI
    except Exception as e:
        print(f"  [FAIL] S3 check failed: {e}")
        return False


def verify_disk_thresholds() -> bool:
    """Check disk proactive threshold = 70%."""
    try:
        from app.config.thresholds import DISK_SYNC_TARGET_PERCENT

        passed = DISK_SYNC_TARGET_PERCENT == 70
        print(f"  [{check_mark(passed)}] DISK_SYNC_TARGET_PERCENT: {DISK_SYNC_TARGET_PERCENT}")
        return passed
    except Exception as e:
        print(f"  [FAIL] Disk threshold check failed: {e}")
        return False


def verify_raft_not_in_gpu_nodes() -> bool:
    """Check that non-coordinator GPU nodes are NOT in Raft members."""
    try:
        from app.config.cluster_config import get_raft_members, get_gpu_nodes, get_coordinator_nodes

        members = set(get_raft_members())
        coordinators = set(get_coordinator_nodes())
        gpu_nodes = get_gpu_nodes()
        # Only flag GPU nodes that are NOT coordinators
        non_coord_gpu = {n.name for n in gpu_nodes} - coordinators
        overlap = members & non_coord_gpu
        passed = len(overlap) == 0
        print(f"  [{check_mark(passed)}] No non-coordinator GPU nodes in Raft (checked {len(non_coord_gpu)} GPU workers)")
        if not passed:
            print(f"         GPU workers in Raft: {sorted(overlap)}")
        return passed
    except Exception as e:
        print(f"  [FAIL] Raft/GPU overlap check failed: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("Architecture Overhaul Verification")
    print("=" * 60)

    results = []

    print("\n1. Raft Configuration")
    results.append(verify_raft_members())
    results.append(verify_coordinator_nodes())
    results.append(verify_raft_not_in_gpu_nodes())

    print("\n2. Local-Mac Safety")
    results.append(verify_local_mac_blocked())

    print("\n3. S3 Access")
    results.append(verify_s3_accessible())

    print("\n4. Disk Thresholds")
    results.append(verify_disk_thresholds())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    all_passed = all(results)
    status = "ALL PASSED" if all_passed else "SOME FAILED"
    print(f"Result: {passed}/{total} checks passed - {status}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
