"""Compatibility wrapper for unified work orchestrator.

Deprecated: use scripts/resource_aware_router.py or the P2P orchestrator.
This wrapper translates legacy CLI flags to resource_aware_router where
possible so existing cron jobs keep working.
"""

from __future__ import annotations

import argparse
import runpy
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Legacy unified work orchestrator (compatibility wrapper)."
    )
    parser.add_argument("--node-id", help="Legacy flag (unused).")
    parser.add_argument("--once", action="store_true", help="Run a single rebalance pass.")
    parser.add_argument("--status", action="store_true", help="Show cluster status.")
    parser.add_argument("--rebalance", action="store_true", help="Rebalance work across cluster.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done.")
    parser.add_argument("--daemon", action="store_true", help="Run continuously.")
    parser.add_argument("--interval", type=int, default=None, help="Daemon interval (seconds).")
    args, extra = parser.parse_known_args()

    print(
        "unified_work_orchestrator.py is deprecated. "
        "Use scripts/resource_aware_router.py or scripts/p2p_orchestrator.py instead.",
        file=sys.stderr,
    )

    target_argv = ["resource_aware_router.py"]
    if args.daemon:
        target_argv.append("--daemon")
    elif args.status:
        target_argv.append("--status")
    else:
        if args.once or args.rebalance:
            target_argv.append("--rebalance")
        else:
            target_argv.append("--status")

    if args.dry_run:
        target_argv.append("--dry-run")
    if args.interval is not None:
        target_argv.extend(["--interval", str(args.interval)])

    sys.argv = target_argv + extra
    runpy.run_module("scripts.resource_aware_router", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
