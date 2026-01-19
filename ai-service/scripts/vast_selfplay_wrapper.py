#!/usr/bin/env python3
"""Vast.ai selfplay wrapper with zombie prevention.

Designed to be invoked via SSH as a single command:
    ssh vast-node 'cd ~/ringrift/ai-service && python scripts/vast_selfplay_wrapper.py --board hex8 --num-players 2 --num-games 500'

Features:
1. Pre-cleanup: Kills existing selfplay processes before starting
2. Singleton lock: Prevents concurrent invocations
3. Process limit: Refuses to start if too many processes exist
4. Lower threshold: Uses 32 (not 128) for single-GPU Vast nodes

This prevents the zombie accumulation problem where 500-1000+ selfplay
processes accumulate on Vast.ai nodes due to:
- No process supervision (no P2P/DaemonManager on Vast nodes)
- SSH+nohup being fire-and-forget with no tracking
- Multiple manual SSH invocations stacking up
"""

import argparse
import fcntl
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vast_selfplay_wrapper")

# Lower threshold for Vast single-GPU nodes (vs 128 default)
VAST_RUNAWAY_THRESHOLD = int(
    os.environ.get("RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD", "32")
)
VAST_LOCK_FILE = Path("/tmp/ringrift_vast_selfplay.lock")

# Patterns to match selfplay processes
SELFPLAY_PATTERNS = [
    "run_self_play_soak",
    "selfplay.py",
    "generate_data",
    "gpu_parallel",
    "run_hybrid_selfplay",
    "generate_canonical_selfplay",
]


def count_selfplay_processes() -> int:
    """Count running selfplay processes using pgrep."""
    my_pid = os.getpid()
    count = 0

    for pattern in SELFPLAY_PATTERNS:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
                # Exclude our own PID
                pids = [p for p in pids if p and int(p) != my_pid]
                count += len(pids)
        except (subprocess.TimeoutExpired, ValueError):
            pass

    return count


def cleanup_existing_selfplay(force: bool = False) -> int:
    """Kill existing selfplay processes.

    Args:
        force: If True, use SIGKILL immediately. Otherwise SIGTERM then SIGKILL.

    Returns:
        Number of patterns that had processes killed.
    """
    killed = 0

    for pattern in SELFPLAY_PATTERNS:
        try:
            sig = "-9" if force else "-TERM"
            result = subprocess.run(
                ["pkill", sig, "-f", pattern],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                killed += 1
        except subprocess.TimeoutExpired:
            pass

    if not force and killed > 0:
        # Grace period for SIGTERM
        logger.info("Waiting 5s for graceful shutdown...")
        time.sleep(5)

        # SIGKILL any survivors
        for pattern in SELFPLAY_PATTERNS:
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", pattern],
                    capture_output=True,
                    timeout=5,
                )
            except subprocess.TimeoutExpired:
                pass

    return killed


class SingletonLock:
    """File-based singleton lock to prevent concurrent wrapper invocations."""

    def __init__(self, lock_path: Path = VAST_LOCK_FILE):
        self.lock_path = lock_path
        self._file_handle = None
        self._acquired = False

    def acquire(self) -> bool:
        """Acquire the singleton lock.

        Returns:
            True if lock acquired, False if another instance holds it.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._file_handle = open(self.lock_path, "a+")
            fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write our PID
            self._file_handle.seek(0)
            self._file_handle.truncate()
            self._file_handle.write(str(os.getpid()))
            self._file_handle.flush()

            self._acquired = True
            return True

        except OSError:
            # Lock held by another process
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
            return False

    def release(self) -> None:
        """Release the singleton lock."""
        if self._file_handle:
            try:
                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
                self._file_handle.close()
            except OSError:
                pass
            finally:
                self._file_handle = None
                self._acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


def run_selfplay(args: argparse.Namespace) -> int:
    """Run the actual selfplay command.

    Returns:
        Exit code from selfplay process.
    """
    # Build command
    cmd = [
        sys.executable,
        "scripts/selfplay.py",
        "--board", args.board,
        "--num-players", str(args.num_players),
        "--num-games", str(args.num_games),
        "--engine", args.engine,
        "--output-dir", "data/games",
    ]

    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)
    env["RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD"] = str(VAST_RUNAWAY_THRESHOLD)
    env["RINGRIFT_VAST_NODE"] = "1"

    logger.info(f"Running: {' '.join(cmd)}")

    # Run selfplay
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(Path(__file__).parent.parent),
    )

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vast.ai selfplay wrapper with zombie prevention"
    )
    parser.add_argument(
        "--board", "--board-type",
        required=True,
        help="Board type (hex8, square8, etc.)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=500,
        help="Number of games to generate (default: 500)",
    )
    parser.add_argument(
        "--engine",
        default="gumbel",
        help="Selfplay engine (default: gumbel)",
    )
    parser.add_argument(
        "--force-cleanup",
        action="store_true",
        help="Force kill existing processes with SIGKILL",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup step (dangerous)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    args = parser.parse_args()

    # Step 1: Acquire singleton lock
    lock = SingletonLock()
    if not lock.acquire():
        logger.error("Another vast_selfplay_wrapper is already running")
        return 1

    try:
        # Step 2: Check process count BEFORE cleanup
        current_count = count_selfplay_processes()
        logger.info(f"Current selfplay process count: {current_count}")

        if current_count >= VAST_RUNAWAY_THRESHOLD:
            logger.warning(
                f"RUNAWAY DETECTED: {current_count} processes (>= {VAST_RUNAWAY_THRESHOLD})"
            )
            logger.info("Forcing cleanup before proceeding...")

            if args.dry_run:
                logger.info("[DRY RUN] Would force cleanup all selfplay processes")
            else:
                cleanup_existing_selfplay(force=True)
                time.sleep(2)

        # Step 3: Cleanup existing (unless skipped)
        if not args.skip_cleanup:
            if args.dry_run:
                logger.info("[DRY RUN] Would cleanup existing selfplay processes")
            else:
                killed = cleanup_existing_selfplay(force=args.force_cleanup)
                if killed > 0:
                    logger.info(f"Cleaned up processes matching {killed} patterns")

        # Step 4: Verify count is now safe
        final_count = count_selfplay_processes()
        logger.info(f"Post-cleanup process count: {final_count}")

        if final_count >= VAST_RUNAWAY_THRESHOLD:
            logger.error(
                f"Still {final_count} processes after cleanup (>= {VAST_RUNAWAY_THRESHOLD})"
            )
            return 1

        # Step 5: Run selfplay
        if args.dry_run:
            logger.info(
                f"[DRY RUN] Would run selfplay: "
                f"board={args.board}, players={args.num_players}, "
                f"games={args.num_games}, engine={args.engine}"
            )
            return 0

        return run_selfplay(args)

    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
