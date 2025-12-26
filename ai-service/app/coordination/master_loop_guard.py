"""Master Loop Guard - Ensures master loop is running for full automation (December 2025).

This module provides utilities to check if the master loop is running and enforce
it as a requirement for critical coordination operations.

Usage:
    from app.coordination.master_loop_guard import ensure_master_loop_running

    # Will raise RuntimeError if master loop is not active
    ensure_master_loop_running()

    # Or check without raising
    if not is_master_loop_running():
        print("Master loop is not running!")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# PID file path - same location as master_loop.py
PID_FILE_PATH = Path(__file__).parent.parent.parent / "data" / "coordination" / "master_loop.pid"


def is_master_loop_running(pid_file: Path | None = None) -> bool:
    """Check if master loop is running by checking PID file.

    Args:
        pid_file: Path to PID file (defaults to standard location)

    Returns:
        True if master loop process is active, False otherwise
    """
    pid_file = pid_file or PID_FILE_PATH

    if not pid_file.exists():
        return False

    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())

        # Check if process with this PID exists
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
            return True
        except OSError:
            # Process doesn't exist, remove stale PID file
            try:
                pid_file.unlink()
            except OSError:
                pass
            return False
    except (ValueError, IOError, OSError):
        return False


def ensure_master_loop_running(
    require_for_automation: bool = True,
    operation_name: str = "full automation",
) -> None:
    """Ensure master loop is running for full automation.

    Args:
        require_for_automation: If True, raises RuntimeError when master loop is not running
        operation_name: Name of operation requiring master loop (for error message)

    Raises:
        RuntimeError: If master loop is not running and require_for_automation is True
    """
    if not require_for_automation:
        return

    if not is_master_loop_running():
        raise RuntimeError(
            f"Master loop must be running for {operation_name}. "
            f"Start with: python scripts/master_loop.py\n"
            f"Or skip this check by setting RINGRIFT_SKIP_MASTER_LOOP_CHECK=1"
        )


def check_or_warn(operation_name: str = "this operation") -> bool:
    """Check if master loop is running and warn if not.

    Args:
        operation_name: Name of operation (for warning message)

    Returns:
        True if master loop is running, False otherwise
    """
    if is_master_loop_running():
        return True

    logger.warning(
        f"[MasterLoopGuard] Master loop is not running for {operation_name}. "
        f"Full automation requires: python scripts/master_loop.py"
    )
    return False


__all__ = [
    "is_master_loop_running",
    "ensure_master_loop_running",
    "check_or_warn",
    "PID_FILE_PATH",
]
