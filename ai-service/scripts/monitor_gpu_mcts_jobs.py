#!/usr/bin/env python3
"""Monitor GPU MCTS selfplay jobs across cluster.

Watches for job completion, validates data, and triggers next steps.

Usage:
    python scripts/monitor_gpu_mcts_jobs.py --watch
    python scripts/monitor_gpu_mcts_jobs.py --once
"""

import argparse
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

CLUSTER_NODES = [
    "lambda-gh200-c",
    "lambda-gh200-g",
    "lambda-2xh100",
]

SSH_KEY = "~/.ssh/id_cluster"
REMOTE_DIR = "~/ringrift/ai-service"


@dataclass
class JobStatus:
    node: str
    config: str
    status: str  # running, completed, failed
    samples: int = 0
    file_size: str = ""
    output_path: str = ""


def run_ssh(node: str, cmd: str, timeout: int = 10) -> str:
    """Run SSH command and return output."""
    try:
        result = subprocess.run(
            ["ssh", "-i", Path(SSH_KEY).expanduser(),
             "-o", "ConnectTimeout=5",
             "-o", "StrictHostKeyChecking=no",
             f"ubuntu@{node}", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def check_node_status(node: str) -> list[JobStatus]:
    """Check status of GPU MCTS jobs on a node."""
    statuses = []

    # Check running processes
    procs = run_ssh(node, "pgrep -f gpu_mcts_selfplay | wc -l")
    try:
        n_procs = int(procs)
    except:
        n_procs = 0

    # Check log files for job configs
    logs = run_ssh(node, f"cd {REMOTE_DIR} && ls logs/gpu_mcts_*.log 2>/dev/null")

    for log in logs.split("\n"):
        if not log or log.startswith("ERROR"):
            continue

        config = log.replace("logs/gpu_mcts_", "").replace(".log", "")

        # Check if corresponding NPZ exists
        npz_path = f"data/training/gpu_mcts_{config}.npz"
        npz_info = run_ssh(node, f"cd {REMOTE_DIR} && ls -lh {npz_path} 2>/dev/null")

        if npz_info and not npz_info.startswith("ERROR"):
            # Job completed
            size = npz_info.split()[4] if len(npz_info.split()) > 4 else "?"

            # Get sample count
            sample_cmd = f"cd {REMOTE_DIR} && python3 -c \"import numpy as np; d=np.load('{npz_path}'); print(len(d['features']))\""
            samples = run_ssh(node, sample_cmd)
            try:
                n_samples = int(samples)
            except:
                n_samples = 0

            statuses.append(JobStatus(
                node=node,
                config=config,
                status="completed",
                samples=n_samples,
                file_size=size,
                output_path=npz_path,
            ))
        else:
            # Check if still running
            is_running = run_ssh(node, f"pgrep -f 'gpu_mcts.*{config}' | wc -l")
            try:
                running = int(is_running) > 0
            except:
                running = False

            statuses.append(JobStatus(
                node=node,
                config=config,
                status="running" if running else "unknown",
            ))

    return statuses


def validate_completed(status: JobStatus) -> dict:
    """Validate a completed job's output."""
    if status.status != "completed":
        return {"valid": False, "error": "Not completed"}

    # Run validation script remotely
    cmd = f"cd {REMOTE_DIR} && PYTHONPATH=. python3 scripts/validate_gpu_mcts_data.py {status.output_path} 2>/dev/null"
    output = run_ssh(status.node, cmd, timeout=30)

    return {
        "valid": "PASS" in output,
        "output": output,
    }


def print_status(all_statuses: list[JobStatus]):
    """Print status table."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"{'Node':<18} {'Config':<15} {'Status':<12} {'Samples':>10} {'Size':>8}")
    logger.info("-" * 70)

    for s in all_statuses:
        samples = str(s.samples) if s.samples else "-"
        size = s.file_size or "-"
        logger.info(f"{s.node:<18} {s.config:<15} {s.status:<12} {samples:>10} {size:>8}")

    logger.info("=" * 70)

    completed = sum(1 for s in all_statuses if s.status == "completed")
    running = sum(1 for s in all_statuses if s.status == "running")
    logger.info(f"Summary: {completed} completed, {running} running")


def download_completed(status: JobStatus, local_dir: str = "data/training/cluster") -> str:
    """Download completed NPZ file from cluster."""
    local_path = Path(local_dir) / f"{status.node}_{status.config}.npz"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    remote_path = f"ubuntu@{status.node}:{REMOTE_DIR}/{status.output_path}"

    try:
        subprocess.run([
            "scp", "-i", Path(SSH_KEY).expanduser(),
            "-o", "StrictHostKeyChecking=no",
            remote_path, str(local_path),
        ], check=True, capture_output=True)
        return str(local_path)
    except Exception as e:
        logger.error(f"Failed to download {status.config} from {status.node}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU MCTS jobs")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--download", action="store_true", help="Download completed files")
    parser.add_argument("--validate", action="store_true", help="Validate completed files")

    args = parser.parse_args()

    while True:
        all_statuses = []

        for node in CLUSTER_NODES:
            statuses = check_node_status(node)
            all_statuses.extend(statuses)

        print_status(all_statuses)

        # Handle completed jobs
        completed = [s for s in all_statuses if s.status == "completed"]

        if args.validate:
            for s in completed:
                result = validate_completed(s)
                if result["valid"]:
                    logger.info(f"  ✓ {s.config} validated")
                else:
                    logger.warning(f"  ✗ {s.config} validation failed")

        if args.download:
            for s in completed:
                local = download_completed(s)
                if local:
                    logger.info(f"  Downloaded: {local}")

        if not args.watch:
            break

        logger.info(f"\nNext check in {args.interval}s... (Ctrl+C to stop)")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
