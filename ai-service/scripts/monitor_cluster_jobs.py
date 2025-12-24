#!/usr/bin/env python3
"""Monitor all active training and selfplay jobs across the cluster."""

import subprocess
import sys
from datetime import datetime


NODES = {
    # Training nodes
    "lambda-h100": "ubuntu@209.20.157.81",
    "lambda-2xh100": "ubuntu@192.222.53.22",
    "lambda-a10": "ubuntu@150.136.65.197",
    "lambda-a10-b": "ubuntu@129.153.159.191",
    "lambda-a10-c": "ubuntu@150.136.56.240",
    # GH200 nodes
    "lambda-gh200-o": "ubuntu@192.222.51.92",
    "lambda-gh200-p": "ubuntu@192.222.51.215",
    "lambda-gh200-t": "ubuntu@192.222.50.211",
    "lambda-gh200-m": "ubuntu@192.222.50.219",
    "lambda-gh200-n": "ubuntu@192.222.51.204",
}


def ssh_cmd(host: str, cmd: str, timeout: int = 10) -> str:
    """Run SSH command and return output."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-i", "~/.ssh/id_cluster", host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def check_training_jobs(host: str) -> list[str]:
    """Check for active training jobs."""
    output = ssh_cmd(host, "pgrep -fa 'app.training.train' | grep -v grep | head -3")
    jobs = []
    for line in output.split("\n"):
        if "train" in line and "board-type" in line:
            # Extract board type and players
            import re
            board = re.search(r"--board-type (\w+)", line)
            players = re.search(r"--num-players (\d+)", line)
            if board and players:
                jobs.append(f"{board.group(1)}_{players.group(1)}p")
    return jobs


def check_selfplay_jobs(host: str) -> list[str]:
    """Check for active selfplay jobs."""
    output = ssh_cmd(host, "pgrep -fa 'selfplay|generate' | grep -v grep | head -5")
    jobs = []
    for line in output.split("\n"):
        if "hex8" in line.lower():
            jobs.append("hex8")
        elif "hexagonal" in line.lower():
            jobs.append("hexagonal")
        elif "square19" in line.lower():
            jobs.append("square19")
        elif "square8" in line.lower():
            jobs.append("square8")
    return list(set(jobs))


def check_gauntlet_jobs(host: str) -> list[str]:
    """Check for active gauntlet jobs."""
    output = ssh_cmd(host, "pgrep -fa 'gauntlet|auto_promote' | grep -v grep | head -3")
    jobs = []
    for line in output.split("\n"):
        if "gauntlet" in line or "auto_promote" in line:
            import re
            board = re.search(r"--board-type (\w+)", line)
            if board:
                jobs.append(f"gauntlet_{board.group(1)}")
    return jobs


def get_training_progress(host: str, config: str) -> str:
    """Get training progress from log."""
    log_patterns = [
        f"logs/train_{config}.log",
        f"logs/train_{config}_*.log",
        f"logs/*{config}*.log",
    ]
    for pattern in log_patterns:
        output = ssh_cmd(host, f"tail -1 ~/ringrift/ai-service/{pattern} 2>/dev/null | head -1")
        if "Epoch" in output:
            import re
            match = re.search(r"Epoch.*?(\d+)/(\d+).*?Loss[=:]([0-9.]+).*?Acc[=:]?\s*([0-9.]+)", output)
            if match:
                return f"Epoch {match.group(1)}/{match.group(2)}, Loss={match.group(3)}, Acc={match.group(4)}%"
            match = re.search(r"Epoch\s+(\d+)", output)
            if match:
                return f"Epoch {match.group(1)}"
    return ""


def main():
    print(f"\n{'='*60}")
    print(f"CLUSTER JOB MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    print("TRAINING JOBS:")
    print("-" * 40)
    for name, host in NODES.items():
        jobs = check_training_jobs(host)
        if jobs:
            for job in jobs:
                progress = get_training_progress(host, job)
                print(f"  {name}: {job} {progress}")

    print("\nSELFPLAY JOBS:")
    print("-" * 40)
    for name, host in NODES.items():
        jobs = check_selfplay_jobs(host)
        if jobs:
            print(f"  {name}: {', '.join(jobs)}")

    print("\nGAUNTLET JOBS:")
    print("-" * 40)
    for name, host in NODES.items():
        jobs = check_gauntlet_jobs(host)
        if jobs:
            print(f"  {name}: {', '.join(jobs)}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
