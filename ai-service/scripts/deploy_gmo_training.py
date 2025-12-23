#!/usr/bin/env python3
"""Deploy GMO variant training to cluster nodes.

Usage:
    python scripts/deploy_gmo_training.py sync     # Sync code and data
    python scripts/deploy_gmo_training.py train    # Start all training
    python scripts/deploy_gmo_training.py status   # Check status
    python scripts/deploy_gmo_training.py logs gmo # Show logs
    python scripts/deploy_gmo_training.py fetch    # Fetch trained models
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Node:
    name: str
    user: str
    host: str
    port: int
    ssh_key: str
    repo_path: str


NODES = {
    "gh200_e": Node("gh200_e", "ubuntu", "100.88.176.74", 22, "~/.ssh/id_ed25519", "~/ringrift"),
    "gh200_f": Node("gh200_f", "ubuntu", "100.104.165.116", 22, "~/.ssh/id_ed25519", "~/ringrift"),
    "vast_a40": Node("vast_a40", "root", "ssh8.vast.ai", 38742, "~/.ssh/id_cluster", "/workspace/ringrift"),
    "vast_5090": Node("vast_5090", "root", "ssh1.vast.ai", 15166, "~/.ssh/id_cluster", "/workspace/ringrift"),
}

TRAINING_ASSIGNMENT = {
    "gmo": "gh200_e",
    "gmo_v2": "vast_5090",
    "ig_gmo": "vast_a40",
}

LOCAL_PATH = Path(__file__).resolve().parent.parent


def ssh_cmd(node: Node, cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Run SSH command on node."""
    ssh_key = os.path.expanduser(node.ssh_key)
    ssh = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={timeout}",
        "-i", ssh_key,
        "-p", str(node.port),
        f"{node.user}@{node.host}",
        cmd
    ]
    try:
        result = subprocess.run(ssh, capture_output=True, text=True, timeout=timeout + 10)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, "Timeout"


def rsync_to_node(node: Node, src: str, dst: str, exclude: list[str] | None = None) -> bool:
    """Rsync files to node."""
    ssh_key = os.path.expanduser(node.ssh_key)
    rsync = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -o StrictHostKeyChecking=no -i {ssh_key} -p {node.port}",
    ]
    for ex in (exclude or []):
        rsync.extend(["--exclude", ex])
    rsync.extend([src, f"{node.user}@{node.host}:{dst}"])

    result = subprocess.run(rsync, capture_output=False)
    return result.returncode == 0


def sync_to_node(node: Node) -> bool:
    """Sync code and data to node."""
    print(f"\n=== Syncing to {node.name} ({node.host}:{node.port}) ===")

    # Create directories
    repo = node.repo_path
    ret, out = ssh_cmd(node, f"mkdir -p {repo}/data/training {repo}/models/gmo {repo}/models/gmo_v2 {repo}/models/ig_gmo")
    if ret != 0:
        print(f"  Failed to create directories: {out}")
        return False

    # Sync app code
    excludes = [".git", "*.pt", "__pycache__", ".pytest_cache", "venv",
                "node_modules", "checkpoints", "data/training/*.npz", "data/training/*.db"]
    if not rsync_to_node(node, f"{LOCAL_PATH}/app/", f"{repo}/app/", excludes):
        print("  Failed to sync app/")
        return False

    # Sync scripts
    if not rsync_to_node(node, f"{LOCAL_PATH}/scripts/", f"{repo}/scripts/", excludes):
        print("  Failed to sync scripts/")
        return False

    # Sync training data (mega file with ~7K games / 720K samples)
    if not rsync_to_node(node, f"{LOCAL_PATH}/data/training/gmo_mega_sq8_2p.jsonl", f"{repo}/data/training/"):
        print("  Failed to sync training data")
        return False

    # Sync pyproject.toml
    if not rsync_to_node(node, f"{LOCAL_PATH}/pyproject.toml", f"{repo}/"):
        print("  Failed to sync pyproject.toml")
        return False

    # Install dependencies
    ret, out = ssh_cmd(node, f"cd {repo} && pip install -q -e . 2>/dev/null || pip install -q torch numpy tqdm pydantic", timeout=120)
    if ret != 0:
        print(f"  Warning: pip install returned non-zero: {out[:200]}")

    print(f"=== {node.name} synced ===")
    return True


def start_training(model: str) -> bool:
    """Start training for a model."""
    node_name = TRAINING_ASSIGNMENT.get(model)
    if not node_name:
        print(f"No node assigned for model: {model}")
        return False

    node = NODES[node_name]
    repo = node.repo_path
    print(f"\n=== Starting {model} training on {node.name} ===")

    if model == "gmo":
        cmd = f"""cd {repo} && nohup python -m app.training.train_gmo \
            --data-path data/training/gmo_mega_sq8_2p.jsonl \
            --output-dir models/gmo/sq8_2p_mega \
            --epochs 100 --batch-size 256 --lr 0.0003 --device cuda \
            > /tmp/training_gmo.log 2>&1 &"""
    elif model == "gmo_v2":
        cmd = f"""cd {repo} && nohup python -m app.training.train_gmo_v2 \
            --data-path data/training/gmo_mega_sq8_2p.jsonl \
            --output-dir models/gmo_v2/sq8_2p_mega \
            --epochs 100 --batch-size 128 --lr 0.0002 --device cuda \
            > /tmp/training_gmo_v2.log 2>&1 &"""
    elif model == "ig_gmo":
        cmd = f"""cd {repo} && nohup python -m app.training.train_ig_gmo \
            --data-path data/training/gmo_mega_sq8_2p.jsonl \
            --output-dir models/ig_gmo/sq8_2p_mega \
            --epochs 80 --batch-size 128 --lr 0.0002 --device cuda \
            > /tmp/training_ig_gmo.log 2>&1 &"""
    else:
        print(f"Unknown model: {model}")
        return False

    ret, out = ssh_cmd(node, cmd, timeout=30)
    if ret != 0:
        print(f"  Failed to start training: {out}")
        return False

    print(f"=== {model} training started (log: /tmp/training_{model}.log) ===")
    return True


def check_status() -> None:
    """Check training status on all nodes."""
    for name, node in NODES.items():
        print(f"\n=== {name} ===")
        ret, out = ssh_cmd(node, "nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader", timeout=10)
        if ret == 0:
            print(out.strip())
        else:
            print("  GPU check failed")

        ret, out = ssh_cmd(node, "ps aux | grep 'train_' | grep python | grep -v grep | head -5", timeout=10)
        if ret == 0 and out.strip():
            print("  Running:")
            for line in out.strip().split('\n'):
                parts = line.split()
                if len(parts) > 10:
                    print(f"    {parts[10]}")
        else:
            print("  No training running")


def show_logs(model: str, lines: int = 100) -> None:
    """Show training logs."""
    node_name = TRAINING_ASSIGNMENT.get(model)
    if not node_name:
        print(f"No node assigned for model: {model}")
        return

    node = NODES[node_name]
    print(f"\n=== Logs for {model} on {node.name} ===")
    ret, out = ssh_cmd(node, f"tail -{lines} /tmp/training_{model}.log", timeout=30)
    if ret == 0:
        print(out)
    else:
        print(f"  Failed to get logs: {out}")


def fetch_models() -> None:
    """Fetch trained models back to local."""
    for model, node_name in TRAINING_ASSIGNMENT.items():
        node = NODES[node_name]
        local_dir = LOCAL_PATH / "models" / model / "sq8_2p_full"
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Fetching {model} from {node.name} ===")

        ssh_key = os.path.expanduser(node.ssh_key)
        rsync = [
            "rsync", "-avz", "--progress",
            "-e", f"ssh -o StrictHostKeyChecking=no -i {ssh_key} -p {node.port}",
            f"{node.user}@{node.host}:{node.repo_path}/models/{model}/sq8_2p_full/",
            f"{local_dir}/"
        ]
        result = subprocess.run(rsync, capture_output=False)
        if result.returncode != 0:
            print(f"  No models found or fetch failed")


def main():
    parser = argparse.ArgumentParser(description="Deploy GMO training to cluster")
    parser.add_argument("action", choices=["sync", "train", "status", "logs", "fetch"])
    parser.add_argument("model", nargs="?", default="all", help="Model to train/log (gmo, gmo_v2, ig_gmo, all)")
    parser.add_argument("--lines", type=int, default=100, help="Number of log lines to show")

    args = parser.parse_args()

    if args.action == "sync":
        for node in NODES.values():
            sync_to_node(node)

    elif args.action == "train":
        if args.model == "all":
            for model in TRAINING_ASSIGNMENT:
                start_training(model)
        else:
            start_training(args.model)

    elif args.action == "status":
        check_status()

    elif args.action == "logs":
        if args.model == "all":
            for model in TRAINING_ASSIGNMENT:
                show_logs(model, args.lines)
        else:
            show_logs(args.model, args.lines)

    elif args.action == "fetch":
        fetch_models()


if __name__ == "__main__":
    main()
