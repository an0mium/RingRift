#!/usr/bin/env python3
"""Recover high-Elo models from backup locations.

December 2025 - Phase 3C: Model Distribution & Elo Fairness

This script scans backup locations (e.g., OWC external drive on mac-studio)
for model files that may have been lost when nodes were terminated.

Usage:
    # Dry run - show what would be recovered
    python scripts/recover_backup_models.py --dry-run

    # Full recovery with distribution
    python scripts/recover_backup_models.py --distribute

    # Scan specific backup host
    python scripts/recover_backup_models.py --backup-host mac-studio

    # Scan specific path
    python scripts/recover_backup_models.py --backup-path /Volumes/RingRift-Data/
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add parent to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class BackupModel:
    """Represents a model found in backup locations."""

    filename: str
    elo: int | None
    size_bytes: int
    remote_path: str
    board_type: str
    num_players: int


# Default backup locations
DEFAULT_BACKUP_HOSTS = ["mac-studio"]
DEFAULT_BACKUP_PATHS = [
    "/Volumes/RingRift-Data/selfplay_repository/models/checkpoints",
    "/Volumes/RingRift-Data/trained_models",
    "/Volumes/RingRift-Data/models",
]

# Pattern to extract Elo from filenames like:
#   sq8_2p_best_1725elo.pth
#   model_elo1597_20251224.pth
#   canonical_hex8_2p_elo1145.pth
ELO_PATTERNS = [
    r"_(\d+)elo",          # _1725elo
    r"elo(\d+)",           # elo1597
    r"_elo_?(\d+)",        # _elo_1145 or _elo1145
]

# Pattern to extract board type and players
BOARD_PATTERNS = {
    r"sq8_?(\d)p": ("square8", None),
    r"square8_?(\d)p": ("square8", None),
    r"hex8_?(\d)p": ("hex8", None),
    r"sq19_?(\d)p": ("square19", None),
    r"square19_?(\d)p": ("square19", None),
    r"hexagonal_?(\d)p": ("hexagonal", None),
}


def extract_elo(filename: str) -> int | None:
    """Extract Elo rating from filename."""
    for pattern in ELO_PATTERNS:
        match = re.search(pattern, filename.lower())
        if match:
            return int(match.group(1))
    return None


def extract_board_config(filename: str) -> tuple[str, int]:
    """Extract board type and player count from filename."""
    for pattern, (board_type, _) in BOARD_PATTERNS.items():
        match = re.search(pattern, filename.lower())
        if match:
            num_players = int(match.group(1))
            return (board_type, num_players)

    # Fallback: try to guess from common patterns
    if "sq8" in filename.lower() or "square8" in filename.lower():
        return ("square8", 2)
    if "hex8" in filename.lower():
        return ("hex8", 2)
    if "sq19" in filename.lower() or "square19" in filename.lower():
        return ("square19", 2)
    if "hexagonal" in filename.lower():
        return ("hexagonal", 2)

    return ("unknown", 2)


def scan_backup_location(
    host: str,
    path: str,
    ssh_key: str | None = None,
) -> list[BackupModel]:
    """Scan a backup location for model files.

    Args:
        host: Remote host to SSH into
        path: Path to scan on remote host
        ssh_key: Optional SSH key path

    Returns:
        List of BackupModel objects found
    """
    models = []

    # Build SSH command
    ssh_opts = [
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=30",
        "-o", "BatchMode=yes",
    ]
    if ssh_key:
        ssh_opts.extend(["-i", ssh_key])

    # Find all .pth files
    find_cmd = f"find {path} -name '*.pth' -type f 2>/dev/null | head -100"

    try:
        result = subprocess.run(
            ["ssh", *ssh_opts, host, find_cmd],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to scan {host}:{path}: {result.stderr}")
            return []

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            remote_path = line.strip()
            filename = Path(remote_path).name

            # Get file size
            stat_cmd = f"stat -f '%z' '{remote_path}' 2>/dev/null || stat -c '%s' '{remote_path}' 2>/dev/null"
            stat_result = subprocess.run(
                ["ssh", *ssh_opts, host, stat_cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )

            size_bytes = 0
            if stat_result.returncode == 0:
                try:
                    size_bytes = int(stat_result.stdout.strip())
                except ValueError:
                    pass

            # Extract metadata
            elo = extract_elo(filename)
            board_type, num_players = extract_board_config(filename)

            models.append(BackupModel(
                filename=filename,
                elo=elo,
                size_bytes=size_bytes,
                remote_path=remote_path,
                board_type=board_type,
                num_players=num_players,
            ))

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout scanning {host}:{path}")
    except subprocess.SubprocessError as e:
        logger.error(f"Error scanning {host}:{path}: {e}")

    return models


def copy_model_to_local(
    host: str,
    remote_path: str,
    local_dir: Path,
    ssh_key: str | None = None,
) -> Path | None:
    """Copy a model from backup to local directory.

    Args:
        host: Remote host
        remote_path: Path on remote host
        local_dir: Local directory to copy to
        ssh_key: Optional SSH key path

    Returns:
        Local path if successful, None otherwise
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(remote_path).name
    local_path = local_dir / filename

    # Build SCP command
    scp_opts = [
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=30",
    ]
    if ssh_key:
        scp_opts.extend(["-i", ssh_key])

    try:
        subprocess.run(
            ["scp", *scp_opts, f"{host}:{remote_path}", str(local_path)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        logger.info(f"Copied {filename} to {local_path}")
        return local_path

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout copying {filename}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to copy {filename}: {e}")

    return None


def distribute_model(model_path: Path, board_type: str, num_players: int) -> bool:
    """Trigger distribution of a model to the cluster.

    Args:
        model_path: Local path to model
        board_type: Board type
        num_players: Number of players

    Returns:
        True if distribution triggered successfully
    """
    try:
        # Try to use the distribution script
        distribute_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "distribute_models_aria2.py"),
            "--model", str(model_path),
        ]

        result = subprocess.run(
            distribute_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )

        if result.returncode == 0:
            logger.info(f"Distribution triggered for {model_path.name}")
            return True
        else:
            logger.warning(f"Distribution returned non-zero: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Distribution timeout")
    except subprocess.SubprocessError as e:
        logger.error(f"Distribution error: {e}")

    return False


def main():
    parser = argparse.ArgumentParser(description="Recover models from backup locations")
    parser.add_argument(
        "--backup-host",
        nargs="+",
        default=DEFAULT_BACKUP_HOSTS,
        help="Backup hosts to scan (default: mac-studio)",
    )
    parser.add_argument(
        "--backup-path",
        nargs="+",
        default=DEFAULT_BACKUP_PATHS,
        help="Paths to scan on backup hosts",
    )
    parser.add_argument(
        "--ssh-key",
        help="SSH key to use for remote connections",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=1000,
        help="Minimum Elo to consider for recovery (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be recovered without copying",
    )
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Trigger distribution after copying",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models",
        help="Local directory to copy models to",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Scan all backup locations
    all_models: list[BackupModel] = []

    print("\n" + "=" * 70)
    print("SCANNING BACKUP LOCATIONS FOR HIGH-ELO MODELS")
    print("=" * 70 + "\n")

    for host in args.backup_host:
        for path in args.backup_path:
            logger.info(f"Scanning {host}:{path}...")
            models = scan_backup_location(host, path, args.ssh_key)
            all_models.extend(models)

    if not all_models:
        print("No models found in backup locations.")
        return

    # Filter and sort by Elo
    high_elo_models = [
        m for m in all_models
        if m.elo is not None and m.elo >= args.min_elo
    ]
    high_elo_models.sort(key=lambda m: m.elo or 0, reverse=True)

    print(f"\nFound {len(all_models)} total models, "
          f"{len(high_elo_models)} with Elo >= {args.min_elo}\n")

    print("-" * 70)
    print(f"{'Filename':<45} {'Elo':>6} {'Size':>10} {'Config':<12}")
    print("-" * 70)

    for model in high_elo_models:
        size_mb = model.size_bytes / (1024 * 1024)
        config = f"{model.board_type}_{model.num_players}p"
        elo_str = str(model.elo) if model.elo else "N/A"
        print(f"{model.filename:<45} {elo_str:>6} {size_mb:>8.1f}MB {config:<12}")

    print("-" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would recover the above models.")
        return

    # Copy and optionally distribute
    print("\n" + "=" * 70)
    print("RECOVERING MODELS")
    print("=" * 70 + "\n")

    recovered = 0
    distributed = 0

    for model in high_elo_models:
        host = args.backup_host[0]  # Use first host for now
        local_path = copy_model_to_local(
            host, model.remote_path, args.output_dir, args.ssh_key
        )

        if local_path:
            recovered += 1

            if args.distribute:
                if distribute_model(local_path, model.board_type, model.num_players):
                    distributed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {recovered} recovered, {distributed} distributed")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
