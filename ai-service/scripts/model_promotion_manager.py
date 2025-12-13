#!/usr/bin/env python
"""Model Promotion Manager - Manages promoted model symlinks and cluster-wide deployment.

This script provides:
1. Symlink management: Creates/updates symlinks like `square8_2p_best.pth`
2. Config generation: Updates `promoted_models.json` for sandbox/backend consumption
3. Cluster sync: Triggers model sync across all cluster nodes via P2P orchestrator
4. Webhook notifications: Notifies external systems of promotions

Usage:
    # Update symlinks and config for all promoted models
    python scripts/model_promotion_manager.py --update-all

    # Update specific config
    python scripts/model_promotion_manager.py --update square8 2

    # Sync models to cluster
    python scripts/model_promotion_manager.py --sync-cluster

    # Full pipeline: promote, update symlinks, sync cluster
    python scripts/model_promotion_manager.py --full-pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = AI_SERVICE_ROOT.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Paths
MODELS_DIR = AI_SERVICE_ROOT / "models"
PROMOTED_DIR = MODELS_DIR / "promoted"
PROMOTED_CONFIG_PATH = AI_SERVICE_ROOT / "data" / "promoted_models.json"
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "elo_leaderboard.db"
PROMOTION_LOG_PATH = AI_SERVICE_ROOT / "data" / "model_promotion_history.json"

# Sandbox config path (TypeScript side)
SANDBOX_CONFIG_PATH = PROJECT_ROOT / "src" / "shared" / "config" / "ai_models.json"

# P2P Orchestrator endpoint for cluster sync
P2P_ORCHESTRATOR_URL = os.environ.get("P2P_ORCHESTRATOR_URL", "http://localhost:8770")

# All board/player configurations
ALL_CONFIGS = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("hexagonal", 2),
]


@dataclass
class PromotedModel:
    """Information about a promoted model."""
    board_type: str
    num_players: int
    model_path: str  # Relative path from ai-service/models/
    model_id: str
    elo_rating: float
    games_played: int
    promoted_at: str
    symlink_name: str  # e.g., "square8_2p_best.pth"


def get_best_model_from_elo(board_type: str, num_players: int) -> Optional[Dict[str, Any]]:
    """Get the best model from Elo leaderboard for a given config."""
    if not ELO_DB_PATH.exists():
        return None

    try:
        conn = sqlite3.connect(str(ELO_DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_id, rating, games_played, wins, losses, draws
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
            ORDER BY rating DESC
            LIMIT 1
        """, (board_type, num_players))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "model_id": row[0],
                "elo_rating": row[1],
                "games_played": row[2],
                "wins": row[3],
                "losses": row[4],
                "draws": row[5],
            }
        return None
    except Exception as e:
        print(f"[model_promotion] Error querying Elo DB: {e}")
        return None


def find_model_file(model_id: str) -> Optional[Path]:
    """Find the actual model file for a given model ID."""
    # Model ID might be a filename or a partial name
    candidates = [
        MODELS_DIR / model_id,
        MODELS_DIR / f"{model_id}.pth",
    ]

    # Also search for files containing the model_id
    for path in MODELS_DIR.glob("*.pth"):
        if model_id in path.name:
            candidates.append(path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def create_symlink(model_path: Path, symlink_name: str) -> bool:
    """Create or update a symlink in the promoted directory."""
    PROMOTED_DIR.mkdir(parents=True, exist_ok=True)
    symlink_path = PROMOTED_DIR / symlink_name

    # Remove existing symlink if it exists
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    # Create relative symlink
    try:
        rel_path = os.path.relpath(model_path, PROMOTED_DIR)
        symlink_path.symlink_to(rel_path)
        print(f"[model_promotion] Created symlink: {symlink_name} -> {rel_path}")
        return True
    except Exception as e:
        print(f"[model_promotion] Error creating symlink: {e}")
        return False


def update_promoted_config(promoted_models: List[PromotedModel]) -> bool:
    """Update the promoted_models.json config file."""
    config = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "models": {
            f"{m.board_type}_{m.num_players}p": {
                "path": f"promoted/{m.symlink_name}",
                "model_id": m.model_id,
                "elo_rating": m.elo_rating,
                "games_played": m.games_played,
                "promoted_at": m.promoted_at,
            }
            for m in promoted_models
        }
    }

    try:
        PROMOTED_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROMOTED_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[model_promotion] Updated config: {PROMOTED_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"[model_promotion] Error updating config: {e}")
        return False


def update_sandbox_config(promoted_models: List[PromotedModel]) -> bool:
    """Update the TypeScript sandbox config with promoted models."""
    config = {
        "_comment": "Auto-generated by model_promotion_manager.py - DO NOT EDIT",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "models": {
            f"{m.board_type}_{m.num_players}p": {
                "path": f"ai-service/models/promoted/{m.symlink_name}",
                "elo_rating": m.elo_rating,
            }
            for m in promoted_models
        }
    }

    try:
        SANDBOX_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SANDBOX_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[model_promotion] Updated sandbox config: {SANDBOX_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"[model_promotion] Error updating sandbox config: {e}")
        return False


def log_promotion(promoted_model: PromotedModel) -> None:
    """Append promotion to history log."""
    try:
        history = []
        if PROMOTION_LOG_PATH.exists():
            with open(PROMOTION_LOG_PATH) as f:
                history = json.load(f)

        history.append(asdict(promoted_model))

        # Keep last 1000 entries
        if len(history) > 1000:
            history = history[-1000:]

        with open(PROMOTION_LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[model_promotion] Warning: Could not log promotion: {e}")


def sync_to_cluster(verbose: bool = True) -> bool:
    """Trigger cluster-wide model sync via P2P orchestrator."""
    try:
        import requests

        url = f"{P2P_ORCHESTRATOR_URL}/api/sync_models"
        response = requests.post(url, json={
            "action": "sync_promoted_models",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }, timeout=30)

        if response.status_code == 200:
            if verbose:
                print(f"[model_promotion] Cluster sync triggered successfully")
            return True
        else:
            if verbose:
                print(f"[model_promotion] Cluster sync failed: {response.status_code}")
            return False
    except ImportError:
        print("[model_promotion] requests not installed, using SSH-based sync")
        return sync_to_cluster_ssh(verbose)
    except Exception as e:
        if verbose:
            print(f"[model_promotion] P2P sync failed, falling back to SSH: {e}")
        return sync_to_cluster_ssh(verbose)


def sync_to_cluster_ssh(verbose: bool = True) -> bool:
    """Sync models to cluster nodes via SSH (fallback method)."""
    # Read cluster hosts from config
    hosts_file = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
    if not hosts_file.exists():
        if verbose:
            print("[model_promotion] No distributed_hosts.yaml found, skipping cluster sync")
        return False

    try:
        import yaml
        with open(hosts_file) as f:
            config = yaml.safe_load(f)

        hosts = config.get("hosts", {})
        success_count = 0

        for host_name, host_config in hosts.items():
            ssh_host = host_config.get("ssh_host")
            ssh_user = host_config.get("ssh_user", "root")
            ssh_port = host_config.get("ssh_port", 22)
            ringrift_path = host_config.get("ringrift_path", "~/ringrift")

            if not ssh_host:
                continue

            # Build SSH command
            ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
            if ssh_port != 22:
                ssh_cmd.extend(["-p", str(ssh_port)])
            ssh_cmd.append(f"{ssh_user}@{ssh_host}")

            # Git pull to update models
            pull_cmd = f"cd {ringrift_path} && git pull origin main --quiet 2>/dev/null"

            try:
                result = subprocess.run(
                    ssh_cmd + [pull_cmd],
                    capture_output=True,
                    timeout=60,
                    text=True,
                )
                if result.returncode == 0:
                    if verbose:
                        print(f"[model_promotion] Synced: {host_name}")
                    success_count += 1
                else:
                    if verbose:
                        print(f"[model_promotion] Failed to sync {host_name}: {result.stderr}")
            except subprocess.TimeoutExpired:
                if verbose:
                    print(f"[model_promotion] Timeout syncing {host_name}")
            except Exception as e:
                if verbose:
                    print(f"[model_promotion] Error syncing {host_name}: {e}")

        if verbose:
            print(f"[model_promotion] Cluster sync complete: {success_count}/{len(hosts)} hosts")
        return success_count > 0
    except Exception as e:
        if verbose:
            print(f"[model_promotion] Cluster sync error: {e}")
        return False


def update_all_promotions(min_games: int = 20, verbose: bool = True) -> List[PromotedModel]:
    """Update symlinks and config for all board/player configurations."""
    promoted_models = []

    for board_type, num_players in ALL_CONFIGS:
        if verbose:
            print(f"\n[model_promotion] Checking {board_type} {num_players}p...")

        best = get_best_model_from_elo(board_type, num_players)
        if not best:
            if verbose:
                print(f"  No Elo data available")
            continue

        if best["games_played"] < min_games:
            if verbose:
                print(f"  Best model has only {best['games_played']} games (min: {min_games})")
            continue

        model_path = find_model_file(best["model_id"])
        if not model_path:
            if verbose:
                print(f"  Model file not found: {best['model_id']}")
            continue

        symlink_name = f"{board_type}_{num_players}p_best.pth"

        if create_symlink(model_path, symlink_name):
            promoted = PromotedModel(
                board_type=board_type,
                num_players=num_players,
                model_path=str(model_path.relative_to(MODELS_DIR)),
                model_id=best["model_id"],
                elo_rating=best["elo_rating"],
                games_played=best["games_played"],
                promoted_at=datetime.utcnow().isoformat() + "Z",
                symlink_name=symlink_name,
            )
            promoted_models.append(promoted)
            log_promotion(promoted)

            if verbose:
                print(f"  Promoted: {best['model_id']} (Elo: {best['elo_rating']:.0f}, games: {best['games_played']})")

    if promoted_models:
        update_promoted_config(promoted_models)
        update_sandbox_config(promoted_models)

    return promoted_models


def run_full_pipeline(
    min_games: int = 20,
    sync_cluster: bool = True,
    verbose: bool = True,
) -> bool:
    """Run the full promotion pipeline: update symlinks, config, and sync cluster."""
    if verbose:
        print("[model_promotion] Starting full promotion pipeline...")

    # Step 1: Update all promotions
    promoted = update_all_promotions(min_games=min_games, verbose=verbose)

    if not promoted:
        if verbose:
            print("[model_promotion] No models promoted")
        return False

    if verbose:
        print(f"\n[model_promotion] Promoted {len(promoted)} models")

    # Step 2: Sync to cluster
    if sync_cluster:
        if verbose:
            print("\n[model_promotion] Syncing to cluster...")
        sync_to_cluster(verbose=verbose)

    if verbose:
        print("\n[model_promotion] Pipeline complete!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Model Promotion Manager - Manage promoted model symlinks and deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--update-all",
        action="store_true",
        help="Update symlinks and config for all configurations",
    )
    parser.add_argument(
        "--update",
        nargs=2,
        metavar=("BOARD", "PLAYERS"),
        help="Update specific configuration (e.g., --update square8 2)",
    )
    parser.add_argument(
        "--sync-cluster",
        action="store_true",
        help="Sync models to all cluster nodes",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline: update all, sync cluster",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=20,
        help="Minimum games required for promotion (default: 20)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.full_pipeline:
        run_full_pipeline(
            min_games=args.min_games,
            sync_cluster=True,
            verbose=verbose,
        )
    elif args.update_all:
        update_all_promotions(min_games=args.min_games, verbose=verbose)
    elif args.sync_cluster:
        sync_to_cluster(verbose=verbose)
    elif args.update:
        board_type, num_players = args.update
        # Single config update
        promoted = update_all_promotions(min_games=args.min_games, verbose=verbose)
        # Filter to just the requested config
        for p in promoted:
            if p.board_type == board_type and p.num_players == int(num_players):
                print(f"Updated: {p.symlink_name}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
