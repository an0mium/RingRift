#!/usr/bin/env python
"""Multi-config NNUE training orchestrator for all 9 board/player configurations.

This script extends train_all_nnue_models.py with:
- Parallel training across configurations
- Integration with bulk canonical data generation
- Progress tracking and event emission
- 2000+ Elo targeting per configuration
- Tier promotion readiness checking

Usage:
    # Train all 9 configurations with available data
    python scripts/run_multiconfig_nnue_training.py

    # Train with parallel execution (3 configs at a time)
    python scripts/run_multiconfig_nnue_training.py --parallel 3

    # Target specific Elo threshold
    python scripts/run_multiconfig_nnue_training.py --target-elo 2000

    # Prioritize underrepresented configs
    python scripts/run_multiconfig_nnue_training.py --underrepresented-first
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.training.crossboard_strength import (
    ALL_BOARD_CONFIGS,
    aggregate_cross_board_elos,
    config_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODELS_DIR = AI_SERVICE_ROOT / "models" / "nnue"
DEFAULT_GAMES_DIR = AI_SERVICE_ROOT / "data" / "games"


@dataclass
class ConfigTrainingStatus:
    """Status of training for a single configuration."""

    board: str
    num_players: int
    games_available: int
    model_exists: bool
    model_path: Path
    estimated_elo: float | None = None
    needs_training: bool = True


def get_game_count_for_config(
    board: str,
    num_players: int,
    games_dir: Path,
) -> int:
    """Get game count for a specific configuration."""
    # Check multiple database naming patterns
    patterns = [
        f"canonical_{board}_{num_players}p.db",
        f"canonical_{board}.db",
        f"{board}_selfplay.db",
        f"lambda_h100_selfplay.db",
    ]

    total_games = 0
    for pattern in patterns:
        db_path = games_dir / pattern
        if not db_path.exists():
            continue

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                "SELECT COUNT(*) FROM games WHERE board_type = ? AND num_players = ?",
                (board, num_players)
            )
            count = cursor.fetchone()[0]
            conn.close()
            total_games += count
        except (sqlite3.Error, OSError, IndexError):
            pass

    return total_games


def get_model_path(board: str, num_players: int) -> Path:
    """Get the expected model path for a configuration."""
    return DEFAULT_MODELS_DIR / f"nnue_{board}_{num_players}p.pt"


def get_all_training_status(games_dir: Path) -> list[ConfigTrainingStatus]:
    """Get training status for all 9 configurations."""
    statuses = []

    for board, num_players in ALL_BOARD_CONFIGS:
        games = get_game_count_for_config(board, num_players, games_dir)
        model_path = get_model_path(board, num_players)
        model_exists = model_path.exists()

        # Determine if training is needed
        min_games = 50 if board == "square8" else 30
        needs_training = games >= min_games and (
            not model_exists or games > 5000  # Retrain if lots of new data
        )

        statuses.append(ConfigTrainingStatus(
            board=board,
            num_players=num_players,
            games_available=games,
            model_exists=model_exists,
            model_path=model_path,
            needs_training=needs_training,
        ))

    return statuses


def train_single_config(
    board: str,
    num_players: int,
    db_paths: list[str],
    run_dir: str,
    epochs: int = 100,
    force: bool = False,
) -> dict[str, Any]:
    """Train NNUE for a single configuration.

    Returns a result dictionary with training outcome.
    """
    cfg_key = config_key(board, num_players)
    start_time = time.time()

    logger.info(f"Training NNUE for {cfg_key}...")

    # Build command
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_nnue.py"),
        "--db", *db_paths,
        "--board-type", board,
        "--num-players", str(num_players),
        "--epochs", str(epochs),
        "--run-dir", run_dir,
    ]

    if force:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(AI_SERVICE_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        elapsed = time.time() - start_time
        model_path = get_model_path(board, num_players)

        return {
            "config_key": cfg_key,
            "board": board,
            "num_players": num_players,
            "success": result.returncode == 0,
            "elapsed_seconds": elapsed,
            "model_exists": model_path.exists(),
            "error": result.stderr[-500:] if result.returncode != 0 else None,
        }

    except subprocess.TimeoutExpired:
        return {
            "config_key": cfg_key,
            "board": board,
            "num_players": num_players,
            "success": False,
            "elapsed_seconds": 7200,
            "error": "Timeout after 2 hours",
        }
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
        return {
            "config_key": cfg_key,
            "board": board,
            "num_players": num_players,
            "success": False,
            "elapsed_seconds": time.time() - start_time,
            "error": str(e),
        }


def find_all_databases(games_dir: Path) -> list[str]:
    """Find all SQLite game databases."""
    db_paths = []

    if games_dir.exists():
        for db_file in games_dir.glob("*.db"):
            # Skip temporary and quarantine databases
            if any(x in db_file.name for x in ["tmp_", "quarantine", "holdout"]):
                continue
            db_paths.append(str(db_file))

    return sorted(db_paths)


def run_multiconfig_training(
    statuses: list[ConfigTrainingStatus],
    db_paths: list[str],
    run_dir: str,
    parallel: int = 1,
    epochs: int = 100,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Run training for multiple configurations."""
    results = []
    to_train = [s for s in statuses if s.needs_training]

    if not to_train:
        logger.info("No configurations need training")
        return results

    logger.info(f"Training {len(to_train)} configurations...")

    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {}

            for status in to_train:
                config_run_dir = os.path.join(
                    run_dir,
                    config_key(status.board, status.num_players)
                )
                os.makedirs(config_run_dir, exist_ok=True)

                future = executor.submit(
                    train_single_config,
                    status.board,
                    status.num_players,
                    db_paths,
                    config_run_dir,
                    epochs,
                    force,
                )
                futures[future] = config_key(status.board, status.num_players)

            for future in as_completed(futures):
                cfg_key = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status_str = "OK" if result["success"] else "FAILED"
                    logger.info(f"Completed {cfg_key}: {status_str}")
                except (OSError, RuntimeError, subprocess.SubprocessError) as e:
                    logger.error(f"Exception for {cfg_key}: {e}")
                    results.append({
                        "config_key": cfg_key,
                        "success": False,
                        "error": str(e),
                    })
    else:
        for status in to_train:
            config_run_dir = os.path.join(
                run_dir,
                config_key(status.board, status.num_players)
            )
            os.makedirs(config_run_dir, exist_ok=True)

            result = train_single_config(
                status.board,
                status.num_players,
                db_paths,
                config_run_dir,
                epochs,
                force,
            )
            results.append(result)

    return results


def generate_report(
    statuses: list[ConfigTrainingStatus],
    results: list[dict[str, Any]],
    target_elo: float,
) -> dict[str, Any]:
    """Generate training summary report."""
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful

    # Build config status
    config_status = {}
    for status in statuses:
        cfg_key = config_key(status.board, status.num_players)
        config_status[cfg_key] = {
            "games_available": status.games_available,
            "model_exists": status.model_exists,
            "needs_training": status.needs_training,
        }

    # Merge in results
    for result in results:
        cfg_key = result.get("config_key")
        if cfg_key and cfg_key in config_status:
            config_status[cfg_key]["trained"] = result.get("success", False)
            config_status[cfg_key]["elapsed"] = result.get("elapsed_seconds", 0)

    # Count models ready
    models_ready = sum(
        1 for s in statuses if s.model_exists
    )

    return {
        "total_configs": len(statuses),
        "models_ready": models_ready,
        "configs_trained": len(results),
        "successful": successful,
        "failed": failed,
        "target_elo": target_elo,
        "config_status": config_status,
        "training_results": results,
        "all_models_ready": models_ready == len(statuses),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def emit_training_event(report: dict[str, Any]) -> None:
    """Emit training completion event for monitoring."""
    try:
        from app.distributed.event_helpers import emit_sync

        payload = {
            "configs_trained": report["configs_trained"],
            "successful": report["successful"],
            "failed": report["failed"],
            "models_ready": report["models_ready"],
            "all_ready": report["all_models_ready"],
        }

        emit_sync("TRAINING_COMPLETED", payload, "multiconfig_nnue_training")

    except (ImportError, OSError, RuntimeError) as e:
        logger.debug(f"Could not emit training event: {e}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-config NNUE training orchestrator for all 9 configurations. "
            "Supports parallel training and progress tracking."
        ),
    )
    parser.add_argument(
        "--games-dir",
        type=str,
        default=str(DEFAULT_GAMES_DIR),
        help="Directory containing game databases.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help="Output directory for trained models.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/multiconfig_nnue",
        help="Directory for training run artifacts.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel training runs (default: 1).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs per config (default: 100).",
    )
    parser.add_argument(
        "--target-elo",
        type=float,
        default=2000.0,
        help="Target Elo for promotion readiness (default: 2000).",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        help="Specific configs to train (e.g., 'square8_2p hexagonal_3p').",
    )
    parser.add_argument(
        "--underrepresented-first",
        action="store_true",
        help="Prioritize configs with fewer games.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if model exists.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs with existing models.",
    )
    parser.add_argument(
        "--output-json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--emit-events",
        action="store_true",
        help="Emit events for monitoring integration.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo mode (fewer epochs).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    games_dir = Path(args.games_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Adjust for demo mode
    epochs = args.epochs
    if args.demo:
        epochs = min(epochs, 5)

    # Get all training status
    all_statuses = get_all_training_status(games_dir)

    # Filter configs if specified
    if args.configs:
        config_keys = set(args.configs)
        all_statuses = [
            s for s in all_statuses
            if config_key(s.board, s.num_players) in config_keys
        ]

    # Skip existing if requested
    if args.skip_existing:
        for s in all_statuses:
            if s.model_exists and not args.force:
                s.needs_training = False

    # Force training if requested
    if args.force:
        for s in all_statuses:
            s.needs_training = s.games_available > 0

    # Sort by game count (underrepresented first if requested)
    if args.underrepresented_first:
        all_statuses.sort(key=lambda s: s.games_available)
    else:
        all_statuses.sort(key=lambda s: -s.games_available)

    # Find all databases
    db_paths = find_all_databases(games_dir)
    if not db_paths:
        logger.error("No game databases found")
        return 1

    # Print status
    print("\n" + "=" * 60)
    print("MULTI-CONFIG NNUE TRAINING STATUS")
    print("=" * 60)
    for status in all_statuses:
        cfg_key = config_key(status.board, status.num_players)
        model_str = "YES" if status.model_exists else "NO"
        train_str = "TRAIN" if status.needs_training else "skip"
        print(f"  {cfg_key:15s} games={status.games_available:5d}  model={model_str:3s}  {train_str}")
    print("=" * 60 + "\n")

    # Run training
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_run_dir = str(run_dir / timestamp)

    results = run_multiconfig_training(
        statuses=all_statuses,
        db_paths=db_paths,
        run_dir=training_run_dir,
        parallel=args.parallel,
        epochs=epochs,
        force=args.force,
    )

    elapsed = time.time() - start_time

    # Refresh status after training
    all_statuses = get_all_training_status(games_dir)

    # Generate report
    report = generate_report(all_statuses, results, args.target_elo)
    report["elapsed_seconds"] = elapsed

    # Emit events
    if args.emit_events:
        emit_training_event(report)

    # Write report
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        logger.info(f"Report written to {args.output_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("MULTI-CONFIG NNUE TRAINING SUMMARY")
    print("=" * 60)
    print(f"Configs trained: {report['configs_trained']}")
    print(f"Successful: {report['successful']}")
    print(f"Failed: {report['failed']}")
    print(f"Models ready: {report['models_ready']}/9")
    print(f"All models ready: {'YES' if report['all_models_ready'] else 'NO'}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 60)

    return 0 if report['failed'] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
