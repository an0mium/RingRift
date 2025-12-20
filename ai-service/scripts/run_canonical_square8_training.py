#!/usr/bin/env python
"""
Run a small, canonical Square-8 training job for v2 models.

This script is intended as a safe entrypoint for the first round of
training experiments on canonical Square-8 data (default:
`canonical_square8_2p.db`).

It assumes that:
  - Training data (.npz) has been exported from a canonical Square-8 DB
    (for example via `scripts/export_replay_dataset.py`), and
  - The canonical gate summary for the source DB has been refreshed
    via `scripts/generate_canonical_selfplay.py` and written to
    `data/games/db_health.<db>.json` with `canonical_ok=true`.

Usage (from ai-service/):

  PYTHONPATH=. python scripts/run_canonical_square8_training.py \\
    --data-path data/training/canonical_square8_2p.npz \\
    --save-path checkpoints/ringrift_v2_square8_demo.pth \\
    --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import BoardType  # type: ignore[import]
from app.training.config import TrainConfig, get_training_config_for_board  # type: ignore[import]
from app.training.train import train_model  # type: ignore[import]


def _resolve_db_path(db_path: Path) -> Path:
    if db_path.is_absolute():
        return db_path
    return (PROJECT_ROOT / db_path).resolve()


def _validate_source_db(db_path: Path) -> None:
    """Best-effort guard: ensure source DB is canonical Square-8 with clean gate.

    This mirrors the policy documented in TRAINING_DATA_REGISTRY.md and
    AI_IMPROVEMENT_PLAN.md §1.3.8 without attempting to parse the registry
    markdown at runtime.
    """
    db_path = _resolve_db_path(db_path)
    if not db_path.name.startswith("canonical_square8"):
        raise SystemExit(
            f"[canonical-training] Refusing to train from non-canonical DB: {db_path}\n"
            "Expected basename starting with 'canonical_square8'. Update "
            "TRAINING_DATA_REGISTRY.md and this script together if you intend to "
            "promote a new canonical DB."
        )
    if not db_path.exists():
        raise SystemExit(
            f"[canonical-training] Source DB not found: {db_path}\n"
            "Generate canonical Square-8 data before training."
        )

    summary_path = db_path.parent / f"db_health.{db_path.stem}.json"
    if not summary_path.exists():
        raise SystemExit(
            f"[canonical-training] {summary_path.name} not found.\n"
            "Run scripts/generate_canonical_selfplay.py --board square8 --num-games 0 "
            f"--db {db_path} --summary {summary_path} from ai-service/ before training."
        )

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"[canonical-training] Failed to parse {summary_path}: {exc}") from exc

    canonical_ok = bool(summary.get("canonical_ok"))
    parity_gate = summary.get("parity_gate") or {}
    sem = int(parity_gate.get("games_with_semantic_divergence", 1))
    struct = int(parity_gate.get("games_with_structural_issues", 1))
    total = int((summary.get("canonical_history") or {}).get("games_checked", 0))
    non_canonical = int((summary.get("canonical_history") or {}).get("non_canonical_games", 0))

    if not canonical_ok:
        raise SystemExit(
            "[canonical-training] Canonical gate summary is not clean:\n"
            f"  canonical_ok={canonical_ok}\n"
            f"  games_with_semantic_divergence={sem}\n"
            f"  games_with_structural_issues={struct}\n"
            f"  canonical_history.games_checked={total}\n"
            f"  canonical_history.non_canonical_games={non_canonical}\n"
            "Investigate TS↔Python replay parity and canonical history for the "
            "source DB and rerun the canonical gate before training."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small, canonical Square-8 training job on data exported "
            "from a canonical Square-8 DB."
        ),
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to .npz training data exported from a canonical Square-8 DB.",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Path to write the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--source-db",
        default="data/games/canonical_square8_2p.db",
        help=(
            "Source GameReplayDB used to generate data-path. Defaults to "
            "data/games/canonical_square8_2p.db and is validated against the "
            "canonical gate summary."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=("Use a very small configuration (few epochs, small batch) for " "smoke-testing and local experiments."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_path = args.data_path
    save_path = args.save_path

    # Validate source DB parity if provided.
    if args.source_db:
        _validate_source_db(Path(args.source_db))

    # Build a Square-8 training config from board presets.
    board_enum = BoardType.SQUARE8
    config = get_training_config_for_board(board_enum, TrainConfig())
    config.seed = args.seed

    # In demo mode, keep the run intentionally small/safe.
    if args.demo:
        config.epochs_per_iter = max(1, min(config.epochs_per_iter, 3))
        config.batch_size = max(32, min(config.batch_size, 128))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Single-device, non-distributed training; callers can wrap this in
    # torchrun/SLURM as needed for larger jobs.
    train_model(
        config=config,
        data_path=data_path,
        save_path=save_path,
        early_stopping_patience=0 if args.demo else 10,
        checkpoint_dir=os.path.dirname(save_path),
        checkpoint_interval=max(1, config.epochs_per_iter // 2),
        warmup_epochs=0,
        lr_scheduler="none",
        use_streaming=False,
        multi_player=False,
        num_players=2,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
