#!/usr/bin/env python3
"""
Tier-based training pipeline for AI difficulty ladder.

This script trains AI models for specific difficulty tiers (D2, D4, D6, D8, D9, D10)
using different training modes based on the tier configuration.

Usage:
    python scripts/run_tier_training_pipeline.py --tier D6 --board square8 --num-players 2
    python scripts/run_tier_training_pipeline.py --tier D8 --config config/tier_training_pipeline.square8_2p.json

For gating after training, see: python scripts/run_full_tier_gating.py --help

Training modes by tier (configurable):
    D2: heuristic_cmaes - CMA-ES optimization of heuristic weights
    D4: search_persona - MCTS with tuned parameters
    D6-D10: neural - Neural network training with increasing strength
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AI models for a specific difficulty tier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier",
        required=True,
        choices=["D2", "D4", "D6", "D8", "D9", "D10"],
        help="Difficulty tier to train (D2=easiest, D10=hardest).",
    )
    parser.add_argument(
        "--board",
        default="square8",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type (default: square8).",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to tier training config JSON (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "tier_training",
        help="Output directory for training artifacts.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to training data NPZ file (for neural tiers).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs for neural tiers (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for neural training (default: 256).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with minimal training for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123).",
    )
    return parser.parse_args(argv)


def load_tier_config(config_path: Path | None, tier: str) -> dict[str, Any]:
    """Load tier-specific configuration."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            full_config = json.load(f)
        return full_config.get("tiers", {}).get(tier, {})

    # Default configurations by tier
    defaults = {
        "D2": {"training": {"mode": "heuristic_cmaes"}},
        "D4": {"training": {"mode": "search_persona"}},
        "D6": {"training": {"mode": "neural"}},
        "D8": {"training": {"mode": "neural"}},
        "D9": {"training": {"mode": "neural"}},
        "D10": {"training": {"mode": "neural"}},
    }
    return defaults.get(tier, {"training": {"mode": "neural"}})


def run_neural_training(args: argparse.Namespace, tier_config: dict) -> dict[str, Any]:
    """Run neural network training for the tier."""
    # Build training command
    cmd = [
        sys.executable, "-m", "app.training.train",
        "--board-type", args.board,
        "--num-players", str(args.num_players),
        "--epochs", str(args.epochs if not args.demo else 2),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
    ]

    if args.data_path:
        cmd.extend(["--data-path", str(args.data_path)])

    # Add tier-specific save path
    run_dir = args.output_dir / f"{args.tier}_{args.board}_{args.num_players}p_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--save-path", str(run_dir / "model.pth")])

    print(f"Running neural training for tier {args.tier}...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    return {
        "mode": "neural",
        "run_dir": str(run_dir),
        "exit_code": result.returncode,
        "success": result.returncode == 0,
    }


def run_heuristic_cmaes(args: argparse.Namespace, tier_config: dict) -> dict[str, Any]:
    """Run CMA-ES optimization for heuristic weights."""
    print(f"Running CMA-ES optimization for tier {args.tier}...")

    # Check if CMA-ES script exists
    cmaes_script = PROJECT_ROOT / "scripts" / "run_cmaes_optimization.py"
    if not cmaes_script.exists():
        print(f"Warning: {cmaes_script} not found, using default heuristic weights")
        return {
            "mode": "heuristic_cmaes",
            "success": True,
            "note": "Using default weights (CMA-ES script not available)",
        }

    run_dir = args.output_dir / f"{args.tier}_{args.board}_{args.num_players}p_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(cmaes_script),
        "--board", args.board,
        "--num-players", str(args.num_players),
        "--output-dir", str(run_dir),
    ]

    if args.demo:
        cmd.extend(["--generations", "5", "--population-size", "10"])

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    return {
        "mode": "heuristic_cmaes",
        "run_dir": str(run_dir),
        "exit_code": result.returncode,
        "success": result.returncode == 0,
    }


def run_search_persona(args: argparse.Namespace, tier_config: dict) -> dict[str, Any]:
    """Run search persona training (MCTS parameter tuning)."""
    print(f"Running search persona training for tier {args.tier}...")

    run_dir = args.output_dir / f"{args.tier}_{args.board}_{args.num_players}p_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # For now, create a placeholder - search persona tuning would go here
    persona_config = {
        "tier": args.tier,
        "mcts_simulations": 100 if args.tier == "D4" else 200,
        "exploration_constant": 1.4,
        "temperature": 0.5,
    }

    with open(run_dir / "search_persona.json", "w") as f:
        json.dump(persona_config, f, indent=2)

    return {
        "mode": "search_persona",
        "run_dir": str(run_dir),
        "success": True,
        "persona_config": persona_config,
    }


def main(argv: list[str] | None = None) -> int:
    """Main entry point for tier training pipeline."""
    args = parse_args(argv)

    print(f"=" * 60)
    print(f"Tier Training Pipeline")
    print(f"=" * 60)
    print(f"Tier: {args.tier}")
    print(f"Board: {args.board}")
    print(f"Players: {args.num_players}")
    print(f"Demo mode: {args.demo}")
    print(f"=" * 60)

    # Load tier config
    tier_config = load_tier_config(args.config, args.tier)
    training_mode = tier_config.get("training", {}).get("mode", "neural")

    print(f"Training mode: {training_mode}")

    # Run appropriate training
    if training_mode == "heuristic_cmaes":
        result = run_heuristic_cmaes(args, tier_config)
    elif training_mode == "search_persona":
        result = run_search_persona(args, tier_config)
    else:  # neural
        result = run_neural_training(args, tier_config)

    # Create training report
    candidate_id = f"{args.tier}_{args.board}_{args.num_players}p_{uuid.uuid4().hex[:8]}"
    training_report = {
        "tier": args.tier,
        "candidate_id": candidate_id,
        "board": args.board,
        "num_players": args.num_players,
        "training_mode": training_mode,
        "demo": args.demo,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }

    # Save training report
    if "run_dir" in result:
        report_path = Path(result["run_dir"]) / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(training_report, f, indent=2)
        print(f"\nTraining report saved to: {report_path}")

    print(f"\n{'=' * 60}")
    print(f"Training {'completed successfully' if result.get('success') else 'failed'}")
    print(f"Candidate ID: {candidate_id}")
    print(f"{'=' * 60}")

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
