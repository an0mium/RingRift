#!/usr/bin/env python
"""Iterative training loop that closes the self-play cycle.

This script orchestrates the full training loop:
1. Generate selfplay data using current best model
2. Train on new data
3. Evaluate trained model vs champion
4. If better, promote and loop

This is the key to AlphaZero-style improvement: the model generates
its own training data, which is used to train a better model.

Usage:
    # Run one iteration
    python scripts/run_iterative_training_loop.py \
        --board square8 --num-players 2 \
        --games-per-iteration 500 \
        --iterations 1

    # Run continuous loop (stop with Ctrl+C)
    python scripts/run_iterative_training_loop.py \
        --board square8 --num-players 2 \
        --games-per-iteration 500 \
        --continuous

    # Run with custom thresholds
    python scripts/run_iterative_training_loop.py \
        --board square8 --num-players 2 \
        --games-per-iteration 1000 \
        --mcts-sims 800 \
        --elo-threshold 30 \
        --iterations 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_iterative_training_loop")


class IterativeTrainingLoop:
    """Orchestrates the iterative training loop."""

    def __init__(
        self,
        board_type: str,
        num_players: int,
        games_per_iteration: int = 500,
        mcts_sims: int = 800,
        parallel_batch: int = 8,
        elo_threshold: float = 30.0,
        epochs: int = 50,
        output_base: str = "data/iterative_loop",
    ):
        """Initialize the training loop.

        Args:
            board_type: Board type (square8, hex8, etc.)
            num_players: Number of players (2, 3, or 4)
            games_per_iteration: Games to generate per iteration
            mcts_sims: MCTS simulations per move
            parallel_batch: Batch size for parallel selfplay
            elo_threshold: Minimum Elo improvement for promotion
            epochs: Training epochs per iteration
            output_base: Base directory for outputs
        """
        self.board_type = board_type
        self.num_players = num_players
        self.games_per_iteration = games_per_iteration
        self.mcts_sims = mcts_sims
        self.parallel_batch = parallel_batch
        self.elo_threshold = elo_threshold
        self.epochs = epochs

        self._config_key = f"{board_type.lower()}_{num_players}p"
        self._output_base = Path(output_base) / self._config_key
        self._output_base.mkdir(parents=True, exist_ok=True)

        # State file for resuming
        self._state_file = self._output_base / "loop_state.json"
        self._state = self._load_state()

        # Paths
        self._models_dir = self._output_base / "models"
        self._data_dir = self._output_base / "data"
        self._logs_dir = self._output_base / "logs"
        for d in [self._models_dir, self._data_dir, self._logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"IterativeTrainingLoop initialized: config={self._config_key}, "
            f"games_per_iter={games_per_iteration}, mcts_sims={mcts_sims}"
        )

    def _load_state(self) -> dict:
        """Load loop state from file."""
        if self._state_file.exists():
            with open(self._state_file) as f:
                return json.load(f)
        return {
            "iteration": 0,
            "champion_model": None,
            "champion_elo": 1500.0,
            "total_games": 0,
            "history": [],
        }

    def _save_state(self) -> None:
        """Save loop state to file."""
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def run_iteration(self) -> dict:
        """Run a single iteration of the loop.

        Returns:
            Dictionary with iteration results.
        """
        iteration = self._state["iteration"] + 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")

        start_time = time.time()
        result = {
            "iteration": iteration,
            "start_time": datetime.now().isoformat(),
            "phases": {},
        }

        # Phase 1: Generate selfplay data
        logger.info("\n[Phase 1] Generating selfplay data...")
        selfplay_result = self._run_selfplay()
        result["phases"]["selfplay"] = selfplay_result

        if not selfplay_result.get("success"):
            logger.error("Selfplay failed, aborting iteration")
            return result

        # Phase 2: Train model
        logger.info("\n[Phase 2] Training model...")
        training_result = self._run_training(iteration)
        result["phases"]["training"] = training_result

        if not training_result.get("success"):
            logger.error("Training failed, aborting iteration")
            return result

        # Phase 3: Evaluate model
        logger.info("\n[Phase 3] Evaluating model...")
        eval_result = self._run_evaluation(training_result["model_path"])
        result["phases"]["evaluation"] = eval_result

        # Phase 4: Promote if better
        new_elo = eval_result.get("estimated_elo", self._state["champion_elo"])
        elo_delta = new_elo - self._state["champion_elo"]

        logger.info(f"\nElo delta: {elo_delta:+.1f}")

        if elo_delta >= self.elo_threshold:
            logger.info(f"PROMOTING new model (Elo: {new_elo:.0f})")
            self._promote_model(training_result["model_path"], new_elo)
            result["promoted"] = True
        else:
            logger.info(f"Not promoting (threshold: {self.elo_threshold})")
            result["promoted"] = False

        # Update state
        self._state["iteration"] = iteration
        self._state["total_games"] += selfplay_result.get("games_generated", 0)
        self._state["history"].append({
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "games": selfplay_result.get("games_generated", 0),
            "val_accuracy": training_result.get("val_accuracy"),
            "elo_delta": elo_delta,
            "promoted": result["promoted"],
        })
        self._save_state()

        elapsed = time.time() - start_time
        result["elapsed_time"] = elapsed
        result["success"] = True

        logger.info(f"\nIteration {iteration} complete in {elapsed:.1f}s")
        logger.info(f"Champion Elo: {self._state['champion_elo']:.0f}")
        logger.info(f"Total games: {self._state['total_games']}")

        return result

    def run_continuous(self, max_iterations: int | None = None) -> None:
        """Run the loop continuously.

        Args:
            max_iterations: Maximum iterations (None for infinite).
        """
        iteration_count = 0

        while max_iterations is None or iteration_count < max_iterations:
            try:
                result = self.run_iteration()
                iteration_count += 1

                if not result.get("success"):
                    logger.error("Iteration failed, retrying in 60s...")
                    time.sleep(60)
                    continue

                # Brief pause between iterations
                time.sleep(5)

            except KeyboardInterrupt:
                logger.info("\nLoop stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)

    def _run_selfplay(self) -> dict:
        """Run selfplay to generate training data."""
        output_dir = self._data_dir / f"iter_{self._state['iteration'] + 1}"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/run_parallel_mcts_selfplay.py",
            "--board", self.board_type,
            "--num-players", str(self.num_players),
            "--num-games", str(self.games_per_iteration),
            "--parallel-batch", str(self.parallel_batch),
            "--mcts-sims", str(self.mcts_sims),
            "--output-dir", str(output_dir),
        ]

        # Add model if we have a champion
        if self._state["champion_model"]:
            cmd.extend(["--model-path", self._state["champion_model"]])

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=str(Path(__file__).parent.parent),
            )

            if result.returncode != 0:
                logger.error(f"Selfplay failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

            # Parse stats
            stats_file = output_dir / "stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                return {
                    "success": True,
                    "games_generated": stats.get("total_games", 0),
                    "games_per_second": stats.get("games_per_second", 0),
                    "output_dir": str(output_dir),
                }

            return {"success": True, "output_dir": str(output_dir)}

        except subprocess.TimeoutExpired:
            logger.error("Selfplay timed out")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Selfplay error: {e}")
            return {"success": False, "error": str(e)}

    def _run_training(self, iteration: int) -> dict:
        """Run training on generated data."""
        from app.models import BoardType

        # Map board type
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hex": BoardType.HEXAGONAL,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type_enum = board_type_map.get(self.board_type.lower(), BoardType.SQUARE8)

        model_path = self._models_dir / f"model_iter_{iteration}.pt"

        # Find all data from this and previous iterations
        data_dirs = sorted(self._data_dir.glob("iter_*"))

        try:
            # Use NNUE training
            from app.ai.nnue import RingRiftNNUE, create_nnue_model

            # Find game data files
            game_files = []
            for data_dir in data_dirs:
                games_file = data_dir / "games.jsonl"
                if games_file.exists():
                    game_files.append(games_file)

            if not game_files:
                logger.warning("No game data found for training")
                return {"success": False, "error": "no_data"}

            logger.info(f"Training on data from {len(game_files)} iterations")

            # Simple training using NNUE trainer
            # In practice, this would use the full training pipeline
            model = create_nnue_model(board_type_enum, self.num_players)

            # TODO: Implement actual training on game data
            # For now, just save the model as placeholder
            import torch
            torch.save({
                "model_state_dict": model.state_dict(),
                "iteration": iteration,
                "config": {
                    "board_type": self.board_type,
                    "num_players": self.num_players,
                },
            }, model_path)

            logger.info(f"Model saved to {model_path}")

            return {
                "success": True,
                "model_path": str(model_path),
                "val_accuracy": None,  # Would be filled by actual training
            }

        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def _run_evaluation(self, model_path: str) -> dict:
        """Evaluate model against baseline."""
        # Quick evaluation using win rate against heuristic
        try:
            # Run a quick tournament
            cmd = [
                sys.executable,
                "scripts/evaluate_nnue.py",
                "--model", model_path,
                "--board", self.board_type,
                "--num-players", str(self.num_players),
                "--games", "20",
                "--depth", "2",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(Path(__file__).parent.parent),
            )

            if result.returncode != 0:
                logger.warning(f"Evaluation failed: {result.stderr}")
                # Fall back to estimating from training loss
                return {"success": True, "estimated_elo": self._state["champion_elo"]}

            # Parse win rate from output
            for line in result.stdout.split("\n"):
                if "win rate" in line.lower():
                    try:
                        # Extract percentage
                        import re
                        match = re.search(r"(\d+\.?\d*)%", line)
                        if match:
                            win_rate = float(match.group(1)) / 100
                            # Convert to Elo estimate
                            # 60% win rate ≈ +70 Elo
                            elo_delta = 400 * (win_rate - 0.5)
                            estimated_elo = self._state["champion_elo"] + elo_delta
                            return {
                                "success": True,
                                "win_rate": win_rate,
                                "estimated_elo": estimated_elo,
                            }
                    except Exception:
                        pass

            return {"success": True, "estimated_elo": self._state["champion_elo"]}

        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            return {"success": True, "estimated_elo": self._state["champion_elo"]}

    def _promote_model(self, model_path: str, new_elo: float) -> None:
        """Promote model to champion."""
        champion_path = self._models_dir / f"champion_{self._config_key}.pt"

        # Copy model
        shutil.copy(model_path, champion_path)

        # Update state
        self._state["champion_model"] = str(champion_path)
        self._state["champion_elo"] = new_elo

        # Also copy to NNUE models directory for auto-detection
        nnue_path = Path(__file__).parent.parent / "models" / "nnue" / f"nnue_{self._config_key}.pt"
        nnue_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, nnue_path)

        logger.info(f"Champion promoted: {champion_path}")
        logger.info(f"NNUE model updated: {nnue_path}")

        # Emit promotion event
        try:
            from app.coordination.event_router import emit, DataEventType

            emit(
                DataEventType.MODEL_PROMOTED,
                {
                    "model_path": str(champion_path),
                    "config_key": self._config_key,
                    "elo": new_elo,
                    "iteration": self._state["iteration"],
                },
            )
            logger.info("MODEL_PROMOTED event emitted")
        except Exception as e:
            logger.debug(f"Could not emit promotion event: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run iterative training loop"
    )
    parser.add_argument(
        "--board", "-b",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex8", "hex", "hexagonal"],
        help="Board type",
    )
    parser.add_argument(
        "--num-players", "-p",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=500,
        help="Games to generate per iteration",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=800,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--parallel-batch",
        type=int,
        default=8,
        help="Parallel batch size for selfplay",
    )
    parser.add_argument(
        "--elo-threshold",
        type=float,
        default=30.0,
        help="Minimum Elo improvement for promotion",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per iteration",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously until stopped",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="data/iterative_loop",
        help="Base output directory",
    )

    args = parser.parse_args()

    loop = IterativeTrainingLoop(
        board_type=args.board,
        num_players=args.num_players,
        games_per_iteration=args.games_per_iteration,
        mcts_sims=args.mcts_sims,
        parallel_batch=args.parallel_batch,
        elo_threshold=args.elo_threshold,
        epochs=args.epochs,
        output_base=args.output_base,
    )

    if args.continuous:
        logger.info("Starting continuous training loop (Ctrl+C to stop)")
        loop.run_continuous()
    else:
        logger.info(f"Running {args.iterations} iteration(s)")
        for i in range(args.iterations):
            result = loop.run_iteration()
            if not result.get("success"):
                logger.error(f"Iteration {i+1} failed")
                break

    logger.info("Loop complete")


if __name__ == "__main__":
    main()
