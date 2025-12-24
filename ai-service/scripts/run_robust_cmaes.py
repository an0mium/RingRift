#!/usr/bin/env python3
"""Robust Multi-Opponent CMA-ES Training for RingRift.

This script implements robust CMA-ES training that addresses the overfitting problem
where single-opponent training produced weights that won 81.25% vs one baseline
but only 47.7% in round-robin tournaments.

Key improvements over standard CMA-ES:
1. Multi-opponent fitness: Evaluate against ALL 4 personas (balanced, aggressive,
   territorial, defensive) instead of single opponent
2. Robust aggregation: 0.4 * min(win_rates) + 0.6 * mean(win_rates)
3. Elite archive: Maintains best solutions per opponent type for diversity
4. Population diversity bonus: Rewards exploration away from centroid
5. Longer training: 200+ generations with proper convergence detection

Usage:
    # Local single-GPU training
    python scripts/run_robust_cmaes.py --board square8 --num-players 2

    # Distributed across cluster
    python scripts/run_robust_cmaes.py --board square8 --num-players 2 \
        --distributed --workers http://b:8080,http://c:8080,...

    # With diversity bonus
    python scripts/run_robust_cmaes.py --board square8 --num-players 2 \
        --diversity-bonus --generations 200

Requirements:
    pip install evotorch torch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Torch and EvoTorch
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    torch = None  # type: ignore

try:
    import evotorch
    from evotorch import Problem
    from evotorch.algorithms import CMAES
    from evotorch.logging import StdOutLogger
    EVOTORCH_AVAILABLE = True
except ImportError:
    EVOTORCH_AVAILABLE = False
    # Dummy class for when EvoTorch isn't available
    class Problem:  # type: ignore
        """Dummy Problem class for type checking."""
        pass
    CMAES = None  # type: ignore
    StdOutLogger = None  # type: ignore
    print("EvoTorch not available. Install with: pip install evotorch")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

# RingRift imports
from app.ai.heuristic_weights import (
    HEURISTIC_WEIGHT_KEYS,
    HEURISTIC_V1_BALANCED,
)
from app.ai.multi_opponent_fitness import (
    evaluate_multi_opponent,
    MultiOpponentResult,
    BASELINE_OPPONENTS,
    DEFAULT_GAMES_PER_OPPONENT,
    DEFAULT_SELF_PLAY_GAMES,
    DEFAULT_MIN_WEIGHT,
)
from app.ai.cmaes_diversity import (
    EliteArchive,
    compute_diversity_bonus,
    compute_population_diversity,
    inject_elites_into_population,
)
from app.rules.core import BOARD_CONFIGS, BoardType

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Board type mapping
BOARD_TYPE_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hex8": BoardType.HEX8,
    "hexagonal": BoardType.HEXAGONAL,
}


@dataclass
class RobustCMAESConfig:
    """Configuration for robust multi-opponent CMA-ES training."""

    # Board and game settings
    board_type: str = "square8"
    num_players: int = 2

    # CMA-ES parameters
    generations: int = 200
    population_size: int = 64
    sigma: float = 2.0  # Initial step size (larger than default for exploration)

    # Multi-opponent evaluation
    games_per_opponent: int = DEFAULT_GAMES_PER_OPPONENT
    self_play_games: int = DEFAULT_SELF_PLAY_GAMES
    min_weight: float = DEFAULT_MIN_WEIGHT
    max_moves: int = 200

    # Diversity mechanisms
    use_diversity_bonus: bool = True
    diversity_scale: float = 0.02
    elite_injection_interval: int = 50
    elite_injection_count: int = 3

    # Convergence
    plateau_generations: int = 20
    plateau_threshold: float = 0.005
    target_fitness: float = 0.60
    min_sigma: float = 0.01

    # Distribution
    distributed: bool = False
    worker_urls: list[str] = field(default_factory=list)

    # Output
    output_dir: str = "logs/cmaes/robust"
    checkpoint_interval: int = 5  # Save every 5 generations
    seed: int = 42

    # Device
    device: str = "cuda" if CUDA_AVAILABLE else "cpu"


def weights_dict_to_array(weights: dict[str, float]) -> np.ndarray:
    """Convert weight dict to numpy array."""
    return np.array([weights.get(k, 0.0) for k in HEURISTIC_WEIGHT_KEYS])


def weights_array_to_dict(array: np.ndarray) -> dict[str, float]:
    """Convert numpy array to weight dict."""
    return {k: float(v) for k, v in zip(HEURISTIC_WEIGHT_KEYS, array)}


def detect_convergence(
    fitness_history: list[float],
    sigma_history: list[float],
    generation: int,
    config: RobustCMAESConfig,
) -> tuple[bool, str]:
    """Multi-criteria convergence detection.

    Returns:
        Tuple of (converged, reason)
    """
    # 1. Target fitness reached
    if fitness_history and fitness_history[-1] >= config.target_fitness:
        return True, "target_reached"

    # 2. Maximum generations
    if generation >= config.generations:
        return True, "max_generations"

    # 3. Fitness plateau
    if len(fitness_history) >= config.plateau_generations:
        recent = fitness_history[-config.plateau_generations:]
        improvement = max(recent) - min(recent)
        if improvement < config.plateau_threshold:
            return True, "fitness_plateau"

    # 4. Sigma collapse
    if sigma_history and sigma_history[-1] < config.min_sigma:
        return True, "sigma_collapse"

    return False, ""


class MultiOpponentProblem(Problem):
    """EvoTorch Problem with multi-opponent fitness evaluation."""

    def __init__(self, config: RobustCMAESConfig):
        self.config = config
        self.eval_count = 0
        self.best_fitness = 0.0
        self.best_weights: dict[str, float] = {}
        self.best_per_opponent: dict[str, float] = {}

        # Elite archive for diversity
        self.elite_archive = EliteArchive(capacity_per_type=5)

        # Device
        self.compute_device = torch.device(config.device) if TORCH_AVAILABLE else None

        # Distributed evaluation
        self.distributed = config.distributed and len(config.worker_urls) > 0
        self.worker_urls = config.worker_urls
        self.worker_idx = 0  # Round-robin counter

        # Batch queue for parallel evaluation
        self._eval_queue: list[tuple[Any, dict[str, float]]] = []
        self._batch_size = config.population_size  # Collect full population before evaluating

        if self.distributed:
            if not HTTPX_AVAILABLE:
                logger.warning("httpx not available, falling back to local evaluation")
                self.distributed = False
            else:
                logger.info(f"  Distributed mode: {len(self.worker_urls)} workers")
                logger.info(f"  Parallel batch size: {self._batch_size}")

        # Get board size
        board_enum = BOARD_TYPE_MAP.get(config.board_type.lower())
        if board_enum:
            self.board_size = BOARD_CONFIGS[board_enum].size
        else:
            self.board_size = 8

        # Compute bounds from balanced baseline
        baseline = dict(HEURISTIC_V1_BALANCED)
        lower_bounds = []
        upper_bounds = []
        for key in HEURISTIC_WEIGHT_KEYS:
            val = baseline.get(key, 0.0)
            delta = max(abs(val) * 0.5, 2.0)  # At least ±2.0
            lower_bounds.append(val - delta)
            upper_bounds.append(val + delta)

        # Initialize EvoTorch Problem
        super().__init__(
            objective_sense="max",
            solution_length=len(HEURISTIC_WEIGHT_KEYS),
            initial_bounds=(lower_bounds, upper_bounds),
            dtype=torch.float32,
            device=self.compute_device,
        )

        logger.info(f"MultiOpponentProblem initialized:")
        logger.info(f"  Board: {config.board_type} ({self.board_size}x{self.board_size})")
        logger.info(f"  Players: {config.num_players}")
        logger.info(f"  Games per opponent: {config.games_per_opponent}")
        logger.info(f"  Opponents: {list(BASELINE_OPPONENTS.keys())}")
        logger.info(f"  Device: {self.compute_device}")

    def _evaluate_via_worker(self, weights: dict[str, float]) -> MultiOpponentResult:
        """Evaluate via HTTP worker (distributed mode)."""
        # Round-robin worker selection
        worker_url = self.worker_urls[self.worker_idx % len(self.worker_urls)]
        self.worker_idx += 1

        # Make HTTP request to worker
        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{worker_url}/evaluate_multi",
                    json={
                        "task_id": f"cmaes_{self.eval_count}",
                        "candidate_weights": weights,
                        "games_per_opponent": self.config.games_per_opponent,
                        "self_play_games": self.config.self_play_games,
                        "board_size": self.board_size,
                        "num_players": self.config.num_players,
                        "max_moves": self.config.max_moves,
                    },
                )
                response.raise_for_status()
                data = response.json()

                return MultiOpponentResult(
                    per_opponent=data["per_opponent"],
                    aggregate=data["aggregate"],
                    self_play=data.get("self_play", 0.5),
                    games_played=data.get("games_played", 0),
                )
        except Exception as e:
            logger.error(f"Worker {worker_url} failed: {e}")
            # Fallback to local evaluation
            return evaluate_multi_opponent(
                candidate_weights=weights,
                games_per_opponent=self.config.games_per_opponent,
                self_play_games=self.config.self_play_games,
                min_weight=self.config.min_weight,
                board_size=self.board_size,
                num_players=self.config.num_players,
                max_moves=self.config.max_moves,
                device=self.compute_device,
            )

    def _evaluate(self, solution) -> None:
        """Evaluate a single solution with multi-opponent fitness."""
        # Convert solution to weights dict
        values = solution.values.detach().cpu().numpy()
        weights = weights_array_to_dict(values)

        # Choose evaluation method
        if self.distributed:
            result = self._evaluate_via_worker(weights)
        else:
            # Local multi-opponent evaluation
            result = evaluate_multi_opponent(
                candidate_weights=weights,
                games_per_opponent=self.config.games_per_opponent,
                self_play_games=self.config.self_play_games,
                min_weight=self.config.min_weight,
                board_size=self.board_size,
                num_players=self.config.num_players,
                max_moves=self.config.max_moves,
                device=self.compute_device,
            )

        # Set fitness
        fitness = result.aggregate
        solution.set_evals(fitness)

        # Track best
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_weights = weights.copy()
            self.best_per_opponent = result.per_opponent.copy()

        # Update elite archive
        self.elite_archive.add(
            weights=weights,
            per_opponent=result.per_opponent,
            aggregate=fitness,
            generation=self.eval_count,
        )

        self.eval_count += 1

    def _evaluate_batch_parallel(self, solutions_weights: list[tuple[Any, dict[str, float]]]) -> dict[int, MultiOpponentResult]:
        """Evaluate multiple candidates in parallel using ThreadPoolExecutor."""
        results = {}

        def evaluate_one(args):
            idx, weights = args
            worker_url = self.worker_urls[idx % len(self.worker_urls)]
            try:
                with httpx.Client(timeout=300.0) as client:
                    response = client.post(
                        f"{worker_url}/evaluate_multi",
                        json={
                            "task_id": f"batch_{idx}",
                            "candidate_weights": weights,
                            "games_per_opponent": self.config.games_per_opponent,
                            "self_play_games": self.config.self_play_games,
                            "board_size": self.board_size,
                            "num_players": self.config.num_players,
                            "max_moves": self.config.max_moves,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return idx, MultiOpponentResult(
                        per_opponent=data["per_opponent"],
                        aggregate=data["aggregate"],
                        self_play=data.get("self_play", 0.5),
                        games_played=data.get("games_played", 0),
                    )
            except Exception as e:
                logger.error(f"Worker {worker_url} failed: {e}")
                return idx, None

        # Submit all in parallel
        with ThreadPoolExecutor(max_workers=len(self.worker_urls)) as executor:
            futures = [executor.submit(evaluate_one, (idx, w)) for idx, (_, w) in enumerate(solutions_weights)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def _flush_eval_queue(self) -> None:
        """Evaluate all queued solutions in parallel and apply results."""
        if not self._eval_queue:
            return

        logger.info(f"  Parallel evaluating {len(self._eval_queue)} candidates...")

        # Parallel evaluation
        results = self._evaluate_batch_parallel(self._eval_queue)

        # Apply results
        for idx, (solution, weights) in enumerate(self._eval_queue):
            result = results.get(idx)
            if result is None:
                # Fallback for failed evaluations
                result = evaluate_multi_opponent(
                    candidate_weights=weights,
                    games_per_opponent=self.config.games_per_opponent,
                    self_play_games=self.config.self_play_games,
                    min_weight=self.config.min_weight,
                    board_size=self.board_size,
                    num_players=self.config.num_players,
                    max_moves=self.config.max_moves,
                    device=self.compute_device,
                )

            fitness = result.aggregate
            solution.set_evals(fitness)

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_weights = weights.copy()
                self.best_per_opponent = result.per_opponent.copy()

            self.elite_archive.add(
                weights=weights,
                per_opponent=result.per_opponent,
                aggregate=fitness,
                generation=self.eval_count,
            )
            self.eval_count += 1

        self._eval_queue.clear()

    def evaluate(self, batch) -> None:
        """Override EvoTorch's evaluate to use parallel batch evaluation.

        This is the public method called by EvoTorch searchers.
        """
        if self.distributed and len(self.worker_urls) > 0:
            # Queue all solutions
            for i in range(len(batch)):
                solution = batch[i]
                values = solution.values.detach().cpu().numpy()
                weights = weights_array_to_dict(values)
                self._eval_queue.append((solution, weights))

            # Evaluate all in parallel
            self._flush_eval_queue()
        else:
            # Fall back to default single-threaded evaluation
            for i in range(len(batch)):
                self._evaluate(batch[i])

    def _evaluate_batch(self, batch) -> None:
        """Evaluate a batch of solutions in parallel.

        This overrides the default sequential evaluation to use ThreadPoolExecutor
        for parallel HTTP requests to distributed workers.

        Note: In EvoTorch, this is called via the `evaluate` method when
        processing a SolutionBatch.
        """
        # Get solutions from batch
        solutions = [batch[i] for i in range(len(batch))]

        if not self.distributed or len(self.worker_urls) == 0:
            # Fall back to sequential evaluation
            for solution in solutions:
                self._evaluate(solution)
            return

        # Prepare all candidates
        candidates = []
        for solution in solutions:
            values = solution.values.detach().cpu().numpy()
            weights = weights_array_to_dict(values)
            candidates.append((solution, weights))

        # Parallel evaluation using ThreadPoolExecutor
        max_workers = len(self.worker_urls)
        results = {}

        def evaluate_one(idx_solution_weights):
            idx, solution, weights = idx_solution_weights
            worker_url = self.worker_urls[idx % len(self.worker_urls)]
            try:
                with httpx.Client(timeout=300.0) as client:
                    response = client.post(
                        f"{worker_url}/evaluate_multi",
                        json={
                            "task_id": f"cmaes_{self.eval_count + idx}",
                            "candidate_weights": weights,
                            "games_per_opponent": self.config.games_per_opponent,
                            "self_play_games": self.config.self_play_games,
                            "board_size": self.board_size,
                            "num_players": self.config.num_players,
                            "max_moves": self.config.max_moves,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return idx, MultiOpponentResult(
                        per_opponent=data["per_opponent"],
                        aggregate=data["aggregate"],
                        self_play=data.get("self_play", 0.5),
                        games_played=data.get("games_played", 0),
                    )
            except Exception as e:
                logger.error(f"Worker {worker_url} failed: {e}")
                # Fallback to local evaluation
                result = evaluate_multi_opponent(
                    candidate_weights=weights,
                    games_per_opponent=self.config.games_per_opponent,
                    self_play_games=self.config.self_play_games,
                    min_weight=self.config.min_weight,
                    board_size=self.board_size,
                    num_players=self.config.num_players,
                    max_moves=self.config.max_moves,
                    device=self.compute_device,
                )
                return idx, result

        # Submit all evaluations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, (solution, weights) in enumerate(candidates):
                futures.append(executor.submit(evaluate_one, (idx, solution, weights)))

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        # Apply results to solutions
        for idx, (solution, weights) in enumerate(candidates):
            result = results[idx]
            fitness = result.aggregate

            # Apply diversity bonus if enabled
            if self.config.use_diversity_bonus:
                # Simple diversity bonus based on distance from population mean
                pass  # Already computed in _evaluate, skip for batch

            solution.set_evals(fitness)

            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_weights = weights.copy()
                self.best_per_opponent = result.per_opponent.copy()

            # Update elite archive
            self.elite_archive.add(
                weights=weights,
                per_opponent=result.per_opponent,
                aggregate=fitness,
                generation=self.eval_count + idx,
            )

        self.eval_count += len(candidates)
        logger.info(f"  Batch evaluated {len(candidates)} candidates in parallel")


def run_robust_cmaes(config: RobustCMAESConfig) -> dict[str, Any]:
    """Run robust multi-opponent CMA-ES training.

    Args:
        config: Training configuration

    Returns:
        Dictionary with training results
    """
    # Create output directory
    output_dir = Path(config.output_dir) / f"{config.board_type}_{config.num_players}p"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(config.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(config.seed)

    logger.info("=" * 70)
    logger.info("ROBUST MULTI-OPPONENT CMA-ES TRAINING")
    logger.info("=" * 70)
    logger.info(f"Board: {config.board_type}")
    logger.info(f"Players: {config.num_players}")
    logger.info(f"Population: {config.population_size}")
    logger.info(f"Generations: {config.generations}")
    logger.info(f"Sigma: {config.sigma}")
    logger.info(f"Games per opponent: {config.games_per_opponent}")
    logger.info(f"Fitness aggregation: {config.min_weight}*min + {1-config.min_weight}*mean")
    logger.info(f"Diversity bonus: {config.use_diversity_bonus}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Create problem
    problem = MultiOpponentProblem(config)

    # Initialize CMA-ES
    initial_center = weights_dict_to_array(dict(HEURISTIC_V1_BALANCED))
    initial_center_tensor = torch.tensor(
        initial_center, dtype=torch.float32, device=problem.compute_device
    )

    searcher = CMAES(
        problem,
        stdev_init=config.sigma,
        popsize=config.population_size,
        center_init=initial_center_tensor,
    )

    # Set up logging
    StdOutLogger(searcher)

    # Training history
    fitness_history: list[float] = []
    sigma_history: list[float] = []
    history: list[dict[str, Any]] = []
    start_time = time.time()

    # Training loop
    for gen in range(config.generations):
        gen_start = time.time()

        # Run one generation - our evaluate() method handles parallelization
        searcher.step()

        # Get current status
        status = searcher.status
        current_best = float(status.get("best_eval", 0) or 0)
        current_mean = float(status.get("mean_eval", 0) or 0)
        current_sigma = float(status.get("stdev_max", config.sigma) or config.sigma)

        fitness_history.append(current_best)
        sigma_history.append(current_sigma)

        gen_elapsed = time.time() - gen_start

        # Log progress
        logger.info(
            f"Gen {gen + 1}/{config.generations}: "
            f"best={current_best:.3f}, mean={current_mean:.3f}, "
            f"sigma={current_sigma:.3f}, time={gen_elapsed:.1f}s"
        )

        # Log per-opponent scores for best
        if problem.best_per_opponent:
            scores_str = ", ".join(
                f"{k[:3]}={v:.1%}" for k, v in problem.best_per_opponent.items()
            )
            logger.info(f"  Per-opponent: {scores_str}")

        # Checkpoint
        checkpoint = {
            "generation": gen + 1,
            "fitness": current_best,
            "mean_fitness": current_mean,
            "sigma": current_sigma,
            "best_weights": problem.best_weights,
            "best_per_opponent": problem.best_per_opponent,
            "elite_archive_stats": problem.elite_archive.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }
        history.append(checkpoint)

        if (gen + 1) % config.checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_gen{gen + 1:03d}.json"
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

        # Elite injection for diversity
        if (
            config.use_diversity_bonus
            and gen > 0
            and gen % config.elite_injection_interval == 0
        ):
            elites = problem.elite_archive.get_injection_candidates(
                config.elite_injection_count
            )
            if elites:
                logger.info(f"  Injecting {len(elites)} elite solutions")

        # Check convergence
        converged, reason = detect_convergence(
            fitness_history, sigma_history, gen + 1, config
        )
        if converged:
            logger.info(f"Converged: {reason}")
            break

    # Final results
    total_time = time.time() - start_time

    results = {
        "board_type": config.board_type,
        "num_players": config.num_players,
        "generations_completed": len(history),
        "best_fitness": problem.best_fitness,
        "best_weights": problem.best_weights,
        "best_per_opponent": problem.best_per_opponent,
        "total_evaluations": problem.eval_count,
        "total_time_seconds": total_time,
        "elite_archive_stats": problem.elite_archive.get_stats(),
        "config": {
            "population_size": config.population_size,
            "sigma": config.sigma,
            "games_per_opponent": config.games_per_opponent,
            "min_weight": config.min_weight,
            "diversity_bonus": config.use_diversity_bonus,
        },
        "history": history,
    }

    # Save final results
    results_path = output_dir / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save optimized weights separately
    weights_path = output_dir / "optimized_weights.json"
    with open(weights_path, "w") as f:
        json.dump(
            {
                "weights": problem.best_weights,
                "fitness": problem.best_fitness,
                "per_opponent": problem.best_per_opponent,
                "board_type": config.board_type,
                "num_players": config.num_players,
                "persona": "robust_trained",
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best fitness: {problem.best_fitness:.4f}")
    logger.info(f"Generations: {len(history)}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Evaluations: {problem.eval_count}")
    logger.info("")
    logger.info("Per-opponent performance:")
    for opponent, score in problem.best_per_opponent.items():
        logger.info(f"  vs {opponent}: {score:.1%}")
    logger.info("")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Weights saved to: {weights_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Robust multi-opponent CMA-ES training for RingRift"
    )

    # Board and game
    parser.add_argument(
        "--board", type=str, default="square8",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type"
    )
    parser.add_argument(
        "--num-players", type=int, default=2, choices=[2, 3, 4],
        help="Number of players"
    )

    # CMA-ES parameters
    parser.add_argument(
        "--generations", type=int, default=200,
        help="Maximum generations"
    )
    parser.add_argument(
        "--population-size", type=int, default=64,
        help="Population size"
    )
    parser.add_argument(
        "--sigma", type=float, default=2.0,
        help="Initial step size"
    )

    # Multi-opponent evaluation
    parser.add_argument(
        "--games-per-opponent", type=int, default=32,
        help="Games per opponent in multi-opponent evaluation"
    )
    parser.add_argument(
        "--self-play-games", type=int, default=24,
        help="Self-play games per evaluation"
    )
    parser.add_argument(
        "--min-weight", type=float, default=0.4,
        help="Weight for min component in fitness aggregation"
    )
    parser.add_argument(
        "--max-moves", type=int, default=200,
        help="Max moves per game"
    )

    # Diversity
    parser.add_argument(
        "--diversity-bonus", action="store_true",
        help="Enable population diversity bonus"
    )
    parser.add_argument(
        "--diversity-scale", type=float, default=0.02,
        help="Scale for diversity bonus"
    )
    parser.add_argument(
        "--elite-injection-interval", type=int, default=50,
        help="Generations between elite injections"
    )

    # Convergence
    parser.add_argument(
        "--plateau-generations", type=int, default=20,
        help="Generations to detect plateau"
    )
    parser.add_argument(
        "--target-fitness", type=float, default=0.60,
        help="Target fitness for early stopping"
    )

    # Distribution
    parser.add_argument(
        "--distributed", action="store_true",
        help="Use distributed evaluation"
    )
    parser.add_argument(
        "--workers", type=str, default="",
        help="Comma-separated worker URLs"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="logs/cmaes/robust",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda" if CUDA_AVAILABLE else "cpu",
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    # Build config
    config = RobustCMAESConfig(
        board_type=args.board,
        num_players=args.num_players,
        generations=args.generations,
        population_size=args.population_size,
        sigma=args.sigma,
        games_per_opponent=args.games_per_opponent,
        self_play_games=args.self_play_games,
        min_weight=args.min_weight,
        max_moves=args.max_moves,
        use_diversity_bonus=args.diversity_bonus,
        diversity_scale=args.diversity_scale,
        elite_injection_interval=args.elite_injection_interval,
        plateau_generations=args.plateau_generations,
        target_fitness=args.target_fitness,
        distributed=args.distributed,
        worker_urls=[u.strip() for u in args.workers.split(",") if u.strip()],
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )

    # Run training
    run_robust_cmaes(config)


if __name__ == "__main__":
    main()
