"""Diversity mechanisms for CMA-ES training.

Provides:
1. EliteArchive: Maintains historically best solutions per opponent type
2. DiversityBonus: Rewards solutions that are distant from population centroid
3. NoveltySearch: Optional behavioral characterization for novelty-based selection

These mechanisms help prevent premature convergence and local optima traps
that are common with standard CMA-ES on noisy fitness landscapes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EliteSolution:
    """A solution stored in the elite archive."""

    weights: dict[str, float]
    fitness: float
    per_opponent: dict[str, float]
    generation: int = 0

    def to_array(self, weight_keys: list[str]) -> np.ndarray:
        """Convert weights to numpy array for CMA-ES injection."""
        return np.array([self.weights.get(k, 0.0) for k in weight_keys])

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        weight_keys: list[str],
        fitness: float,
        per_opponent: dict[str, float],
        generation: int = 0,
    ) -> "EliteSolution":
        """Create from numpy array."""
        weights = {k: float(v) for k, v in zip(weight_keys, array)}
        return cls(
            weights=weights,
            fitness=fitness,
            per_opponent=per_opponent,
            generation=generation,
        )


class EliteArchive:
    """Archive of best solutions per opponent type.

    Maintains separate archives for:
    - Each opponent persona (balanced, aggressive, territorial, defensive)
    - Overall aggregate fitness

    This enables diverse elite injection to maintain exploration even
    when the main population has converged.
    """

    def __init__(self, capacity_per_type: int = 5):
        """Initialize archive.

        Args:
            capacity_per_type: Maximum solutions per archive category
        """
        self.capacity = capacity_per_type
        self.archives: dict[str, list[EliteSolution]] = {
            "balanced": [],
            "aggressive": [],
            "territorial": [],
            "defensive": [],
            "overall": [],
        }
        self.total_added = 0

    def add(
        self,
        weights: dict[str, float],
        per_opponent: dict[str, float],
        aggregate: float,
        generation: int = 0,
    ) -> list[str]:
        """Add solution to relevant archives if it qualifies.

        Args:
            weights: Heuristic weight dictionary
            per_opponent: Win rates per opponent
            aggregate: Aggregate fitness score
            generation: Current generation number

        Returns:
            List of archive categories the solution was added to
        """
        added_to: list[str] = []

        # Check each opponent-specific archive
        for opponent_name, score in per_opponent.items():
            if opponent_name in self.archives:
                if self._insert_if_better(
                    self.archives[opponent_name],
                    weights,
                    score,
                    per_opponent,
                    generation,
                ):
                    added_to.append(opponent_name)

        # Check overall archive
        if self._insert_if_better(
            self.archives["overall"],
            weights,
            aggregate,
            per_opponent,
            generation,
        ):
            added_to.append("overall")

        if added_to:
            self.total_added += 1
            logger.debug(f"Added elite to archives: {added_to}")

        return added_to

    def _insert_if_better(
        self,
        archive: list[EliteSolution],
        weights: dict[str, float],
        fitness: float,
        per_opponent: dict[str, float],
        generation: int,
    ) -> bool:
        """Insert solution if it qualifies for the archive."""
        solution = EliteSolution(
            weights=weights.copy(),
            fitness=fitness,
            per_opponent=per_opponent.copy(),
            generation=generation,
        )

        # Always insert if archive not full
        if len(archive) < self.capacity:
            archive.append(solution)
            archive.sort(key=lambda x: x.fitness, reverse=True)
            return True

        # Check if better than worst in archive
        if fitness > archive[-1].fitness:
            archive[-1] = solution
            archive.sort(key=lambda x: x.fitness, reverse=True)
            return True

        return False

    def get_injection_candidates(
        self,
        n: int = 3,
        weight_keys: list[str] | None = None,
    ) -> list[EliteSolution]:
        """Get diverse elite solutions for population injection.

        Selects best solutions from different archive categories
        to maximize diversity in the injected population.

        Args:
            n: Number of candidates to return
            weight_keys: Optional key ordering for array conversion

        Returns:
            List of up to n elite solutions
        """
        candidates: list[EliteSolution] = []
        seen_fitnesses: set[float] = set()

        # Round-robin selection from each archive
        for archive_name in ["overall", "balanced", "aggressive", "territorial", "defensive"]:
            archive = self.archives[archive_name]
            if archive and len(candidates) < n:
                best = archive[0]
                # Avoid duplicates (by fitness as proxy)
                if best.fitness not in seen_fitnesses:
                    candidates.append(best)
                    seen_fitnesses.add(best.fitness)

        return candidates[:n]

    def get_best_overall(self) -> EliteSolution | None:
        """Get the best overall solution."""
        if self.archives["overall"]:
            return self.archives["overall"][0]
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get archive statistics."""
        return {
            "total_added": self.total_added,
            "archive_sizes": {k: len(v) for k, v in self.archives.items()},
            "best_per_category": {
                k: v[0].fitness if v else None for k, v in self.archives.items()
            },
        }

    def clear(self) -> None:
        """Clear all archives."""
        for archive in self.archives.values():
            archive.clear()
        self.total_added = 0


def compute_diversity_bonus(
    population: list[np.ndarray] | np.ndarray,
    candidate_idx: int,
    scale: float = 0.02,
) -> float:
    """Compute diversity bonus for a candidate.

    Rewards solutions that are distant from the population centroid.
    This encourages exploration and prevents premature convergence.

    Args:
        population: Population of weight vectors (list or 2D array)
        candidate_idx: Index of candidate to compute bonus for
        scale: Maximum bonus scale (default 0.02 = 2% fitness boost)

    Returns:
        Diversity bonus to add to fitness (0.0 to scale)
    """
    if isinstance(population, list):
        population = np.array(population)

    if len(population) < 2:
        return 0.0

    candidate = population[candidate_idx]
    centroid = np.mean(population, axis=0)

    # Normalized L2 distance from centroid
    distance = np.linalg.norm(candidate - centroid)
    dimension = len(candidate)

    # Normalize by expected distance in high dimensions
    # For uniform distribution, expected distance ~ sqrt(dimension/12)
    expected_distance = np.sqrt(dimension / 12.0)
    normalized_distance = distance / max(expected_distance, 1e-6)

    # Sigmoid-like mapping to bound bonus
    bonus = scale * min(normalized_distance, 1.0)

    return float(bonus)


def compute_population_diversity(
    population: list[np.ndarray] | np.ndarray,
) -> dict[str, float]:
    """Compute population diversity metrics.

    Args:
        population: Population of weight vectors

    Returns:
        Dictionary with diversity metrics:
        - mean_distance: Average pairwise distance
        - std_distance: Standard deviation of distances
        - centroid_spread: Average distance from centroid
        - dimension: Weight vector dimension
    """
    if isinstance(population, list):
        population = np.array(population)

    n = len(population)
    if n < 2:
        return {
            "mean_distance": 0.0,
            "std_distance": 0.0,
            "centroid_spread": 0.0,
            "dimension": len(population[0]) if n > 0 else 0,
        }

    # Pairwise distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(np.linalg.norm(population[i] - population[j]))

    # Centroid spread
    centroid = np.mean(population, axis=0)
    centroid_distances = [np.linalg.norm(p - centroid) for p in population]

    return {
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "centroid_spread": float(np.mean(centroid_distances)),
        "dimension": len(population[0]),
    }


@dataclass
class BehavioralVector:
    """Behavioral characterization of a solution for novelty search."""

    avg_game_length: float = 0.0
    territory_control: float = 0.0
    elimination_rate: float = 0.0
    consistency: float = 0.0  # Std dev of game lengths

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for distance calculations."""
        return np.array([
            self.avg_game_length / 200.0,  # Normalize
            self.territory_control,
            self.elimination_rate,
            1.0 - self.consistency / 50.0,  # Invert so higher = better
        ])


class NoveltyArchive:
    """Archive for novelty search (optional, for behavioral diversity).

    Instead of just optimizing fitness, novelty search rewards solutions
    that behave differently from previously seen solutions.
    """

    def __init__(self, capacity: int = 100, k_nearest: int = 15):
        """Initialize novelty archive.

        Args:
            capacity: Maximum solutions to store
            k_nearest: Number of neighbors for novelty calculation
        """
        self.capacity = capacity
        self.k_nearest = k_nearest
        self.behaviors: list[np.ndarray] = []
        self.solutions: list[dict[str, float]] = []

    def add(self, behavior: BehavioralVector, weights: dict[str, float]) -> float:
        """Add solution and return its novelty score.

        Args:
            behavior: Behavioral characterization
            weights: Solution weights

        Returns:
            Novelty score (average distance to k nearest neighbors)
        """
        behavior_array = behavior.to_array()
        novelty = self._compute_novelty(behavior_array)

        # Always add to archive (will evict oldest if full)
        if len(self.behaviors) >= self.capacity:
            self.behaviors.pop(0)
            self.solutions.pop(0)

        self.behaviors.append(behavior_array)
        self.solutions.append(weights.copy())

        return novelty

    def _compute_novelty(self, behavior: np.ndarray) -> float:
        """Compute novelty as average distance to k nearest neighbors."""
        if not self.behaviors:
            return 1.0  # Maximum novelty for first solution

        distances = [np.linalg.norm(behavior - b) for b in self.behaviors]
        k = min(self.k_nearest, len(distances))
        k_nearest_distances = sorted(distances)[:k]

        return float(np.mean(k_nearest_distances))

    def get_novelty_bonus(
        self,
        behavior: BehavioralVector,
        scale: float = 0.03,
    ) -> float:
        """Get novelty bonus for fitness modification.

        Args:
            behavior: Behavioral characterization
            scale: Maximum bonus scale

        Returns:
            Novelty bonus to add to fitness
        """
        novelty = self._compute_novelty(behavior.to_array())
        # Normalize: novelty > 0.5 gets full bonus, < 0.1 gets none
        normalized = min(max((novelty - 0.1) / 0.4, 0.0), 1.0)
        return scale * normalized


def inject_elites_into_population(
    population: np.ndarray,
    elites: list[EliteSolution],
    weight_keys: list[str],
    injection_rate: float = 0.1,
) -> np.ndarray:
    """Inject elite solutions into population.

    Replaces worst members of population with elite solutions.

    Args:
        population: Current population (2D array)
        elites: Elite solutions to inject
        weight_keys: Key ordering for weight vectors
        injection_rate: Fraction of population to replace (default 10%)

    Returns:
        Modified population with elites injected
    """
    n_inject = min(len(elites), int(len(population) * injection_rate))
    if n_inject == 0:
        return population

    # Convert elites to arrays
    elite_arrays = [e.to_array(weight_keys) for e in elites[:n_inject]]

    # Replace last n_inject members (assumed to be worst after sorting)
    new_population = population.copy()
    for i, elite_array in enumerate(elite_arrays):
        new_population[-(i + 1)] = elite_array

    logger.info(f"Injected {n_inject} elites into population")
    return new_population
