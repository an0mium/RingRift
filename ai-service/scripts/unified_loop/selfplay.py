"""Unified Loop Local Selfplay Generation.

This module provides local selfplay data generation for the unified AI loop.
Uses the parallel_selfplay module for efficient multi-process game generation.

Supports:
- Descent AI (default)
- MCTS AI (standard tree search)
- Gumbel-MCTS AI (soft policy targets for improved training)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from scripts.unified_ai_loop import EventBus, UnifiedLoopState

# Path constants
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]

# Coordinator-only mode - skip local CPU-intensive work
DISABLE_LOCAL_TASKS = os.environ.get("RINGRIFT_DISABLE_LOCAL_TASKS", "").lower() in ("1", "true", "yes", "on")

logger = logging.getLogger(__name__)

# Import parallel selfplay module
try:
    from app.training.parallel_selfplay import (
        generate_dataset_parallel,
        SelfplayConfig,
        GameResult,
    )
    HAS_PARALLEL_SELFPLAY = True
except ImportError:
    HAS_PARALLEL_SELFPLAY = False
    generate_dataset_parallel = None

# Prometheus metrics - avoid duplicate registration
try:
    from prometheus_client import Counter, Gauge, Histogram, REGISTRY
    HAS_PROMETHEUS = True

    if 'ringrift_local_selfplay_games_total' in REGISTRY._names_to_collectors:
        LOCAL_SELFPLAY_GAMES = REGISTRY._names_to_collectors['ringrift_local_selfplay_games_total']
    else:
        LOCAL_SELFPLAY_GAMES = Counter(
            'ringrift_local_selfplay_games_total',
            'Total local selfplay games generated',
            ['config', 'engine']
        )

    if 'ringrift_local_selfplay_samples_total' in REGISTRY._names_to_collectors:
        LOCAL_SELFPLAY_SAMPLES = REGISTRY._names_to_collectors['ringrift_local_selfplay_samples_total']
    else:
        LOCAL_SELFPLAY_SAMPLES = Counter(
            'ringrift_local_selfplay_samples_total',
            'Total local selfplay samples generated',
            ['config', 'engine']
        )

    if 'ringrift_local_selfplay_duration_seconds' in REGISTRY._names_to_collectors:
        LOCAL_SELFPLAY_DURATION = REGISTRY._names_to_collectors['ringrift_local_selfplay_duration_seconds']
    else:
        LOCAL_SELFPLAY_DURATION = Histogram(
            'ringrift_local_selfplay_duration_seconds',
            'Local selfplay generation duration',
            ['config', 'engine'],
            buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
        )
except ImportError:
    HAS_PROMETHEUS = False
    LOCAL_SELFPLAY_GAMES = None
    LOCAL_SELFPLAY_SAMPLES = None
    LOCAL_SELFPLAY_DURATION = None


class LocalSelfplayGenerator:
    """Generates selfplay data locally using parallel workers.

    Supports multiple AI engines:
    - descent: Fast descent-based AI (default)
    - mcts: Standard MCTS tree search
    - gumbel: Gumbel-MCTS with soft policy targets (best for training)

    Enhanced features (2025-12):
    - PFSP opponent selection for diverse training
    - Priority-based config selection based on training proximity
    - Curriculum weight integration for adaptive generation
    """

    def __init__(
        self,
        state: "UnifiedLoopState",
        event_bus: "EventBus",
        output_dir: Optional[Path] = None,
        num_workers: Optional[int] = None,
        training_scheduler: Optional[Any] = None,
    ):
        self.state = state
        self.event_bus = event_bus
        self.output_dir = output_dir or AI_SERVICE_ROOT / "data" / "games" / "local_selfplay"
        self.num_workers = num_workers
        self._running = False
        self._generation_task: Optional[asyncio.Task] = None
        # Reference to training scheduler for PFSP and priority access
        self._training_scheduler = training_scheduler

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not HAS_PARALLEL_SELFPLAY:
            logger.warning("Parallel selfplay module not available")

    def set_training_scheduler(self, scheduler: Any) -> None:
        """Set reference to training scheduler for PFSP integration."""
        self._training_scheduler = scheduler

    def get_pfsp_opponent(self, config_key: str, current_elo: float = 1500.0) -> Optional[str]:
        """Get PFSP-selected opponent for selfplay.

        Args:
            config_key: Config identifier
            current_elo: Current model Elo for matchmaking

        Returns:
            Opponent model path or None if PFSP not available
        """
        if self._training_scheduler is not None:
            return self._training_scheduler.get_pfsp_opponent(config_key, current_elo)
        return None

    def get_prioritized_config(self) -> Optional[str]:
        """Get the config that should have highest selfplay priority.

        Priority is based on:
        1. Proximity to training threshold (closer = higher priority)
        2. Curriculum weights (higher weight = higher priority)
        3. Time since last training (longer = higher priority)

        Returns:
            Config key with highest priority, or None if no configs
        """
        if not self.state.configs:
            return None

        best_config = None
        best_priority = -1.0

        for config_key, config_state in self.state.configs.items():
            priority = 0.0

            # Factor 1: Proximity to training threshold (0-1)
            # Closer to threshold = higher priority
            if self._training_scheduler:
                threshold = self._training_scheduler._get_dynamic_threshold(config_key)
                if threshold > 0:
                    proximity = min(1.0, config_state.games_since_training / threshold)
                    priority += proximity * 0.5

            # Factor 2: Curriculum weight (0.5-2.0 normalized to 0-1)
            curriculum_weight = getattr(config_state, 'curriculum_weight', 1.0)
            normalized_curriculum = (curriculum_weight - 0.5) / 1.5
            priority += normalized_curriculum * 0.3

            # Factor 3: Time since last training (0-1)
            time_since_training = time.time() - config_state.last_training_time
            hours_since = time_since_training / 3600
            staleness_factor = min(1.0, hours_since / 6.0)  # Cap at 6 hours
            priority += staleness_factor * 0.2

            if priority > best_priority:
                best_priority = priority
                best_config = config_key

        return best_config

    def get_config_priorities(self) -> Dict[str, float]:
        """Get priority scores for all configs.

        Returns:
            Dict mapping config_key to priority score (0-1)
        """
        priorities = {}

        for config_key, config_state in self.state.configs.items():
            priority = 0.0

            # Proximity to threshold
            if self._training_scheduler:
                threshold = self._training_scheduler._get_dynamic_threshold(config_key)
                if threshold > 0:
                    proximity = min(1.0, config_state.games_since_training / threshold)
                    priority += proximity * 0.5

            # Curriculum weight
            curriculum_weight = getattr(config_state, 'curriculum_weight', 1.0)
            normalized_curriculum = (curriculum_weight - 0.5) / 1.5
            priority += max(0, normalized_curriculum * 0.3)

            # Staleness
            time_since_training = time.time() - config_state.last_training_time
            hours_since = time_since_training / 3600
            staleness_factor = min(1.0, hours_since / 6.0)
            priority += staleness_factor * 0.2

            priorities[config_key] = priority

        return priorities

    def get_adaptive_engine(self, config_key: str, quality_threshold: float = 0.7) -> str:
        """Select selfplay engine based on training proximity.

        Uses higher quality (but slower) 'gumbel' engine when config is
        close to training threshold, and faster 'descent' engine when
        far from threshold to maximize throughput.

        Args:
            config_key: Config identifier
            quality_threshold: Proximity threshold to switch to gumbel (0-1)

        Returns:
            Engine name: 'gumbel' for high quality, 'descent' for throughput
        """
        priorities = self.get_config_priorities()
        priority = priorities.get(config_key, 0.0)

        # Higher priority = closer to training threshold = use higher quality engine
        if priority >= quality_threshold:
            logger.info(f"[AdaptiveEngine] {config_key} priority={priority:.2f} >= {quality_threshold} -> using 'gumbel'")
            return "gumbel"
        else:
            logger.debug(f"[AdaptiveEngine] {config_key} priority={priority:.2f} < {quality_threshold} -> using 'descent'")
            return "descent"

    def get_all_adaptive_engines(self) -> Dict[str, str]:
        """Get recommended engine for all configs.

        Returns:
            Dict mapping config_key to recommended engine
        """
        return {
            config_key: self.get_adaptive_engine(config_key)
            for config_key in self.state.configs
        }

    async def generate_games(
        self,
        num_games: int,
        config_key: str,
        engine: str = "descent",
        nn_model_id: Optional[str] = None,
        gumbel_simulations: int = 64,
        gumbel_top_k: int = 16,
        progress_callback: Optional[callable] = None,
        use_pfsp_opponent: bool = False,
        current_elo: float = 1500.0,
        temperature: float = 1.0,
        use_temperature_decay: bool = True,
        opening_temperature: float = 1.5,
    ) -> Dict[str, Any]:
        """Generate selfplay games locally.

        Args:
            num_games: Number of games to generate
            config_key: Config identifier (e.g., "square8_2p")
            engine: AI engine ("descent", "mcts", or "gumbel")
            nn_model_id: Neural network model ID for AI
            gumbel_simulations: Simulations per move for Gumbel-MCTS
            gumbel_top_k: Top-k for sequential halving
            progress_callback: Optional callback(completed, total)
            use_pfsp_opponent: Whether to use PFSP for opponent selection
            current_elo: Current model Elo for PFSP matchmaking
            temperature: Base temperature for move selection (1.0 = standard)
            use_temperature_decay: Decay temperature from opening to base
            opening_temperature: Higher temperature for opening moves

        Returns:
            Dict with generation results
        """
        # Guard against local work in coordinator-only mode
        if DISABLE_LOCAL_TASKS:
            logger.info("[LocalSelfplay] Skipping local selfplay (RINGRIFT_DISABLE_LOCAL_TASKS=true)")
            return {
                "success": False,
                "error": "Coordinator-only mode",
                "games": 0,
                "samples": 0,
            }

        if not HAS_PARALLEL_SELFPLAY:
            return {
                "success": False,
                "error": "Parallel selfplay module not available",
                "games": 0,
                "samples": 0,
            }

        # Parse config key
        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2

        # Convert board_type to BoardType enum
        from app.models import BoardType
        board_type_enum = BoardType(board_type)

        # PFSP opponent selection (2025-12 enhancement)
        opponent_model = None
        if use_pfsp_opponent:
            opponent_model = self.get_pfsp_opponent(config_key, current_elo)
            if opponent_model:
                logger.info(f"[PFSP] Selected opponent: {opponent_model} for {config_key}")
            else:
                logger.debug(f"[PFSP] No opponent available, using self-play")

        # Generate output filename
        timestamp = int(time.time())
        output_file = self.output_dir / config_key / f"selfplay_{engine}_{timestamp}.npz"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            opponent_info = f", opponent={opponent_model}" if opponent_model else ""
            logger.info(f"Starting local selfplay: {num_games} games, config={config_key}, engine={engine}{opponent_info}")

            # Run parallel selfplay in executor to avoid blocking event loop
            # Temperature scheduling: higher temp early in game for diverse positions
            temp_info = f", temp={temperature:.2f}" if temperature != 1.0 else ""
            if use_temperature_decay:
                temp_info = f", temp={opening_temperature:.1f}->{temperature:.2f}"
            logger.info(f"[Temperature] {config_key}: {temp_info or 'default (1.0)'}")

            loop = asyncio.get_event_loop()
            total_samples = await loop.run_in_executor(
                None,
                lambda: generate_dataset_parallel(
                    num_games=num_games,
                    output_file=str(output_file),
                    num_workers=self.num_workers,
                    board_type=board_type_enum,
                    num_players=num_players,
                    engine=engine,
                    nn_model_id=nn_model_id,
                    multi_player_values=(num_players > 2),
                    max_players=max(4, num_players),
                    progress_callback=progress_callback,
                    gumbel_simulations=gumbel_simulations,
                    gumbel_top_k=gumbel_top_k,
                    temperature=temperature,
                    use_temperature_decay=use_temperature_decay,
                    opening_temperature=opening_temperature,
                )
            )

            duration = time.time() - start_time
            games_per_sec = num_games / duration if duration > 0 else 0

            logger.info(
                f"Local selfplay complete: {num_games} games, {total_samples} samples "
                f"in {duration:.1f}s ({games_per_sec:.1f} games/sec)"
            )

            # Update Prometheus metrics
            if HAS_PROMETHEUS:
                LOCAL_SELFPLAY_GAMES.labels(config=config_key, engine=engine).inc(num_games)
                LOCAL_SELFPLAY_SAMPLES.labels(config=config_key, engine=engine).inc(total_samples)
                LOCAL_SELFPLAY_DURATION.labels(config=config_key, engine=engine).observe(duration)

            return {
                "success": True,
                "games": num_games,
                "samples": total_samples,
                "output_file": str(output_file),
                "duration_seconds": duration,
                "games_per_second": games_per_sec,
                "engine": engine,
                "config": config_key,
                "pfsp_opponent": opponent_model,
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Local selfplay failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "games": 0,
                "samples": 0,
                "duration_seconds": duration,
                "engine": engine,
                "config": config_key,
            }

    async def generate_gumbel_games(
        self,
        num_games: int,
        config_key: str,
        nn_model_id: Optional[str] = None,
        simulations: int = 64,
        top_k: int = 16,
    ) -> Dict[str, Any]:
        """Convenience method to generate games using Gumbel-MCTS.

        Gumbel-MCTS produces higher quality training data with soft policy
        targets based on visit counts from sequential halving search.

        Args:
            num_games: Number of games to generate
            config_key: Config identifier
            nn_model_id: Neural network model ID
            simulations: Simulations per move
            top_k: Top-k actions for sequential halving

        Returns:
            Generation results
        """
        return await self.generate_games(
            num_games=num_games,
            config_key=config_key,
            engine="gumbel",
            nn_model_id=nn_model_id,
            gumbel_simulations=simulations,
            gumbel_top_k=top_k,
        )

    async def run_continuous_generation(
        self,
        config_key: str,
        target_games_per_hour: int = 100,
        engine: str = "descent",
        batch_size: int = 20,
        nn_model_id: Optional[str] = None,
    ) -> None:
        """Run continuous selfplay generation in the background.

        Generates games in batches to maintain the target rate while
        allowing for model updates between batches.

        Args:
            config_key: Config identifier
            target_games_per_hour: Target generation rate
            engine: AI engine to use
            batch_size: Games per batch
            nn_model_id: Neural network model ID
        """
        self._running = True
        games_per_batch = batch_size

        # Calculate delay between batches to achieve target rate
        batches_per_hour = target_games_per_hour / games_per_batch
        seconds_per_batch = 3600 / batches_per_hour if batches_per_hour > 0 else 60

        logger.info(
            f"Starting continuous selfplay: target={target_games_per_hour}/hour, "
            f"batch_size={games_per_batch}, interval={seconds_per_batch:.1f}s"
        )

        while self._running:
            try:
                batch_start = time.time()

                result = await self.generate_games(
                    num_games=games_per_batch,
                    config_key=config_key,
                    engine=engine,
                    nn_model_id=nn_model_id,
                )

                if result["success"]:
                    # Update state with new games
                    if config_key in self.state.configs:
                        self.state.configs[config_key].games_since_training += result["games"]
                        self.state.total_games_pending += result["games"]

                # Wait for next batch
                elapsed = time.time() - batch_start
                wait_time = max(0, seconds_per_batch - elapsed)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Continuous selfplay error: {e}")
                await asyncio.sleep(60)  # Back off on error

        logger.info("Continuous selfplay stopped")

    def stop_continuous_generation(self) -> None:
        """Stop continuous selfplay generation."""
        self._running = False
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()

    def start_continuous_generation_task(
        self,
        config_key: str,
        target_games_per_hour: int = 100,
        engine: str = "descent",
    ) -> asyncio.Task:
        """Start continuous generation as an async task.

        Returns:
            The asyncio Task for the generation loop
        """
        self._generation_task = asyncio.create_task(
            self.run_continuous_generation(
                config_key=config_key,
                target_games_per_hour=target_games_per_hour,
                engine=engine,
            )
        )
        return self._generation_task

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about local selfplay generation."""
        stats = {
            "output_dir": str(self.output_dir),
            "num_workers": self.num_workers,
            "running": self._running,
            "parallel_selfplay_available": HAS_PARALLEL_SELFPLAY,
        }

        # Count generated files per config
        if self.output_dir.exists():
            for config_dir in self.output_dir.iterdir():
                if config_dir.is_dir():
                    npz_files = list(config_dir.glob("*.npz"))
                    stats[f"{config_dir.name}_files"] = len(npz_files)

        return stats
