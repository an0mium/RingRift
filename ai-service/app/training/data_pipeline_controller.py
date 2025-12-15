"""Unified Data Pipeline Controller for RingRift AI.

Provides a single entry point for all data operations, consolidating:
- Real-time streaming from SQLite databases (streaming_pipeline.py)
- Batch loading from NPZ/HDF5 files (data_loader.py)
- Data sync and aggregation orchestration (run_data_pipeline.py)

This controller exposes a clean API for the unified AI loop to consume,
handling data preparation, loading, and streaming in a consistent manner.

Usage:
    from app.training.data_pipeline_controller import DataPipelineController

    # Initialize controller
    controller = DataPipelineController(
        db_paths=["data/games/selfplay.db"],
        npz_paths=["data/training/training_data.npz"],
    )

    # Get training data (automatically selects best source)
    for batch in controller.get_training_batches(batch_size=256):
        train_on_batch(batch)

    # Or use specific pipeline
    async for batch in controller.stream_from_database(batch_size=256):
        train_on_batch(batch)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of data sources supported by the pipeline."""

    DATABASE = "database"  # SQLite game databases
    NPZ = "npz"  # NumPy compressed files
    HDF5 = "hdf5"  # HDF5 files
    STREAMING = "streaming"  # Real-time streaming
    REMOTE = "remote"  # Remote sources (via rsync)


class PipelineMode(Enum):
    """Operating modes for the data pipeline."""

    BATCH = "batch"  # Load fixed batches from files
    STREAMING = "streaming"  # Real-time streaming from database
    HYBRID = "hybrid"  # Combine batch and streaming


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""

    source_type: DataSourceType
    path: str
    weight: float = 1.0  # Sampling weight for this source
    board_type: Optional[str] = None
    num_players: Optional[int] = None
    enabled: bool = True
    priority: int = 0  # Higher = preferred when multiple sources available


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""

    # Operating mode
    mode: PipelineMode = PipelineMode.BATCH

    # Batch settings
    batch_size: int = 256
    shuffle: bool = True
    drop_last: bool = False

    # Streaming settings
    poll_interval_seconds: float = 5.0
    buffer_size: int = 10000
    min_buffer_fill: float = 0.2

    # Sampling settings
    priority_sampling: bool = True
    recency_weight: float = 0.3
    late_game_exponent: float = 2.0

    # Performance settings
    prefetch_count: int = 2
    pin_memory: bool = False
    num_workers: int = 0

    # Filtering
    board_type: Optional[str] = None
    num_players: Optional[int] = None

    # Multi-player value support
    use_multi_player_values: bool = False


@dataclass
class PipelineStats:
    """Statistics from the data pipeline."""

    total_samples_loaded: int = 0
    total_batches_yielded: int = 0
    total_samples_ingested: int = 0
    active_sources: int = 0
    buffer_size: int = 0
    buffer_capacity: int = 0
    last_batch_time: Optional[float] = None
    avg_batch_load_time_ms: float = 0.0
    source_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_samples_loaded": self.total_samples_loaded,
            "total_batches_yielded": self.total_batches_yielded,
            "total_samples_ingested": self.total_samples_ingested,
            "active_sources": self.active_sources,
            "buffer_size": self.buffer_size,
            "buffer_capacity": self.buffer_capacity,
            "last_batch_time": self.last_batch_time,
            "avg_batch_load_time_ms": self.avg_batch_load_time_ms,
            "source_stats": self.source_stats,
        }


class DataPipelineController:
    """Unified controller for all data pipeline operations.

    Provides a single entry point for:
    - Loading training data from various sources
    - Real-time streaming during training
    - Data synchronization from remote sources
    - Statistics and monitoring
    """

    def __init__(
        self,
        db_paths: Optional[List[str]] = None,
        npz_paths: Optional[List[str]] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """Initialize the data pipeline controller.

        Args:
            db_paths: Paths to SQLite game databases
            npz_paths: Paths to NPZ training data files
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()

        # Data sources
        self._sources: List[DataSourceConfig] = []
        self._db_paths = db_paths or []
        self._npz_paths = npz_paths or []

        # Initialize sources
        self._init_sources()

        # Lazy-loaded pipeline components
        self._streaming_pipeline = None
        self._batch_loader = None
        self._is_running = False
        self._lock = threading.Lock()

        # Batch timing
        self._batch_times: List[float] = []

        logger.info(
            f"DataPipelineController initialized with {len(self._sources)} sources "
            f"(mode={self.config.mode.value})"
        )

    def _init_sources(self):
        """Initialize data sources from paths."""
        # Add database sources
        for i, db_path in enumerate(self._db_paths):
            if os.path.exists(db_path):
                self._sources.append(DataSourceConfig(
                    source_type=DataSourceType.DATABASE,
                    path=db_path,
                    priority=100 - i,  # Earlier paths have higher priority
                ))

        # Add NPZ sources
        for i, npz_path in enumerate(self._npz_paths):
            if os.path.exists(npz_path):
                ext = os.path.splitext(npz_path)[1].lower()
                source_type = DataSourceType.HDF5 if ext in ('.h5', '.hdf5') else DataSourceType.NPZ
                self._sources.append(DataSourceConfig(
                    source_type=source_type,
                    path=npz_path,
                    priority=50 - i,
                ))

        self.stats.active_sources = len([s for s in self._sources if s.enabled])

    def add_source(self, source: DataSourceConfig):
        """Add a data source to the pipeline.

        Args:
            source: Data source configuration
        """
        self._sources.append(source)
        self.stats.active_sources = len([s for s in self._sources if s.enabled])
        logger.info(f"Added data source: {source.source_type.value} at {source.path}")

    def remove_source(self, path: str):
        """Remove a data source by path.

        Args:
            path: Path of the source to remove
        """
        self._sources = [s for s in self._sources if s.path != path]
        self.stats.active_sources = len([s for s in self._sources if s.enabled])

    def get_sources(self) -> List[DataSourceConfig]:
        """Get all configured data sources."""
        return self._sources.copy()

    def _get_streaming_pipeline(self):
        """Lazy-load the streaming pipeline."""
        if self._streaming_pipeline is None:
            try:
                from app.training.streaming_pipeline import (
                    MultiDBStreamingPipeline,
                    StreamingConfig,
                    StreamingDataPipeline,
                )

                db_sources = [
                    s for s in self._sources
                    if s.source_type == DataSourceType.DATABASE and s.enabled
                ]

                if not db_sources:
                    logger.warning("No database sources available for streaming")
                    return None

                config = StreamingConfig(
                    poll_interval_seconds=self.config.poll_interval_seconds,
                    buffer_size=self.config.buffer_size,
                    min_buffer_fill=self.config.min_buffer_fill,
                    priority_sampling=self.config.priority_sampling,
                    recency_weight=self.config.recency_weight,
                )

                if len(db_sources) == 1:
                    self._streaming_pipeline = StreamingDataPipeline(
                        db_path=Path(db_sources[0].path),
                        board_type=self.config.board_type,
                        num_players=self.config.num_players,
                        config=config,
                    )
                else:
                    self._streaming_pipeline = MultiDBStreamingPipeline(
                        db_paths=[Path(s.path) for s in db_sources],
                        board_type=self.config.board_type,
                        num_players=self.config.num_players,
                        config=config,
                    )

                self.stats.buffer_capacity = self.config.buffer_size

            except ImportError as e:
                logger.error(f"Failed to import streaming pipeline: {e}")
                return None

        return self._streaming_pipeline

    def _get_batch_loader(self):
        """Lazy-load the batch data loader."""
        if self._batch_loader is None:
            try:
                from app.training.data_loader import (
                    StreamingDataLoader,
                    WeightedStreamingDataLoader,
                )

                npz_sources = [
                    s for s in self._sources
                    if s.source_type in (DataSourceType.NPZ, DataSourceType.HDF5)
                    and s.enabled
                ]

                if not npz_sources:
                    logger.warning("No NPZ/HDF5 sources available for batch loading")
                    return None

                data_paths = [s.path for s in npz_sources]

                # Use weighted loader if late-game bias is configured
                if self.config.late_game_exponent != 1.0:
                    self._batch_loader = WeightedStreamingDataLoader(
                        data_paths=data_paths,
                        batch_size=self.config.batch_size,
                        shuffle=self.config.shuffle,
                        drop_last=self.config.drop_last,
                        sampling_weights='late_game',
                        late_game_exponent=self.config.late_game_exponent,
                    )
                else:
                    self._batch_loader = StreamingDataLoader(
                        data_paths=data_paths,
                        batch_size=self.config.batch_size,
                        shuffle=self.config.shuffle,
                        drop_last=self.config.drop_last,
                    )

            except ImportError as e:
                logger.error(f"Failed to import data loader: {e}")
                return None

        return self._batch_loader

    def get_training_batches(
        self,
        batch_size: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        """Get training batches from the best available source.

        Automatically selects the appropriate data source based on
        configuration and availability.

        Args:
            batch_size: Override batch size
            max_batches: Maximum number of batches to yield

        Yields:
            Tuples of (inputs, targets) for training
        """
        batch_size = batch_size or self.config.batch_size
        batches_yielded = 0

        if self.config.mode == PipelineMode.BATCH:
            # Use batch loader
            loader = self._get_batch_loader()
            if loader is None:
                logger.error("No batch loader available")
                return

            # Optionally wrap with prefetching
            if self.config.prefetch_count > 0:
                try:
                    from app.training.data_loader import prefetch_loader
                    source = prefetch_loader(
                        loader,
                        prefetch_count=self.config.prefetch_count,
                        pin_memory=self.config.pin_memory,
                        use_mp=self.config.use_multi_player_values,
                    )
                except ImportError:
                    source = iter(loader)
            else:
                source = iter(loader)

            for batch in source:
                start_time = time.time()
                yield batch

                # Track timing
                elapsed_ms = (time.time() - start_time) * 1000
                self._batch_times.append(elapsed_ms)
                if len(self._batch_times) > 100:
                    self._batch_times = self._batch_times[-100:]

                self.stats.total_batches_yielded += 1
                self.stats.last_batch_time = time.time()
                self.stats.avg_batch_load_time_ms = np.mean(self._batch_times)

                batches_yielded += 1
                if max_batches and batches_yielded >= max_batches:
                    break

        elif self.config.mode == PipelineMode.STREAMING:
            # Use streaming via asyncio bridge
            logger.warning(
                "Streaming mode requested but sync iterator called. "
                "Use stream_from_database() for async streaming."
            )
            return

        else:
            # Hybrid mode - combine batch and streaming
            logger.info("Hybrid mode not yet fully implemented, using batch mode")
            yield from self.get_training_batches(batch_size, max_batches)

    async def stream_from_database(
        self,
        batch_size: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> AsyncIterator[Any]:
        """Stream training batches from database in real-time.

        Args:
            batch_size: Override batch size
            max_batches: Maximum number of batches to yield

        Yields:
            Batches of GameSample objects
        """
        batch_size = batch_size or self.config.batch_size

        pipeline = self._get_streaming_pipeline()
        if pipeline is None:
            logger.error("No streaming pipeline available")
            return

        self._is_running = True

        try:
            async for batch in pipeline.stream_batches(
                batch_size=batch_size,
                max_batches=max_batches,
            ):
                if not self._is_running:
                    break

                self.stats.total_batches_yielded += 1
                self.stats.last_batch_time = time.time()

                # Update buffer stats
                if hasattr(pipeline, 'buffer'):
                    self.stats.buffer_size = len(pipeline.buffer)

                yield batch

        finally:
            self._is_running = False

    async def start_streaming(self):
        """Start the streaming pipeline in the background."""
        pipeline = self._get_streaming_pipeline()
        if pipeline is not None:
            await pipeline.start()
            self._is_running = True
            logger.info("Streaming pipeline started")

    async def stop_streaming(self):
        """Stop the streaming pipeline."""
        if self._streaming_pipeline is not None:
            await self._streaming_pipeline.stop()
            self._is_running = False
            logger.info("Streaming pipeline stopped")

    def get_sample_count(self) -> int:
        """Get total sample count across all sources."""
        total = 0

        for source in self._sources:
            if not source.enabled:
                continue

            if source.source_type in (DataSourceType.NPZ, DataSourceType.HDF5):
                try:
                    from app.training.data_loader import get_sample_count
                    total += get_sample_count(source.path)
                except Exception as e:
                    logger.warning(f"Failed to count samples in {source.path}: {e}")

            elif source.source_type == DataSourceType.DATABASE:
                try:
                    import sqlite3
                    conn = sqlite3.connect(source.path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM games WHERE status = 'completed'")
                    count = cursor.fetchone()[0]
                    conn.close()
                    total += count
                except Exception as e:
                    logger.warning(f"Failed to count games in {source.path}: {e}")

        return total

    def get_stats(self) -> PipelineStats:
        """Get current pipeline statistics."""
        # Update streaming stats if available
        if self._streaming_pipeline is not None:
            if hasattr(self._streaming_pipeline, 'get_stats'):
                stream_stats = self._streaming_pipeline.get_stats()
                self.stats.buffer_size = stream_stats.get('buffer_size', 0)
                self.stats.total_samples_ingested = stream_stats.get('total_samples_ingested', 0)
            elif hasattr(self._streaming_pipeline, 'get_aggregate_stats'):
                stream_stats = self._streaming_pipeline.get_aggregate_stats()
                self.stats.buffer_size = stream_stats.get('total_buffer_size', 0)
                self.stats.total_samples_ingested = stream_stats.get('total_samples_ingested', 0)

        # Update batch loader stats
        if self._batch_loader is not None:
            self.stats.total_samples_loaded = getattr(
                self._batch_loader, 'total_samples', 0
            )

        return self.stats

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling.

        Args:
            epoch: Current epoch number
        """
        if self._batch_loader is not None:
            self._batch_loader.set_epoch(epoch)

    def reset(self):
        """Reset the pipeline state."""
        self.stats = PipelineStats(active_sources=len([s for s in self._sources if s.enabled]))
        self._batch_times.clear()

        if self._streaming_pipeline is not None and hasattr(self._streaming_pipeline, 'reset'):
            self._streaming_pipeline.reset()

    def close(self):
        """Close all data sources and release resources."""
        if self._batch_loader is not None:
            self._batch_loader.close()
            self._batch_loader = None

        if self._streaming_pipeline is not None and self._is_running:
            # Schedule stop in event loop if running
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.stop_streaming())
                else:
                    loop.run_until_complete(self.stop_streaming())
            except RuntimeError:
                pass

        self._streaming_pipeline = None
        logger.info("DataPipelineController closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()


def create_pipeline_from_config(
    config_path: Optional[str] = None,
    **overrides,
) -> DataPipelineController:
    """Create a DataPipelineController from configuration file.

    Args:
        config_path: Path to YAML configuration file
        **overrides: Override specific configuration values

    Returns:
        Configured DataPipelineController instance
    """
    config = PipelineConfig()
    db_paths = []
    npz_paths = []

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            # Load data paths
            data_cfg = cfg.get('data', {})
            db_paths = data_cfg.get('db_paths', [])
            npz_paths = data_cfg.get('npz_paths', [])

            # Load pipeline config
            pipeline_cfg = cfg.get('pipeline', {})
            if 'mode' in pipeline_cfg:
                config.mode = PipelineMode(pipeline_cfg['mode'])
            if 'batch_size' in pipeline_cfg:
                config.batch_size = pipeline_cfg['batch_size']
            if 'shuffle' in pipeline_cfg:
                config.shuffle = pipeline_cfg['shuffle']
            if 'prefetch_count' in pipeline_cfg:
                config.prefetch_count = pipeline_cfg['prefetch_count']

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif key == 'db_paths':
            db_paths = value
        elif key == 'npz_paths':
            npz_paths = value

    return DataPipelineController(
        db_paths=db_paths,
        npz_paths=npz_paths,
        config=config,
    )


# Convenience function for backward compatibility
def get_training_data_loader(
    data_paths: List[str],
    batch_size: int = 256,
    shuffle: bool = True,
    **kwargs,
) -> DataPipelineController:
    """Create a data pipeline for training.

    This is a convenience function that provides backward compatibility
    with older code expecting a simple data loader interface.

    Args:
        data_paths: Paths to training data files
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        **kwargs: Additional configuration options

    Returns:
        DataPipelineController configured for batch loading
    """
    # Categorize paths by type
    db_paths = []
    npz_paths = []

    for path in data_paths:
        if not os.path.exists(path):
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext == '.db':
            db_paths.append(path)
        elif ext in ('.npz', '.npy', '.h5', '.hdf5'):
            npz_paths.append(path)

    config = PipelineConfig(
        mode=PipelineMode.BATCH,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return DataPipelineController(
        db_paths=db_paths,
        npz_paths=npz_paths,
        config=config,
    )
