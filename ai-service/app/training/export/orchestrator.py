"""High-level orchestrator for training data export.

This module provides a unified export pipeline using the modular components:
- GameIterator: Database iteration and filtering
- SampleCollector: Sample encoding and collection
- ArrayBuilder: Array accumulation and stacking
- NPZExportWriter: NPZ output with validation

Usage:
    from app.training.export.orchestrator import export_dataset, ExportOrchestrator

    # Simple usage
    result = export_dataset(
        db_paths=["data/games/selfplay.db"],
        board_type="hex8",
        num_players=2,
        output_path=Path("data/training/hex8_2p.npz"),
    )

    # Advanced usage with full configuration
    orchestrator = ExportOrchestrator()
    result = orchestrator.export(
        db_paths=db_paths,
        export_config=ExportConfig(...),
        filter_config=FilterConfig(...),
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.training.export.array_builder import ArrayBuilder, BuiltArrays, Sample
from app.training.export.config import ExportConfig, ExportResult, FilterConfig
from app.training.export.game_processor import (
    GameData,
    GameIterator,
    GameIteratorConfig,
    IterationStats,
)
from app.training.export.npz_writer import NPZExportWriter, WriteResult
from app.training.export.sample_collector import (
    CollectionResult,
    GameMetadata,
    SampleCollector,
    SampleCollectorConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ExportProgress:
    """Progress tracking for export operation.

    Attributes:
        databases_processed: Number of databases processed
        databases_total: Total number of databases
        games_processed: Number of games processed
        samples_collected: Number of samples collected
        current_database: Current database being processed
    """

    databases_processed: int = 0
    databases_total: int = 0
    games_processed: int = 0
    samples_collected: int = 0
    current_database: str = ""


class ExportOrchestrator:
    """Orchestrates the export pipeline using modular components.

    This class coordinates the interaction between:
    - GameIterator for database iteration
    - SampleCollector for sample encoding
    - ArrayBuilder for array accumulation
    - NPZExportWriter for output

    Example:
        orchestrator = ExportOrchestrator()
        result = orchestrator.export(
            db_paths=[Path("data/games/selfplay.db")],
            export_config=ExportConfig(
                board_type="hex8",
                num_players=2,
                output_path=Path("data/training/hex8_2p.npz"),
            ),
        )
        if result.success:
            print(f"Exported {result.total_samples} samples")
    """

    def __init__(self) -> None:
        """Initialize orchestrator."""
        self._iterator: GameIterator | None = None
        self._collector: SampleCollector | None = None
        self._builder: ArrayBuilder | None = None
        self._writer: NPZExportWriter | None = None

    def export(
        self,
        db_paths: list[str | Path],
        export_config: ExportConfig,
        filter_config: FilterConfig | None = None,
        *,
        progress_callback: Any | None = None,
    ) -> ExportResult:
        """Execute the full export pipeline.

        Args:
            db_paths: List of database paths to process
            export_config: Export configuration
            filter_config: Optional filter configuration
            progress_callback: Optional callback for progress updates

        Returns:
            ExportResult with outcome and statistics
        """
        import time

        start_time = time.time()

        if filter_config is None:
            filter_config = FilterConfig()

        # Initialize components
        iterator_config = GameIteratorConfig(
            board_type=export_config.board_type,
            num_players=export_config.num_players,
            require_moves=True,
            require_completed=filter_config.require_completed,
            min_moves=filter_config.min_moves,
            max_moves=filter_config.max_moves,
            min_quality=filter_config.min_quality_score,
            exclude_recovery=filter_config.exclude_recovery,
            parity_fixtures_dir=export_config.parity_fixtures_dir,
            max_games=export_config.max_games,
        )

        collector_config = SampleCollectorConfig(
            board_type=export_config.board_type,
            num_players=export_config.num_players,
            history_length=export_config.history_length,
            feature_version=export_config.feature_version,
            encoder_version=export_config.encoder_version,
            use_board_aware_encoding=export_config.use_board_aware_encoding,
            use_rank_aware_values=export_config.use_rank_aware_values,
            sample_every=export_config.sample_every,
            include_heuristics=export_config.include_heuristics,
            full_heuristics=export_config.full_heuristics,
        )

        self._iterator = GameIterator(iterator_config)
        self._collector = SampleCollector(collector_config)
        self._builder = ArrayBuilder()
        self._writer = NPZExportWriter(export_config)

        # Track progress
        progress = ExportProgress(
            databases_total=len(db_paths),
        )
        iteration_stats = IterationStats()

        # Process games
        db_paths_str = [str(p) for p in db_paths]

        try:
            for game_data in self._iterator.iterate_databases(
                db_paths_str, stats=iteration_stats
            ):
                # Create metadata for collector
                metadata = GameMetadata(
                    game_id=game_data.game_id,
                    victory_type=game_data.victory_type,
                    engine_mode=game_data.engine_mode,
                    opponent_elo=game_data.opponent_elo,
                    opponent_type=game_data.opponent_type,
                    quality_score=game_data.quality_score,
                    timestamp=game_data.timestamp,
                    db_winner=game_data.db_winner,
                    move_probs=game_data.move_probs,
                )

                # Collect samples from game
                result = self._collector.collect_from_game(
                    initial_state=game_data.initial_state,
                    moves=game_data.moves,
                    metadata=metadata,
                    max_move_index=game_data.max_safe_move_index,
                )

                if result.success and result.samples:
                    for sample in result.samples:
                        self._builder.add_sample(sample)
                    progress.samples_collected += len(result.samples)

                progress.games_processed += 1

                # Report progress
                if progress_callback is not None:
                    progress_callback(progress)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                output_path=export_config.output_path,
                total_games=iteration_stats.games_processed,
                total_samples=progress.samples_collected,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

        # Check if we have any samples
        if progress.samples_collected == 0:
            return ExportResult(
                success=False,
                output_path=export_config.output_path,
                total_games=iteration_stats.games_processed,
                total_samples=0,
                duration_seconds=time.time() - start_time,
                error="No samples collected from databases",
            )

        # Build arrays
        try:
            arrays = self._builder.build_arrays()
        except Exception as e:
            logger.error(f"Failed to build arrays: {e}")
            return ExportResult(
                success=False,
                output_path=export_config.output_path,
                total_games=iteration_stats.games_processed,
                total_samples=progress.samples_collected,
                duration_seconds=time.time() - start_time,
                error=f"Array building failed: {e}",
            )

        # Write NPZ file
        write_result = self._writer.write(
            arrays=arrays,
            games_processed=iteration_stats.games_processed,
            newest_game_time=iteration_stats.newest_game_time,
        )

        duration = time.time() - start_time

        return ExportResult(
            success=write_result.success,
            output_path=export_config.output_path,
            total_games=iteration_stats.games_processed,
            total_samples=arrays.sample_count,
            duration_seconds=duration,
            games_skipped=iteration_stats.games_skipped,
            games_deduplicated=iteration_stats.games_deduplicated,
            error=write_result.error if not write_result.success else None,
        )


def export_dataset(
    db_paths: list[str | Path],
    board_type: str,
    num_players: int,
    output_path: Path | str,
    *,
    history_length: int = 3,
    feature_version: int = 2,
    encoder_version: str = "default",
    sample_every: int = 1,
    max_games: int | None = None,
    require_completed: bool = False,
    min_moves: int | None = None,
    max_moves: int | None = None,
    min_quality: float | None = None,
    use_rank_aware_values: bool = True,
    use_board_aware_encoding: bool = True,
    include_heuristics: bool = False,
    full_heuristics: bool = False,
    exclude_recovery: bool = False,
    parity_fixtures_dir: str | None = None,
    append: bool = False,
    progress_callback: Any | None = None,
) -> ExportResult:
    """Export training dataset from game databases.

    This is a convenience function that wraps ExportOrchestrator with
    a simpler interface for common use cases.

    Args:
        db_paths: List of database paths to process
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players (2, 3, 4)
        output_path: Path for output NPZ file
        history_length: Number of history frames (default: 3)
        feature_version: Feature encoding version (default: 2)
        encoder_version: Encoder version (default, v2, v3)
        sample_every: Sample every Nth move (default: 1)
        max_games: Maximum games to process
        require_completed: Only include completed games
        min_moves: Minimum moves per game
        max_moves: Maximum moves per game
        min_quality: Minimum quality score
        use_rank_aware_values: Use rank-aware values for multiplayer
        use_board_aware_encoding: Use board-specific policy encoding
        include_heuristics: Include heuristic features
        full_heuristics: Use full 49-feature extraction
        exclude_recovery: Exclude games with recovery moves
        parity_fixtures_dir: Directory with parity fixtures
        append: Append to existing output
        progress_callback: Optional progress callback

    Returns:
        ExportResult with outcome and statistics
    """
    export_config = ExportConfig(
        board_type=board_type,
        num_players=num_players,
        output_path=Path(output_path),
        history_length=history_length,
        feature_version=feature_version,
        encoder_version=encoder_version,
        sample_every=sample_every,
        max_games=max_games,
        use_rank_aware_values=use_rank_aware_values,
        use_board_aware_encoding=use_board_aware_encoding,
        include_heuristics=include_heuristics,
        full_heuristics=full_heuristics,
        parity_fixtures_dir=parity_fixtures_dir,
        append=append,
    )

    filter_config = FilterConfig(
        require_completed=require_completed,
        min_moves=min_moves,
        max_moves=max_moves,
        min_quality_score=min_quality,
        exclude_recovery=exclude_recovery,
    )

    orchestrator = ExportOrchestrator()
    return orchestrator.export(
        db_paths=db_paths,
        export_config=export_config,
        filter_config=filter_config,
        progress_callback=progress_callback,
    )
