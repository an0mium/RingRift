"""Tournament recording helpers for canonical GameReplayDB output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.db.unified_recording import RecordSource, RecordingConfig


@dataclass(frozen=True)
class TournamentRecordingOptions:
    """Configuration for tournament game recording."""

    enabled: bool = True
    source: str = RecordSource.TOURNAMENT
    engine_mode: str | None = None  # None = derive from actual engine config
    db_prefix: str = "tournament"
    db_dir: str = "data/games"
    fsm_validation: bool = True
    parity_mode: str | None = None
    snapshot_interval: int | None = None
    store_history_entries: bool = True
    tags: list[str] = field(default_factory=list)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def build_recording_config(
        self,
        board_type: str,
        num_players: int,
        *,
        model_id: str | None = None,
        difficulty: int | None = None,
        actual_engine_mode: str | None = None,
    ) -> RecordingConfig:
        """Build a RecordingConfig for the given match settings.

        Args:
            board_type: Board configuration (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)
            model_id: Optional model identifier
            difficulty: Optional difficulty level
            actual_engine_mode: The actual engine mode used (e.g., "gumbel-mcts").
                               If provided, overrides the default engine_mode.
        """
        # Use actual engine mode if provided, otherwise fall back to configured default
        effective_engine_mode = actual_engine_mode or self.engine_mode or "tournament"
        return RecordingConfig(
            board_type=board_type,
            num_players=num_players,
            source=self.source,
            engine_mode=effective_engine_mode,
            model_id=model_id,
            difficulty=difficulty,
            tags=list(self.tags),
            db_prefix=self.db_prefix,
            db_dir=self.db_dir,
            store_history_entries=self.store_history_entries,
            snapshot_interval=self.snapshot_interval,
            parity_mode=self.parity_mode,
            fsm_validation=self.fsm_validation,
        )
