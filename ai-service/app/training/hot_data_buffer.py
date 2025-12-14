"""Hot data buffer for streaming training data.

This module provides an in-memory buffer for recently generated game data,
enabling faster training cycles by avoiding full DB roundtrips.

Key features:
1. LRU eviction when buffer exceeds max size
2. Thread-safe operations for concurrent access
3. Periodic flush to canonical DB (background)
4. Feature extraction compatible with StreamingDataLoader

Usage:
    buffer = HotDataBuffer(max_size=1000)

    # Add games as they complete
    buffer.add_game(game_record)

    # Get training samples
    features, policies, values = buffer.get_training_batch(batch_size=256)

    # Flush to DB periodically
    buffer.flush_to_db(db_path)
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class GameRecord:
    """Lightweight game record for hot buffer storage."""

    game_id: str
    board_type: str
    num_players: int
    moves: List[Dict[str, Any]]  # List of move records with state/action
    outcome: Dict[str, float]  # Player ID -> final score
    timestamp: float = field(default_factory=time.time)
    source: str = "hot_buffer"

    def to_training_samples(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Convert game to training samples (state, policy, value)."""
        # This is a placeholder - actual implementation depends on encoder
        samples = []
        for move in self.moves:
            state = move.get("state_features")
            policy = move.get("policy_target")
            value = move.get("value_target")
            if state is not None and policy is not None and value is not None:
                samples.append((
                    np.array(state, dtype=np.float32),
                    np.array(policy, dtype=np.float32),
                    float(value),
                ))
        return samples


class HotDataBuffer:
    """Thread-safe in-memory buffer for recent game data.

    Implements LRU eviction when buffer exceeds max_size.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 500,
    ):
        """Initialize the hot data buffer.

        Args:
            max_size: Maximum number of games to keep in buffer
            max_memory_mb: Soft memory limit in MB (triggers eviction)
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._buffer: OrderedDict[str, GameRecord] = OrderedDict()
        self._lock = threading.RLock()
        self._sample_cache: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self._cache_dirty = True
        self._total_samples = 0
        self._flushed_game_ids: set = set()

    def add_game(self, game: GameRecord) -> None:
        """Add a game to the buffer (thread-safe)."""
        with self._lock:
            # Remove if already exists (to update LRU position)
            if game.game_id in self._buffer:
                del self._buffer[game.game_id]

            self._buffer[game.game_id] = game
            self._cache_dirty = True

            # Evict oldest entries if over capacity
            while len(self._buffer) > self.max_size:
                self._buffer.popitem(last=False)

    def add_game_from_dict(self, data: Dict[str, Any]) -> None:
        """Add a game from a dictionary representation."""
        game = GameRecord(
            game_id=str(data.get("game_id", "")),
            board_type=str(data.get("board_type", "square8")),
            num_players=int(data.get("num_players", 2)),
            moves=data.get("moves", []),
            outcome=data.get("outcome", {}),
            timestamp=float(data.get("timestamp", time.time())),
            source=str(data.get("source", "hot_buffer")),
        )
        self.add_game(game)

    def get_game(self, game_id: str) -> Optional[GameRecord]:
        """Get a game by ID (thread-safe)."""
        with self._lock:
            return self._buffer.get(game_id)

    def remove_game(self, game_id: str) -> bool:
        """Remove a game from the buffer (thread-safe)."""
        with self._lock:
            if game_id in self._buffer:
                del self._buffer[game_id]
                self._cache_dirty = True
                return True
            return False

    def get_all_games(self) -> List[GameRecord]:
        """Get all games in the buffer (thread-safe copy)."""
        with self._lock:
            return list(self._buffer.values())

    def get_unflushed_games(self) -> List[GameRecord]:
        """Get games that haven't been flushed to DB yet."""
        with self._lock:
            return [
                g for g in self._buffer.values()
                if g.game_id not in self._flushed_game_ids
            ]

    def __len__(self) -> int:
        """Return number of games in buffer."""
        with self._lock:
            return len(self._buffer)

    def __contains__(self, game_id: str) -> bool:
        """Check if game is in buffer."""
        with self._lock:
            return game_id in self._buffer

    def _rebuild_sample_cache(self) -> None:
        """Rebuild the flattened sample cache from all games."""
        samples = []
        for game in self._buffer.values():
            samples.extend(game.to_training_samples())
        self._sample_cache = samples
        self._total_samples = len(samples)
        self._cache_dirty = False

    def get_training_batch(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a batch of training samples.

        Args:
            batch_size: Number of samples to return
            shuffle: Whether to randomly sample (vs sequential)

        Returns:
            Tuple of (states, policies, values) as numpy arrays
        """
        with self._lock:
            if self._cache_dirty:
                self._rebuild_sample_cache()

            if not self._sample_cache:
                # Return empty arrays with correct shapes
                return (
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )

            if shuffle:
                indices = np.random.choice(
                    len(self._sample_cache),
                    size=min(batch_size, len(self._sample_cache)),
                    replace=False,
                )
            else:
                indices = list(range(min(batch_size, len(self._sample_cache))))

            states = []
            policies = []
            values = []

            for idx in indices:
                s, p, v = self._sample_cache[idx]
                states.append(s)
                policies.append(p)
                values.append(v)

            return (
                np.array(states, dtype=np.float32),
                np.array(policies, dtype=np.float32),
                np.array(values, dtype=np.float32),
            )

    def get_sample_iterator(
        self,
        batch_size: int = 256,
        epochs: int = 1,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Iterator over training samples for multiple epochs.

        Yields batches of (states, policies, values).
        """
        with self._lock:
            if self._cache_dirty:
                self._rebuild_sample_cache()

            if not self._sample_cache:
                return

            for _ in range(epochs):
                # Shuffle samples for each epoch
                indices = np.random.permutation(len(self._sample_cache))

                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    states = []
                    policies = []
                    values = []

                    for idx in batch_indices:
                        s, p, v = self._sample_cache[idx]
                        states.append(s)
                        policies.append(p)
                        values.append(v)

                    yield (
                        np.array(states, dtype=np.float32),
                        np.array(policies, dtype=np.float32),
                        np.array(values, dtype=np.float32),
                    )

    @property
    def total_samples(self) -> int:
        """Total number of training samples in buffer."""
        with self._lock:
            if self._cache_dirty:
                self._rebuild_sample_cache()
            return self._total_samples

    @property
    def game_count(self) -> int:
        """Number of games in buffer."""
        return len(self)

    def mark_flushed(self, game_ids: List[str]) -> None:
        """Mark games as flushed to persistent storage."""
        with self._lock:
            self._flushed_game_ids.update(game_ids)

    def clear_flushed(self) -> int:
        """Remove games that have been flushed to persistent storage.

        Returns number of games removed.
        """
        with self._lock:
            to_remove = [
                gid for gid in self._buffer.keys()
                if gid in self._flushed_game_ids
            ]
            for gid in to_remove:
                del self._buffer[gid]
            self._flushed_game_ids.clear()
            self._cache_dirty = True
            return len(to_remove)

    def flush_to_jsonl(self, path: Path) -> int:
        """Flush unflushed games to JSONL file.

        Returns number of games written.
        """
        games = self.get_unflushed_games()
        if not games:
            return 0

        path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with open(path, "a", encoding="utf-8") as f:
            for game in games:
                record = {
                    "game_id": game.game_id,
                    "board_type": game.board_type,
                    "num_players": game.num_players,
                    "moves": game.moves,
                    "outcome": game.outcome,
                    "timestamp": game.timestamp,
                    "source": game.source,
                }
                f.write(json.dumps(record) + "\n")
                written += 1

        self.mark_flushed([g.game_id for g in games])
        return written

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            if self._cache_dirty:
                self._rebuild_sample_cache()

            return {
                "game_count": len(self._buffer),
                "max_size": self.max_size,
                "total_samples": self._total_samples,
                "flushed_count": len(self._flushed_game_ids),
                "unflushed_count": len(self.get_unflushed_games()),
                "utilization": len(self._buffer) / self.max_size if self.max_size > 0 else 0,
            }

    def clear(self) -> None:
        """Clear all games from buffer."""
        with self._lock:
            self._buffer.clear()
            self._sample_cache.clear()
            self._flushed_game_ids.clear()
            self._total_samples = 0
            self._cache_dirty = True


def create_hot_buffer(
    *,
    max_size: int = 1000,
    max_memory_mb: int = 500,
) -> HotDataBuffer:
    """Factory function to create a hot data buffer."""
    return HotDataBuffer(max_size=max_size, max_memory_mb=max_memory_mb)
