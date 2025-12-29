"""Tests for hot_data_buffer.py - streaming training data management.

Tests cover:
- GameRecord dataclass operations
- HotDataBuffer initialization and configuration
- Game addition, retrieval, and LRU eviction
- Training batch generation
- Priority experience replay
- Quality lookup functionality
- Flush operations to JSONL
- Thread safety
- Edge cases and error handling

December 2025 - Test coverage for critical untested module.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.training.hot_data_buffer import (
    GameRecord,
    HotDataBuffer,
)


# =============================================================================
# Test Helpers
# =============================================================================


def create_test_move(
    state_features: list[float] | None = None,
    policy_target: list[float] | None = None,
    value_target: float = 0.5,
) -> dict[str, Any]:
    """Create a test move with training data."""
    if state_features is None:
        state_features = list(np.random.randn(64).astype(float))
    if policy_target is None:
        policy_target = list(np.random.dirichlet(np.ones(64)).astype(float))
    return {
        "state_features": state_features,
        "global_features": list(np.random.randn(10).astype(float)),
        "policy_target": policy_target,
        "value_target": value_target,
        "action": {"type": "PLACE_RING", "position": 0},
    }


def create_test_game_record(
    game_id: str = "test-game",
    board_type: str = "hex8",
    num_players: int = 2,
    num_moves: int = 10,
    avg_elo: float = 1500.0,
    priority: float = 1.0,
    manifest_quality: float = 0.5,
) -> GameRecord:
    """Create a test GameRecord with valid training data."""
    # Use varied value_targets to enable shuffle detection in tests
    moves = [create_test_move(value_target=i / max(num_moves, 1)) for i in range(num_moves)]
    outcome = {f"player_{i}": 1.0 / num_players for i in range(num_players)}

    return GameRecord(
        game_id=game_id,
        board_type=board_type,
        num_players=num_players,
        moves=moves,
        outcome=outcome,
        timestamp=time.time(),
        source="test",
        avg_elo=avg_elo,
        priority=priority,
        from_promoted_model=False,
        manifest_quality=manifest_quality,
    )


# =============================================================================
# GameRecord Tests
# =============================================================================


class TestGameRecord:
    """Tests for GameRecord dataclass."""

    def test_game_record_creation_minimal(self) -> None:
        """Test creating GameRecord with minimal required fields."""
        record = GameRecord(
            game_id="test-game-001",
            board_type="hex8",
            num_players=2,
            moves=[],
            outcome={},
        )
        assert record.game_id == "test-game-001"
        assert record.board_type == "hex8"
        assert record.num_players == 2
        assert record.moves == []
        assert record.outcome == {}

    def test_game_record_creation_full(self) -> None:
        """Test creating GameRecord with all fields."""
        now = time.time()
        record = GameRecord(
            game_id="test-game-002",
            board_type="square8",
            num_players=4,
            moves=[{"action": "test"}],
            outcome={"player_0": 1.0, "player_1": 0.0},
            timestamp=now,
            source="selfplay",
            avg_elo=1600.0,
            priority=2.0,
            from_promoted_model=True,
            manifest_quality=0.85,
        )
        assert record.timestamp == now
        assert record.source == "selfplay"
        assert record.avg_elo == 1600.0
        assert record.priority == 2.0
        assert record.from_promoted_model is True
        assert record.manifest_quality == 0.85

    def test_game_record_to_training_samples(self) -> None:
        """Test converting GameRecord to training samples."""
        moves = [create_test_move(value_target=i * 0.25) for i in range(4)]

        record = GameRecord(
            game_id="test-game-003",
            board_type="hex8",
            num_players=2,
            moves=moves,
            outcome={"player_0": 1.0, "player_1": 0.0},
        )

        samples = record.to_training_samples()

        assert len(samples) == 4
        for i, sample in enumerate(samples):
            # Each sample is (board_features, global_features, policy, value)
            assert len(sample) == 4
            assert isinstance(sample[0], np.ndarray)  # board_features
            assert isinstance(sample[1], np.ndarray)  # global_features
            assert isinstance(sample[2], np.ndarray)  # policy
            assert isinstance(sample[3], float)       # value
            assert sample[3] == i * 0.25

    def test_game_record_to_training_samples_legacy(self) -> None:
        """Test legacy training sample format."""
        record = create_test_game_record(num_moves=3)
        samples = record.to_training_samples_legacy()

        assert len(samples) == 3
        for sample in samples:
            # Legacy format is (board_features, policy, value)
            assert len(sample) == 3
            assert isinstance(sample[0], np.ndarray)
            assert isinstance(sample[1], np.ndarray)
            assert isinstance(sample[2], float)

    def test_game_record_empty_moves(self) -> None:
        """Test GameRecord with no moves."""
        record = GameRecord(
            game_id="empty-game",
            board_type="hex8",
            num_players=2,
            moves=[],
            outcome={},
        )
        samples = record.to_training_samples()
        assert samples == []

    def test_game_record_default_values(self) -> None:
        """Test GameRecord default values."""
        record = GameRecord(
            game_id="defaults",
            board_type="hex8",
            num_players=2,
            moves=[],
            outcome={},
        )
        assert record.source == "hot_buffer"
        assert record.avg_elo == 1500.0
        assert record.priority == 1.0
        assert record.from_promoted_model is False
        assert record.manifest_quality == 0.5


# =============================================================================
# HotDataBuffer Initialization Tests
# =============================================================================


class TestHotDataBufferInit:
    """Tests for HotDataBuffer initialization."""

    def test_default_initialization(self) -> None:
        """Test buffer with default parameters."""
        buffer = HotDataBuffer()
        assert buffer.max_size == 1000
        assert buffer.max_memory_mb == 500
        assert buffer.buffer_name == "default"
        assert buffer.training_threshold == 100

    def test_custom_initialization(self) -> None:
        """Test buffer with custom parameters."""
        buffer = HotDataBuffer(
            max_size=500,
            max_memory_mb=250,
            buffer_name="test_buffer",
            training_threshold=50,
            batch_notification_size=25,
        )
        assert buffer.max_size == 500
        assert buffer.max_memory_mb == 250
        assert buffer.buffer_name == "test_buffer"
        assert buffer.training_threshold == 50
        assert buffer.batch_notification_size == 25

    def test_buffer_starts_empty(self) -> None:
        """Test that buffer starts empty."""
        buffer = HotDataBuffer()
        assert len(buffer) == 0
        assert buffer.game_count == 0


# =============================================================================
# Game Operations Tests
# =============================================================================


class TestGameOperations:
    """Tests for adding, getting, and removing games."""

    def test_add_game(self) -> None:
        """Test adding a game to buffer."""
        buffer = HotDataBuffer(enable_events=False)
        record = create_test_game_record("game-001")

        buffer.add_game(record)

        assert len(buffer) == 1
        assert buffer.game_count == 1
        assert "game-001" in buffer

    def test_add_multiple_games(self) -> None:
        """Test adding multiple games."""
        buffer = HotDataBuffer(enable_events=False)

        for i in range(5):
            record = create_test_game_record(f"game-{i:03d}")
            buffer.add_game(record)

        assert len(buffer) == 5

    def test_add_game_from_dict(self) -> None:
        """Test adding a game from dictionary representation."""
        buffer = HotDataBuffer(enable_events=False)

        data = {
            "game_id": "dict-game",
            "board_type": "square8",
            "num_players": 4,
            "moves": [create_test_move()],
            "outcome": {"player_0": 1.0},
            "timestamp": time.time(),
        }

        buffer.add_game_from_dict(data)

        assert len(buffer) == 1
        game = buffer.get_game("dict-game")
        assert game is not None
        assert game.board_type == "square8"
        assert game.num_players == 4

    def test_get_game(self) -> None:
        """Test retrieving a game by ID."""
        buffer = HotDataBuffer(enable_events=False)
        record = create_test_game_record("game-001")
        buffer.add_game(record)

        retrieved = buffer.get_game("game-001")

        assert retrieved is not None
        assert retrieved.game_id == "game-001"

    def test_get_nonexistent_game(self) -> None:
        """Test retrieving a game that doesn't exist."""
        buffer = HotDataBuffer(enable_events=False)

        retrieved = buffer.get_game("nonexistent")

        assert retrieved is None

    def test_remove_game(self) -> None:
        """Test removing a game."""
        buffer = HotDataBuffer(enable_events=False)
        record = create_test_game_record("game-001")
        buffer.add_game(record)

        removed = buffer.remove_game("game-001")

        assert removed is True
        assert len(buffer) == 0
        assert buffer.get_game("game-001") is None

    def test_remove_nonexistent_game(self) -> None:
        """Test removing a game that doesn't exist."""
        buffer = HotDataBuffer(enable_events=False)

        removed = buffer.remove_game("nonexistent")

        assert removed is False

    def test_get_all_games(self) -> None:
        """Test getting all games."""
        buffer = HotDataBuffer(enable_events=False)

        for i in range(5):
            buffer.add_game(create_test_game_record(f"game-{i}"))

        games = buffer.get_all_games()

        assert len(games) == 5
        assert all(isinstance(g, GameRecord) for g in games)

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when buffer is full."""
        buffer = HotDataBuffer(max_size=3, enable_events=False)

        # Add 3 games
        for i in range(3):
            buffer.add_game(create_test_game_record(f"game-{i:03d}"))

        # Add 4th game - should evict oldest (game-000)
        buffer.add_game(create_test_game_record("game-003"))

        assert len(buffer) == 3
        assert buffer.get_game("game-000") is None  # Evicted
        assert buffer.get_game("game-003") is not None  # Newest

    def test_contains(self) -> None:
        """Test __contains__ method."""
        buffer = HotDataBuffer(enable_events=False)
        buffer.add_game(create_test_game_record("game-001"))

        assert "game-001" in buffer
        assert "game-002" not in buffer


# =============================================================================
# Training Batch Tests
# =============================================================================


class TestTrainingBatch:
    """Tests for training batch generation."""

    def test_get_training_batch_basic(self) -> None:
        """Test basic batch retrieval."""
        buffer = HotDataBuffer(enable_events=False)
        for i in range(5):
            buffer.add_game(create_test_game_record(f"game-{i:03d}", num_moves=10))

        batch = buffer.get_training_batch(batch_size=32)

        # Batch is (board_features, global_features, policies, values)
        assert len(batch) == 4
        assert len(batch[0]) <= 32  # board_features
        assert len(batch[1]) <= 32  # global_features
        assert len(batch[2]) <= 32  # policies
        assert len(batch[3]) <= 32  # values

    def test_get_training_batch_empty_buffer(self) -> None:
        """Test batch retrieval from empty buffer."""
        buffer = HotDataBuffer(enable_events=False)

        batch = buffer.get_training_batch(batch_size=32)

        # Should return empty arrays
        assert len(batch) == 4
        assert len(batch[0]) == 0
        assert len(batch[1]) == 0
        assert len(batch[2]) == 0
        assert len(batch[3]) == 0

    def test_get_training_batch_shuffle(self) -> None:
        """Test that shuffled batches are randomized."""
        buffer = HotDataBuffer(enable_events=False)
        for i in range(10):
            buffer.add_game(create_test_game_record(f"game-{i}", num_moves=5))

        # Get multiple batches and check they're not identical
        batches = [buffer.get_training_batch(batch_size=10, shuffle=True) for _ in range(5)]

        # At least some batches should be different (probabilistic)
        values_lists = [tuple(b[3].tolist()) for b in batches]
        assert len(set(values_lists)) > 1  # At least some variety

    def test_get_training_batch_no_shuffle(self) -> None:
        """Test sequential (non-shuffled) batch retrieval."""
        buffer = HotDataBuffer(enable_events=False)
        buffer.add_game(create_test_game_record("game-001", num_moves=20))

        batch1 = buffer.get_training_batch(batch_size=10, shuffle=False)
        batch2 = buffer.get_training_batch(batch_size=10, shuffle=False)

        # Sequential batches should be identical
        np.testing.assert_array_equal(batch1[3], batch2[3])

    def test_total_samples_property(self) -> None:
        """Test total_samples property."""
        buffer = HotDataBuffer(enable_events=False)

        buffer.add_game(create_test_game_record("game-1", num_moves=10))
        buffer.add_game(create_test_game_record("game-2", num_moves=15))

        assert buffer.total_samples == 25


# =============================================================================
# Priority Experience Replay Tests
# =============================================================================


class TestPriorityExperienceReplay:
    """Tests for priority experience replay functionality."""

    def test_compute_game_priority_basic(self) -> None:
        """Test basic priority computation."""
        buffer = HotDataBuffer(enable_events=False)
        record = create_test_game_record(avg_elo=1500.0, manifest_quality=0.5)

        priority = buffer.compute_game_priority(record)

        assert priority > 0
        assert isinstance(priority, float)

    def test_priority_increases_with_elo(self) -> None:
        """Test that higher Elo increases priority."""
        buffer = HotDataBuffer(enable_events=False)

        low_elo = create_test_game_record(game_id="low", avg_elo=1200.0)
        high_elo = create_test_game_record(game_id="high", avg_elo=1800.0)

        low_priority = buffer.compute_game_priority(low_elo)
        high_priority = buffer.compute_game_priority(high_elo)

        assert high_priority > low_priority

    def test_priority_increases_with_quality(self) -> None:
        """Test that higher quality increases priority."""
        buffer = HotDataBuffer(enable_events=False)

        low_quality = create_test_game_record(game_id="low", manifest_quality=0.1)
        high_quality = create_test_game_record(game_id="high", manifest_quality=0.9)

        low_priority = buffer.compute_game_priority(low_quality)
        high_priority = buffer.compute_game_priority(high_quality)

        assert high_priority > low_priority

    def test_priority_promotion_bonus(self) -> None:
        """Test promotion bonus for games from promoted models."""
        buffer = HotDataBuffer(enable_events=False)

        normal = GameRecord(
            game_id="normal",
            board_type="hex8",
            num_players=2,
            moves=[],
            outcome={},
            from_promoted_model=False,
        )
        promoted = GameRecord(
            game_id="promoted",
            board_type="hex8",
            num_players=2,
            moves=[],
            outcome={},
            from_promoted_model=True,
        )

        normal_priority = buffer.compute_game_priority(normal)
        promoted_priority = buffer.compute_game_priority(promoted)

        assert promoted_priority > normal_priority

    def test_update_game_priority(self) -> None:
        """Test updating priority for a specific game."""
        buffer = HotDataBuffer(enable_events=False)
        record = create_test_game_record("update-test")
        buffer.add_game(record)

        original_priority = buffer.get_game("update-test").priority

        # Update with TD error
        buffer.update_game_priority("update-test", td_error=0.5)

        updated_priority = buffer.get_game("update-test").priority
        assert updated_priority > original_priority

    def test_update_all_priorities(self) -> None:
        """Test updating priorities for all games."""
        buffer = HotDataBuffer(enable_events=False)

        for i in range(5):
            record = create_test_game_record(f"game-{i}", avg_elo=1400 + i * 100)
            buffer.add_game(record)

        buffer.update_all_priorities(priority_alpha=0.8)

        # Just verify it doesn't crash and priorities are set
        for game in buffer.get_all_games():
            assert game.priority > 0


# =============================================================================
# Quality Lookup Tests
# =============================================================================


class TestQualityLookup:
    """Tests for quality score lookup and updates."""

    def test_set_quality_lookup(self) -> None:
        """Test setting quality lookup table."""
        buffer = HotDataBuffer(enable_events=False)

        record = create_test_game_record("game-001", manifest_quality=0.5)
        buffer.add_game(record)

        # Set quality lookup
        updated = buffer.set_quality_lookup(
            quality_lookup={"game-001": 0.9},
            elo_lookup={"game-001": 1800.0},
        )

        assert updated == 1
        game = buffer.get_game("game-001")
        assert game.manifest_quality == 0.9
        assert game.avg_elo == 1800.0

    def test_quality_lookup_updates_priority(self) -> None:
        """Test that quality lookup updates recompute priorities."""
        buffer = HotDataBuffer(enable_events=False)

        record = create_test_game_record("game-001", manifest_quality=0.5)
        buffer.add_game(record)

        original_priority = buffer.get_game("game-001").priority

        buffer.set_quality_lookup(quality_lookup={"game-001": 0.95})

        new_priority = buffer.get_game("game-001").priority
        assert new_priority != original_priority


# =============================================================================
# Flush Operations Tests
# =============================================================================


class TestFlushOperations:
    """Tests for flushing buffer to disk."""

    def test_flush_to_jsonl(self) -> None:
        """Test flushing buffer to JSONL file."""
        buffer = HotDataBuffer(enable_events=False)
        for i in range(3):
            buffer.add_game(create_test_game_record(f"flush-game-{i}"))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            output_path = Path(f.name)

        try:
            count = buffer.flush_to_jsonl(output_path)

            assert count == 3
            assert output_path.exists()

            # Verify JSONL format
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 3
            for line in lines:
                data = json.loads(line)
                assert "game_id" in data
                assert "board_type" in data
                assert "moves" in data
        finally:
            output_path.unlink(missing_ok=True)

    def test_flush_marks_games_as_flushed(self) -> None:
        """Test that flush marks games as flushed."""
        buffer = HotDataBuffer(enable_events=False)
        for i in range(3):
            buffer.add_game(create_test_game_record(f"game-{i}"))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            output_path = Path(f.name)

        try:
            buffer.flush_to_jsonl(output_path)

            # Flushing again should return 0 (already flushed)
            count = buffer.flush_to_jsonl(output_path)
            assert count == 0
        finally:
            output_path.unlink(missing_ok=True)

    def test_get_unflushed_games(self) -> None:
        """Test getting unflushed games."""
        buffer = HotDataBuffer(enable_events=False)
        for i in range(5):
            buffer.add_game(create_test_game_record(f"game-{i}"))

        unflushed = buffer.get_unflushed_games()
        assert len(unflushed) == 5

        # Mark some as flushed
        buffer.mark_flushed(["game-0", "game-1"])

        unflushed = buffer.get_unflushed_games()
        assert len(unflushed) == 3

    def test_clear_flushed(self) -> None:
        """Test clearing flushed games from buffer."""
        buffer = HotDataBuffer(enable_events=False)
        for i in range(5):
            buffer.add_game(create_test_game_record(f"game-{i}"))

        buffer.mark_flushed(["game-0", "game-1", "game-2"])

        removed = buffer.clear_flushed()

        assert removed == 3
        assert len(buffer) == 2

    def test_flush_empty_buffer(self) -> None:
        """Test flushing empty buffer."""
        buffer = HotDataBuffer(enable_events=False)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            output_path = Path(f.name)

        try:
            count = buffer.flush_to_jsonl(output_path)
            assert count == 0
        finally:
            output_path.unlink(missing_ok=True)


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for buffer statistics."""

    def test_get_statistics(self) -> None:
        """Test getting buffer statistics."""
        buffer = HotDataBuffer(enable_events=False, max_size=100)
        for i in range(5):
            buffer.add_game(create_test_game_record(
                f"stats-game-{i}",
                num_moves=10,
                avg_elo=1500 + i * 50,
            ))

        stats = buffer.get_statistics()

        assert stats["game_count"] == 5
        assert stats["max_size"] == 100
        assert stats["total_samples"] == 50
        assert "avg_priority" in stats
        assert "avg_elo" in stats
        assert stats["utilization"] == 0.05  # 5/100

    def test_statistics_with_promoted_games(self) -> None:
        """Test statistics tracking promoted model games."""
        buffer = HotDataBuffer(enable_events=False)

        # Add normal games
        for i in range(3):
            buffer.add_game(create_test_game_record(f"normal-{i}"))

        # Add promoted model games
        for i in range(2):
            record = GameRecord(
                game_id=f"promoted-{i}",
                board_type="hex8",
                num_players=2,
                moves=[create_test_move()],
                outcome={},
                from_promoted_model=True,
            )
            buffer.add_game(record)

        stats = buffer.get_statistics()

        assert stats["promoted_model_games"] == 2


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_adds(self) -> None:
        """Test concurrent game additions."""
        buffer = HotDataBuffer(enable_events=False, max_size=1000)
        errors: list[Exception] = []

        def add_games(start_idx: int) -> None:
            try:
                for i in range(10):
                    record = create_test_game_record(
                        f"thread-{start_idx}-game-{i}",
                        num_moves=5,
                    )
                    buffer.add_game(record)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_games, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(buffer) == 50

    def test_concurrent_read_write(self) -> None:
        """Test concurrent reading and writing."""
        buffer = HotDataBuffer(enable_events=False, max_size=1000)
        errors: list[Exception] = []

        # Pre-populate
        for i in range(10):
            buffer.add_game(create_test_game_record(f"pre-game-{i}", num_moves=5))

        def reader() -> None:
            try:
                for _ in range(50):
                    buffer.get_training_batch(batch_size=16)
                    buffer.get_statistics()
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(20):
                    buffer.add_game(create_test_game_record(f"new-game-{i}", num_moves=5))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_add_duplicate_game_id(self) -> None:
        """Test adding game with duplicate ID updates LRU position."""
        buffer = HotDataBuffer(max_size=5, enable_events=False)

        # Add initial games
        for i in range(3):
            buffer.add_game(create_test_game_record(f"game-{i}"))

        # Re-add game-0 (should update LRU position)
        buffer.add_game(create_test_game_record("game-0"))

        # game_count should still be 3
        assert len(buffer) == 3

    def test_very_large_batch_request(self) -> None:
        """Test requesting batch larger than buffer."""
        buffer = HotDataBuffer(enable_events=False)
        buffer.add_game(create_test_game_record("small-game", num_moves=5))

        batch = buffer.get_training_batch(batch_size=1000)

        # Should return whatever samples are available
        assert len(batch[0]) == 5

    def test_special_characters_in_game_id(self) -> None:
        """Test game IDs with special characters."""
        buffer = HotDataBuffer(enable_events=False)
        special_id = "game/with:special-chars_123"

        buffer.add_game(create_test_game_record(special_id))

        retrieved = buffer.get_game(special_id)
        assert retrieved is not None
        assert retrieved.game_id == special_id

    def test_move_without_training_data(self) -> None:
        """Test handling moves without training data."""
        buffer = HotDataBuffer(enable_events=False)

        # Moves without required fields
        record = GameRecord(
            game_id="incomplete",
            board_type="hex8",
            num_players=2,
            moves=[{"action": "test"}],  # Missing state_features, policy_target, etc.
            outcome={},
        )
        buffer.add_game(record)

        # Should still be stored
        assert len(buffer) == 1
        # But samples should be empty
        samples = record.to_training_samples()
        assert len(samples) == 0


# =============================================================================
# Sample Iterator Tests
# =============================================================================


class TestSampleIterator:
    """Tests for sample iterator functionality."""

    def test_get_sample_iterator(self) -> None:
        """Test sample iterator basic functionality."""
        buffer = HotDataBuffer(enable_events=False)
        for i in range(3):
            buffer.add_game(create_test_game_record(f"game-{i}", num_moves=10))

        batches = list(buffer.get_sample_iterator(batch_size=10, epochs=1))

        # With 30 total samples and batch_size=10, expect 3 batches
        assert len(batches) == 3

    def test_get_sample_iterator_multiple_epochs(self) -> None:
        """Test sample iterator with multiple epochs."""
        buffer = HotDataBuffer(enable_events=False)
        buffer.add_game(create_test_game_record("game-1", num_moves=20))

        batches = list(buffer.get_sample_iterator(batch_size=10, epochs=3))

        # 20 samples, batch_size=10, 3 epochs = 6 batches
        assert len(batches) == 6

    def test_get_sample_iterator_empty_buffer(self) -> None:
        """Test sample iterator on empty buffer."""
        buffer = HotDataBuffer(enable_events=False)

        batches = list(buffer.get_sample_iterator(batch_size=10, epochs=1))

        assert len(batches) == 0
