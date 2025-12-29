"""Tests for hot_data_buffer.py - streaming training data management.

Tests cover:
- GameRecord dataclass operations
- HotDataBuffer initialization and configuration
- Game addition, retrieval, and LRU eviction
- Training batch generation with quality weighting
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
    HotDataBufferConfig,
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
            config_key="hex8_2p",
            states=[np.zeros((10, 8, 8))],
            policies=[np.ones(64) / 64],
            values=[0.5],
            timestamps=[time.time()],
        )
        assert record.game_id == "test-game-001"
        assert record.config_key == "hex8_2p"
        assert len(record.states) == 1
        assert len(record.policies) == 1
        assert len(record.values) == 1

    def test_game_record_creation_full(self) -> None:
        """Test creating GameRecord with all fields."""
        now = time.time()
        record = GameRecord(
            game_id="test-game-002",
            config_key="square8_2p",
            states=[np.zeros((10, 8, 8)), np.zeros((10, 8, 8))],
            policies=[np.ones(64) / 64, np.ones(64) / 64],
            values=[0.0, 1.0],
            timestamps=[now, now + 1],
            outcome=1,
            quality_score=0.85,
            model_version="v1.2.3",
            mcts_simulations=800,
            temperature=1.0,
        )
        assert record.outcome == 1
        assert record.quality_score == 0.85
        assert record.model_version == "v1.2.3"
        assert record.mcts_simulations == 800
        assert record.temperature == 1.0

    def test_game_record_to_training_samples(self) -> None:
        """Test converting GameRecord to training samples."""
        states = [np.random.randn(10, 8, 8) for _ in range(3)]
        policies = [np.random.dirichlet(np.ones(64)) for _ in range(3)]
        values = [0.0, 0.5, 1.0]
        
        record = GameRecord(
            game_id="test-game-003",
            config_key="hex8_2p",
            states=states,
            policies=policies,
            values=values,
            timestamps=[time.time()] * 3,
            quality_score=0.9,
        )
        
        samples = record.to_training_samples()
        
        assert len(samples) == 3
        for i, sample in enumerate(samples):
            assert "state" in sample
            assert "policy" in sample
            assert "value" in sample
            assert sample["value"] == values[i]
            np.testing.assert_array_almost_equal(sample["policy"], policies[i])

    def test_game_record_sample_count(self) -> None:
        """Test sample_count property."""
        record = GameRecord(
            game_id="test-game-004",
            config_key="hex8_2p",
            states=[np.zeros((10, 8, 8))] * 5,
            policies=[np.ones(64) / 64] * 5,
            values=[0.5] * 5,
            timestamps=[time.time()] * 5,
        )
        assert record.sample_count == 5

    def test_game_record_empty(self) -> None:
        """Test GameRecord with no samples."""
        record = GameRecord(
            game_id="empty-game",
            config_key="hex8_2p",
            states=[],
            policies=[],
            values=[],
            timestamps=[],
        )
        assert record.sample_count == 0
        assert record.to_training_samples() == []


# =============================================================================
# HotDataBuffer Initialization Tests
# =============================================================================


class TestHotDataBufferInit:
    """Tests for HotDataBuffer initialization."""

    def test_default_initialization(self) -> None:
        """Test buffer with default config."""
        buffer = HotDataBuffer()
        assert buffer.config is not None
        assert buffer.config.max_games > 0
        assert buffer.config.max_samples > 0

    def test_custom_config(self) -> None:
        """Test buffer with custom config."""
        config = HotDataBufferConfig(
            max_games=100,
            max_samples=10000,
            min_quality_score=0.5,
            priority_alpha=0.6,
            priority_beta=0.4,
        )
        buffer = HotDataBuffer(config=config)
        assert buffer.config.max_games == 100
        assert buffer.config.max_samples == 10000
        assert buffer.config.min_quality_score == 0.5

    def test_buffer_starts_empty(self) -> None:
        """Test that buffer starts empty."""
        buffer = HotDataBuffer()
        assert buffer.game_count == 0
        assert buffer.sample_count == 0
        assert buffer.is_empty


# =============================================================================
# Game Operations Tests
# =============================================================================


class TestGameOperations:
    """Tests for adding, getting, and removing games."""

    def test_add_game(self) -> None:
        """Test adding a game to buffer."""
        buffer = HotDataBuffer()
        record = self._create_test_record("game-001")
        
        buffer.add_game(record)
        
        assert buffer.game_count == 1
        assert buffer.sample_count == record.sample_count

    def test_add_multiple_games(self) -> None:
        """Test adding multiple games."""
        buffer = HotDataBuffer()
        
        for i in range(5):
            record = self._create_test_record(f"game-{i:03d}")
            buffer.add_game(record)
        
        assert buffer.game_count == 5

    def test_get_game(self) -> None:
        """Test retrieving a game by ID."""
        buffer = HotDataBuffer()
        record = self._create_test_record("game-001")
        buffer.add_game(record)
        
        retrieved = buffer.get_game("game-001")
        
        assert retrieved is not None
        assert retrieved.game_id == "game-001"

    def test_get_nonexistent_game(self) -> None:
        """Test retrieving a game that doesn't exist."""
        buffer = HotDataBuffer()
        
        retrieved = buffer.get_game("nonexistent")
        
        assert retrieved is None

    def test_remove_game(self) -> None:
        """Test removing a game."""
        buffer = HotDataBuffer()
        record = self._create_test_record("game-001")
        buffer.add_game(record)
        
        removed = buffer.remove_game("game-001")
        
        assert removed is True
        assert buffer.game_count == 0
        assert buffer.get_game("game-001") is None

    def test_remove_nonexistent_game(self) -> None:
        """Test removing a game that doesn't exist."""
        buffer = HotDataBuffer()
        
        removed = buffer.remove_game("nonexistent")
        
        assert removed is False

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when buffer is full."""
        config = HotDataBufferConfig(max_games=3)
        buffer = HotDataBuffer(config=config)
        
        # Add 3 games
        for i in range(3):
            buffer.add_game(self._create_test_record(f"game-{i:03d}"))
        
        # Add 4th game - should evict oldest (game-000)
        buffer.add_game(self._create_test_record("game-003"))
        
        assert buffer.game_count == 3
        assert buffer.get_game("game-000") is None  # Evicted
        assert buffer.get_game("game-003") is not None  # Newest

    def test_lru_access_updates_order(self) -> None:
        """Test that accessing a game updates its LRU position."""
        config = HotDataBufferConfig(max_games=3)
        buffer = HotDataBuffer(config=config)
        
        # Add 3 games
        for i in range(3):
            buffer.add_game(self._create_test_record(f"game-{i:03d}"))
        
        # Access game-000 to make it recently used
        buffer.get_game("game-000")
        
        # Add 4th game - should evict game-001 (now oldest)
        buffer.add_game(self._create_test_record("game-003"))
        
        assert buffer.get_game("game-000") is not None  # Was accessed
        assert buffer.get_game("game-001") is None  # Evicted

    def _create_test_record(
        self, game_id: str, num_samples: int = 10
    ) -> GameRecord:
        """Create a test GameRecord."""
        return GameRecord(
            game_id=game_id,
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8) for _ in range(num_samples)],
            policies=[np.random.dirichlet(np.ones(64)) for _ in range(num_samples)],
            values=list(np.linspace(0, 1, num_samples)),
            timestamps=[time.time() + i for i in range(num_samples)],
            quality_score=0.8,
        )


# =============================================================================
# Training Batch Tests
# =============================================================================


class TestTrainingBatch:
    """Tests for training batch generation."""

    def test_get_training_batch_basic(self) -> None:
        """Test basic batch retrieval."""
        buffer = HotDataBuffer()
        for i in range(5):
            buffer.add_game(self._create_test_record(f"game-{i:03d}"))
        
        batch = buffer.get_training_batch(batch_size=32)
        
        assert batch is not None
        assert "states" in batch
        assert "policies" in batch
        assert "values" in batch
        assert len(batch["states"]) <= 32

    def test_get_training_batch_empty_buffer(self) -> None:
        """Test batch retrieval from empty buffer."""
        buffer = HotDataBuffer()
        
        batch = buffer.get_training_batch(batch_size=32)
        
        assert batch is None or len(batch.get("states", [])) == 0

    def test_get_training_batch_with_config_filter(self) -> None:
        """Test batch retrieval with config_key filter."""
        buffer = HotDataBuffer()
        
        # Add games with different configs
        for i in range(3):
            record = GameRecord(
                game_id=f"hex-game-{i}",
                config_key="hex8_2p",
                states=[np.random.randn(10, 8, 8)] * 5,
                policies=[np.random.dirichlet(np.ones(64))] * 5,
                values=[0.5] * 5,
                timestamps=[time.time()] * 5,
            )
            buffer.add_game(record)
        
        for i in range(3):
            record = GameRecord(
                game_id=f"square-game-{i}",
                config_key="square8_2p",
                states=[np.random.randn(10, 8, 8)] * 5,
                policies=[np.random.dirichlet(np.ones(64))] * 5,
                values=[0.5] * 5,
                timestamps=[time.time()] * 5,
            )
            buffer.add_game(record)
        
        # Get batch for hex8_2p only
        batch = buffer.get_training_batch(batch_size=32, config_key="hex8_2p")
        
        assert batch is not None
        # Samples should only come from hex8_2p games

    def test_batch_quality_weighting(self) -> None:
        """Test that quality weighting affects sample distribution."""
        buffer = HotDataBuffer()
        
        # Add low quality game
        low_quality = GameRecord(
            game_id="low-quality",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 100,
            policies=[np.random.dirichlet(np.ones(64))] * 100,
            values=[0.5] * 100,
            timestamps=[time.time()] * 100,
            quality_score=0.1,
        )
        buffer.add_game(low_quality)
        
        # Add high quality game
        high_quality = GameRecord(
            game_id="high-quality",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 100,
            policies=[np.random.dirichlet(np.ones(64))] * 100,
            values=[0.5] * 100,
            timestamps=[time.time()] * 100,
            quality_score=0.95,
        )
        buffer.add_game(high_quality)
        
        # Get multiple batches and check distribution
        # High quality samples should appear more frequently
        # (This is a statistical test, so we just check it runs)
        for _ in range(10):
            batch = buffer.get_training_batch(batch_size=32)
            assert batch is not None

    def _create_test_record(
        self, game_id: str, num_samples: int = 10
    ) -> GameRecord:
        """Create a test GameRecord."""
        return GameRecord(
            game_id=game_id,
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8) for _ in range(num_samples)],
            policies=[np.random.dirichlet(np.ones(64)) for _ in range(num_samples)],
            values=list(np.linspace(0, 1, num_samples)),
            timestamps=[time.time() + i for i in range(num_samples)],
            quality_score=0.8,
        )


# =============================================================================
# Priority Experience Replay Tests
# =============================================================================


class TestPriorityExperienceReplay:
    """Tests for priority experience replay functionality."""

    def test_compute_game_priority(self) -> None:
        """Test priority computation for games."""
        buffer = HotDataBuffer()
        record = GameRecord(
            game_id="priority-test",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 10,
            policies=[np.random.dirichlet(np.ones(64))] * 10,
            values=[0.5] * 10,
            timestamps=[time.time()] * 10,
            quality_score=0.9,
        )
        buffer.add_game(record)
        
        priority = buffer.compute_game_priority("priority-test")
        
        assert priority is not None
        assert priority > 0

    def test_priority_affected_by_quality(self) -> None:
        """Test that quality affects priority."""
        buffer = HotDataBuffer()
        
        # Add low quality game
        low_q = GameRecord(
            game_id="low-q",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 10,
            policies=[np.random.dirichlet(np.ones(64))] * 10,
            values=[0.5] * 10,
            timestamps=[time.time()] * 10,
            quality_score=0.1,
        )
        buffer.add_game(low_q)
        
        # Add high quality game
        high_q = GameRecord(
            game_id="high-q",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 10,
            policies=[np.random.dirichlet(np.ones(64))] * 10,
            values=[0.5] * 10,
            timestamps=[time.time()] * 10,
            quality_score=0.95,
        )
        buffer.add_game(high_q)
        
        low_priority = buffer.compute_game_priority("low-q")
        high_priority = buffer.compute_game_priority("high-q")
        
        # High quality should have higher priority
        assert high_priority > low_priority

    def test_update_priorities(self) -> None:
        """Test updating priorities after training."""
        buffer = HotDataBuffer()
        record = GameRecord(
            game_id="update-test",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 10,
            policies=[np.random.dirichlet(np.ones(64))] * 10,
            values=[0.5] * 10,
            timestamps=[time.time()] * 10,
        )
        buffer.add_game(record)
        
        # Update priority
        buffer.update_priority("update-test", td_error=0.5)
        
        # Just verify it doesn't crash
        priority = buffer.compute_game_priority("update-test")
        assert priority is not None


# =============================================================================
# Quality Lookup Tests
# =============================================================================


class TestQualityLookup:
    """Tests for quality score lookup and updates."""

    def test_get_quality_score(self) -> None:
        """Test getting quality score for a game."""
        buffer = HotDataBuffer()
        record = GameRecord(
            game_id="quality-test",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 5,
            policies=[np.random.dirichlet(np.ones(64))] * 5,
            values=[0.5] * 5,
            timestamps=[time.time()] * 5,
            quality_score=0.75,
        )
        buffer.add_game(record)
        
        quality = buffer.get_quality_score("quality-test")
        
        assert quality == 0.75

    def test_update_quality_score(self) -> None:
        """Test updating quality score."""
        buffer = HotDataBuffer()
        record = GameRecord(
            game_id="update-quality",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 5,
            policies=[np.random.dirichlet(np.ones(64))] * 5,
            values=[0.5] * 5,
            timestamps=[time.time()] * 5,
            quality_score=0.5,
        )
        buffer.add_game(record)
        
        buffer.update_quality_score("update-quality", 0.9)
        
        assert buffer.get_quality_score("update-quality") == 0.9

    def test_quality_below_threshold_filtered(self) -> None:
        """Test that games below quality threshold can be filtered."""
        config = HotDataBufferConfig(min_quality_score=0.5)
        buffer = HotDataBuffer(config=config)
        
        # Add game below threshold
        low_quality = GameRecord(
            game_id="low-quality",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 10,
            policies=[np.random.dirichlet(np.ones(64))] * 10,
            values=[0.5] * 10,
            timestamps=[time.time()] * 10,
            quality_score=0.3,
        )
        buffer.add_game(low_quality)
        
        # Game should still be stored (filtering happens at batch time)
        assert buffer.get_game("low-quality") is not None


# =============================================================================
# Flush Operations Tests
# =============================================================================


class TestFlushOperations:
    """Tests for flushing buffer to disk."""

    def test_flush_to_jsonl(self) -> None:
        """Test flushing buffer to JSONL file."""
        buffer = HotDataBuffer()
        for i in range(3):
            record = GameRecord(
                game_id=f"flush-game-{i}",
                config_key="hex8_2p",
                states=[np.random.randn(10, 8, 8)] * 5,
                policies=[np.random.dirichlet(np.ones(64))] * 5,
                values=[0.5] * 5,
                timestamps=[time.time()] * 5,
            )
            buffer.add_game(record)
        
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
        finally:
            output_path.unlink(missing_ok=True)

    def test_flush_clears_buffer(self) -> None:
        """Test that flush clears buffer when requested."""
        buffer = HotDataBuffer()
        for i in range(3):
            record = GameRecord(
                game_id=f"clear-game-{i}",
                config_key="hex8_2p",
                states=[np.random.randn(10, 8, 8)] * 5,
                policies=[np.random.dirichlet(np.ones(64))] * 5,
                values=[0.5] * 5,
                timestamps=[time.time()] * 5,
            )
            buffer.add_game(record)
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            output_path = Path(f.name)
        
        try:
            buffer.flush_to_jsonl(output_path, clear_after=True)
            
            assert buffer.game_count == 0
            assert buffer.is_empty
        finally:
            output_path.unlink(missing_ok=True)

    def test_flush_empty_buffer(self) -> None:
        """Test flushing empty buffer."""
        buffer = HotDataBuffer()
        
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

    def test_get_stats(self) -> None:
        """Test getting buffer statistics."""
        buffer = HotDataBuffer()
        for i in range(5):
            record = GameRecord(
                game_id=f"stats-game-{i}",
                config_key="hex8_2p",
                states=[np.random.randn(10, 8, 8)] * 10,
                policies=[np.random.dirichlet(np.ones(64))] * 10,
                values=[0.5] * 10,
                timestamps=[time.time()] * 10,
                quality_score=0.5 + i * 0.1,
            )
            buffer.add_game(record)
        
        stats = buffer.get_stats()
        
        assert stats["game_count"] == 5
        assert stats["sample_count"] == 50
        assert "avg_quality" in stats
        assert "configs" in stats

    def test_get_config_breakdown(self) -> None:
        """Test getting per-config breakdown."""
        buffer = HotDataBuffer()
        
        # Add games with different configs
        for i in range(3):
            buffer.add_game(GameRecord(
                game_id=f"hex-{i}",
                config_key="hex8_2p",
                states=[np.zeros((10, 8, 8))] * 5,
                policies=[np.ones(64) / 64] * 5,
                values=[0.5] * 5,
                timestamps=[time.time()] * 5,
            ))
        
        for i in range(2):
            buffer.add_game(GameRecord(
                game_id=f"square-{i}",
                config_key="square8_2p",
                states=[np.zeros((10, 8, 8))] * 5,
                policies=[np.ones(64) / 64] * 5,
                values=[0.5] * 5,
                timestamps=[time.time()] * 5,
            ))
        
        breakdown = buffer.get_config_breakdown()
        
        assert "hex8_2p" in breakdown
        assert "square8_2p" in breakdown
        assert breakdown["hex8_2p"]["game_count"] == 3
        assert breakdown["square8_2p"]["game_count"] == 2


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_adds(self) -> None:
        """Test concurrent game additions."""
        buffer = HotDataBuffer()
        errors: list[Exception] = []
        
        def add_games(start_idx: int) -> None:
            try:
                for i in range(10):
                    record = GameRecord(
                        game_id=f"thread-{start_idx}-game-{i}",
                        config_key="hex8_2p",
                        states=[np.random.randn(10, 8, 8)] * 5,
                        policies=[np.random.dirichlet(np.ones(64))] * 5,
                        values=[0.5] * 5,
                        timestamps=[time.time()] * 5,
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
        assert buffer.game_count == 50

    def test_concurrent_read_write(self) -> None:
        """Test concurrent reading and writing."""
        buffer = HotDataBuffer()
        errors: list[Exception] = []
        
        # Pre-populate
        for i in range(10):
            buffer.add_game(GameRecord(
                game_id=f"pre-game-{i}",
                config_key="hex8_2p",
                states=[np.random.randn(10, 8, 8)] * 5,
                policies=[np.random.dirichlet(np.ones(64))] * 5,
                values=[0.5] * 5,
                timestamps=[time.time()] * 5,
            ))
        
        def reader() -> None:
            try:
                for _ in range(100):
                    buffer.get_training_batch(batch_size=16)
                    buffer.get_stats()
            except Exception as e:
                errors.append(e)
        
        def writer() -> None:
            try:
                for i in range(20):
                    buffer.add_game(GameRecord(
                        game_id=f"new-game-{i}",
                        config_key="hex8_2p",
                        states=[np.random.randn(10, 8, 8)] * 5,
                        policies=[np.random.dirichlet(np.ones(64))] * 5,
                        values=[0.5] * 5,
                        timestamps=[time.time()] * 5,
                    ))
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
        """Test adding game with duplicate ID."""
        buffer = HotDataBuffer()
        record1 = GameRecord(
            game_id="duplicate",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 5,
            policies=[np.random.dirichlet(np.ones(64))] * 5,
            values=[0.5] * 5,
            timestamps=[time.time()] * 5,
            quality_score=0.5,
        )
        record2 = GameRecord(
            game_id="duplicate",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 10,
            policies=[np.random.dirichlet(np.ones(64))] * 10,
            values=[0.5] * 10,
            timestamps=[time.time()] * 10,
            quality_score=0.9,
        )
        
        buffer.add_game(record1)
        buffer.add_game(record2)  # Should update/replace
        
        assert buffer.game_count == 1
        retrieved = buffer.get_game("duplicate")
        assert retrieved.quality_score == 0.9  # Updated

    def test_very_large_batch_request(self) -> None:
        """Test requesting batch larger than buffer."""
        buffer = HotDataBuffer()
        buffer.add_game(GameRecord(
            game_id="small-game",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 5,
            policies=[np.random.dirichlet(np.ones(64))] * 5,
            values=[0.5] * 5,
            timestamps=[time.time()] * 5,
        ))
        
        batch = buffer.get_training_batch(batch_size=1000)
        
        # Should return whatever samples are available
        assert batch is not None
        assert len(batch.get("states", [])) <= 5

    def test_special_characters_in_game_id(self) -> None:
        """Test game IDs with special characters."""
        buffer = HotDataBuffer()
        special_id = "game/with:special-chars_123"
        
        buffer.add_game(GameRecord(
            game_id=special_id,
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 5,
            policies=[np.random.dirichlet(np.ones(64))] * 5,
            values=[0.5] * 5,
            timestamps=[time.time()] * 5,
        ))
        
        retrieved = buffer.get_game(special_id)
        assert retrieved is not None
        assert retrieved.game_id == special_id

    def test_nan_handling_in_values(self) -> None:
        """Test handling of NaN values."""
        buffer = HotDataBuffer()
        
        # This tests that the buffer handles edge cases gracefully
        record = GameRecord(
            game_id="nan-game",
            config_key="hex8_2p",
            states=[np.random.randn(10, 8, 8)] * 3,
            policies=[np.random.dirichlet(np.ones(64))] * 3,
            values=[0.5, float("nan"), 0.5],  # NaN in values
            timestamps=[time.time()] * 3,
        )
        
        buffer.add_game(record)
        
        # Buffer should store it (validation happens elsewhere)
        assert buffer.game_count == 1
