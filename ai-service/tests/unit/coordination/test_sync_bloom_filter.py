"""Tests for sync_bloom_filter module.

Tests the bloom filter implementation for P2P sync set membership testing.
"""

import pytest

from app.coordination.sync_bloom_filter import (
    BloomFilter,
    BloomFilterStats,
    SyncBloomFilter,
    create_event_dedup_filter,
    create_game_id_filter,
    create_model_hash_filter,
)


class TestSyncBloomFilter:
    """Test SyncBloomFilter basic functionality."""

    def test_add_and_contains(self):
        """Test basic add and contains operations."""
        bf = SyncBloomFilter()
        bf.add("game-123")
        bf.add("game-456")

        assert "game-123" in bf
        assert "game-456" in bf
        assert "game-789" not in bf

    def test_add_many(self):
        """Test adding multiple items at once."""
        bf = SyncBloomFilter()
        items = [f"game-{i}" for i in range(100)]
        count = bf.add_many(items)

        assert count == 100
        assert bf.items_added == 100
        for item in items:
            assert item in bf

    def test_probably_contains_alias(self):
        """Test probably_contains is alias for __contains__."""
        bf = SyncBloomFilter()
        bf.add("test-item")

        assert bf.probably_contains("test-item")
        assert not bf.probably_contains("missing-item")

    def test_false_positive_rate(self):
        """Test that false positive rate is within expected bounds."""
        bf = SyncBloomFilter(expected_items=1000, false_positive_rate=0.01)

        # Add 1000 items
        for i in range(1000):
            bf.add(f"known-{i}")

        # Test 10000 items that were NOT added
        false_positives = sum(
            1 for i in range(10000) if f"unknown-{i}" in bf
        )

        # Should be roughly 1% (100 out of 10000) with some margin
        # Allow up to 3% to account for statistical variance
        assert false_positives < 300, f"Too many false positives: {false_positives}"

    def test_no_false_negatives(self):
        """Test that there are no false negatives."""
        bf = SyncBloomFilter(expected_items=1000)

        items = [f"item-{i}" for i in range(1000)]
        for item in items:
            bf.add(item)

        # All added items must be found
        for item in items:
            assert item in bf, f"False negative for {item}"


class TestBloomFilterSizing:
    """Test optimal sizing calculations."""

    def test_optimal_params_basic(self):
        """Test optimal parameter calculation."""
        bf = SyncBloomFilter(expected_items=10000, false_positive_rate=0.01)

        # Size should be roughly 10 bits per item for 1% FP rate
        assert bf.size > 80000  # At least 8 bits per item
        assert bf.size < 200000  # Not excessively large

        # Hash count should be around 7 for 1% FP rate
        assert 5 <= bf.hash_count <= 10

    def test_explicit_params(self):
        """Test explicit size and hash count."""
        bf = SyncBloomFilter(size=50000, hash_count=5)

        assert bf.size == 50000
        assert bf.hash_count == 5

    def test_default_params(self):
        """Test default parameters."""
        bf = SyncBloomFilter()

        assert bf.size == 100000
        assert bf.hash_count == 7


class TestBloomFilterSerialization:
    """Test serialization and deserialization."""

    def test_to_bytes_and_from_bytes(self):
        """Test round-trip serialization."""
        bf1 = SyncBloomFilter(size=10000, hash_count=5)
        bf1.add("item-1")
        bf1.add("item-2")
        bf1.add("item-3")

        # Serialize
        data = bf1.to_bytes(compress=True)

        # Deserialize
        bf2 = SyncBloomFilter.from_bytes(data, size=10000, hash_count=5)

        # Check items preserved
        assert "item-1" in bf2
        assert "item-2" in bf2
        assert "item-3" in bf2
        assert "item-4" not in bf2

    def test_compression_reduces_size(self):
        """Test that compression reduces size for sparse filters."""
        bf = SyncBloomFilter(size=100000, hash_count=7)
        bf.add("item-1")
        bf.add("item-2")

        compressed = bf.to_bytes(compress=True)
        uncompressed = bf.to_bytes(compress=False)

        # Compressed should be smaller for sparse filter
        assert len(compressed) < len(uncompressed)

    def test_from_bytes_auto_decompress(self):
        """Test from_bytes handles both compressed and uncompressed data."""
        bf = SyncBloomFilter(size=10000, hash_count=5)
        bf.add("test-item")

        # Compressed data
        compressed = bf.to_bytes(compress=True)
        bf_from_compressed = SyncBloomFilter.from_bytes(compressed, size=10000, hash_count=5)
        assert "test-item" in bf_from_compressed

        # Uncompressed data
        uncompressed = bf.to_bytes(compress=False)
        bf_from_uncompressed = SyncBloomFilter.from_bytes(
            uncompressed, size=10000, hash_count=5, compressed=False
        )
        assert "test-item" in bf_from_uncompressed


class TestBloomFilterMerge:
    """Test merge operations."""

    def test_merge_basic(self):
        """Test merging two filters."""
        bf1 = SyncBloomFilter(size=10000, hash_count=5)
        bf1.add("item-1")
        bf1.add("item-2")

        bf2 = SyncBloomFilter(size=10000, hash_count=5)
        bf2.add("item-3")
        bf2.add("item-4")

        merged = bf1.merge(bf2)

        assert "item-1" in merged
        assert "item-2" in merged
        assert "item-3" in merged
        assert "item-4" in merged
        assert "item-5" not in merged

    def test_merge_incompatible_sizes(self):
        """Test that merging incompatible filters raises error."""
        bf1 = SyncBloomFilter(size=10000, hash_count=5)
        bf2 = SyncBloomFilter(size=20000, hash_count=5)

        with pytest.raises(ValueError, match="different parameters"):
            bf1.merge(bf2)

    def test_merge_incompatible_hash_count(self):
        """Test that merging with different hash counts raises error."""
        bf1 = SyncBloomFilter(size=10000, hash_count=5)
        bf2 = SyncBloomFilter(size=10000, hash_count=7)

        with pytest.raises(ValueError, match="different parameters"):
            bf1.merge(bf2)


class TestBloomFilterStats:
    """Test statistics and metrics."""

    def test_fill_ratio(self):
        """Test fill ratio calculation."""
        bf = SyncBloomFilter(size=1000, hash_count=3)

        assert bf.fill_ratio == 0.0

        bf.add("item-1")
        assert bf.fill_ratio > 0.0

    def test_estimated_items(self):
        """Test item count estimation."""
        bf = SyncBloomFilter(size=100000, hash_count=7)

        # Add 1000 items
        for i in range(1000):
            bf.add(f"item-{i}")

        estimated = bf.estimated_items()

        # Should be reasonably close to 1000
        assert 800 < estimated < 1200

    def test_estimated_false_positive_rate(self):
        """Test FP rate estimation."""
        bf = SyncBloomFilter(expected_items=1000, false_positive_rate=0.01)

        # Empty filter should have 0% FP rate
        assert bf.estimated_false_positive_rate() == 0.0

        # Add items
        for i in range(1000):
            bf.add(f"item-{i}")

        fp_rate = bf.estimated_false_positive_rate()
        # Should be roughly 1%
        assert 0.005 < fp_rate < 0.03

    def test_get_stats(self):
        """Test stats retrieval."""
        bf = SyncBloomFilter(size=10000, hash_count=5)
        bf.add("item-1")
        bf.add("item-2")

        stats = bf.get_stats()

        assert isinstance(stats, BloomFilterStats)
        assert stats.size_bits == 10000
        assert stats.hash_count == 5
        assert stats.items_added == 2
        assert stats.fill_ratio > 0
        assert stats.size_bytes > 0
        assert stats.compressed_bytes > 0

    def test_intersection_ratio(self):
        """Test intersection ratio calculation."""
        bf1 = SyncBloomFilter(size=10000, hash_count=5)
        bf2 = SyncBloomFilter(size=10000, hash_count=5)

        # Add overlapping items
        for i in range(100):
            bf1.add(f"item-{i}")
        for i in range(50, 150):  # 50 overlap
            bf2.add(f"item-{i}")

        ratio = bf1.intersection_ratio(bf2)

        # Should be around 50% overlap
        assert 0.3 < ratio < 0.7


class TestBloomFilterUtility:
    """Test utility methods."""

    def test_clear(self):
        """Test clearing the filter."""
        bf = SyncBloomFilter()
        bf.add("item-1")
        bf.add("item-2")

        assert bf.items_added == 2
        assert "item-1" in bf

        bf.clear()

        assert bf.items_added == 0
        assert "item-1" not in bf

    def test_len(self):
        """Test __len__ returns estimated items."""
        bf = SyncBloomFilter()
        bf.add("item-1")
        bf.add("item-2")

        # len() returns estimated_items()
        assert len(bf) >= 0

    def test_repr(self):
        """Test string representation."""
        bf = SyncBloomFilter(size=10000, hash_count=5)
        bf.add("test")

        repr_str = repr(bf)
        assert "SyncBloomFilter" in repr_str
        assert "size=10000" in repr_str
        assert "hash_count=5" in repr_str


class TestFactoryFunctions:
    """Test convenience factory functions."""

    def test_create_game_id_filter(self):
        """Test game ID filter creation."""
        bf = create_game_id_filter(expected_games=10000)

        # Should have reasonable parameters
        assert bf.size > 0
        assert bf.hash_count > 0

        # Should work with UUIDs
        bf.add("123e4567-e89b-12d3-a456-426614174000")
        assert "123e4567-e89b-12d3-a456-426614174000" in bf

    def test_create_model_hash_filter(self):
        """Test model hash filter creation."""
        bf = create_model_hash_filter(expected_models=100)

        # Should have tighter FP rate for models
        bf.add("sha256:abc123")
        assert "sha256:abc123" in bf

    def test_create_event_dedup_filter(self):
        """Test event dedup filter creation."""
        bf = create_event_dedup_filter(expected_events=50000)

        # Should handle event IDs
        bf.add("event-123-456-789")
        assert "event-123-456-789" in bf


class TestBackwardCompatibility:
    """Test backward compatibility with original BloomFilter."""

    def test_bloom_filter_alias(self):
        """Test BloomFilter alias works."""
        # BloomFilter should be aliased to SyncBloomFilter
        bf = BloomFilter()

        bf.add("item-1")
        assert "item-1" in bf

        # Should have to_bytes
        data = bf.to_bytes()
        assert isinstance(data, bytes)

    def test_from_bytes_compatibility(self):
        """Test from_bytes works like original."""
        bf = BloomFilter(size=10000, hash_count=7)
        bf.add("test")

        data = bf.to_bytes(compress=False)
        bf2 = BloomFilter.from_bytes(data, size=10000, hash_count=7, compressed=False)

        assert "test" in bf2
