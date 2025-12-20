"""Tests for scripts/lib/metrics.py module.

Tests cover:
- TimingStats for timing measurements
- RateCalculator for throughput calculations
- Counter and WinLossCounter
- ProgressTracker with ETA
- RunningStats for online statistics
- MetricsCollection for aggregate metrics
"""

import time
from unittest.mock import patch

import pytest

from scripts.lib.metrics import (
    Counter,
    MetricsCollection,
    ProgressTracker,
    RateCalculator,
    RunningStats,
    TimingStats,
    WinLossCounter,
)


class TestTimingStats:
    """Tests for TimingStats class."""

    def test_record_single(self):
        """Test recording a single timing."""
        stats = TimingStats()
        stats.record(0.1)

        assert stats.count == 1
        assert stats.total_time == 0.1
        assert stats.avg_time == 0.1
        assert stats.min_time == 0.1
        assert stats.max_time == 0.1

    def test_record_multiple(self):
        """Test recording multiple timings."""
        stats = TimingStats()
        stats.record(0.1)
        stats.record(0.2)
        stats.record(0.3)

        assert stats.count == 3
        assert stats.total_time == pytest.approx(0.6)
        assert stats.avg_time == pytest.approx(0.2)
        assert stats.min_time == 0.1
        assert stats.max_time == 0.3

    def test_time_context_manager(self):
        """Test timing with context manager."""
        stats = TimingStats()

        with stats.time():
            time.sleep(0.01)

        assert stats.count == 1
        assert stats.total_time >= 0.01
        assert stats.total_time < 0.1

    def test_millisecond_properties(self):
        """Test millisecond conversion properties."""
        stats = TimingStats()
        stats.record(0.1)  # 100ms

        assert stats.avg_time_ms == pytest.approx(100.0)
        assert stats.total_time_ms == pytest.approx(100.0)
        assert stats.min_time_ms == pytest.approx(100.0)
        assert stats.max_time_ms == pytest.approx(100.0)

    def test_empty_stats(self):
        """Test empty stats return sensible defaults."""
        stats = TimingStats()

        assert stats.count == 0
        assert stats.avg_time == 0.0
        assert stats.min_time_ms == 0.0  # Converts inf to 0

    def test_reset(self):
        """Test reset clears all values."""
        stats = TimingStats()
        stats.record(0.1)
        stats.record(0.2)

        stats.reset()

        assert stats.count == 0
        assert stats.total_time == 0.0
        assert stats.min_time == float("inf")
        assert stats.max_time == 0.0

    def test_merge(self):
        """Test merging two TimingStats."""
        stats1 = TimingStats()
        stats1.record(0.1)
        stats1.record(0.2)

        stats2 = TimingStats()
        stats2.record(0.05)
        stats2.record(0.3)

        stats1.merge(stats2)

        assert stats1.count == 4
        assert stats1.total_time == pytest.approx(0.65)
        assert stats1.min_time == 0.05
        assert stats1.max_time == 0.3

    def test_to_dict(self):
        """Test dictionary serialization."""
        stats = TimingStats(name="test")
        stats.record(0.1)

        d = stats.to_dict()

        assert d["name"] == "test"
        assert d["count"] == 1
        assert d["avg_time_ms"] == pytest.approx(100.0)

    def test_str_representation(self):
        """Test string representation."""
        stats = TimingStats(name="query")
        stats.record(0.1)

        s = str(stats)

        assert "query:" in s
        assert "1 calls" in s
        assert "avg=" in s

    def test_str_empty(self):
        """Test string for empty stats."""
        stats = TimingStats(name="test")
        assert "no data" in str(stats)


class TestRateCalculator:
    """Tests for RateCalculator class."""

    def test_record_with_explicit_time(self):
        """Test recording with explicit elapsed time."""
        rate = RateCalculator()
        rate.record(items=100, elapsed=10.0)

        assert rate.total_items == 100
        assert rate.total_time == 10.0
        assert rate.rate_per_second == 10.0

    def test_start_stop(self):
        """Test start/stop timing."""
        rate = RateCalculator()
        rate.start()
        time.sleep(0.01)
        rate.record(100)
        rate.stop()

        assert rate.total_items == 100
        assert rate.total_time >= 0.01
        assert rate.rate_per_second > 0

    def test_rate_conversions(self):
        """Test rate unit conversions."""
        rate = RateCalculator()
        rate.record(items=60, elapsed=1.0)

        assert rate.rate_per_second == 60.0
        assert rate.rate_per_minute == 3600.0
        assert rate.rate_per_hour == 216000.0

    def test_format_rate_high(self):
        """Test formatting high rates."""
        rate = RateCalculator()
        rate.record(items=10000, elapsed=1.0)

        assert "10.0k" in rate.format_rate("items")

    def test_format_rate_low(self):
        """Test formatting low rates."""
        rate = RateCalculator()
        rate.record(items=1, elapsed=60.0)

        assert "/min" in rate.format_rate("items")

    def test_reset(self):
        """Test reset clears all values."""
        rate = RateCalculator()
        rate.record(items=100, elapsed=10.0)
        rate.reset()

        assert rate.total_items == 0
        assert rate.total_time == 0.0

    def test_to_dict(self):
        """Test dictionary serialization."""
        rate = RateCalculator()
        rate.record(items=100, elapsed=10.0)

        d = rate.to_dict()

        assert d["total_items"] == 100
        assert d["rate_per_second"] == 10.0


class TestCounter:
    """Tests for Counter class."""

    def test_increment(self):
        """Test basic increment."""
        counter = Counter()
        counter.increment()
        counter.increment()

        assert counter.value == 2

    def test_increment_by_amount(self):
        """Test increment by specific amount."""
        counter = Counter()
        counter.increment(10)

        assert counter.value == 10

    def test_increment_returns_value(self):
        """Test increment returns new value."""
        counter = Counter()
        result = counter.increment(5)

        assert result == 5

    def test_reset(self):
        """Test reset to zero."""
        counter = Counter()
        counter.increment(10)
        counter.reset()

        assert counter.value == 0

    def test_str_with_name(self):
        """Test string with name."""
        counter = Counter(name="games", value=42)
        assert str(counter) == "games: 42"

    def test_str_without_name(self):
        """Test string without name."""
        counter = Counter(value=42)
        assert str(counter) == "42"


class TestWinLossCounter:
    """Tests for WinLossCounter class."""

    def test_record_wins(self):
        """Test recording wins."""
        counter = WinLossCounter()
        counter.record_win()
        counter.record_win(2)

        assert counter.wins == 3
        assert counter.total_games == 3

    def test_record_losses(self):
        """Test recording losses."""
        counter = WinLossCounter()
        counter.record_loss()
        counter.record_loss(2)

        assert counter.losses == 3

    def test_record_draws(self):
        """Test recording draws."""
        counter = WinLossCounter()
        counter.record_draw()
        counter.record_draw(2)

        assert counter.draws == 3

    def test_record_result_by_string(self):
        """Test recording by result string."""
        counter = WinLossCounter()
        counter.record_result("win")
        counter.record_result("WIN")  # Case insensitive
        counter.record_result("loss")
        counter.record_result("draw")

        assert counter.wins == 2
        assert counter.losses == 1
        assert counter.draws == 1

    def test_win_rate(self):
        """Test win rate calculation (draws = 0.5)."""
        counter = WinLossCounter(wins=7, losses=2, draws=2)

        # (7 + 0.5*2) / 11 = 8 / 11
        assert counter.win_rate == pytest.approx(8.0 / 11.0)

    def test_win_rate_strict(self):
        """Test strict win rate (draws don't count)."""
        counter = WinLossCounter(wins=7, losses=2, draws=2)

        assert counter.win_rate_strict == pytest.approx(7.0 / 11.0)

    def test_loss_rate(self):
        """Test loss rate."""
        counter = WinLossCounter(wins=7, losses=2, draws=1)
        assert counter.loss_rate == pytest.approx(0.2)

    def test_draw_rate(self):
        """Test draw rate."""
        counter = WinLossCounter(wins=7, losses=2, draws=1)
        assert counter.draw_rate == pytest.approx(0.1)

    def test_empty_rates(self):
        """Test rates for empty counter."""
        counter = WinLossCounter()
        assert counter.win_rate == 0.0
        assert counter.loss_rate == 0.0

    def test_reset(self):
        """Test reset clears all."""
        counter = WinLossCounter(wins=5, losses=3, draws=2)
        counter.reset()

        assert counter.wins == 0
        assert counter.losses == 0
        assert counter.draws == 0

    def test_merge(self):
        """Test merging counters."""
        c1 = WinLossCounter(wins=5, losses=3, draws=2)
        c2 = WinLossCounter(wins=3, losses=2, draws=1)

        c1.merge(c2)

        assert c1.wins == 8
        assert c1.losses == 5
        assert c1.draws == 3

    def test_to_dict(self):
        """Test dictionary serialization."""
        counter = WinLossCounter(wins=5, losses=3, draws=2)
        d = counter.to_dict()

        assert d["wins"] == 5
        assert d["losses"] == 3
        assert d["total_games"] == 10

    def test_str_representation(self):
        """Test string representation."""
        counter = WinLossCounter(wins=10, losses=5, draws=2)
        assert str(counter) == "10-5-2 (W-L-D)"


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_update(self):
        """Test updating progress."""
        progress = ProgressTracker(total=100)
        progress.update(10)
        progress.update(5)

        assert progress.current == 15
        assert progress.remaining == 85

    def test_set(self):
        """Test setting progress directly."""
        progress = ProgressTracker(total=100)
        progress.set(50)

        assert progress.current == 50

    def test_percentage(self):
        """Test percentage calculation."""
        progress = ProgressTracker(total=200)
        progress.set(50)

        assert progress.percentage == 25.0
        assert progress.fraction == 0.25

    def test_percentage_capped(self):
        """Test percentage doesn't exceed 100."""
        progress = ProgressTracker(total=100)
        progress.set(150)

        assert progress.percentage == 100.0
        assert progress.fraction == 1.0

    def test_is_complete(self):
        """Test completion detection."""
        progress = ProgressTracker(total=100)

        assert not progress.is_complete
        progress.set(100)
        assert progress.is_complete

    def test_elapsed(self):
        """Test elapsed time tracking."""
        progress = ProgressTracker(total=100)
        time.sleep(0.01)

        assert progress.elapsed >= 0.01
        assert "s" in progress.elapsed_str

    def test_eta(self):
        """Test ETA calculation."""
        with patch("scripts.lib.metrics.time.perf_counter") as mock_time:
            mock_time.return_value = 0.0
            progress = ProgressTracker(total=100)
            # Reset to use our mocked start time
            progress._start_time = 0.0

            mock_time.return_value = 10.0
            progress.set(50)

            # 50 items in 10 seconds = 5/sec
            # 50 remaining / 5/sec = 10 seconds ETA
            assert progress.eta_seconds == pytest.approx(10.0)

    def test_eta_no_progress(self):
        """Test ETA with no progress."""
        progress = ProgressTracker(total=100)
        assert progress.eta_seconds is None
        assert "calculating" in progress.eta_str

    def test_rate_per_second(self):
        """Test rate calculation."""
        with patch("scripts.lib.metrics.time.perf_counter") as mock_time:
            mock_time.return_value = 0.0
            progress = ProgressTracker(total=100)
            # Reset to use our mocked start time
            progress._start_time = 0.0

            mock_time.return_value = 10.0
            progress.set(50)

            assert progress.rate_per_second == pytest.approx(5.0)

    def test_reset(self):
        """Test reset with new total."""
        progress = ProgressTracker(total=100)
        progress.set(50)
        progress.reset(total=200)

        assert progress.current == 0
        assert progress.total == 200

    def test_format_status(self):
        """Test status formatting."""
        progress = ProgressTracker(total=100)
        progress.set(25)

        status = progress.format_status()

        assert "25/100" in status
        assert "25.0%" in status
        assert "ETA:" in status

    def test_to_dict(self):
        """Test dictionary serialization."""
        progress = ProgressTracker(total=100)
        progress.set(50)

        d = progress.to_dict()

        assert d["current"] == 50
        assert d["total"] == 100
        assert d["percentage"] == 50.0


class TestRunningStats:
    """Tests for RunningStats class."""

    def test_single_value(self):
        """Test with a single value."""
        stats = RunningStats()
        stats.update(10.0)

        assert stats.count == 1
        assert stats.mean == 10.0
        assert stats.min_value == 10.0
        assert stats.max_value == 10.0

    def test_multiple_values(self):
        """Test with multiple values."""
        stats = RunningStats()
        for v in [2, 4, 6, 8, 10]:
            stats.update(v)

        assert stats.count == 5
        assert stats.mean == 6.0
        assert stats.min_value == 2.0
        assert stats.max_value == 10.0

    def test_variance_stddev(self):
        """Test variance and standard deviation."""
        stats = RunningStats()
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            stats.update(v)

        # Mean = 5.0
        assert stats.mean == 5.0
        # Sample variance = 32/7 = 4.571..., stddev = sqrt(32/7) = 2.138...
        assert stats.variance == pytest.approx(32 / 7)
        assert stats.stddev == pytest.approx(2.138, rel=0.01)

    def test_population_variance(self):
        """Test population variance."""
        stats = RunningStats()
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            stats.update(v)

        # Population variance = 32/8 = 4.0
        assert stats.population_variance == pytest.approx(4.0)

    def test_empty_stats(self):
        """Test empty stats return defaults."""
        stats = RunningStats()

        assert stats.mean == 0.0
        assert stats.variance == 0.0
        assert stats.stddev == 0.0

    def test_single_value_variance(self):
        """Test single value has zero variance."""
        stats = RunningStats()
        stats.update(5.0)

        assert stats.variance == 0.0

    def test_reset(self):
        """Test reset clears all values."""
        stats = RunningStats()
        stats.update(10.0)
        stats.update(20.0)

        stats.reset()

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.min_value == float("inf")
        assert stats.max_value == float("-inf")

    def test_merge(self):
        """Test merging two RunningStats."""
        stats1 = RunningStats()
        for v in [1, 2, 3]:
            stats1.update(v)

        stats2 = RunningStats()
        for v in [4, 5, 6]:
            stats2.update(v)

        stats1.merge(stats2)

        assert stats1.count == 6
        assert stats1.mean == pytest.approx(3.5)
        assert stats1.min_value == 1.0
        assert stats1.max_value == 6.0

    def test_merge_empty(self):
        """Test merging empty stats."""
        stats1 = RunningStats()
        stats1.update(5.0)

        stats2 = RunningStats()
        stats1.merge(stats2)

        assert stats1.count == 1
        assert stats1.mean == 5.0

    def test_merge_into_empty(self):
        """Test merging into empty stats."""
        stats1 = RunningStats()

        stats2 = RunningStats()
        stats2.update(5.0)

        stats1.merge(stats2)

        assert stats1.count == 1
        assert stats1.mean == 5.0

    def test_to_dict(self):
        """Test dictionary serialization."""
        stats = RunningStats(name="latency")
        stats.update(10.0)
        stats.update(20.0)

        d = stats.to_dict()

        assert d["name"] == "latency"
        assert d["count"] == 2
        assert d["mean"] == 15.0

    def test_str_representation(self):
        """Test string representation."""
        stats = RunningStats(name="latency")
        stats.update(10.0)
        stats.update(20.0)

        s = str(stats)

        assert "latency:" in s
        assert "n=2" in s
        assert "mean=" in s

    def test_str_empty(self):
        """Test string for empty stats."""
        stats = RunningStats(name="test")
        assert "no data" in str(stats)


class TestMetricsCollection:
    """Tests for MetricsCollection class."""

    def test_timing(self):
        """Test getting/creating timing stats."""
        metrics = MetricsCollection()

        timing = metrics.timing("query")
        timing.record(0.1)

        assert metrics.timing("query").count == 1

    def test_counter(self):
        """Test getting/creating counter."""
        metrics = MetricsCollection()

        counter = metrics.counter("games")
        counter.increment(10)

        assert metrics.counter("games").value == 10

    def test_stats(self):
        """Test getting/creating running stats."""
        metrics = MetricsCollection()

        stats = metrics.stats("latency")
        stats.update(5.0)

        assert metrics.stats("latency").mean == 5.0

    def test_reset(self):
        """Test resetting all metrics."""
        metrics = MetricsCollection()
        metrics.timing("query").record(0.1)
        metrics.counter("games").increment(10)
        metrics.stats("latency").update(5.0)

        metrics.reset()

        assert metrics.timing("query").count == 0
        assert metrics.counter("games").value == 0
        assert metrics.stats("latency").count == 0

    def test_summary(self):
        """Test summary generation."""
        metrics = MetricsCollection()
        metrics.timing("query").record(0.1)
        metrics.counter("games").increment(10)
        metrics.stats("latency").update(5.0)

        summary = metrics.summary()

        assert "Timings:" in summary
        assert "Counters:" in summary
        assert "Statistics:" in summary
        assert "query" in summary
        assert "games" in summary
        assert "latency" in summary

    def test_summary_empty(self):
        """Test summary for empty collection."""
        metrics = MetricsCollection()
        assert "No metrics recorded" in metrics.summary()

    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = MetricsCollection()
        metrics.timing("query").record(0.1)
        metrics.counter("games").increment(10)

        d = metrics.to_dict()

        assert "timings" in d
        assert "counters" in d
        assert "stats" in d
        assert d["counters"]["games"] == 10
