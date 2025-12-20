"""Tests for app/metrics/registry.py - safe metric registration.

Tests cover:
- safe_metric function for duplicate prevention
- safe_counter, safe_gauge, safe_histogram, safe_summary helpers
- get_metric and is_metric_registered introspection
- Thread safety of metric registration
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Summary


class TestSafeMetric:
    """Tests for safe_metric function."""

    def test_creates_new_metric(self):
        """Test that safe_metric creates a new metric when none exists."""
        from app.metrics.registry import safe_metric

        # Use unique name to avoid conflicts with other tests
        name = "test_creates_new_metric_counter"

        # Clean up if exists from previous run
        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        metric = safe_metric(Counter, name, "Test counter")

        assert metric is not None
        assert isinstance(metric, Counter)

    def test_returns_existing_metric(self):
        """Test that safe_metric returns existing metric on duplicate."""
        from app.metrics.registry import safe_metric

        name = "test_returns_existing_counter"

        # Clean up if exists
        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        # Create first time
        metric1 = safe_metric(Counter, name, "Test counter")
        # Try to create again
        metric2 = safe_metric(Counter, name, "Test counter different doc")

        # Should be the same instance
        assert metric1 is metric2

    def test_metric_with_labels(self):
        """Test creating metrics with labels."""
        from app.metrics.registry import safe_metric

        name = "test_metric_with_labels_counter"

        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        metric = safe_metric(
            Counter,
            name,
            "Counter with labels",
            labelnames=["label1", "label2"],
        )

        # Should be able to use labels
        labeled = metric.labels(label1="a", label2="b")
        labeled.inc()

        assert labeled._value.get() == 1.0

    def test_histogram_with_buckets(self):
        """Test creating histogram with custom buckets."""
        from app.metrics.registry import safe_metric

        name = "test_histogram_with_buckets"

        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        buckets = [0.1, 0.5, 1.0, 5.0, 10.0]
        metric = safe_metric(
            Histogram,
            name,
            "Histogram with custom buckets",
            buckets=buckets,
        )

        metric.observe(0.3)
        metric.observe(2.0)

        # Verify it works - histogram doesn't error on observe
        # Cannot easily check count without labels


class TestSafeHelpers:
    """Tests for type-specific safe helpers."""

    def test_safe_counter(self):
        """Test safe_counter helper."""
        from app.metrics.registry import safe_counter

        name = "test_safe_counter_helper"

        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        counter = safe_counter(name, "Test counter")

        assert isinstance(counter, Counter)
        counter.inc()
        assert counter._value.get() == 1.0

    def test_safe_gauge(self):
        """Test safe_gauge helper."""
        from app.metrics.registry import safe_gauge

        name = "test_safe_gauge_helper"

        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        gauge = safe_gauge(name, "Test gauge")

        assert isinstance(gauge, Gauge)
        gauge.set(42.0)
        assert gauge._value.get() == 42.0

    def test_safe_histogram(self):
        """Test safe_histogram helper."""
        from app.metrics.registry import safe_histogram

        name = "test_safe_histogram_helper"

        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        hist = safe_histogram(name, "Test histogram")

        assert isinstance(hist, Histogram)
        hist.observe(1.5)
        # Histogram doesn't error on observe - that's the test

    def test_safe_summary(self):
        """Test safe_summary helper."""
        from app.metrics.registry import safe_summary

        name = "test_safe_summary_helper"

        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        summary = safe_summary(name, "Test summary")

        assert isinstance(summary, Summary)
        summary.observe(2.5)
        assert summary._count.get() == 1


class TestMetricIntrospection:
    """Tests for metric introspection functions."""

    def test_get_metric_exists(self):
        """Test get_metric returns metric when it exists."""
        from app.metrics.registry import get_metric, safe_counter

        name = "test_get_metric_exists"

        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        created = safe_counter(name, "Test")
        retrieved = get_metric(name)

        assert retrieved is created

    def test_get_metric_not_exists(self):
        """Test get_metric returns None for non-existent metric."""
        from app.metrics.registry import get_metric

        result = get_metric("nonexistent_metric_12345")

        assert result is None

    def test_is_metric_registered_true(self):
        """Test is_metric_registered returns True when metric exists."""
        from app.metrics.registry import is_metric_registered, safe_counter

        name = "test_is_registered_true"

        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        safe_counter(name, "Test")

        assert is_metric_registered(name) is True

    def test_is_metric_registered_false(self):
        """Test is_metric_registered returns False for non-existent metric."""
        from app.metrics.registry import is_metric_registered

        assert is_metric_registered("nonexistent_metric_67890") is False

    def test_list_registered_metrics(self):
        """Test list_registered_metrics returns tracked metrics."""
        from app.metrics.registry import (
            list_registered_metrics,
            safe_counter,
            safe_gauge,
        )

        # Create some test metrics
        name1 = "test_list_registered_1"
        name2 = "test_list_registered_2"

        for name in [name1, name2]:
            if name in REGISTRY._names_to_collectors:
                del REGISTRY._names_to_collectors[name]

        safe_counter(name1, "Counter")
        safe_gauge(name2, "Gauge")

        registered = list_registered_metrics()

        assert name1 in registered
        assert name2 in registered
        assert registered[name1] == "Counter"
        assert registered[name2] == "Gauge"


class TestThreadSafety:
    """Tests for thread safety of metric registration."""

    def test_concurrent_registration_same_metric(self):
        """Test that concurrent registration of same metric is safe."""
        from app.metrics.registry import safe_counter

        name = "test_concurrent_same_metric"

        if name in REGISTRY._names_to_collectors:
            del REGISTRY._names_to_collectors[name]

        results = []
        errors = []

        def register_metric():
            try:
                metric = safe_counter(name, "Concurrent test")
                results.append(metric)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=register_metric) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # All should return same instance
        assert len(results) == 10
        assert all(r is results[0] for r in results)

    def test_concurrent_registration_different_metrics(self):
        """Test concurrent registration of different metrics."""
        from app.metrics.registry import safe_counter

        base_name = "test_concurrent_different"
        num_metrics = 10

        # Clean up
        for i in range(num_metrics):
            name = f"{base_name}_{i}"
            if name in REGISTRY._names_to_collectors:
                del REGISTRY._names_to_collectors[name]

        results = {}
        errors = []
        lock = threading.Lock()

        def register_metric(idx):
            try:
                name = f"{base_name}_{idx}"
                metric = safe_counter(name, f"Test {idx}")
                with lock:
                    results[idx] = metric
            except Exception as e:
                with lock:
                    errors.append((idx, e))

        threads = [
            threading.Thread(target=register_metric, args=(i,))
            for i in range(num_metrics)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == num_metrics


class TestBackwardsCompatibility:
    """Tests for backwards compatibility."""

    def test_safe_metric_alias(self):
        """Test _safe_metric alias exists for backwards compatibility."""
        from app.metrics.registry import _safe_metric, safe_metric

        assert _safe_metric is safe_metric
