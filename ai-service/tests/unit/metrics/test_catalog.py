"""Tests for app/metrics/catalog.py - MetricCatalog and MetricInfo.

Tests cover:
- MetricInfo dataclass
- MetricCatalog singleton pattern
- Metric registration and retrieval
- Category filtering
- Search functionality
- Documentation generation
"""

import pytest

from app.metrics.catalog import (
    MetricCatalog,
    get_metric_catalog,
    MetricCategory,
    MetricType,
    MetricInfo,
    register_metric,
)


class TestMetricInfo:
    """Tests for MetricInfo dataclass."""

    def test_basic_creation(self):
        """Test creating a MetricInfo with required fields."""
        info = MetricInfo(
            name="test_counter_total",
            description="A test counter",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.TRAINING,
        )

        assert info.name == "test_counter_total"
        assert info.description == "A test counter"
        assert info.metric_type == MetricType.COUNTER
        assert info.category == MetricCategory.TRAINING
        assert info.labels == []
        assert info.unit is None

    def test_creation_with_all_fields(self):
        """Test creating MetricInfo with all optional fields."""
        info = MetricInfo(
            name="ringrift_test_histogram_seconds",
            description="Test histogram",
            metric_type=MetricType.HISTOGRAM,
            category=MetricCategory.PIPELINE,
            labels=["label1", "label2"],
            unit="seconds",
            buckets=[0.1, 0.5, 1.0],
            module="app.metrics.test",
        )

        assert info.labels == ["label1", "label2"]
        assert info.unit == "seconds"
        assert info.buckets == [0.1, 0.5, 1.0]
        assert info.module == "app.metrics.test"

    def test_full_name_property(self):
        """Test full_name returns the metric name."""
        info = MetricInfo(
            name="ringrift_test_gauge",
            description="Test",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.TRAINING,
        )

        assert info.full_name == "ringrift_test_gauge"

    def test_short_name_with_prefix(self):
        """Test short_name strips ringrift_ prefix."""
        info = MetricInfo(
            name="ringrift_selfplay_games_total",
            description="Test",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.SELFPLAY,
        )

        assert info.short_name == "selfplay_games_total"

    def test_short_name_without_prefix(self):
        """Test short_name returns full name when no prefix."""
        info = MetricInfo(
            name="custom_metric_total",
            description="Test",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.TRAINING,
        )

        assert info.short_name == "custom_metric_total"

    def test_hash(self):
        """Test MetricInfo is hashable by name."""
        info1 = MetricInfo(
            name="test_metric",
            description="Test 1",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.TRAINING,
        )
        info2 = MetricInfo(
            name="test_metric",
            description="Test 2 - different",
            metric_type=MetricType.GAUGE,  # Different type
            category=MetricCategory.SELFPLAY,  # Different category
        )

        # Same name = same hash
        assert hash(info1) == hash(info2)

        # Can be used in sets
        metric_set = {info1}
        assert info1 in metric_set


class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_types_exist(self):
        """Test all expected metric types exist."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"
        assert MetricType.INFO.value == "info"


class TestMetricCategory:
    """Tests for MetricCategory enum."""

    def test_pipeline_categories_exist(self):
        """Test pipeline-related categories exist."""
        assert MetricCategory.SELFPLAY.value == "selfplay"
        assert MetricCategory.TRAINING.value == "training"
        assert MetricCategory.EVALUATION.value == "evaluation"
        assert MetricCategory.PROMOTION.value == "promotion"

    def test_infrastructure_categories_exist(self):
        """Test infrastructure categories exist."""
        assert MetricCategory.SYNC.value == "sync"
        assert MetricCategory.COORDINATOR.value == "coordinator"
        assert MetricCategory.PIPELINE.value == "pipeline"

    def test_application_categories_exist(self):
        """Test application categories exist."""
        assert MetricCategory.API.value == "api"
        assert MetricCategory.AI.value == "ai"
        assert MetricCategory.CACHE.value == "cache"


class TestMetricCatalog:
    """Tests for MetricCatalog class."""

    def test_singleton_pattern(self):
        """Test get_instance returns same instance."""
        catalog1 = MetricCatalog.get_instance()
        catalog2 = MetricCatalog.get_instance()

        assert catalog1 is catalog2

    def test_get_metric_catalog_helper(self):
        """Test get_metric_catalog returns singleton."""
        catalog = get_metric_catalog()
        singleton = MetricCatalog.get_instance()

        assert catalog is singleton

    def test_register_and_get(self):
        """Test registering and retrieving a metric."""
        catalog = get_metric_catalog()

        info = MetricInfo(
            name="test_register_get_metric",
            description="Test metric",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.TRAINING,
        )

        catalog.register(info)
        retrieved = catalog.get("test_register_get_metric")

        assert retrieved is info

    def test_get_nonexistent(self):
        """Test get returns None for non-existent metric."""
        catalog = get_metric_catalog()

        result = catalog.get("nonexistent_metric_xyz123")

        assert result is None

    def test_get_by_category(self):
        """Test filtering metrics by category."""
        catalog = get_metric_catalog()

        # Catalog should have pre-registered metrics
        selfplay_metrics = catalog.get_by_category(MetricCategory.SELFPLAY)

        # Should have some selfplay metrics
        assert len(selfplay_metrics) > 0
        # All should be selfplay category
        assert all(m.category == MetricCategory.SELFPLAY for m in selfplay_metrics)

    def test_get_by_category_empty(self):
        """Test get_by_category returns empty list for unused category."""
        # Create fresh catalog instance for this test
        catalog = MetricCatalog()

        result = catalog.get_by_category(MetricCategory.CACHE)

        assert result == []

    def test_search_by_name(self):
        """Test searching metrics by name."""
        catalog = get_metric_catalog()

        results = catalog.search("selfplay")

        assert len(results) > 0
        # Search also matches description, so check at least some match in name
        name_matches = [m for m in results if "selfplay" in m.name.lower()]
        assert len(name_matches) > 0

    def test_search_by_description(self):
        """Test searching metrics by description."""
        catalog = get_metric_catalog()

        results = catalog.search("games")

        assert len(results) > 0

    def test_search_case_insensitive(self):
        """Test search is case-insensitive."""
        catalog = get_metric_catalog()

        results_lower = catalog.search("selfplay")
        results_upper = catalog.search("SELFPLAY")
        results_mixed = catalog.search("SelfPlay")

        assert results_lower == results_upper == results_mixed

    def test_search_no_results(self):
        """Test search returns empty list for no matches."""
        catalog = get_metric_catalog()

        results = catalog.search("nonexistent_term_xyz123")

        assert results == []

    def test_list_all(self):
        """Test list_all returns all metrics."""
        catalog = get_metric_catalog()

        all_metrics = catalog.list_all()

        # Should have pre-registered metrics
        assert len(all_metrics) > 0
        assert all(isinstance(m, MetricInfo) for m in all_metrics)

    def test_list_names(self):
        """Test list_names returns sorted metric names."""
        catalog = get_metric_catalog()

        names = catalog.list_names()

        # Should be sorted
        assert names == sorted(names)
        # Should have some metrics
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


class TestMetricCatalogDocumentation:
    """Tests for documentation generation."""

    def test_get_documentation(self):
        """Test documentation generation."""
        catalog = get_metric_catalog()

        docs = catalog.get_documentation()

        assert isinstance(docs, str)
        assert "# RingRift Metrics Catalog" in docs
        assert "**Type**:" in docs
        assert "**Description**:" in docs

    def test_documentation_has_categories(self):
        """Test documentation includes category sections."""
        catalog = get_metric_catalog()

        docs = catalog.get_documentation()

        # Should have at least selfplay and training sections
        assert "## Selfplay Metrics" in docs or "selfplay" in docs.lower()

    def test_documentation_has_labels(self):
        """Test documentation includes labels for labeled metrics."""
        catalog = get_metric_catalog()

        docs = catalog.get_documentation()

        # Should mention labels somewhere
        assert "Labels" in docs or "labels" in docs.lower()


class TestRegisterMetricHelper:
    """Tests for register_metric helper function."""

    def test_register_metric_basic(self):
        """Test register_metric helper creates and registers."""
        info = register_metric(
            name="test_helper_registered_metric",
            description="Test metric via helper",
            metric_type=MetricType.GAUGE,
            category=MetricCategory.TRAINING,
        )

        assert isinstance(info, MetricInfo)
        assert info.name == "test_helper_registered_metric"

        # Should be in catalog
        catalog = get_metric_catalog()
        retrieved = catalog.get("test_helper_registered_metric")
        assert retrieved is info

    def test_register_metric_with_labels(self):
        """Test register_metric with labels."""
        info = register_metric(
            name="test_helper_with_labels",
            description="Test",
            metric_type=MetricType.COUNTER,
            category=MetricCategory.SELFPLAY,
            labels=["board_type", "num_players"],
        )

        assert info.labels == ["board_type", "num_players"]

    def test_register_metric_with_unit(self):
        """Test register_metric with unit."""
        info = register_metric(
            name="test_helper_with_unit_seconds",
            description="Test",
            metric_type=MetricType.HISTOGRAM,
            category=MetricCategory.PIPELINE,
            unit="seconds",
        )

        assert info.unit == "seconds"


class TestPreRegisteredMetrics:
    """Tests for pre-registered metrics in the catalog."""

    def test_selfplay_games_total_registered(self):
        """Test selfplay games counter is pre-registered."""
        catalog = get_metric_catalog()

        info = catalog.get("ringrift_selfplay_games_total")

        assert info is not None
        assert info.metric_type == MetricType.COUNTER
        assert info.category == MetricCategory.SELFPLAY
        assert "board_type" in info.labels

    def test_training_loss_registered(self):
        """Test training loss gauge is pre-registered."""
        catalog = get_metric_catalog()

        info = catalog.get("ringrift_orchestrator_training_loss")

        assert info is not None
        assert info.metric_type == MetricType.GAUGE
        assert info.category == MetricCategory.TRAINING

    def test_pipeline_state_registered(self):
        """Test pipeline state gauge is pre-registered."""
        catalog = get_metric_catalog()

        info = catalog.get("ringrift_pipeline_state")

        assert info is not None
        assert info.metric_type == MetricType.GAUGE
        assert info.category == MetricCategory.PIPELINE

    def test_coordinator_status_registered(self):
        """Test coordinator status is pre-registered."""
        catalog = get_metric_catalog()

        info = catalog.get("ringrift_coordinator_status")

        assert info is not None
        assert info.metric_type == MetricType.GAUGE
        assert info.category == MetricCategory.COORDINATOR
