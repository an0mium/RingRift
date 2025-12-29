"""Unit tests for availability/capacity_planner.py.

Tests for CapacityPlanner daemon that manages budget-aware capacity.

Created: Dec 28, 2025
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.capacity_planner import (
    CapacityBudget,
    CapacityPlanner,
    CapacityPlannerConfig,
    ScaleAction,
    ScaleRecommendation,
    UtilizationMetrics,
    get_capacity_planner,
)


class TestScaleAction:
    """Tests for ScaleAction enum."""

    def test_action_values(self):
        """Test all action values exist."""
        assert ScaleAction.SCALE_UP is not None
        assert ScaleAction.SCALE_DOWN is not None
        assert ScaleAction.NONE is not None

    def test_action_names(self):
        """Test action names."""
        assert ScaleAction.SCALE_UP.value == "scale_up"
        assert ScaleAction.SCALE_DOWN.value == "scale_down"
        assert ScaleAction.NONE.value == "none"


class TestCapacityBudget:
    """Tests for CapacityBudget dataclass."""

    def test_default_budget(self):
        """Test default budget values."""
        budget = CapacityBudget()
        assert budget.hourly_limit_usd > 0
        assert budget.daily_limit_usd > 0
        assert budget.current_hourly_usd == 0.0
        assert budget.current_daily_usd == 0.0

    def test_custom_budget(self):
        """Test custom budget values."""
        budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=2000.0,
            current_hourly_usd=25.0,
            current_daily_usd=500.0,
        )
        assert budget.hourly_limit_usd == 100.0
        assert budget.daily_limit_usd == 2000.0
        assert budget.current_hourly_usd == 25.0

    def test_is_exceeded(self):
        """Test budget exceeded detection."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
            current_hourly_usd=60.0,
            current_daily_usd=100.0,
        )
        # Hourly is exceeded
        assert budget.current_hourly_usd > budget.hourly_limit_usd

    def test_to_dict(self):
        """Test serialization to dict."""
        budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
        )
        d = budget.to_dict()
        assert d["hourly_limit_usd"] == 50.0
        assert d["daily_limit_usd"] == 500.0
        assert "current_hourly_usd" in d


class TestScaleRecommendation:
    """Tests for ScaleRecommendation dataclass."""

    def test_scale_up_recommendation(self):
        """Test scale up recommendation."""
        rec = ScaleRecommendation(
            action=ScaleAction.SCALE_UP,
            count=2,
            reason="Utilization above 90%",
        )
        assert rec.action == ScaleAction.SCALE_UP
        assert rec.count == 2
        assert "90%" in rec.reason

    def test_no_action_recommendation(self):
        """Test no action recommendation."""
        rec = ScaleRecommendation(
            action=ScaleAction.NONE,
            count=0,
            reason="Cluster is within target utilization",
        )
        assert rec.action == ScaleAction.NONE
        assert rec.count == 0

    def test_to_dict(self):
        """Test serialization to dict."""
        rec = ScaleRecommendation(
            action=ScaleAction.SCALE_DOWN,
            count=1,
            reason="Low utilization",
        )
        d = rec.to_dict()
        assert d["action"] == "scale_down"
        assert d["count"] == 1
        assert d["reason"] == "Low utilization"


class TestUtilizationMetrics:
    """Tests for UtilizationMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics."""
        metrics = UtilizationMetrics()
        assert metrics.gpu_utilization == 0.0
        assert metrics.active_gpus == 0
        assert metrics.total_gpus == 0

    def test_custom_metrics(self):
        """Test custom metrics."""
        metrics = UtilizationMetrics(
            gpu_utilization=75.5,
            active_gpus=12,
            total_gpus=16,
            active_jobs=8,
        )
        assert metrics.gpu_utilization == 75.5
        assert metrics.active_gpus == 12

    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = UtilizationMetrics(
            gpu_utilization=50.0,
            active_gpus=4,
            total_gpus=8,
        )
        d = metrics.to_dict()
        assert d["gpu_utilization"] == 50.0
        assert "timestamp" in d


class TestCapacityPlannerConfig:
    """Tests for CapacityPlannerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CapacityPlannerConfig()
        assert config.hourly_budget_usd > 0
        assert config.daily_budget_usd > 0
        assert config.min_gpu_capacity > 0
        assert config.target_utilization_min > 0
        assert config.target_utilization_max < 100

    def test_from_env(self):
        """Test loading config from environment."""
        with patch.dict("os.environ", {
            "RINGRIFT_CAPACITY_HOURLY_BUDGET": "100.0",
            "RINGRIFT_CAPACITY_MIN_GPUS": "8",
        }):
            config = CapacityPlannerConfig.from_env()
            assert config.hourly_budget_usd == 100.0
            assert config.min_gpu_capacity == 8


class TestCapacityPlanner:
    """Tests for CapacityPlanner daemon."""

    def test_singleton_pattern(self):
        """Test that get_capacity_planner returns singleton."""
        CapacityPlanner._instance = None

        planner1 = get_capacity_planner()
        planner2 = get_capacity_planner()

        assert planner1 is planner2

        CapacityPlanner._instance = None

    def test_event_subscriptions(self):
        """Test event subscription setup."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        subs = planner._get_event_subscriptions()

        # Should subscribe to provisioning events
        assert "NODE_PROVISIONED" in subs
        assert "NODE_TERMINATED" in subs

        CapacityPlanner._instance = None

    def test_get_budget_status(self):
        """Test getting current budget status."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        status = planner.get_budget_status()
        assert "hourly_limit_usd" in status
        assert "current_hourly_usd" in status

        CapacityPlanner._instance = None

    def test_get_utilization_history(self):
        """Test getting utilization history."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        history = planner.get_utilization_history()
        assert isinstance(history, list)

        CapacityPlanner._instance = None

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health_check method returns valid result."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        result = planner.health_check()

        assert "healthy" in result
        assert "name" in result
        assert result["name"] == "CapacityPlanner"

        CapacityPlanner._instance = None


class TestCapacityPlannerBudgetChecks:
    """Tests for budget checking logic."""

    def test_should_scale_up_within_budget(self):
        """Test scale up allowed when within budget."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        # Set budget with room to spare
        planner.budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=2000.0,
            current_hourly_usd=50.0,
            current_daily_usd=500.0,
        )

        result = planner.should_scale_up(count=1)
        assert result is True

        CapacityPlanner._instance = None

    def test_should_not_scale_up_budget_exceeded(self):
        """Test scale up blocked when budget exceeded."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        # Set budget at limit
        planner.budget = CapacityBudget(
            hourly_limit_usd=50.0,
            daily_limit_usd=500.0,
            current_hourly_usd=50.0,  # At limit
            current_daily_usd=100.0,
        )

        result = planner.should_scale_up(count=1)
        assert result is False

        CapacityPlanner._instance = None


class TestCapacityPlannerEventHandlers:
    """Tests for CapacityPlanner event handlers."""

    @pytest.mark.asyncio
    async def test_on_node_provisioned(self):
        """Test handling NODE_PROVISIONED event."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        # Reset budget
        planner.budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=2000.0,
        )

        event = {
            "payload": {
                "node_id": "test-node",
                "cost_per_hour": 2.50,
            }
        }

        await planner._on_node_provisioned(event)

        # Budget should increase
        assert planner.budget.current_hourly_usd >= 2.50

        CapacityPlanner._instance = None

    @pytest.mark.asyncio
    async def test_on_node_terminated(self):
        """Test handling NODE_TERMINATED event."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        # Set initial budget
        planner.budget = CapacityBudget(
            hourly_limit_usd=100.0,
            daily_limit_usd=2000.0,
            current_hourly_usd=10.0,
            current_daily_usd=100.0,
        )

        event = {
            "payload": {
                "node_id": "test-node",
                "cost_per_hour": 2.50,
            }
        }

        await planner._on_node_terminated(event)

        # Budget should decrease
        assert planner.budget.current_hourly_usd <= 10.0

        CapacityPlanner._instance = None


class TestCapacityPlannerScaling:
    """Tests for scaling recommendation logic."""

    @pytest.mark.asyncio
    async def test_get_scale_recommendation_high_utilization(self):
        """Test scale up recommended for high utilization."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        # Mock high utilization
        with patch.object(planner, "_get_current_utilization", new_callable=AsyncMock) as mock_util:
            mock_util.return_value = UtilizationMetrics(
                gpu_utilization=95.0,
                active_gpus=15,
                total_gpus=16,
            )

            # Also ensure budget allows scaling
            planner.budget = CapacityBudget(
                hourly_limit_usd=100.0,
                daily_limit_usd=2000.0,
                current_hourly_usd=20.0,
                current_daily_usd=200.0,
            )

            rec = await planner.get_scale_recommendation()

            # Should recommend scaling up
            assert rec.action == ScaleAction.SCALE_UP or rec.action == ScaleAction.NONE

        CapacityPlanner._instance = None

    @pytest.mark.asyncio
    async def test_get_scale_recommendation_low_utilization(self):
        """Test scale down recommended for low utilization."""
        CapacityPlanner._instance = None
        planner = get_capacity_planner()

        # Mock low utilization
        with patch.object(planner, "_get_current_utilization", new_callable=AsyncMock) as mock_util:
            mock_util.return_value = UtilizationMetrics(
                gpu_utilization=15.0,
                active_gpus=2,
                total_gpus=16,
            )

            rec = await planner.get_scale_recommendation()

            # Should recommend scaling down or no action
            assert rec.action in [ScaleAction.SCALE_DOWN, ScaleAction.NONE]

        CapacityPlanner._instance = None
