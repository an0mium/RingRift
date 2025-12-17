"""Tests for notification_config module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.training.notification_config import (
    DEFAULT_CONFIG_PATH,
    FilteredWebhookHook,
    OpsGenieNotificationHook,
    PagerDutyNotificationHook,
    RollbackConfig,
    load_config_yaml,
    load_notification_hooks,
    load_rollback_config,
    load_rollback_criteria,
)
from app.training.promotion_controller import (
    LoggingNotificationHook,
    RollbackCriteria,
    RollbackEvent,
)


class TestLoadConfigYaml:
    """Test YAML config loading."""

    def test_load_default_config(self):
        """Test loading the default config file."""
        if DEFAULT_CONFIG_PATH.exists():
            config = load_config_yaml()
            assert isinstance(config, dict)
            assert "version" in config

    def test_load_custom_config(self):
        """Test loading a custom config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"enabled": True, "test_key": "test_value"}, f)
            temp_path = Path(f.name)

        try:
            config = load_config_yaml(temp_path)
            assert config["enabled"] is True
            assert config["test_key"] == "test_value"
        finally:
            temp_path.unlink()

    def test_load_missing_config_returns_defaults(self):
        """Test loading a missing config file returns defaults."""
        config = load_config_yaml(Path("/nonexistent/path.yaml"))
        assert config["enabled"] is True
        assert config["logging"]["enabled"] is True

    def test_env_var_override(self):
        """Test environment variable config path override."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"enabled": False, "from_env": True}, f)
            temp_path = Path(f.name)

        try:
            with patch.dict("os.environ", {"RINGRIFT_NOTIFICATION_CONFIG": str(temp_path)}):
                config = load_config_yaml()
                assert config["enabled"] is False
                assert config["from_env"] is True
        finally:
            temp_path.unlink()


class TestLoadNotificationHooks:
    """Test loading notification hooks from config."""

    def test_load_disabled_returns_empty(self):
        """Test disabled config returns no hooks."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"enabled": False}, f)
            temp_path = Path(f.name)

        try:
            hooks = load_notification_hooks(temp_path)
            assert hooks == []
        finally:
            temp_path.unlink()

    def test_load_logging_hook_by_default(self):
        """Test logging hook is loaded by default."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"enabled": True}, f)
            temp_path = Path(f.name)

        try:
            hooks = load_notification_hooks(temp_path)
            assert len(hooks) == 1
            assert isinstance(hooks[0], LoggingNotificationHook)
        finally:
            temp_path.unlink()

    def test_load_slack_hook(self):
        """Test loading Slack webhook hook."""
        config = {
            "enabled": True,
            "logging": {"enabled": False},
            "slack": {
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/test",
                "timeout_seconds": 5,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            hooks = load_notification_hooks(temp_path)
            assert len(hooks) == 1
            assert isinstance(hooks[0], FilteredWebhookHook)
            assert hooks[0].webhook_type == "slack"
        finally:
            temp_path.unlink()

    def test_load_pagerduty_hook(self):
        """Test loading PagerDuty hook."""
        config = {
            "enabled": True,
            "logging": {"enabled": False},
            "pagerduty": {
                "enabled": True,
                "routing_key": "test-routing-key",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            hooks = load_notification_hooks(temp_path)
            assert len(hooks) == 1
            assert isinstance(hooks[0], PagerDutyNotificationHook)
        finally:
            temp_path.unlink()

    def test_load_opsgenie_hook(self):
        """Test loading OpsGenie hook."""
        config = {
            "enabled": True,
            "logging": {"enabled": False},
            "opsgenie": {
                "enabled": True,
                "api_key": "test-api-key",
                "region": "eu",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            hooks = load_notification_hooks(temp_path)
            assert len(hooks) == 1
            assert isinstance(hooks[0], OpsGenieNotificationHook)
            assert "eu.opsgenie.com" in hooks[0].base_url
        finally:
            temp_path.unlink()

    def test_load_multiple_hooks(self):
        """Test loading multiple hooks from config."""
        config = {
            "enabled": True,
            "logging": {"enabled": True},
            "slack": {
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/test",
            },
            "discord": {
                "enabled": True,
                "webhook_url": "https://discord.com/api/webhooks/test",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            hooks = load_notification_hooks(temp_path)
            assert len(hooks) == 3  # logging + slack + discord
        finally:
            temp_path.unlink()


class TestLoadRollbackCriteria:
    """Test loading rollback criteria from config."""

    def test_load_default_criteria(self):
        """Test loading default criteria when no overrides."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"enabled": True}, f)
            temp_path = Path(f.name)

        try:
            criteria = load_rollback_criteria(temp_path)
            assert isinstance(criteria, RollbackCriteria)
            assert criteria.elo_regression_threshold == -30.0
        finally:
            temp_path.unlink()

    def test_load_custom_criteria(self):
        """Test loading custom criteria from config."""
        config = {
            "criteria_overrides": {
                "elo_regression_threshold": -50.0,
                "min_games_for_regression": 30,
                "cooldown_seconds": 7200,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            criteria = load_rollback_criteria(temp_path)
            assert criteria.elo_regression_threshold == -50.0
            assert criteria.min_games_for_regression == 30
            assert criteria.cooldown_seconds == 7200
        finally:
            temp_path.unlink()


class TestLoadRollbackConfig:
    """Test loading complete rollback config."""

    def test_load_complete_config(self):
        """Test loading complete config with hooks and criteria."""
        config = {
            "enabled": True,
            "logging": {"enabled": True},
            "criteria_overrides": {
                "max_rollbacks_per_day": 5,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)

        try:
            rollback_config = load_rollback_config(temp_path)
            assert isinstance(rollback_config, RollbackConfig)
            assert rollback_config.enabled is True
            assert len(rollback_config.hooks) == 1
            assert rollback_config.criteria.max_rollbacks_per_day == 5
        finally:
            temp_path.unlink()


class TestFilteredWebhookHook:
    """Test filtered webhook hook."""

    def test_event_filtering(self):
        """Test that events are filtered based on config."""
        hook = FilteredWebhookHook(
            webhook_url="https://test.com",
            events={
                "at_risk": False,
                "rollback_triggered": True,
                "rollback_completed": True,
            },
        )

        # Mock the _send_webhook method
        hook._send_webhook = MagicMock(return_value=True)

        # at_risk should not trigger (disabled)
        hook.on_at_risk("model_v1", {"consecutive_regressions": 2})
        hook._send_webhook.assert_not_called()

        # rollback_triggered should trigger
        event = RollbackEvent(
            triggered_at="2024-01-01T00:00:00",
            current_model_id="model_v2",
            rollback_model_id="model_v1",
            reason="test",
        )
        hook.on_rollback_triggered(event)
        hook._send_webhook.assert_called_once()


class TestPagerDutyNotificationHook:
    """Test PagerDuty notification hook."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = PagerDutyNotificationHook(
            routing_key="test-key",
            severity_mapping={"at_risk": "info"},
        )
        assert hook.routing_key == "test-key"
        assert hook.severity_mapping["at_risk"] == "info"

    @patch("urllib.request.urlopen")
    def test_send_event(self, mock_urlopen):
        """Test sending an event to PagerDuty."""
        mock_urlopen.return_value.__enter__ = MagicMock()
        mock_urlopen.return_value.__exit__ = MagicMock()

        hook = PagerDutyNotificationHook(routing_key="test-key")
        result = hook._send_event(
            summary="Test alert",
            severity="critical",
            dedup_key="test-dedup",
        )

        assert result is True
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_on_rollback_triggered(self, mock_urlopen):
        """Test on_rollback_triggered sends event."""
        mock_urlopen.return_value.__enter__ = MagicMock()
        mock_urlopen.return_value.__exit__ = MagicMock()

        hook = PagerDutyNotificationHook(routing_key="test-key")
        event = RollbackEvent(
            triggered_at="2024-01-01T00:00:00",
            current_model_id="model_v2",
            rollback_model_id="model_v1",
            reason="test regression",
        )

        hook.on_rollback_triggered(event)
        mock_urlopen.assert_called_once()


class TestOpsGenieNotificationHook:
    """Test OpsGenie notification hook."""

    def test_initialization_us_region(self):
        """Test hook initialization with US region."""
        hook = OpsGenieNotificationHook(api_key="test-key", region="us")
        assert "api.opsgenie.com" in hook.base_url
        assert "eu" not in hook.base_url

    def test_initialization_eu_region(self):
        """Test hook initialization with EU region."""
        hook = OpsGenieNotificationHook(api_key="test-key", region="eu")
        assert "api.eu.opsgenie.com" in hook.base_url

    @patch("urllib.request.urlopen")
    def test_send_alert(self, mock_urlopen):
        """Test sending an alert to OpsGenie."""
        mock_urlopen.return_value.__enter__ = MagicMock()
        mock_urlopen.return_value.__exit__ = MagicMock()

        hook = OpsGenieNotificationHook(api_key="test-key")
        result = hook._send_alert(
            message="Test alert",
            priority="P1",
            alias="test-alias",
        )

        assert result is True
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_on_rollback_completed_success_closes_alert(self, mock_urlopen):
        """Test successful rollback closes OpsGenie alert."""
        mock_urlopen.return_value.__enter__ = MagicMock()
        mock_urlopen.return_value.__exit__ = MagicMock()

        hook = OpsGenieNotificationHook(api_key="test-key")
        event = RollbackEvent(
            triggered_at="2024-01-01T00:00:00",
            current_model_id="model_v2",
            rollback_model_id="model_v1",
            reason="test",
        )

        hook.on_rollback_completed(event, success=True)
        # Should call close for both rollback and at-risk alerts
        assert mock_urlopen.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
