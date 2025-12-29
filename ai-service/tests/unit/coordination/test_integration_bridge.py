"""Unit tests for integration_bridge.py.

Tests the critical event wiring infrastructure that connects integration modules
(ModelLifecycleManager, P2PIntegrationManager, PipelineFeedbackController)
to the unified event router.

Created: December 2025
Purpose: Provide test coverage for critical training pipeline wiring
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.integration_bridge import (
    # Event constants
    EVENT_CLUSTER_HEALTH_CHANGED,
    EVENT_CURRICULUM_UPDATED,
    EVENT_ELO_UPDATED,
    EVENT_EVALUATION_COMPLETE,
    EVENT_EVALUATION_SCHEDULED,
    EVENT_FEEDBACK_SIGNAL,
    EVENT_MODEL_PROMOTED,
    EVENT_MODEL_REGISTERED,
    EVENT_MODEL_REJECTED,
    EVENT_MODEL_ROLLBACK,
    EVENT_PARITY_VALIDATION_COMPLETE,
    EVENT_REGISTRY_SYNC_NEEDED,
    EVENT_SELFPLAY_SCALED,
    EVENT_TRAINING_COMPLETED,
    EVENT_TRAINING_TRIGGERED,
    # Functions
    get_wiring_status,
    health_check,
    reset_integration_wiring,
    verify_integration_health_sync,
    wire_all_integrations_sync,
)


class TestEventTypeConstants:
    """Tests for event type constants."""

    def test_event_constants_are_strings(self):
        """All event constants should be non-empty strings."""
        events = [
            EVENT_MODEL_REGISTERED,
            EVENT_MODEL_PROMOTED,
            EVENT_MODEL_REJECTED,
            EVENT_MODEL_ROLLBACK,
            EVENT_TRAINING_TRIGGERED,
            EVENT_TRAINING_COMPLETED,
            EVENT_EVALUATION_COMPLETE,
            EVENT_EVALUATION_SCHEDULED,
            EVENT_CLUSTER_HEALTH_CHANGED,
            EVENT_SELFPLAY_SCALED,
            EVENT_FEEDBACK_SIGNAL,
            EVENT_PARITY_VALIDATION_COMPLETE,
            EVENT_ELO_UPDATED,
            EVENT_REGISTRY_SYNC_NEEDED,
            EVENT_CURRICULUM_UPDATED,
        ]
        for event in events:
            assert isinstance(event, str)
            assert len(event) > 0

    def test_evaluation_complete_matches_data_events(self):
        """Event constant should match DataEventType value."""
        assert EVENT_EVALUATION_COMPLETE == "evaluation_completed"

    def test_parity_validation_matches_data_events(self):
        """Event constant should match DataEventType value."""
        assert EVENT_PARITY_VALIDATION_COMPLETE == "parity_validation_completed"

    def test_no_duplicate_event_types(self):
        """All event type constants should be unique."""
        events = [
            EVENT_MODEL_REGISTERED,
            EVENT_MODEL_PROMOTED,
            EVENT_MODEL_REJECTED,
            EVENT_MODEL_ROLLBACK,
            EVENT_TRAINING_TRIGGERED,
            EVENT_TRAINING_COMPLETED,
            EVENT_EVALUATION_COMPLETE,
            EVENT_EVALUATION_SCHEDULED,
            EVENT_CLUSTER_HEALTH_CHANGED,
            EVENT_SELFPLAY_SCALED,
            EVENT_FEEDBACK_SIGNAL,
            EVENT_PARITY_VALIDATION_COMPLETE,
            EVENT_ELO_UPDATED,
            EVENT_REGISTRY_SYNC_NEEDED,
            EVENT_CURRICULUM_UPDATED,
        ]
        assert len(events) == len(set(events)), "Duplicate event type constants found"


class TestCoerceInt:
    """Tests for _coerce_int helper function."""

    def test_coerce_int_with_int(self):
        from app.coordination.integration_bridge import _coerce_int
        assert _coerce_int(42) == 42
        assert _coerce_int(0) == 0
        assert _coerce_int(-10) == -10

    def test_coerce_int_with_float(self):
        from app.coordination.integration_bridge import _coerce_int
        assert _coerce_int(42.0) == 42
        assert _coerce_int(42.7) == 42

    def test_coerce_int_with_string(self):
        from app.coordination.integration_bridge import _coerce_int
        assert _coerce_int("42") == 42
        assert _coerce_int("  42  ") == 42
        assert _coerce_int("0") == 0

    def test_coerce_int_with_invalid_string(self):
        from app.coordination.integration_bridge import _coerce_int
        assert _coerce_int("abc") is None
        assert _coerce_int("42.5") is None
        assert _coerce_int("") is None

    def test_coerce_int_with_none(self):
        from app.coordination.integration_bridge import _coerce_int
        assert _coerce_int(None) is None


class TestInferVersionFromModelId:
    """Tests for _infer_version_from_model_id helper function."""

    def test_infer_version_with_colon_v(self):
        from app.coordination.integration_bridge import _infer_version_from_model_id
        assert _infer_version_from_model_id("model:v1") == 1
        assert _infer_version_from_model_id("hex8_2p:v42") == 42

    def test_infer_version_with_underscore_v(self):
        from app.coordination.integration_bridge import _infer_version_from_model_id
        assert _infer_version_from_model_id("model_v1") == 1
        assert _infer_version_from_model_id("hex8_2p_v42") == 42

    def test_infer_version_with_dash_v(self):
        from app.coordination.integration_bridge import _infer_version_from_model_id
        assert _infer_version_from_model_id("model-v1") == 1

    def test_infer_version_no_version(self):
        from app.coordination.integration_bridge import _infer_version_from_model_id
        assert _infer_version_from_model_id("model") is None
        assert _infer_version_from_model_id("") is None
        assert _infer_version_from_model_id(None) is None


class TestFirstPresent:
    """Tests for _first_present helper function."""

    def test_first_present_in_payload(self):
        from app.coordination.integration_bridge import _first_present
        payload = {"model_id": "test_model", "version": 1}
        metadata = {}
        assert _first_present(payload, metadata, ["model_id", "model"]) == "test_model"

    def test_first_present_in_metadata(self):
        from app.coordination.integration_bridge import _first_present
        payload = {}
        metadata = {"model": "from_meta"}
        assert _first_present(payload, metadata, ["model_id", "model"]) == "from_meta"

    def test_first_present_payload_priority(self):
        from app.coordination.integration_bridge import _first_present
        payload = {"model": "from_payload"}
        metadata = {"model": "from_meta"}
        assert _first_present(payload, metadata, ["model"]) == "from_payload"

    def test_first_present_none_values_skipped(self):
        from app.coordination.integration_bridge import _first_present
        payload = {"model_id": None, "model": "valid"}
        metadata = {}
        assert _first_present(payload, metadata, ["model_id", "model"]) == "valid"

    def test_first_present_not_found(self):
        from app.coordination.integration_bridge import _first_present
        payload = {}
        metadata = {}
        assert _first_present(payload, metadata, ["model_id"]) is None


class TestBuildEvaluationResult:
    """Tests for _build_evaluation_result helper function."""

    def test_build_evaluation_result_basic(self):
        from app.coordination.integration_bridge import _build_evaluation_result
        payload = {
            "model_id": "test_model",
            "version": 5,
            "elo": 1500.0,
            "games_played": 100,
        }
        result = _build_evaluation_result(payload)
        assert result is not None
        assert result.model_id == "test_model"
        assert result.version == 5
        assert result.elo == 1500.0

    def test_build_evaluation_result_from_metadata(self):
        from app.coordination.integration_bridge import _build_evaluation_result
        payload = {
            "metadata": {
                "model_id": "from_meta",
                "version": 3,
            }
        }
        result = _build_evaluation_result(payload)
        assert result is not None
        assert result.model_id == "from_meta"
        assert result.version == 3

    def test_build_evaluation_result_infers_version(self):
        from app.coordination.integration_bridge import _build_evaluation_result
        payload = {"model_id": "model:v7"}
        result = _build_evaluation_result(payload)
        assert result is not None
        assert result.version == 7

    def test_build_evaluation_result_missing_model_id(self):
        from app.coordination.integration_bridge import _build_evaluation_result
        payload = {"version": 5}
        result = _build_evaluation_result(payload)
        assert result is None

    def test_build_evaluation_result_missing_version(self):
        from app.coordination.integration_bridge import _build_evaluation_result
        payload = {"model_id": "test"}
        result = _build_evaluation_result(payload)
        assert result is None


class TestTaskErrorCallback:
    """Tests for _task_error_callback function."""

    def test_task_error_callback_logs_exception(self):
        from app.coordination.integration_bridge import _task_error_callback

        task = MagicMock(spec=asyncio.Task)
        task.exception.return_value = ValueError("Test error")

        with patch("app.coordination.integration_bridge.logger") as mock_logger:
            _task_error_callback(task)
            mock_logger.error.assert_called_once()

    def test_task_error_callback_no_exception(self):
        from app.coordination.integration_bridge import _task_error_callback

        task = MagicMock(spec=asyncio.Task)
        task.exception.return_value = None

        with patch("app.coordination.integration_bridge.logger") as mock_logger:
            _task_error_callback(task)
            mock_logger.error.assert_not_called()

    def test_task_error_callback_cancelled(self):
        from app.coordination.integration_bridge import _task_error_callback

        task = MagicMock(spec=asyncio.Task)
        task.exception.side_effect = asyncio.CancelledError()
        _task_error_callback(task)  # Should not raise


class TestHealthCheck:
    """Tests for health_check function."""

    def test_health_check_not_wired(self):
        reset_integration_wiring()
        result = health_check()
        assert result.healthy is False
        assert result.details.get("wired") is False

    def test_health_check_wired(self):
        import app.coordination.integration_bridge as bridge_module

        bridge_module._integration_wired = True
        bridge_module._events_processed = 100
        bridge_module._errors_count = 0

        result = health_check()
        assert result.healthy is True
        assert result.details.get("wired") is True

        reset_integration_wiring()

    def test_health_check_high_error_rate(self):
        import app.coordination.integration_bridge as bridge_module

        bridge_module._integration_wired = True
        bridge_module._events_processed = 100
        bridge_module._errors_count = 60

        result = health_check()
        assert result.healthy is False
        assert "High error rate" in result.message

        reset_integration_wiring()


class TestGetWiringStatus:
    """Tests for get_wiring_status function."""

    def test_get_wiring_status_returns_dict(self):
        status = get_wiring_status()
        assert isinstance(status, dict)
        assert "wired" in status
        assert "event_types_registered" in status
        assert "components" in status

    def test_get_wiring_status_components(self):
        status = get_wiring_status()
        components = status["components"]
        assert "model_lifecycle" in components
        assert "p2p_integration" in components
        assert "pipeline_feedback" in components


class TestWireModelLifecycleEvents:
    """Tests for wire_model_lifecycle_events function."""

    def test_wire_model_lifecycle_events_registers_callbacks(self):
        from app.coordination.integration_bridge import wire_model_lifecycle_events

        mock_manager = MagicMock()
        mock_manager.register_callback = MagicMock()

        wire_model_lifecycle_events(mock_manager)

        assert mock_manager.register_callback.call_count == 4
        callback_names = [
            call[0][0] for call in mock_manager.register_callback.call_args_list
        ]
        assert "model_registered" in callback_names
        assert "model_promoted" in callback_names


class TestWireP2PIntegrationEvents:
    """Tests for wire_p2p_integration_events function."""

    def test_wire_p2p_integration_events_registers_callbacks(self):
        from app.coordination.integration_bridge import wire_p2p_integration_events

        mock_manager = MagicMock()
        mock_manager.register_callback = MagicMock()

        wire_p2p_integration_events(mock_manager)

        assert mock_manager.register_callback.call_count == 4
        callback_names = [
            call[0][0] for call in mock_manager.register_callback.call_args_list
        ]
        assert "cluster_healthy" in callback_names
        assert "cluster_unhealthy" in callback_names


class TestWirePipelineFeedbackEvents:
    """Tests for wire_pipeline_feedback_events function."""

    def test_wire_pipeline_feedback_events_subscribes(self):
        from app.coordination.integration_bridge import wire_pipeline_feedback_events

        mock_controller = MagicMock()
        mock_controller._emit_signal = None

        with patch("app.coordination.integration_bridge.subscribe") as mock_sub:
            wire_pipeline_feedback_events(mock_controller)
            subscribed_events = [call[0][0] for call in mock_sub.call_args_list]
            assert EVENT_TRAINING_COMPLETED in subscribed_events
            assert EVENT_EVALUATION_COMPLETE in subscribed_events


class TestWireSyncManagerEvents:
    """Tests for wire_sync_manager_events function."""

    def test_wire_sync_manager_events_subscribes(self):
        from app.coordination.integration_bridge import wire_sync_manager_events

        with patch("app.coordination.integration_bridge.subscribe") as mock_sub:
            wire_sync_manager_events()
            subscribed_events = [call[0][0] for call in mock_sub.call_args_list]
            assert EVENT_MODEL_PROMOTED in subscribed_events
            assert EVENT_ELO_UPDATED in subscribed_events


class TestEventHandlerCallbacks:
    """Tests for event handler callback behavior."""

    def test_model_registered_callback_publishes(self):
        from app.coordination.integration_bridge import wire_model_lifecycle_events

        mock_manager = MagicMock()
        registered_callback = None

        def capture_callback(name, cb):
            nonlocal registered_callback
            if name == "model_registered":
                registered_callback = cb

        mock_manager.register_callback = capture_callback

        wire_model_lifecycle_events(mock_manager)

        assert registered_callback is not None

        with patch("app.coordination.integration_bridge.publish_sync") as mock_pub:
            registered_callback(model_id="test", version=1)
            mock_pub.assert_called_once()
            call_args = mock_pub.call_args
            assert call_args[0][0] == EVENT_MODEL_REGISTERED

    def test_cluster_health_callback_includes_status(self):
        from app.coordination.integration_bridge import wire_p2p_integration_events

        mock_manager = MagicMock()
        healthy_callback = None

        def capture_callback(name, cb):
            nonlocal healthy_callback
            if name == "cluster_healthy":
                healthy_callback = cb

        mock_manager.register_callback = capture_callback

        wire_p2p_integration_events(mock_manager)

        with patch("app.coordination.integration_bridge.publish_sync") as mock_pub:
            healthy_callback(node_count=10)
            call_args = mock_pub.call_args
            payload = call_args[0][1]
            assert payload["healthy"] is True
