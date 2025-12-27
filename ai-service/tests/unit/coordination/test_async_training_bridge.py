#!/usr/bin/env python3
"""Unit tests for async_training_bridge.py

Tests the async bridge between TrainingCoordinator and event-driven pipeline.

Test coverage:
- TrainingProgressEvent dataclass (3 tests)
- AsyncTrainingBridge class (22 tests)
- Singleton management (4 tests)
- Convenience functions (6 tests)

Total: 35 tests
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.coordination.async_training_bridge import (
    AsyncTrainingBridge,
    TrainingProgressEvent,
    get_training_bridge,
    reset_training_bridge,
    async_can_train,
    async_complete_training,
    async_get_training_status,
    async_request_training,
    async_update_progress,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_coordinator():
    """Create a mock TrainingCoordinator."""
    coordinator = MagicMock()
    coordinator.can_start_training.return_value = True
    coordinator.start_training.return_value = "square8_2p_1234_5678"
    coordinator.update_progress.return_value = True
    coordinator.complete_training.return_value = True
    coordinator.get_active_jobs.return_value = []
    coordinator.get_status.return_value = {"active_jobs": 0}
    return coordinator


@pytest.fixture
def mock_job():
    """Create a mock TrainingJob."""
    job = MagicMock()
    job.board_type = "square8"
    job.num_players = 2
    job.job_id = "square8_2p_1234_5678"
    return job


@pytest.fixture
def mock_bridge_manager():
    """Create a mock bridge manager."""
    manager = MagicMock()

    async def mock_run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)

    manager.run_sync = mock_run_sync
    return manager


@pytest.fixture
def bridge(mock_coordinator, mock_bridge_manager):
    """Create an AsyncTrainingBridge with mocked dependencies."""
    with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
        with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coordinator):
            bridge = AsyncTrainingBridge(coordinator=mock_coordinator)
            yield bridge


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_training_bridge()
    yield
    reset_training_bridge()


# =============================================================================
# TrainingProgressEvent Tests
# =============================================================================


class TestTrainingProgressEvent:
    """Tests for TrainingProgressEvent dataclass."""

    def test_event_creation(self):
        """Test creating a TrainingProgressEvent."""
        event = TrainingProgressEvent(
            job_id="square8_2p_1234_5678",
            board_type="square8",
            num_players=2,
            epochs_completed=10,
            best_val_loss=0.05,
            current_elo=1500.0,
        )

        assert event.job_id == "square8_2p_1234_5678"
        assert event.board_type == "square8"
        assert event.num_players == 2
        assert event.epochs_completed == 10
        assert event.best_val_loss == 0.05
        assert event.current_elo == 1500.0

    def test_event_with_hex_board(self):
        """Test event with hex board type."""
        event = TrainingProgressEvent(
            job_id="hex8_4p_5678_1234",
            board_type="hex8",
            num_players=4,
            epochs_completed=50,
            best_val_loss=0.01,
            current_elo=1800.0,
        )

        assert event.board_type == "hex8"
        assert event.num_players == 4

    def test_event_equality(self):
        """Test that events with same data are equal."""
        event1 = TrainingProgressEvent(
            job_id="test_job",
            board_type="square8",
            num_players=2,
            epochs_completed=10,
            best_val_loss=0.05,
            current_elo=1500.0,
        )
        event2 = TrainingProgressEvent(
            job_id="test_job",
            board_type="square8",
            num_players=2,
            epochs_completed=10,
            best_val_loss=0.05,
            current_elo=1500.0,
        )

        assert event1 == event2


# =============================================================================
# AsyncTrainingBridge Class Tests
# =============================================================================


class TestAsyncTrainingBridgeInit:
    """Tests for AsyncTrainingBridge initialization."""

    def test_init_with_coordinator(self, mock_coordinator):
        """Test initialization with explicit coordinator."""
        bridge = AsyncTrainingBridge(coordinator=mock_coordinator, emit_events=True)

        assert bridge._coordinator is mock_coordinator
        assert bridge._emit_events is True
        assert bridge._progress_callbacks == []

    def test_init_without_events(self, mock_coordinator):
        """Test initialization with events disabled."""
        bridge = AsyncTrainingBridge(coordinator=mock_coordinator, emit_events=False)

        assert bridge._emit_events is False

    def test_init_uses_global_coordinator(self):
        """Test initialization uses global coordinator when not provided."""
        mock_coord = MagicMock()

        with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coord):
            bridge = AsyncTrainingBridge()
            assert bridge._coordinator is mock_coord


class TestAsyncTrainingBridgeCanStartTraining:
    """Tests for can_start_training method."""

    @pytest.mark.asyncio
    async def test_can_start_training_returns_true(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test can_start_training returns True when slot available."""
        mock_coordinator.can_start_training.return_value = True

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            result = await bridge.can_start_training("square8", 2)

        assert result is True
        mock_coordinator.can_start_training.assert_called_once_with("square8", 2)

    @pytest.mark.asyncio
    async def test_can_start_training_returns_false(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test can_start_training returns False when slot not available."""
        mock_coordinator.can_start_training.return_value = False

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            result = await bridge.can_start_training("hex8", 4)

        assert result is False


class TestAsyncTrainingBridgeRequestSlot:
    """Tests for request_training_slot method."""

    @pytest.mark.asyncio
    async def test_request_slot_success(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test successful slot request."""
        mock_coordinator.start_training.return_value = "square8_2p_1234_5678"

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            job_id = await bridge.request_training_slot("square8", 2, "v2.1", {"lr": 0.001})

        assert job_id == "square8_2p_1234_5678"
        mock_coordinator.start_training.assert_called_once_with("square8", 2, "v2.1", {"lr": 0.001})

    @pytest.mark.asyncio
    async def test_request_slot_failure(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test failed slot request returns None."""
        mock_coordinator.start_training.return_value = None

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            job_id = await bridge.request_training_slot("hex8", 2)

        assert job_id is None


class TestAsyncTrainingBridgeUpdateProgress:
    """Tests for update_progress method."""

    @pytest.mark.asyncio
    async def test_update_progress_success(self, bridge, mock_coordinator, mock_bridge_manager, mock_job):
        """Test successful progress update."""
        mock_coordinator.update_progress.return_value = True
        mock_coordinator.get_job.return_value = mock_job

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            result = await bridge.update_progress(
                "square8_2p_1234_5678",
                epochs_completed=10,
                best_val_loss=0.05,
                current_elo=1500.0,
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_progress_calls_callbacks(self, bridge, mock_coordinator, mock_bridge_manager, mock_job):
        """Test progress update calls registered callbacks."""
        mock_coordinator.update_progress.return_value = True
        mock_coordinator.get_job.return_value = mock_job

        callback_events = []

        def track_callback(event):
            callback_events.append(event)

        bridge.on_progress(track_callback)

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            await bridge.update_progress(
                "square8_2p_1234_5678",
                epochs_completed=10,
                best_val_loss=0.05,
                current_elo=1500.0,
            )

        assert len(callback_events) == 1
        assert callback_events[0].epochs_completed == 10

    @pytest.mark.asyncio
    async def test_update_progress_handles_callback_error(self, bridge, mock_coordinator, mock_bridge_manager, mock_job):
        """Test progress update handles callback errors gracefully."""
        mock_coordinator.update_progress.return_value = True
        mock_coordinator.get_job.return_value = mock_job

        def bad_callback(event):
            raise ValueError("Callback error")

        bridge.on_progress(bad_callback)

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            # Should not raise
            result = await bridge.update_progress("square8_2p_1234_5678", epochs_completed=10)

        assert result is True

    @pytest.mark.asyncio
    async def test_update_progress_failure(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test failed progress update."""
        mock_coordinator.update_progress.return_value = False

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            result = await bridge.update_progress("invalid_job", epochs_completed=10)

        assert result is False


class TestAsyncTrainingBridgeCompleteTraining:
    """Tests for complete_training method."""

    @pytest.mark.asyncio
    async def test_complete_training_success(self, bridge, mock_coordinator, mock_bridge_manager, mock_job):
        """Test successful training completion."""
        mock_coordinator.complete_training.return_value = True
        mock_coordinator.get_job.return_value = mock_job

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.emit_training_complete") as mock_emit:
                mock_emit.return_value = None  # Make it awaitable
                result = await bridge.complete_training(
                    "square8_2p_1234_5678",
                    status="completed",
                    final_val_loss=0.02,
                    final_elo=1600.0,
                    model_path="/models/best.pt",
                )

        assert result is True
        mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_training_no_emit_when_disabled(self, mock_coordinator, mock_bridge_manager, mock_job):
        """Test no event emission when disabled."""
        mock_coordinator.complete_training.return_value = True
        mock_coordinator.get_job.return_value = mock_job

        # Create bridge with emit_events=False
        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            bridge = AsyncTrainingBridge(coordinator=mock_coordinator, emit_events=False)

            with patch("app.coordination.async_training_bridge.emit_training_complete") as mock_emit:
                result = await bridge.complete_training("square8_2p_1234_5678")

        assert result is True
        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_training_failed_status(self, bridge, mock_coordinator, mock_bridge_manager, mock_job):
        """Test training completion with failed status."""
        mock_coordinator.complete_training.return_value = True
        mock_coordinator.get_job.return_value = mock_job

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.emit_training_complete") as mock_emit:
                mock_emit.return_value = None
                result = await bridge.complete_training(
                    "square8_2p_1234_5678",
                    status="failed",
                )

        assert result is True
        call_kwargs = mock_emit.call_args.kwargs
        assert call_kwargs["success"] is False
        assert call_kwargs["status"] == "failed"


class TestAsyncTrainingBridgeGetJob:
    """Tests for get_job_by_id method."""

    @pytest.mark.asyncio
    async def test_get_job_by_id_success(self, bridge, mock_coordinator, mock_bridge_manager, mock_job):
        """Test getting job by ID."""
        mock_coordinator.get_job.return_value = mock_job

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            job = await bridge.get_job_by_id("square8_2p_1234_5678")

        assert job is mock_job
        mock_coordinator.get_job.assert_called_once_with("square8", 2)

    @pytest.mark.asyncio
    async def test_get_job_by_id_invalid_format(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test getting job with invalid ID format returns None."""
        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            job = await bridge.get_job_by_id("invalid")

        assert job is None

    @pytest.mark.asyncio
    async def test_get_job_by_id_hex_board(self, bridge, mock_coordinator, mock_bridge_manager, mock_job):
        """Test getting hex board job by ID."""
        mock_job.board_type = "hex8"
        mock_job.num_players = 4
        mock_coordinator.get_job.return_value = mock_job

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            job = await bridge.get_job_by_id("hex8_4p_1234_5678")

        mock_coordinator.get_job.assert_called_once_with("hex8", 4)


class TestAsyncTrainingBridgeGetActiveJobs:
    """Tests for get_active_jobs method."""

    @pytest.mark.asyncio
    async def test_get_active_jobs_empty(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test getting active jobs when none exist."""
        mock_coordinator.get_active_jobs.return_value = []

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            jobs = await bridge.get_active_jobs()

        assert jobs == []

    @pytest.mark.asyncio
    async def test_get_active_jobs_with_jobs(self, bridge, mock_coordinator, mock_bridge_manager, mock_job):
        """Test getting active jobs when some exist."""
        mock_coordinator.get_active_jobs.return_value = [mock_job, mock_job]

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            jobs = await bridge.get_active_jobs()

        assert len(jobs) == 2


class TestAsyncTrainingBridgeGetStatus:
    """Tests for get_training_status method."""

    @pytest.mark.asyncio
    async def test_get_training_status(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test getting training status."""
        mock_coordinator.get_status.return_value = {
            "active_jobs": 2,
            "pending_jobs": 1,
            "cluster_healthy": True,
        }

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            status = await bridge.get_training_status()

        assert status["active_jobs"] == 2
        assert status["cluster_healthy"] is True


class TestAsyncTrainingBridgeCallbacks:
    """Tests for progress callback registration."""

    def test_on_progress_registers_callback(self, bridge):
        """Test registering a progress callback."""
        callback = MagicMock()
        bridge.on_progress(callback)

        assert callback in bridge._progress_callbacks

    def test_on_progress_multiple_callbacks(self, bridge):
        """Test registering multiple callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        bridge.on_progress(callback1)
        bridge.on_progress(callback2)

        assert len(bridge._progress_callbacks) == 2

    def test_off_progress_removes_callback(self, bridge):
        """Test removing a progress callback."""
        callback = MagicMock()
        bridge.on_progress(callback)
        bridge.off_progress(callback)

        assert callback not in bridge._progress_callbacks

    def test_off_progress_nonexistent_callback(self, bridge):
        """Test removing a callback that was never added."""
        callback = MagicMock()
        # Should not raise
        bridge.off_progress(callback)

        assert callback not in bridge._progress_callbacks


# =============================================================================
# Singleton Management Tests
# =============================================================================


class TestSingletonManagement:
    """Tests for singleton pattern."""

    def test_get_training_bridge_creates_singleton(self, mock_coordinator, mock_bridge_manager):
        """Test that get_training_bridge creates a singleton."""
        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coordinator):
                bridge1 = get_training_bridge()
                bridge2 = get_training_bridge()

        assert bridge1 is bridge2

    def test_reset_clears_singleton(self, mock_coordinator, mock_bridge_manager):
        """Test that reset clears the singleton."""
        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coordinator):
                bridge1 = get_training_bridge()
                reset_training_bridge()
                bridge2 = get_training_bridge()

        assert bridge1 is not bridge2

    def test_get_training_bridge_with_custom_coordinator(self, mock_coordinator, mock_bridge_manager):
        """Test creating bridge with custom coordinator."""
        custom_coord = MagicMock()

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            bridge = get_training_bridge(coordinator=custom_coord)

        assert bridge._coordinator is custom_coord

    def test_get_training_bridge_with_emit_events_false(self, mock_coordinator, mock_bridge_manager):
        """Test creating bridge with events disabled."""
        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coordinator):
                bridge = get_training_bridge(emit_events=False)

        assert bridge._emit_events is False


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_async_can_train(self, mock_coordinator, mock_bridge_manager):
        """Test async_can_train convenience function."""
        mock_coordinator.can_start_training.return_value = True

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coordinator):
                result = await async_can_train("square8", 2)

        assert result is True

    @pytest.mark.asyncio
    async def test_async_request_training(self, mock_coordinator, mock_bridge_manager):
        """Test async_request_training convenience function."""
        mock_coordinator.start_training.return_value = "square8_2p_1234_5678"

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coordinator):
                job_id = await async_request_training("square8", 2, "v2.0")

        assert job_id == "square8_2p_1234_5678"

    @pytest.mark.asyncio
    async def test_async_update_progress(self, mock_coordinator, mock_bridge_manager):
        """Test async_update_progress convenience function."""
        mock_coordinator.update_progress.return_value = True

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coordinator):
                result = await async_update_progress(
                    "square8_2p_1234_5678",
                    epochs_completed=20,
                    best_val_loss=0.03,
                    current_elo=1550.0,
                )

        assert result is True

    @pytest.mark.asyncio
    async def test_async_complete_training(self, mock_coordinator, mock_bridge_manager, mock_job):
        """Test async_complete_training convenience function."""
        mock_coordinator.complete_training.return_value = True
        mock_coordinator.get_job.return_value = mock_job

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coordinator):
                with patch("app.coordination.async_training_bridge.emit_training_complete"):
                    result = await async_complete_training(
                        "square8_2p_1234_5678",
                        status="completed",
                        final_val_loss=0.02,
                        final_elo=1600.0,
                    )

        assert result is True

    @pytest.mark.asyncio
    async def test_async_get_training_status(self, mock_coordinator, mock_bridge_manager):
        """Test async_get_training_status convenience function."""
        mock_coordinator.get_status.return_value = {"active_jobs": 3}

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.get_training_coordinator", return_value=mock_coordinator):
                status = await async_get_training_status()

        assert status["active_jobs"] == 3


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_job_id_parsing_with_extra_underscores(self, bridge, mock_coordinator, mock_bridge_manager, mock_job):
        """Test job ID parsing with extra underscores in board type."""
        mock_coordinator.get_job.return_value = mock_job

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            # square19_2p format should work
            job = await bridge.get_job_by_id("square19_2p_1234_5678")

        mock_coordinator.get_job.assert_called_once_with("square19", 2)

    @pytest.mark.asyncio
    async def test_complete_training_when_job_not_found(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test completing training when job is not found."""
        mock_coordinator.complete_training.return_value = True
        mock_coordinator.get_job.return_value = None

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            with patch("app.coordination.async_training_bridge.emit_training_complete") as mock_emit:
                mock_emit.return_value = None
                result = await bridge.complete_training("unknown_job")

        assert result is True
        # Should still emit with "unknown" board_type
        call_kwargs = mock_emit.call_args.kwargs
        assert call_kwargs["board_type"] == "unknown"
        assert call_kwargs["num_players"] == 0

    @pytest.mark.asyncio
    async def test_update_progress_no_job_found_no_callbacks(self, bridge, mock_coordinator, mock_bridge_manager):
        """Test update_progress with callbacks but job not found."""
        mock_coordinator.update_progress.return_value = True
        mock_coordinator.get_job.return_value = None

        callback = MagicMock()
        bridge.on_progress(callback)

        with patch("app.coordination.async_training_bridge.get_bridge_manager", return_value=mock_bridge_manager):
            result = await bridge.update_progress("square8_2p_1234", epochs_completed=5)

        assert result is True
        callback.assert_not_called()  # No callback because job not found

    def test_multiple_callbacks_same_callback(self, bridge):
        """Test registering same callback multiple times."""
        callback = MagicMock()

        bridge.on_progress(callback)
        bridge.on_progress(callback)

        assert len(bridge._progress_callbacks) == 2
        # Both should be in the list

    def test_off_progress_only_removes_one(self, bridge):
        """Test off_progress only removes one instance."""
        callback = MagicMock()

        bridge.on_progress(callback)
        bridge.on_progress(callback)
        bridge.off_progress(callback)

        assert len(bridge._progress_callbacks) == 1
