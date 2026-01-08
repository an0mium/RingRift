"""Tests for memory_pressure_controller.py.

Tests the 4-tier graduated response system for memory pressure management.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.contracts import CoordinatorStatus
from app.coordination.memory_pressure_controller import (
    MemoryPressureController,
    MemoryPressureState,
    MemoryPressureTier,
    get_memory_pressure_controller,
)
from app.config.coordination_defaults import MemoryPressureDefaults


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    MemoryPressureController._instance = None
    yield
    MemoryPressureController._instance = None


@pytest.fixture
def mock_psutil():
    """Mock psutil for memory readings."""
    with patch("app.coordination.memory_pressure_controller.psutil") as mock:
        # Default: 50% RAM usage (NORMAL tier)
        mock_mem = MagicMock()
        mock_mem.percent = 50.0
        mock_mem.used = 8 * (1024**3)  # 8 GB
        mock_mem.total = 16 * (1024**3)  # 16 GB
        mock.virtual_memory.return_value = mock_mem

        mock_swap = MagicMock()
        mock_swap.percent = 10.0
        mock.swap_memory.return_value = mock_swap

        yield mock


@pytest.fixture
def controller(mock_psutil):
    """Create a controller with mocked psutil."""
    return MemoryPressureController()


# =============================================================================
# MemoryPressureTier Tests
# =============================================================================


class TestMemoryPressureTier:
    """Tests for MemoryPressureTier enum."""

    def test_tier_ordering(self):
        """Test that tiers are ordered by severity."""
        assert MemoryPressureTier.NORMAL < MemoryPressureTier.CAUTION
        assert MemoryPressureTier.CAUTION < MemoryPressureTier.WARNING
        assert MemoryPressureTier.WARNING < MemoryPressureTier.CRITICAL
        assert MemoryPressureTier.CRITICAL < MemoryPressureTier.EMERGENCY

    def test_tier_values(self):
        """Test tier numeric values."""
        assert MemoryPressureTier.NORMAL.value == 0
        assert MemoryPressureTier.CAUTION.value == 1
        assert MemoryPressureTier.WARNING.value == 2
        assert MemoryPressureTier.CRITICAL.value == 3
        assert MemoryPressureTier.EMERGENCY.value == 4


# =============================================================================
# MemoryPressureState Tests
# =============================================================================


class TestMemoryPressureState:
    """Tests for MemoryPressureState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = MemoryPressureState()
        assert state.tier == MemoryPressureTier.NORMAL
        assert state.ram_percent == 0.0
        assert state.ram_used_gb == 0.0
        assert state.ram_total_gb == 0.0
        assert state.swap_percent == 0.0
        assert state.consecutive_samples == 0
        assert state.last_action_time == 0.0
        assert state.actions_taken == []

    def test_age_seconds_property(self):
        """Test age_seconds property calculates correctly."""
        old_time = time.time() - 10.0
        state = MemoryPressureState(timestamp=old_time)
        assert state.age_seconds >= 10.0
        assert state.age_seconds < 11.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        state = MemoryPressureState(
            tier=MemoryPressureTier.WARNING,
            ram_percent=75.0,
            ram_used_gb=12.0,
            ram_total_gb=16.0,
        )
        result = state.to_dict()

        assert result["tier"] == "WARNING"
        assert result["tier_value"] == 2
        assert result["ram_percent"] == 75.0
        assert result["ram_used_gb"] == 12.0
        assert result["ram_total_gb"] == 16.0

    def test_to_dict_truncates_actions(self):
        """Test that to_dict only includes last 10 actions."""
        state = MemoryPressureState()
        state.actions_taken = [f"action_{i}" for i in range(20)]
        result = state.to_dict()
        assert len(result["actions_taken"]) == 10
        assert result["actions_taken"][0] == "action_10"  # Last 10


# =============================================================================
# MemoryPressureController Basic Tests
# =============================================================================


class TestMemoryPressureControllerBasic:
    """Tests for MemoryPressureController basic functionality."""

    def test_initialization_default_config(self, mock_psutil):
        """Test initialization with default config."""
        controller = MemoryPressureController()
        assert controller._pressure_config is not None
        assert controller.current_tier == MemoryPressureTier.NORMAL
        assert controller._selfplay_paused is False

    def test_initialization_custom_config(self, mock_psutil):
        """Test initialization with custom config."""
        # MemoryPressureDefaults is frozen, so we create a mock config
        from dataclasses import dataclass

        @dataclass
        class MockConfig:
            TIER_CAUTION: int = 60
            TIER_WARNING: int = 70
            TIER_CRITICAL: int = 85  # Custom threshold
            TIER_EMERGENCY: int = 90
            CHECK_INTERVAL: int = 10
            HYSTERESIS: int = 3
            ACTION_COOLDOWN: int = 60
            GC_WAIT_TIME: int = 30
            BATCH_SIZE_REDUCTION: float = 0.5
            CONSECUTIVE_SAMPLES_REQUIRED: int = 3

        config = MockConfig()
        controller = MemoryPressureController(config=config)
        assert controller._tier_thresholds[MemoryPressureTier.CRITICAL] == 85

    def test_current_tier_property(self, controller):
        """Test current_tier property returns state tier."""
        assert controller.current_tier == MemoryPressureTier.NORMAL
        controller._pressure_state.tier = MemoryPressureTier.WARNING
        assert controller.current_tier == MemoryPressureTier.WARNING

    def test_current_state_property(self, controller):
        """Test current_state property returns state object."""
        state = controller.current_state
        assert isinstance(state, MemoryPressureState)


# =============================================================================
# Pressure Health Score Tests
# =============================================================================


class TestPressureHealthScore:
    """Tests for pressure_health_score property."""

    def test_normal_tier_score(self, controller):
        """Test NORMAL tier has score 1.0."""
        controller._pressure_state.tier = MemoryPressureTier.NORMAL
        assert controller.pressure_health_score == 1.0

    def test_caution_tier_score(self, controller):
        """Test CAUTION tier has score 0.8."""
        controller._pressure_state.tier = MemoryPressureTier.CAUTION
        assert controller.pressure_health_score == 0.8

    def test_warning_tier_score(self, controller):
        """Test WARNING tier has score 0.5."""
        controller._pressure_state.tier = MemoryPressureTier.WARNING
        assert controller.pressure_health_score == 0.5

    def test_critical_tier_score(self, controller):
        """Test CRITICAL tier has score 0.2."""
        controller._pressure_state.tier = MemoryPressureTier.CRITICAL
        assert controller.pressure_health_score == 0.2

    def test_emergency_tier_score(self, controller):
        """Test EMERGENCY tier has score 0.0."""
        controller._pressure_state.tier = MemoryPressureTier.EMERGENCY
        assert controller.pressure_health_score == 0.0


# =============================================================================
# Memory Usage Tests
# =============================================================================


class TestMemoryUsage:
    """Tests for _get_memory_usage method."""

    def test_get_memory_usage_returns_values(self, controller, mock_psutil):
        """Test that _get_memory_usage returns correct values."""
        mock_psutil.virtual_memory.return_value.percent = 65.0
        mock_psutil.virtual_memory.return_value.used = 10 * (1024**3)
        mock_psutil.virtual_memory.return_value.total = 16 * (1024**3)
        mock_psutil.swap_memory.return_value.percent = 15.0

        ram_pct, ram_used, ram_total, swap_pct = controller._get_memory_usage()

        assert ram_pct == 65.0
        assert ram_used == pytest.approx(10.0, rel=0.01)
        assert ram_total == pytest.approx(16.0, rel=0.01)
        assert swap_pct == 15.0

    def test_get_memory_usage_handles_psutil_none(self, mock_psutil):
        """Test that _get_memory_usage returns zeros when psutil is None."""
        with patch(
            "app.coordination.memory_pressure_controller.psutil", None
        ):
            controller = MemoryPressureController()
            ram_pct, ram_used, ram_total, swap_pct = controller._get_memory_usage()
            assert ram_pct == 0.0
            assert ram_used == 0.0
            assert ram_total == 0.0
            assert swap_pct == 0.0

    def test_get_memory_usage_handles_exception(self, controller, mock_psutil):
        """Test that _get_memory_usage handles exceptions gracefully."""
        mock_psutil.virtual_memory.side_effect = Exception("psutil error")

        ram_pct, ram_used, ram_total, swap_pct = controller._get_memory_usage()

        assert ram_pct == 0.0
        assert ram_used == 0.0


# =============================================================================
# Tier Determination Tests
# =============================================================================


class TestTierDetermination:
    """Tests for _determine_tier method."""

    def test_determine_tier_normal(self, controller):
        """Test NORMAL tier determination."""
        assert controller._determine_tier(50.0) == MemoryPressureTier.NORMAL
        assert controller._determine_tier(59.9) == MemoryPressureTier.NORMAL

    def test_determine_tier_caution(self, controller):
        """Test CAUTION tier determination."""
        assert controller._determine_tier(60.0) == MemoryPressureTier.CAUTION
        assert controller._determine_tier(69.9) == MemoryPressureTier.CAUTION

    def test_determine_tier_warning(self, controller):
        """Test WARNING tier determination."""
        assert controller._determine_tier(70.0) == MemoryPressureTier.WARNING
        assert controller._determine_tier(79.9) == MemoryPressureTier.WARNING

    def test_determine_tier_critical(self, controller):
        """Test CRITICAL tier determination."""
        assert controller._determine_tier(80.0) == MemoryPressureTier.CRITICAL
        assert controller._determine_tier(89.9) == MemoryPressureTier.CRITICAL

    def test_determine_tier_emergency(self, controller):
        """Test EMERGENCY tier determination."""
        assert controller._determine_tier(90.0) == MemoryPressureTier.EMERGENCY
        assert controller._determine_tier(100.0) == MemoryPressureTier.EMERGENCY


# =============================================================================
# Hysteresis Tests
# =============================================================================


class TestHysteresis:
    """Tests for tier downgrade hysteresis."""

    def test_should_downgrade_true_for_escalation(self, controller):
        """Test that escalation is always allowed."""
        controller._pressure_state.tier = MemoryPressureTier.CAUTION
        # Escalating to WARNING should be allowed
        assert controller._should_downgrade_tier(MemoryPressureTier.WARNING) is True

    def test_should_downgrade_with_sufficient_drop(self, controller):
        """Test downgrade with sufficient drop below threshold."""
        controller._pressure_state.tier = MemoryPressureTier.WARNING
        # Hysteresis is typically 5%, so need to drop below 65% (70% - 5%)
        controller._pressure_state.ram_percent = 64.0
        assert controller._should_downgrade_tier(MemoryPressureTier.CAUTION) is True

    def test_should_downgrade_rejected_without_sufficient_drop(self, controller):
        """Test downgrade rejected without sufficient drop."""
        controller._pressure_state.tier = MemoryPressureTier.WARNING
        # RAM at 68% is within hysteresis range of 70% threshold
        controller._pressure_state.ram_percent = 68.0
        assert controller._should_downgrade_tier(MemoryPressureTier.CAUTION) is False


# =============================================================================
# Action Cooldown Tests
# =============================================================================


class TestActionCooldown:
    """Tests for action cooldown logic."""

    def test_can_take_action_initially_true(self, controller):
        """Test that actions are allowed initially."""
        assert controller._can_take_action() is True

    def test_can_take_action_respects_cooldown(self, controller):
        """Test that actions are blocked during cooldown."""
        controller._pressure_state.last_action_time = time.time()
        assert controller._can_take_action() is False

    def test_can_take_action_after_cooldown_expires(self, controller):
        """Test that actions are allowed after cooldown expires."""
        cooldown = controller._pressure_config.ACTION_COOLDOWN
        controller._pressure_state.last_action_time = time.time() - cooldown - 1
        assert controller._can_take_action() is True

    def test_record_action_updates_state(self, controller):
        """Test that record_action updates state."""
        controller._record_action("test_action")
        assert controller._pressure_state.last_action_time > 0
        assert len(controller._pressure_state.actions_taken) == 1
        assert "test_action" in controller._pressure_state.actions_taken[0]


# =============================================================================
# Tier Handler Tests
# =============================================================================


class TestTierHandlers:
    """Tests for tier-specific handlers."""

    @pytest.mark.asyncio
    async def test_handle_caution_tier_emits_event(self, controller, mock_psutil):
        """Test CAUTION tier handler emits event."""
        controller._pressure_state.ram_percent = 65.0
        controller._pressure_state.ram_used_gb = 10.0
        controller._pressure_state.ram_total_gb = 16.0

        with patch(
            "app.coordination.memory_pressure_controller.MemoryPressureController._emit_event"
        ) as mock_emit:
            await controller._handle_caution_tier()
            mock_emit.assert_called_once()
            args = mock_emit.call_args
            assert args[0][0] == "MEMORY_PRESSURE_CAUTION"

    @pytest.mark.asyncio
    async def test_handle_warning_tier_pauses_selfplay(self, controller, mock_psutil):
        """Test WARNING tier handler pauses selfplay."""
        controller._pressure_state.ram_percent = 75.0

        with patch.object(
            controller, "_pause_selfplay", new_callable=AsyncMock
        ) as mock_pause:
            with patch.object(controller, "_emit_event"):
                await controller._handle_warning_tier()
                mock_pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_warning_tier_respects_cooldown(self, controller, mock_psutil):
        """Test WARNING tier handler respects cooldown."""
        controller._pressure_state.last_action_time = time.time()

        with patch.object(
            controller, "_pause_selfplay", new_callable=AsyncMock
        ) as mock_pause:
            await controller._handle_warning_tier()
            mock_pause.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_critical_tier_kills_daemons(self, controller, mock_psutil):
        """Test CRITICAL tier handler kills non-essential daemons."""
        controller._pressure_state.ram_percent = 85.0

        with patch.object(
            controller, "_pause_selfplay", new_callable=AsyncMock
        ):
            with patch.object(
                controller, "_kill_non_essential_daemons", new_callable=AsyncMock
            ) as mock_kill:
                with patch.object(controller, "_emit_event"):
                    await controller._handle_critical_tier()
                    mock_kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_emergency_tier_calls_callbacks(self, controller, mock_psutil):
        """Test EMERGENCY tier handler calls registered callbacks."""
        callback = MagicMock()
        controller.register_emergency_callback(callback)
        controller._pressure_state.ram_percent = 95.0

        with patch.object(controller, "_emit_event"):
            await controller._handle_emergency_tier()
            callback.assert_called_once()


# =============================================================================
# Selfplay Pause/Resume Tests
# =============================================================================


class TestSelfplayPauseResume:
    """Tests for selfplay pause/resume functionality."""

    @pytest.mark.asyncio
    async def test_pause_selfplay_sets_flag(self, controller, mock_psutil):
        """Test that _pause_selfplay sets paused flag."""
        mock_scheduler = MagicMock()
        mock_scheduler.pause_spawning = MagicMock()

        # Patch at the source module where it's imported from
        with patch(
            "app.coordination.selfplay_scheduler.get_selfplay_scheduler",
            return_value=mock_scheduler,
        ):
            await controller._pause_selfplay()
            assert controller._selfplay_paused is True
            mock_scheduler.pause_spawning.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_selfplay_idempotent(self, controller, mock_psutil):
        """Test that _pause_selfplay is idempotent."""
        controller._selfplay_paused = True

        # Should not call the scheduler since already paused
        await controller._pause_selfplay()
        # Just verify the flag is still True
        assert controller._selfplay_paused is True

    @pytest.mark.asyncio
    async def test_resume_selfplay_clears_flag(self, controller, mock_psutil):
        """Test that _resume_selfplay clears paused flag."""
        controller._selfplay_paused = True
        mock_scheduler = MagicMock()
        mock_scheduler.resume_spawning = MagicMock()

        with patch(
            "app.coordination.selfplay_scheduler.get_selfplay_scheduler",
            return_value=mock_scheduler,
        ):
            await controller._resume_selfplay()
            assert controller._selfplay_paused is False
            mock_scheduler.resume_spawning.assert_called_once()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_not_running(self, controller):
        """Test health check when not running."""
        controller._running = False
        result = controller.health_check()
        assert result.healthy is False
        assert result.status == CoordinatorStatus.STOPPED

    def test_health_check_normal_tier(self, controller):
        """Test health check at NORMAL tier."""
        controller._running = True
        controller._pressure_state.tier = MemoryPressureTier.NORMAL
        controller._pressure_state.ram_percent = 50.0

        result = controller.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    def test_health_check_warning_tier(self, controller):
        """Test health check at WARNING tier."""
        controller._running = True
        controller._pressure_state.tier = MemoryPressureTier.WARNING
        controller._pressure_state.ram_percent = 75.0

        result = controller.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.DEGRADED
        assert "WARNING" in result.message

    def test_health_check_critical_tier(self, controller):
        """Test health check at CRITICAL tier."""
        controller._running = True
        controller._pressure_state.tier = MemoryPressureTier.CRITICAL
        controller._pressure_state.ram_percent = 85.0

        result = controller.health_check()
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert "CRITICAL" in result.message

    def test_health_check_includes_details(self, controller):
        """Test health check includes state details."""
        controller._running = True
        controller._pressure_state.tier = MemoryPressureTier.WARNING
        controller._pressure_state.ram_percent = 75.0

        result = controller.health_check()
        assert result.details is not None
        assert "ram_percent" in result.details


# =============================================================================
# Callback Registration Tests
# =============================================================================


class TestCallbackRegistration:
    """Tests for callback registration."""

    def test_register_tier_change_callback(self, controller):
        """Test registering tier change callback."""
        callback = MagicMock()
        controller.register_tier_change_callback(callback)
        assert callback in controller._on_tier_change

    def test_register_emergency_callback(self, controller):
        """Test registering emergency callback."""
        callback = MagicMock()
        controller.register_emergency_callback(callback)
        assert callback in controller._on_emergency


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestRunCycle:
    """Tests for _run_cycle method."""

    @pytest.mark.asyncio
    async def test_run_cycle_updates_state(self, controller, mock_psutil):
        """Test that _run_cycle updates state from memory readings."""
        mock_psutil.virtual_memory.return_value.percent = 55.0
        mock_psutil.virtual_memory.return_value.used = 9 * (1024**3)
        mock_psutil.virtual_memory.return_value.total = 16 * (1024**3)
        mock_psutil.swap_memory.return_value.percent = 5.0

        await controller._run_cycle()

        assert controller._pressure_state.ram_percent == 55.0
        assert controller._pressure_state.ram_used_gb == pytest.approx(9.0, rel=0.01)
        assert controller._pressure_state.ram_total_gb == pytest.approx(16.0, rel=0.01)
        assert controller._pressure_state.swap_percent == 5.0

    @pytest.mark.asyncio
    async def test_run_cycle_escalation_requires_consecutive_samples(
        self, mock_psutil
    ):
        """Test that tier escalation requires consecutive samples."""
        from dataclasses import dataclass

        @dataclass
        class MockConfig:
            TIER_CAUTION: int = 60
            TIER_WARNING: int = 70
            TIER_CRITICAL: int = 80
            TIER_EMERGENCY: int = 90
            CHECK_INTERVAL: int = 10
            HYSTERESIS: int = 3
            ACTION_COOLDOWN: int = 60
            GC_WAIT_TIME: int = 30
            BATCH_SIZE_REDUCTION: float = 0.5
            CONSECUTIVE_SAMPLES_REQUIRED: int = 2  # Custom: need 2 samples

        config = MockConfig()
        controller = MemoryPressureController(config=config)

        # First sample at WARNING level
        mock_psutil.virtual_memory.return_value.percent = 75.0
        await controller._run_cycle()
        assert controller._pressure_state.tier == MemoryPressureTier.NORMAL  # Not yet

        # Second sample at WARNING level
        await controller._run_cycle()
        assert controller._pressure_state.tier == MemoryPressureTier.WARNING  # Now changed

    @pytest.mark.asyncio
    async def test_run_cycle_calls_tier_handlers_on_sustained_pressure(
        self, controller, mock_psutil
    ):
        """Test that run_cycle calls handlers when tier is sustained."""
        controller._pressure_state.tier = MemoryPressureTier.WARNING
        mock_psutil.virtual_memory.return_value.percent = 75.0

        with patch.object(
            controller, "_handle_warning_tier", new_callable=AsyncMock
        ) as mock_handler:
            await controller._run_cycle()
            mock_handler.assert_called_once()


# =============================================================================
# Singleton Function Tests
# =============================================================================


class TestSingletonFunction:
    """Tests for get_memory_pressure_controller function."""

    def test_get_memory_pressure_controller_returns_instance(self, mock_psutil):
        """Test that function returns a controller instance."""
        controller = get_memory_pressure_controller()
        assert isinstance(controller, MemoryPressureController)

    def test_get_memory_pressure_controller_returns_same_instance(self, mock_psutil):
        """Test that function returns the same instance."""
        controller1 = get_memory_pressure_controller()
        controller2 = get_memory_pressure_controller()
        assert controller1 is controller2


# =============================================================================
# Tier Change Handler Tests
# =============================================================================


class TestTierChangeHandler:
    """Tests for _handle_tier_change method."""

    @pytest.mark.asyncio
    async def test_handle_tier_change_notifies_callbacks(self, controller, mock_psutil):
        """Test that tier change notifies callbacks."""
        callback = MagicMock()
        controller.register_tier_change_callback(callback)

        with patch.object(controller, "_handle_caution_tier", new_callable=AsyncMock):
            await controller._handle_tier_change(
                MemoryPressureTier.NORMAL, MemoryPressureTier.CAUTION
            )
            callback.assert_called_once_with(MemoryPressureTier.CAUTION)

    @pytest.mark.asyncio
    async def test_handle_tier_change_recovery_resumes_selfplay(
        self, controller, mock_psutil
    ):
        """Test that recovering to NORMAL resumes selfplay."""
        controller._selfplay_paused = True

        with patch.object(
            controller, "_resume_selfplay", new_callable=AsyncMock
        ) as mock_resume:
            with patch.object(controller, "_emit_event"):
                await controller._handle_tier_change(
                    MemoryPressureTier.WARNING, MemoryPressureTier.NORMAL
                )
                mock_resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_tier_change_recovery_emits_event(
        self, controller, mock_psutil
    ):
        """Test that recovering emits MEMORY_PRESSURE_RECOVERED event."""
        with patch.object(
            controller, "_resume_selfplay", new_callable=AsyncMock
        ):
            with patch.object(controller, "_emit_event") as mock_emit:
                await controller._handle_tier_change(
                    MemoryPressureTier.WARNING, MemoryPressureTier.NORMAL
                )
                mock_emit.assert_called_once()
                assert mock_emit.call_args[0][0] == "MEMORY_PRESSURE_RECOVERED"
