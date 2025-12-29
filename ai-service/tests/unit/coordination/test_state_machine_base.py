"""Unit tests for state_machine_base module.

Tests the state machine infrastructure used across coordination components.
"""

from __future__ import annotations

from enum import Enum
from unittest.mock import MagicMock

import pytest

from app.coordination.state_machine_base import (
    AllocationState,
    AllocationStateMachine,
    CompositeStateMachine,
    DaemonLifecycleMachine,
    DaemonLifecycleState,
    FeedbackSignalMachine,
    FeedbackSignalState,
    PriorityState,
    PriorityStateMachine,
    StateHistory,
    StateHistoryEntry,
    StateMachineBase,
    StateTransition,
    state_machine,
)


# =============================================================================
# Test Enums for Tests
# =============================================================================


class SampleState(Enum):
    """Sample state enum for tests."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


# =============================================================================
# StateTransition Tests
# =============================================================================


class TestStateTransition:
    """Tests for StateTransition dataclass."""

    def test_create_simple_transition(self):
        """Should create a transition with just target."""
        trans = StateTransition(target=SampleState.RUNNING)
        assert trans.target == SampleState.RUNNING
        assert trans.guard is None
        assert trans.on_transition is None
        assert trans.name is None

    def test_is_allowed_no_guard(self):
        """Should be allowed when no guard is set."""
        trans = StateTransition(target=SampleState.RUNNING)
        assert trans.is_allowed() is True

    def test_is_allowed_guard_true(self):
        """Should be allowed when guard returns True."""
        trans = StateTransition(
            target=SampleState.RUNNING,
            guard=lambda: True,
        )
        assert trans.is_allowed() is True

    def test_is_allowed_guard_false(self):
        """Should not be allowed when guard returns False."""
        trans = StateTransition(
            target=SampleState.RUNNING,
            guard=lambda: False,
        )
        assert trans.is_allowed() is False

    def test_is_allowed_guard_exception(self):
        """Should not be allowed when guard raises exception."""
        def bad_guard():
            raise ValueError("Guard error")

        trans = StateTransition(target=SampleState.RUNNING, guard=bad_guard)
        assert trans.is_allowed() is False

    def test_with_name(self):
        """Should accept optional name."""
        trans = StateTransition(
            target=SampleState.RUNNING,
            name="start_job",
        )
        assert trans.name == "start_job"

    def test_with_on_transition(self):
        """Should accept on_transition callback."""
        callback = MagicMock()
        trans = StateTransition(
            target=SampleState.RUNNING,
            on_transition=callback,
        )
        assert trans.on_transition is callback


# =============================================================================
# StateHistoryEntry Tests
# =============================================================================


class TestStateHistoryEntry:
    """Tests for StateHistoryEntry dataclass."""

    def test_create_entry(self):
        """Should create a history entry."""
        entry = StateHistoryEntry(
            from_state=SampleState.IDLE,
            to_state=SampleState.RUNNING,
            timestamp=1234567890.0,
        )
        assert entry.from_state == SampleState.IDLE
        assert entry.to_state == SampleState.RUNNING
        assert entry.timestamp == 1234567890.0

    def test_default_reason(self):
        """Should have None reason by default."""
        entry = StateHistoryEntry(
            from_state=SampleState.IDLE,
            to_state=SampleState.RUNNING,
            timestamp=0.0,
        )
        assert entry.reason is None

    def test_with_reason(self):
        """Should accept reason."""
        entry = StateHistoryEntry(
            from_state=SampleState.IDLE,
            to_state=SampleState.RUNNING,
            timestamp=0.0,
            reason="user request",
        )
        assert entry.reason == "user request"

    def test_default_metadata(self):
        """Should have empty metadata by default."""
        entry = StateHistoryEntry(
            from_state=SampleState.IDLE,
            to_state=SampleState.RUNNING,
            timestamp=0.0,
        )
        assert entry.metadata == {}


# =============================================================================
# StateHistory Tests
# =============================================================================


class TestStateHistory:
    """Tests for StateHistory class."""

    def test_create_with_default_size(self):
        """Should create with default max entries."""
        history = StateHistory()
        assert history.max_entries == 100

    def test_create_with_custom_size(self):
        """Should create with custom max entries."""
        history = StateHistory(max_entries=50)
        assert history.max_entries == 50

    def test_record_transition(self):
        """Should record a transition."""
        history = StateHistory()
        history.record(SampleState.IDLE, SampleState.RUNNING)
        assert len(history.entries) == 1

    def test_get_recent(self):
        """Should return recent entries."""
        history = StateHistory()
        history.record(SampleState.IDLE, SampleState.RUNNING)
        history.record(SampleState.RUNNING, SampleState.COMPLETED)

        recent = history.get_recent(1)
        assert len(recent) == 1
        assert recent[0].to_state == SampleState.COMPLETED

    def test_bounded_size(self):
        """Should limit to max_entries entries."""
        history = StateHistory(max_entries=3)
        history.record(SampleState.IDLE, SampleState.RUNNING)
        history.record(SampleState.RUNNING, SampleState.PAUSED)
        history.record(SampleState.PAUSED, SampleState.RUNNING)
        history.record(SampleState.RUNNING, SampleState.COMPLETED)

        assert len(history.entries) == 3

    def test_clear(self):
        """Should clear all entries."""
        history = StateHistory()
        history.record(SampleState.IDLE, SampleState.RUNNING)
        history.clear()
        assert len(history.entries) == 0

    def test_get_transition_count(self):
        """Should track transition counts."""
        history = StateHistory()
        history.record(SampleState.IDLE, SampleState.RUNNING)
        history.record(SampleState.RUNNING, SampleState.IDLE)
        history.record(SampleState.IDLE, SampleState.RUNNING)

        assert history.get_transition_count(SampleState.IDLE, SampleState.RUNNING) == 2
        assert history.get_transition_count(SampleState.RUNNING, SampleState.IDLE) == 1

    def test_to_dict(self):
        """Should export to dictionary."""
        history = StateHistory()
        history.record(SampleState.IDLE, SampleState.RUNNING, reason="start")

        d = history.to_dict()
        assert "entries" in d
        assert "transition_counts" in d
        assert len(d["entries"]) == 1
        assert d["entries"][0]["reason"] == "start"


# =============================================================================
# PriorityState Tests
# =============================================================================


class TestPriorityState:
    """Tests for PriorityState enum."""

    def test_has_expected_states(self):
        """Should have expected priority states."""
        assert PriorityState.CRITICAL.value == "critical"
        assert PriorityState.HIGH.value == "high"
        assert PriorityState.NORMAL.value == "normal"
        assert PriorityState.LOW.value == "low"
        assert PriorityState.PAUSED.value == "paused"


# =============================================================================
# AllocationState Tests
# =============================================================================


class TestAllocationState:
    """Tests for AllocationState enum."""

    def test_has_expected_states(self):
        """Should have expected allocation states."""
        assert AllocationState.IDLE.value == "idle"
        assert AllocationState.SCHEDULED.value == "scheduled"
        assert AllocationState.RUNNING.value == "running"
        assert AllocationState.COMPLETING.value == "completing"
        assert AllocationState.COOLDOWN.value == "cooldown"


# =============================================================================
# PriorityStateMachine Tests
# =============================================================================


class TestPriorityStateMachine:
    """Tests for PriorityStateMachine."""

    def test_initial_state(self):
        """Should start in NORMAL state."""
        machine = PriorityStateMachine()
        assert machine.state == PriorityState.NORMAL

    def test_transition_to_high(self):
        """Should transition to HIGH."""
        machine = PriorityStateMachine()
        success = machine.transition_to(PriorityState.HIGH)
        assert success is True
        assert machine.state == PriorityState.HIGH

    def test_transition_to_critical(self):
        """Should transition to CRITICAL."""
        machine = PriorityStateMachine()
        success = machine.transition_to(PriorityState.CRITICAL)
        assert success is True
        assert machine.state == PriorityState.CRITICAL

    def test_paused_can_return_to_normal(self):
        """Should allow transition from PAUSED to NORMAL."""
        machine = PriorityStateMachine()
        machine.transition_to(PriorityState.PAUSED)
        success = machine.transition_to(PriorityState.NORMAL)
        assert success is True
        assert machine.state == PriorityState.NORMAL


# =============================================================================
# AllocationStateMachine Tests
# =============================================================================


class TestAllocationStateMachine:
    """Tests for AllocationStateMachine."""

    def test_initial_state(self):
        """Should start in IDLE state."""
        machine = AllocationStateMachine()
        assert machine.state == AllocationState.IDLE

    def test_transition_to_scheduled(self):
        """Should transition to SCHEDULED."""
        machine = AllocationStateMachine()
        success = machine.transition_to(AllocationState.SCHEDULED)
        assert success is True
        assert machine.state == AllocationState.SCHEDULED

    def test_full_lifecycle(self):
        """Should follow full allocation lifecycle."""
        machine = AllocationStateMachine()
        assert machine.transition_to(AllocationState.SCHEDULED) is True
        assert machine.transition_to(AllocationState.RUNNING) is True
        assert machine.transition_to(AllocationState.COMPLETING) is True
        assert machine.transition_to(AllocationState.COOLDOWN) is True
        assert machine.transition_to(AllocationState.IDLE) is True

    def test_cannot_skip_states(self):
        """Should not allow skipping states."""
        machine = AllocationStateMachine()
        # Cannot go directly from IDLE to RUNNING
        success = machine.transition_to(AllocationState.RUNNING)
        assert success is False
        assert machine.state == AllocationState.IDLE


# =============================================================================
# DaemonLifecycleState Tests
# =============================================================================


class TestDaemonLifecycleState:
    """Tests for DaemonLifecycleState enum."""

    def test_has_expected_states(self):
        """Should have expected lifecycle states."""
        assert DaemonLifecycleState.UNINITIALIZED.value == "uninitialized"
        assert DaemonLifecycleState.INITIALIZED.value == "initialized"
        assert DaemonLifecycleState.STARTING.value == "starting"
        assert DaemonLifecycleState.RUNNING.value == "running"
        assert DaemonLifecycleState.STOPPING.value == "stopping"
        assert DaemonLifecycleState.STOPPED.value == "stopped"
        assert DaemonLifecycleState.FAILED.value == "failed"
        assert DaemonLifecycleState.RESTARTING.value == "restarting"


# =============================================================================
# DaemonLifecycleMachine Tests
# =============================================================================


class TestDaemonLifecycleMachine:
    """Tests for DaemonLifecycleMachine."""

    def test_initial_state(self):
        """Should start in UNINITIALIZED state."""
        machine = DaemonLifecycleMachine()
        assert machine.state == DaemonLifecycleState.UNINITIALIZED

    def test_start_sequence(self):
        """Should transition through start sequence."""
        machine = DaemonLifecycleMachine()
        assert machine.transition_to(DaemonLifecycleState.INITIALIZED) is True
        assert machine.transition_to(DaemonLifecycleState.STARTING) is True
        assert machine.transition_to(DaemonLifecycleState.RUNNING) is True
        assert machine.state == DaemonLifecycleState.RUNNING

    def test_stop_sequence(self):
        """Should transition through stop sequence."""
        machine = DaemonLifecycleMachine()
        machine.transition_to(DaemonLifecycleState.INITIALIZED)
        machine.transition_to(DaemonLifecycleState.STARTING)
        machine.transition_to(DaemonLifecycleState.RUNNING)
        assert machine.transition_to(DaemonLifecycleState.STOPPING) is True
        assert machine.transition_to(DaemonLifecycleState.STOPPED) is True

    def test_fail_from_running(self):
        """Should allow failure from running state."""
        machine = DaemonLifecycleMachine()
        machine.transition_to(DaemonLifecycleState.INITIALIZED)
        machine.transition_to(DaemonLifecycleState.STARTING)
        machine.transition_to(DaemonLifecycleState.RUNNING)
        assert machine.transition_to(DaemonLifecycleState.FAILED) is True

    def test_restart_from_failed(self):
        """Should allow restart from failed state."""
        machine = DaemonLifecycleMachine()
        machine.transition_to(DaemonLifecycleState.INITIALIZED)
        machine.transition_to(DaemonLifecycleState.STARTING)
        machine.transition_to(DaemonLifecycleState.FAILED)
        assert machine.transition_to(DaemonLifecycleState.RESTARTING) is True


# =============================================================================
# FeedbackSignalState Tests
# =============================================================================


class TestFeedbackSignalState:
    """Tests for FeedbackSignalState enum."""

    def test_has_expected_states(self):
        """Should have expected signal states."""
        assert FeedbackSignalState.COLLECTING.value == "collecting"
        assert FeedbackSignalState.ANALYZING.value == "analyzing"
        assert FeedbackSignalState.READY.value == "ready"
        assert FeedbackSignalState.APPLYING.value == "applying"
        assert FeedbackSignalState.APPLIED.value == "applied"
        assert FeedbackSignalState.COOLDOWN.value == "cooldown"


# =============================================================================
# FeedbackSignalMachine Tests
# =============================================================================


class TestFeedbackSignalMachine:
    """Tests for FeedbackSignalMachine."""

    def test_initial_state(self):
        """Should start in COLLECTING state."""
        machine = FeedbackSignalMachine()
        assert machine.state == FeedbackSignalState.COLLECTING

    def test_full_feedback_cycle(self):
        """Should complete full feedback cycle."""
        machine = FeedbackSignalMachine()
        assert machine.transition_to(FeedbackSignalState.ANALYZING) is True
        assert machine.transition_to(FeedbackSignalState.READY) is True
        assert machine.transition_to(FeedbackSignalState.APPLYING) is True
        assert machine.transition_to(FeedbackSignalState.APPLIED) is True
        assert machine.transition_to(FeedbackSignalState.COOLDOWN) is True
        assert machine.transition_to(FeedbackSignalState.COLLECTING) is True

    def test_can_skip_to_collecting_from_ready(self):
        """Should allow skip from READY to COLLECTING."""
        machine = FeedbackSignalMachine()
        machine.transition_to(FeedbackSignalState.ANALYZING)
        machine.transition_to(FeedbackSignalState.READY)
        # Can skip if feedback is skipped
        assert machine.transition_to(FeedbackSignalState.COLLECTING) is True


# =============================================================================
# StateMachineBase Tests
# =============================================================================


class TestStateMachineBase:
    """Tests for StateMachineBase class."""

    def test_state_property(self):
        """Should expose current state via state property."""
        machine = PriorityStateMachine()
        assert machine.state == PriorityState.NORMAL

    def test_state_name_property(self):
        """Should expose state name as string."""
        machine = PriorityStateMachine()
        assert machine.state_name == "NORMAL"

    def test_time_in_current_state(self):
        """Should track time in current state."""
        machine = PriorityStateMachine()
        import time
        time.sleep(0.01)
        assert machine.time_in_current_state > 0

    def test_can_transition_to(self):
        """Should check if transition is allowed."""
        machine = PriorityStateMachine()
        assert machine.can_transition_to(PriorityState.HIGH) is True
        # Same state is not in allowed list, but transition_to handles it specially
        assert machine.can_transition_to(PriorityState.NORMAL) is False

    def test_get_allowed_transitions(self):
        """Should list allowed transitions."""
        machine = PriorityStateMachine()
        allowed = machine.get_allowed_transitions()
        assert PriorityState.CRITICAL in allowed
        assert PriorityState.HIGH in allowed
        assert PriorityState.LOW in allowed
        assert PriorityState.PAUSED in allowed

    def test_lock_prevents_transitions(self):
        """Should prevent transitions when locked."""
        machine = PriorityStateMachine()
        machine.lock()
        success = machine.transition_to(PriorityState.HIGH)
        assert success is False
        assert machine.state == PriorityState.NORMAL

    def test_unlock_allows_transitions(self):
        """Should allow transitions after unlock."""
        machine = PriorityStateMachine()
        machine.lock()
        machine.unlock()
        success = machine.transition_to(PriorityState.HIGH)
        assert success is True

    def test_reset(self):
        """Should reset to initial state."""
        machine = PriorityStateMachine()
        machine.transition_to(PriorityState.HIGH)
        machine.reset()
        assert machine.state == PriorityState.NORMAL

    def test_reset_to_specific_state(self):
        """Should reset to specific state."""
        machine = PriorityStateMachine()
        machine.reset(to_state=PriorityState.LOW)
        assert machine.state == PriorityState.LOW

    def test_get_stats(self):
        """Should return state machine stats."""
        machine = PriorityStateMachine()
        stats = machine.get_stats()
        assert "current_state" in stats
        assert "time_in_state_seconds" in stats
        assert "locked" in stats
        assert "history_size" in stats
        assert "allowed_transitions" in stats

    def test_history_recorded(self):
        """Should record transitions in history."""
        machine = PriorityStateMachine()
        machine.transition_to(PriorityState.HIGH, reason="test")

        recent = machine.history.get_recent(1)
        assert len(recent) == 1
        assert recent[0].from_state == PriorityState.NORMAL
        assert recent[0].to_state == PriorityState.HIGH
        assert recent[0].reason == "test"

    def test_force_transition(self):
        """Should allow forced transitions that bypass validation."""
        machine = AllocationStateMachine()
        # Cannot normally go IDLE -> RUNNING
        success = machine.transition_to(AllocationState.RUNNING)
        assert success is False

        # But can force it
        success = machine.transition_to(AllocationState.RUNNING, force=True)
        assert success is True
        assert machine.state == AllocationState.RUNNING


# =============================================================================
# CompositeStateMachine Tests
# =============================================================================


class TestCompositeStateMachine:
    """Tests for CompositeStateMachine."""

    def test_add_and_get_machines(self):
        """Should add and retrieve machines."""
        composite = CompositeStateMachine()
        priority_machine = PriorityStateMachine()
        composite.add("priority", priority_machine)

        retrieved = composite.get("priority")
        assert retrieved is priority_machine

    def test_transition_specific_machine(self):
        """Should transition specific machine by name."""
        composite = CompositeStateMachine()
        composite.add("priority", PriorityStateMachine())
        composite.add("allocation", AllocationStateMachine())

        success = composite.transition("priority", PriorityState.HIGH)
        assert success is True

        states = composite.get_combined_state()
        assert states["priority"] == "HIGH"
        assert states["allocation"] == "IDLE"

    def test_get_combined_state(self):
        """Should get combined state of all machines."""
        composite = CompositeStateMachine()
        composite.add("priority", PriorityStateMachine())
        composite.add("allocation", AllocationStateMachine())

        states = composite.get_combined_state()
        assert "priority" in states
        assert "allocation" in states

    def test_get_combined_stats(self):
        """Should get combined stats of all machines."""
        composite = CompositeStateMachine()
        composite.add("priority", PriorityStateMachine())
        composite.add("allocation", AllocationStateMachine())

        stats = composite.get_combined_stats()
        assert "priority" in stats
        assert "allocation" in stats
        assert "current_state" in stats["priority"]


# =============================================================================
# state_machine Decorator Tests
# =============================================================================


class TestStateMachineDecorator:
    """Tests for state_machine decorator."""

    def test_decorates_class(self):
        """Should decorate a class successfully."""

        @state_machine
        class MyMachine(StateMachineBase[SampleState]):
            INITIAL_STATE = SampleState.IDLE
            TRANSITIONS = {
                SampleState.IDLE: [SampleState.RUNNING],
                SampleState.RUNNING: [SampleState.COMPLETED],
            }

        machine = MyMachine()
        assert machine.state == SampleState.IDLE

    def test_decorated_transitions_work(self):
        """Should handle transitions in decorated class."""

        @state_machine
        class MyMachine(StateMachineBase[SampleState]):
            INITIAL_STATE = SampleState.IDLE
            TRANSITIONS = {
                SampleState.IDLE: [SampleState.RUNNING, SampleState.ERROR],
                SampleState.RUNNING: [SampleState.COMPLETED, SampleState.ERROR],
                SampleState.COMPLETED: [],
                SampleState.ERROR: [SampleState.IDLE],
            }

        machine = MyMachine()
        assert machine.transition_to(SampleState.RUNNING) is True
        assert machine.transition_to(SampleState.COMPLETED) is True
        # Cannot transition from COMPLETED (no allowed transitions)
        assert machine.transition_to(SampleState.IDLE) is False

    def test_invalid_transition_rejected(self):
        """Should reject invalid transitions in decorated class."""

        @state_machine
        class MyMachine(StateMachineBase[SampleState]):
            INITIAL_STATE = SampleState.IDLE
            TRANSITIONS = {
                SampleState.IDLE: [SampleState.RUNNING],
                SampleState.RUNNING: [SampleState.COMPLETED],
            }

        machine = MyMachine()
        # Cannot go directly from IDLE to COMPLETED
        assert machine.transition_to(SampleState.COMPLETED) is False
        assert machine.state == SampleState.IDLE
