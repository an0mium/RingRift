#!/usr/bin/env python3
"""Unit tests for app.core.state_machine module (December 2025).

Tests the unified state machine implementation:
- State dataclass with callbacks and metadata
- Transition dataclass with guards and actions
- StateMachine base class with transition validation
- StateHistory recording
- Pre-built state sets (CoordinatorStates, OrchestratorStates, PipelineStates)
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from app.core.state_machine import (
    State,
    Transition,
    StateHistory,
    StateMachine,
    StateMachineError,
    InvalidTransitionError,
    CoordinatorStates,
    OrchestratorStates,
    PipelineStates,
)


class TestState:
    """Tests for State dataclass."""

    def test_state_creation_minimal(self):
        """Test creating a state with just a name."""
        state = State("idle")
        assert state.name == "idle"
        assert state.initial is False
        assert state.terminal is False
        assert state.on_enter is None
        assert state.on_exit is None
        assert state.metadata == {}

    def test_state_creation_initial(self):
        """Test creating an initial state."""
        state = State("init", initial=True)
        assert state.initial is True
        assert state.terminal is False

    def test_state_creation_terminal(self):
        """Test creating a terminal state."""
        state = State("done", terminal=True)
        assert state.terminal is True
        assert state.initial is False

    def test_state_with_callbacks(self):
        """Test state with enter/exit callbacks."""
        on_enter = MagicMock()
        on_exit = MagicMock()

        state = State("active", on_enter=on_enter, on_exit=on_exit)
        assert state.on_enter is on_enter
        assert state.on_exit is on_exit

    def test_state_with_metadata(self):
        """Test state with metadata."""
        state = State("running", metadata={"priority": 1, "timeout": 30})
        assert state.metadata["priority"] == 1
        assert state.metadata["timeout"] == 30

    def test_state_hash(self):
        """Test state hashing is based on name."""
        s1 = State("test")
        s2 = State("test")
        s3 = State("other")

        assert hash(s1) == hash(s2)
        assert hash(s1) != hash(s3)

    def test_state_equality_with_state(self):
        """Test state equality with another State."""
        s1 = State("test", initial=True)
        s2 = State("test", terminal=True)  # Different flags, same name
        s3 = State("other")

        assert s1 == s2  # Equality based on name only
        assert s1 != s3

    def test_state_equality_with_string(self):
        """Test state equality with string."""
        state = State("running")
        assert state == "running"
        assert state != "stopped"

    def test_state_equality_with_other_types(self):
        """Test state inequality with non-state/string types."""
        state = State("test")
        assert state != 123
        assert state != ["test"]
        assert state != {"name": "test"}

    def test_state_str(self):
        """Test state string representation."""
        state = State("my_state")
        assert str(state) == "my_state"

    def test_state_repr(self):
        """Test state repr representation."""
        state = State("my_state")
        assert repr(state) == "State('my_state')"


class TestTransition:
    """Tests for Transition dataclass."""

    def test_transition_creation_minimal(self):
        """Test creating a transition with just states."""
        s1 = State("idle")
        s2 = State("running")

        t = Transition(s1, s2)
        assert t.from_state == s1
        assert t.to_state == s2
        assert t.guard is None
        assert t.action is None
        assert t.name is None

    def test_transition_with_name(self):
        """Test transition with a name."""
        s1 = State("idle")
        s2 = State("running")

        t = Transition(s1, s2, name="start")
        assert t.name == "start"

    def test_transition_with_guard(self):
        """Test transition with a guard function."""
        s1 = State("idle")
        s2 = State("running")

        guard = lambda m: True
        t = Transition(s1, s2, guard=guard)
        assert t.guard is guard

    def test_transition_with_action(self):
        """Test transition with an action function."""
        s1 = State("idle")
        s2 = State("running")

        action = MagicMock()
        t = Transition(s1, s2, action=action)
        assert t.action is action

    def test_can_execute_no_guard(self):
        """Test can_execute returns True when no guard."""
        s1 = State("idle")
        s2 = State("running")

        t = Transition(s1, s2)
        machine = MagicMock()
        assert t.can_execute(machine) is True

    def test_can_execute_guard_passes(self):
        """Test can_execute when guard returns True."""
        s1 = State("idle")
        s2 = State("running")

        guard = MagicMock(return_value=True)
        t = Transition(s1, s2, guard=guard)
        machine = MagicMock()

        assert t.can_execute(machine) is True
        guard.assert_called_once_with(machine)

    def test_can_execute_guard_fails(self):
        """Test can_execute when guard returns False."""
        s1 = State("idle")
        s2 = State("running")

        guard = MagicMock(return_value=False)
        t = Transition(s1, s2, guard=guard)
        machine = MagicMock()

        assert t.can_execute(machine) is False

    def test_can_execute_guard_raises(self):
        """Test can_execute when guard raises exception."""
        s1 = State("idle")
        s2 = State("running")

        guard = MagicMock(side_effect=ValueError("guard error"))
        t = Transition(s1, s2, guard=guard)
        machine = MagicMock()

        assert t.can_execute(machine) is False

    def test_execute_no_action(self):
        """Test execute does nothing when no action."""
        s1 = State("idle")
        s2 = State("running")

        t = Transition(s1, s2)
        machine = MagicMock()
        t.execute(machine)  # Should not raise

    def test_execute_calls_action(self):
        """Test execute calls the action."""
        s1 = State("idle")
        s2 = State("running")

        action = MagicMock()
        t = Transition(s1, s2, action=action)
        machine = MagicMock()

        t.execute(machine)
        action.assert_called_once_with(machine)


class TestStateHistory:
    """Tests for StateHistory dataclass."""

    def test_history_creation(self):
        """Test creating a state history record."""
        history = StateHistory(
            from_state="idle",
            to_state="running",
            timestamp=1000.0,
        )
        assert history.from_state == "idle"
        assert history.to_state == "running"
        assert history.timestamp == 1000.0
        assert history.transition_name is None
        assert history.duration_in_previous == 0.0

    def test_history_with_all_fields(self):
        """Test history with all fields populated."""
        history = StateHistory(
            from_state="idle",
            to_state="running",
            timestamp=1000.0,
            transition_name="start",
            duration_in_previous=5.5,
        )
        assert history.transition_name == "start"
        assert history.duration_in_previous == 5.5


class TestStateMachineBasic:
    """Basic tests for StateMachine class."""

    def test_simple_machine_creation(self):
        """Test creating a simple state machine."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")
            STOPPED = State("stopped", terminal=True)

            TRANSITIONS = [
                Transition(IDLE, RUNNING),
                Transition(RUNNING, STOPPED),
            ]

        machine = SimpleMachine()
        assert machine.state == SimpleMachine.IDLE
        assert machine.state_name == "idle"

    def test_machine_with_explicit_initial_state(self):
        """Test machine with explicit initial state override."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine(initial_state=SimpleMachine.RUNNING)
        assert machine.state == SimpleMachine.RUNNING

    def test_is_terminal_property(self):
        """Test is_terminal property."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            DONE = State("done", terminal=True)

            TRANSITIONS = [Transition(IDLE, DONE)]

        machine = SimpleMachine()
        assert machine.is_terminal is False

        machine.transition_to(SimpleMachine.DONE)
        assert machine.is_terminal is True

    def test_time_in_state_property(self):
        """Test time_in_state property."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)

            TRANSITIONS = []

        machine = SimpleMachine()
        time.sleep(0.1)
        assert machine.time_in_state >= 0.1

    def test_history_property(self):
        """Test history property returns copy."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        machine.transition_to(SimpleMachine.RUNNING)

        history = machine.history
        assert len(history) == 1
        assert history[0].from_state == "idle"
        assert history[0].to_state == "running"

        # Verify it's a copy
        history.clear()
        assert len(machine.history) == 1


class TestStateMachineTransitions:
    """Tests for state machine transitions."""

    def test_can_transition_to_valid(self):
        """Test can_transition_to for valid transition."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        assert machine.can_transition_to(SimpleMachine.RUNNING) is True
        assert machine.can_transition_to("running") is True

    def test_can_transition_to_invalid(self):
        """Test can_transition_to for invalid transition."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")
            STOPPED = State("stopped")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        assert machine.can_transition_to(SimpleMachine.STOPPED) is False

    def test_can_transition_from_terminal(self):
        """Test can_transition_to returns False from terminal state."""
        class SimpleMachine(StateMachine):
            DONE = State("done", initial=True, terminal=True)
            OTHER = State("other")

            TRANSITIONS = [Transition(DONE, OTHER)]

        machine = SimpleMachine()
        assert machine.can_transition_to(SimpleMachine.OTHER) is False

    def test_can_transition_guard_blocks(self):
        """Test can_transition_to when guard blocks."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [
                Transition(IDLE, RUNNING, guard=lambda m: False)
            ]

        machine = SimpleMachine()
        assert machine.can_transition_to(SimpleMachine.RUNNING) is False

    def test_transition_to_success(self):
        """Test successful transition."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        result = machine.transition_to(SimpleMachine.RUNNING)

        assert result is True
        assert machine.state == SimpleMachine.RUNNING

    def test_transition_to_by_string(self):
        """Test transition using state name string."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        machine.transition_to("running")
        assert machine.state_name == "running"

    def test_transition_to_invalid_raises(self):
        """Test transition_to raises for invalid transition."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")
            STOPPED = State("stopped")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()

        with pytest.raises(InvalidTransitionError) as exc_info:
            machine.transition_to(SimpleMachine.STOPPED)

        assert exc_info.value.from_state == "idle"
        assert exc_info.value.to_state == "stopped"
        assert "No valid transition defined" in str(exc_info.value)

    def test_transition_from_terminal_raises(self):
        """Test transition from terminal state raises."""
        class SimpleMachine(StateMachine):
            DONE = State("done", initial=True, terminal=True)
            OTHER = State("other")

            TRANSITIONS = [Transition(DONE, OTHER)]

        machine = SimpleMachine()

        with pytest.raises(InvalidTransitionError) as exc_info:
            machine.transition_to(SimpleMachine.OTHER)

        assert "Cannot transition from terminal state" in str(exc_info.value)

    def test_transition_guard_rejects(self):
        """Test transition when guard rejects."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [
                Transition(IDLE, RUNNING, guard=lambda m: False)
            ]

        machine = SimpleMachine()

        with pytest.raises(InvalidTransitionError) as exc_info:
            machine.transition_to(SimpleMachine.RUNNING)

        assert "Transition guard rejected" in str(exc_info.value)

    def test_transition_force_bypasses_validation(self):
        """Test forced transition bypasses validation."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")
            STOPPED = State("stopped")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        result = machine.transition_to(SimpleMachine.STOPPED, force=True)

        assert result is True
        assert machine.state == SimpleMachine.STOPPED

    def test_transition_unknown_state_raises(self):
        """Test transition to unknown state raises."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)

            TRANSITIONS = []

        machine = SimpleMachine()

        with pytest.raises(StateMachineError) as exc_info:
            machine.transition_to("nonexistent")

        assert "Unknown state: nonexistent" in str(exc_info.value)


class TestStateMachineCallbacks:
    """Tests for state machine callbacks."""

    def test_on_enter_callback(self):
        """Test on_enter callback is called."""
        on_enter = MagicMock()

        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running", on_enter=on_enter)

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        machine.transition_to(SimpleMachine.RUNNING)

        on_enter.assert_called_once_with(machine)

    def test_on_exit_callback(self):
        """Test on_exit callback is called."""
        on_exit = MagicMock()

        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True, on_exit=on_exit)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        machine.transition_to(SimpleMachine.RUNNING)

        on_exit.assert_called_once_with(machine)

    def test_transition_action_called(self):
        """Test transition action is called."""
        action = MagicMock()

        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING, action=action)]

        machine = SimpleMachine()
        machine.transition_to(SimpleMachine.RUNNING)

        action.assert_called_once_with(machine)

    def test_callback_order(self):
        """Test callbacks are called in correct order: exit -> action -> enter."""
        call_order = []

        def on_exit(m):
            call_order.append("exit")

        def action(m):
            call_order.append("action")

        def on_enter(m):
            call_order.append("enter")

        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True, on_exit=on_exit)
            RUNNING = State("running", on_enter=on_enter)

            TRANSITIONS = [Transition(IDLE, RUNNING, action=action)]

        machine = SimpleMachine()
        machine.transition_to(SimpleMachine.RUNNING)

        assert call_order == ["exit", "action", "enter"]

    def test_callback_exception_isolated(self):
        """Test callback exceptions don't prevent transition."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True, on_exit=lambda m: 1/0)
            RUNNING = State("running", on_enter=lambda m: 1/0)

            TRANSITIONS = [
                Transition(IDLE, RUNNING, action=lambda m: 1/0)
            ]

        machine = SimpleMachine()
        # Should not raise, callbacks are isolated
        machine.transition_to(SimpleMachine.RUNNING)
        assert machine.state == SimpleMachine.RUNNING


class TestStateMachineHistory:
    """Tests for state machine history."""

    def test_history_recording(self):
        """Test history is recorded."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING, name="start")]

        machine = SimpleMachine()
        machine.transition_to(SimpleMachine.RUNNING)

        assert len(machine.history) == 1
        record = machine.history[0]
        assert record.from_state == "idle"
        assert record.to_state == "running"
        assert record.transition_name == "start"
        assert record.timestamp > 0

    def test_history_disabled(self):
        """Test history can be disabled."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine(record_history=False)
        machine.transition_to(SimpleMachine.RUNNING)

        assert len(machine.history) == 0

    def test_history_limit(self):
        """Test history is limited."""
        class SimpleMachine(StateMachine):
            A = State("a", initial=True)
            B = State("b")

            TRANSITIONS = [
                Transition(A, B),
                Transition(B, A),
            ]

        machine = SimpleMachine(max_history=3)

        for _ in range(5):
            machine.transition_to("b")
            machine.transition_to("a")

        assert len(machine.history) == 3

    def test_history_duration_recorded(self):
        """Test duration in previous state is recorded."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        time.sleep(0.1)
        machine.transition_to(SimpleMachine.RUNNING)

        record = machine.history[0]
        assert record.duration_in_previous >= 0.1


class TestStateMachineUtilities:
    """Tests for state machine utility methods."""

    def test_get_valid_transitions(self):
        """Test get_valid_transitions returns valid targets."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")
            STOPPED = State("stopped")

            TRANSITIONS = [
                Transition(IDLE, RUNNING),
                Transition(IDLE, STOPPED),
            ]

        machine = SimpleMachine()
        valid = machine.get_valid_transitions()

        assert "running" in valid
        assert "stopped" in valid

    def test_get_valid_transitions_with_guard(self):
        """Test get_valid_transitions respects guards."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")
            STOPPED = State("stopped")

            TRANSITIONS = [
                Transition(IDLE, RUNNING, guard=lambda m: True),
                Transition(IDLE, STOPPED, guard=lambda m: False),
            ]

        machine = SimpleMachine()
        valid = machine.get_valid_transitions()

        assert "running" in valid
        assert "stopped" not in valid

    def test_get_valid_transitions_from_terminal(self):
        """Test get_valid_transitions returns empty from terminal."""
        class SimpleMachine(StateMachine):
            DONE = State("done", initial=True, terminal=True)
            OTHER = State("other")

            TRANSITIONS = [Transition(DONE, OTHER)]

        machine = SimpleMachine()
        assert machine.get_valid_transitions() == []

    def test_reset(self):
        """Test reset returns to initial state."""
        class SimpleMachine(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [Transition(IDLE, RUNNING)]

        machine = SimpleMachine()
        machine.transition_to(SimpleMachine.RUNNING)
        assert len(machine.history) == 1

        machine.reset()

        assert machine.state == SimpleMachine.IDLE
        assert len(machine.history) == 0


class TestInvalidTransitionError:
    """Tests for InvalidTransitionError exception."""

    def test_error_basic(self):
        """Test basic error creation."""
        error = InvalidTransitionError("idle", "running")
        assert error.from_state == "idle"
        assert error.to_state == "running"
        assert error.reason == ""
        assert "Invalid transition from 'idle' to 'running'" in str(error)

    def test_error_with_reason(self):
        """Test error with reason."""
        error = InvalidTransitionError("idle", "running", "guard failed")
        assert error.reason == "guard failed"
        assert "guard failed" in str(error)


class TestCoordinatorStates:
    """Tests for CoordinatorStates pre-built set."""

    def test_states_defined(self):
        """Test all coordinator states are defined."""
        assert CoordinatorStates.INITIALIZING.initial is True
        assert CoordinatorStates.ACTIVE.terminal is False
        assert CoordinatorStates.PAUSED.terminal is False
        assert CoordinatorStates.STOPPING.terminal is False
        assert CoordinatorStates.STOPPED.terminal is True
        assert CoordinatorStates.FAILED.terminal is True

    def test_transitions_defined(self):
        """Test coordinator transitions are defined."""
        transitions = CoordinatorStates.TRANSITIONS
        assert len(transitions) > 0

        # Check key transitions exist
        from_to = [(t.from_state.name, t.to_state.name) for t in transitions]
        assert ("initializing", "active") in from_to
        assert ("active", "paused") in from_to
        assert ("paused", "active") in from_to
        assert ("stopping", "stopped") in from_to


class TestOrchestratorStates:
    """Tests for OrchestratorStates pre-built set."""

    def test_states_defined(self):
        """Test all orchestrator states are defined."""
        assert OrchestratorStates.IDLE.initial is True
        assert OrchestratorStates.STARTING.terminal is False
        assert OrchestratorStates.RUNNING.terminal is False
        assert OrchestratorStates.DRAINING.terminal is False
        assert OrchestratorStates.STOPPED.terminal is True
        assert OrchestratorStates.FAILED.terminal is True

    def test_transitions_defined(self):
        """Test orchestrator transitions are defined."""
        transitions = OrchestratorStates.TRANSITIONS

        from_to = [(t.from_state.name, t.to_state.name) for t in transitions]
        assert ("idle", "starting") in from_to
        assert ("starting", "running") in from_to
        assert ("running", "draining") in from_to
        assert ("draining", "stopped") in from_to


class TestPipelineStates:
    """Tests for PipelineStates pre-built set."""

    def test_states_defined(self):
        """Test all pipeline states are defined."""
        assert PipelineStates.IDLE.initial is True
        assert PipelineStates.SELFPLAY.terminal is False
        assert PipelineStates.DATA_SYNC.terminal is False
        assert PipelineStates.TRAINING.terminal is False
        assert PipelineStates.EVALUATION.terminal is False
        assert PipelineStates.PROMOTION.terminal is False
        assert PipelineStates.FAILED.terminal is False  # Can recover

    def test_transitions_form_pipeline(self):
        """Test pipeline transitions form a logical flow."""
        transitions = PipelineStates.TRANSITIONS

        from_to = [(t.from_state.name, t.to_state.name) for t in transitions]
        # Normal flow
        assert ("idle", "selfplay") in from_to
        assert ("selfplay", "data_sync") in from_to
        assert ("data_sync", "training") in from_to
        assert ("training", "evaluation") in from_to
        assert ("evaluation", "promotion") in from_to
        assert ("promotion", "idle") in from_to
        # Recovery
        assert ("failed", "idle") in from_to


class TestStateMachineEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_no_states_defined(self):
        """Test machine with no states raises error."""
        class EmptyMachine(StateMachine):
            TRANSITIONS = []

        with pytest.raises(StateMachineError):
            EmptyMachine()

    def test_no_initial_state_uses_first(self):
        """Test machine without explicit initial uses first state."""
        class NoInitialMachine(StateMachine):
            ALPHA = State("alpha")  # No initial=True
            BETA = State("beta")

            TRANSITIONS = []

        machine = NoInitialMachine()
        # Should pick first state found
        assert machine.state in [NoInitialMachine.ALPHA, NoInitialMachine.BETA]

    def test_multiple_transitions_same_target(self):
        """Test multiple transitions to same target with different guards.

        Note: The implementation checks transitions in order and returns on
        first match. If first guard fails, second is NOT checked by can_transition_to.
        """
        guard1_result = [True]
        guard2_result = [False]

        class MultiTransition(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")

            TRANSITIONS = [
                Transition(IDLE, RUNNING, guard=lambda m: guard1_result[0], name="fast"),
                Transition(IDLE, RUNNING, guard=lambda m: guard2_result[0], name="slow"),
            ]

        machine = MultiTransition()

        # First guard passes - can transition
        assert machine.can_transition_to(MultiTransition.RUNNING) is True

        # First guard fails - returns False (doesn't check second guard)
        guard1_result[0] = False
        assert machine.can_transition_to(MultiTransition.RUNNING) is False

        # First guard passes again
        guard1_result[0] = True
        assert machine.can_transition_to(MultiTransition.RUNNING) is True

    def test_self_transition(self):
        """Test transition to same state."""
        action_called = [False]

        def action(m):
            action_called[0] = True

        class SelfTransition(StateMachine):
            RUNNING = State("running", initial=True)

            TRANSITIONS = [
                Transition(RUNNING, RUNNING, action=action)
            ]

        machine = SelfTransition()
        machine.transition_to(SelfTransition.RUNNING)

        assert machine.state == SelfTransition.RUNNING
        assert action_called[0] is True


class TestStateMachineIntegration:
    """Integration tests combining multiple features."""

    def test_full_lifecycle(self):
        """Test a complete state machine lifecycle."""
        events = []

        def log_event(name):
            def callback(m):
                events.append(name)
            return callback

        class LifecycleMachine(StateMachine):
            INIT = State("init", initial=True, on_exit=log_event("exit_init"))
            STARTING = State("starting", on_enter=log_event("enter_starting"))
            RUNNING = State("running", on_enter=log_event("enter_running"))
            STOPPING = State("stopping")
            STOPPED = State("stopped", terminal=True, on_enter=log_event("enter_stopped"))

            TRANSITIONS = [
                Transition(INIT, STARTING, action=log_event("action_start")),
                Transition(STARTING, RUNNING),
                Transition(RUNNING, STOPPING),
                Transition(STOPPING, STOPPED),
            ]

        machine = LifecycleMachine()

        # Full lifecycle
        machine.transition_to(LifecycleMachine.STARTING)
        machine.transition_to(LifecycleMachine.RUNNING)
        machine.transition_to(LifecycleMachine.STOPPING)
        machine.transition_to(LifecycleMachine.STOPPED)

        assert machine.is_terminal is True
        assert len(machine.history) == 4

        # Check events were logged in order
        assert events[0] == "exit_init"
        assert events[1] == "action_start"
        assert events[2] == "enter_starting"

    def test_guarded_workflow(self):
        """Test workflow with guards controlling transitions."""
        class GuardedWorkflow(StateMachine):
            IDLE = State("idle", initial=True)
            PROCESSING = State("processing")
            DONE = State("done")

            def __init__(self):
                self.data_ready = False
                self.processing_complete = False
                super().__init__()

            TRANSITIONS = [
                Transition(
                    IDLE, PROCESSING,
                    guard=lambda m: m.data_ready
                ),
                Transition(
                    PROCESSING, DONE,
                    guard=lambda m: m.processing_complete
                ),
            ]

        machine = GuardedWorkflow()

        # Can't transition yet
        assert machine.can_transition_to(GuardedWorkflow.PROCESSING) is False

        # Enable data
        machine.data_ready = True
        assert machine.can_transition_to(GuardedWorkflow.PROCESSING) is True

        machine.transition_to(GuardedWorkflow.PROCESSING)

        # Can't finish yet
        assert machine.can_transition_to(GuardedWorkflow.DONE) is False

        # Complete processing
        machine.processing_complete = True
        machine.transition_to(GuardedWorkflow.DONE)

        assert machine.state == GuardedWorkflow.DONE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
