"""Tests for ModelLifecycleStateMachine - model lifecycle management.

Created: Dec 29, 2025
Phase 3: Test coverage for critical untested modules.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from app.core.state_machine import InvalidTransitionError
from app.training.model_state_machine import (
    MODEL_TRANSITIONS,
    ModelLifecycleStateMachine,
    ModelState,
    ModelStateRecord,
    PromotionControllerIntegration,
    get_model_lifecycle,
    reset_model_lifecycle,
)


# --- ModelState Tests ---


class TestModelState:
    """Tests for ModelState enum."""

    def test_all_states_exist(self) -> None:
        """All expected states should exist."""
        assert ModelState.TRAINING.value == "training"
        assert ModelState.TRAINED.value == "trained"
        assert ModelState.EVALUATING.value == "evaluating"
        assert ModelState.EVALUATED.value == "evaluated"
        assert ModelState.STAGING.value == "staging"
        assert ModelState.SHADOW.value == "shadow"
        assert ModelState.PRODUCTION.value == "production"
        assert ModelState.ARCHIVED.value == "archived"
        assert ModelState.ROLLED_BACK.value == "rolled_back"
        assert ModelState.FAILED.value == "failed"

    def test_state_count(self) -> None:
        """Should have 10 total states."""
        states = list(ModelState)
        assert len(states) == 10


# --- ModelStateRecord Tests ---


class TestModelStateRecord:
    """Tests for ModelStateRecord dataclass."""

    def test_basic_record(self) -> None:
        """Basic record should have correct fields."""
        record = ModelStateRecord(
            model_id="test_model",
            current_state=ModelState.TRAINING,
        )
        assert record.model_id == "test_model"
        assert record.current_state == ModelState.TRAINING
        assert isinstance(record.created_at, float)
        assert isinstance(record.updated_at, float)
        assert record.history == []
        assert record.metadata == {}

    def test_record_with_metadata(self) -> None:
        """Record should accept metadata."""
        record = ModelStateRecord(
            model_id="test_model",
            current_state=ModelState.TRAINING,
            metadata={"version": "v42", "board_type": "hex8"},
        )
        assert record.metadata["version"] == "v42"
        assert record.metadata["board_type"] == "hex8"

    def test_add_transition(self) -> None:
        """add_transition should update state and history."""
        record = ModelStateRecord(
            model_id="test_model",
            current_state=ModelState.TRAINING,
        )

        record.add_transition(
            from_state=ModelState.TRAINING,
            to_state=ModelState.TRAINED,
            reason="Training complete",
            triggered_by="system",
        )

        assert record.current_state == ModelState.TRAINED
        assert len(record.history) == 1
        assert record.history[0]["from_state"] == "training"
        assert record.history[0]["to_state"] == "trained"
        assert record.history[0]["reason"] == "Training complete"
        assert record.history[0]["triggered_by"] == "system"

    def test_add_multiple_transitions(self) -> None:
        """Multiple transitions should accumulate in history."""
        record = ModelStateRecord(
            model_id="test_model",
            current_state=ModelState.TRAINING,
        )

        record.add_transition(ModelState.TRAINING, ModelState.TRAINED)
        record.add_transition(ModelState.TRAINED, ModelState.EVALUATING)
        record.add_transition(ModelState.EVALUATING, ModelState.EVALUATED)

        assert len(record.history) == 3
        assert record.current_state == ModelState.EVALUATED

    def test_to_dict(self) -> None:
        """to_dict should return serializable dictionary."""
        record = ModelStateRecord(
            model_id="test_model",
            current_state=ModelState.PRODUCTION,
            metadata={"key": "value"},
        )
        record.add_transition(
            ModelState.STAGING,
            ModelState.PRODUCTION,
            reason="Promotion",
        )

        d = record.to_dict()
        assert d["model_id"] == "test_model"
        assert d["current_state"] == "production"
        assert isinstance(d["created_at"], float)
        assert isinstance(d["updated_at"], float)
        assert len(d["history"]) == 1
        assert d["metadata"]["key"] == "value"


# --- MODEL_TRANSITIONS Tests ---


class TestModelTransitions:
    """Tests for MODEL_TRANSITIONS configuration."""

    def test_transitions_defined(self) -> None:
        """Transitions should be defined."""
        assert len(MODEL_TRANSITIONS) > 0

    def test_training_transitions(self) -> None:
        """Training should transition to trained or failed."""
        training_transitions = [
            t for t in MODEL_TRANSITIONS
            if t.from_state.name == "training"
        ]
        to_states = {t.to_state.name for t in training_transitions}
        assert "trained" in to_states
        assert "failed" in to_states

    def test_production_transitions(self) -> None:
        """Production should transition to rolled_back or archived."""
        prod_transitions = [
            t for t in MODEL_TRANSITIONS
            if t.from_state.name == "production"
        ]
        to_states = {t.to_state.name for t in prod_transitions}
        assert "rolled_back" in to_states
        assert "archived" in to_states

    def test_all_transitions_have_names(self) -> None:
        """All transitions should have names."""
        for t in MODEL_TRANSITIONS:
            assert t.name is not None and t.name != ""


# --- ModelLifecycleStateMachine Initialization Tests ---


class TestLifecycleInit:
    """Tests for ModelLifecycleStateMachine initialization."""

    def test_default_init(self) -> None:
        """Default initialization should work."""
        lifecycle = ModelLifecycleStateMachine()
        assert lifecycle._models == {}
        assert lifecycle._transition_listeners == []
        assert len(lifecycle._transitions) > 0

    def test_transitions_map_built(self) -> None:
        """Transition map should be built from MODEL_TRANSITIONS."""
        lifecycle = ModelLifecycleStateMachine()
        # Should have entries for training, trained, etc.
        assert "training" in lifecycle._transitions
        assert "trained" in lifecycle._transitions


# --- ModelLifecycleStateMachine Registration Tests ---


class TestLifecycleRegistration:
    """Tests for model registration."""

    def test_register_model(self) -> None:
        """register_model should create new record."""
        lifecycle = ModelLifecycleStateMachine()
        record = lifecycle.register_model("model_v42")

        assert record.model_id == "model_v42"
        assert record.current_state == ModelState.TRAINING
        assert "model_v42" in lifecycle._models

    def test_register_with_initial_state(self) -> None:
        """register_model should accept custom initial state."""
        lifecycle = ModelLifecycleStateMachine()
        record = lifecycle.register_model(
            "model_v42",
            initial_state=ModelState.TRAINED,
        )
        assert record.current_state == ModelState.TRAINED

    def test_register_with_metadata(self) -> None:
        """register_model should accept metadata."""
        lifecycle = ModelLifecycleStateMachine()
        record = lifecycle.register_model(
            "model_v42",
            metadata={"version": "42"},
        )
        assert record.metadata["version"] == "42"

    def test_register_duplicate_raises(self) -> None:
        """Registering same model twice should raise."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42")

        with pytest.raises(ValueError, match="already registered"):
            lifecycle.register_model("model_v42")

    def test_get_or_register_existing(self) -> None:
        """get_or_register should return existing model."""
        lifecycle = ModelLifecycleStateMachine()
        record1 = lifecycle.register_model("model_v42", metadata={"a": 1})
        record2 = lifecycle.get_or_register("model_v42")

        assert record1 is record2
        assert record2.metadata["a"] == 1

    def test_get_or_register_new(self) -> None:
        """get_or_register should create new model if not exists."""
        lifecycle = ModelLifecycleStateMachine()
        record = lifecycle.get_or_register("model_v42")

        assert record.model_id == "model_v42"


# --- State Query Tests ---


class TestStateQueries:
    """Tests for state query methods."""

    def test_get_state(self) -> None:
        """get_state should return current state."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.STAGING)

        state = lifecycle.get_state("model_v42")
        assert state == ModelState.STAGING

    def test_get_state_nonexistent(self) -> None:
        """get_state should return None for unknown model."""
        lifecycle = ModelLifecycleStateMachine()
        assert lifecycle.get_state("unknown") is None

    def test_get_record(self) -> None:
        """get_record should return full record."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", metadata={"key": "value"})

        record = lifecycle.get_record("model_v42")
        assert record is not None
        assert record.metadata["key"] == "value"

    def test_get_record_nonexistent(self) -> None:
        """get_record should return None for unknown model."""
        lifecycle = ModelLifecycleStateMachine()
        assert lifecycle.get_record("unknown") is None


# --- Transition Validation Tests ---


class TestTransitionValidation:
    """Tests for transition validation."""

    def test_get_valid_transitions(self) -> None:
        """get_valid_transitions should return valid targets."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.TRAINING)

        valid = lifecycle.get_valid_transitions("model_v42")
        state_values = [s.value for s in valid]
        assert "trained" in state_values
        assert "failed" in state_values

    def test_get_valid_transitions_nonexistent(self) -> None:
        """get_valid_transitions should return empty for unknown model."""
        lifecycle = ModelLifecycleStateMachine()
        assert lifecycle.get_valid_transitions("unknown") == []

    def test_can_transition_valid(self) -> None:
        """can_transition should return True for valid transition."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.TRAINING)

        can, name = lifecycle.can_transition("model_v42", ModelState.TRAINED)
        assert can is True
        assert name == "training_complete"

    def test_can_transition_invalid(self) -> None:
        """can_transition should return False for invalid transition."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.TRAINING)

        can, msg = lifecycle.can_transition("model_v42", ModelState.PRODUCTION)
        assert can is False
        assert "Invalid transition" in msg

    def test_can_transition_nonexistent(self) -> None:
        """can_transition should handle unknown model."""
        lifecycle = ModelLifecycleStateMachine()

        can, msg = lifecycle.can_transition("unknown", ModelState.TRAINED)
        assert can is False
        assert "not registered" in msg


# --- Transition Execution Tests ---


class TestTransitionExecution:
    """Tests for transition execution."""

    def test_transition_success(self) -> None:
        """Valid transition should succeed."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.TRAINING)

        result = lifecycle.transition("model_v42", ModelState.TRAINED)

        assert result is True
        assert lifecycle.get_state("model_v42") == ModelState.TRAINED

    def test_transition_with_reason(self) -> None:
        """Transition should record reason."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.TRAINING)

        lifecycle.transition(
            "model_v42",
            ModelState.TRAINED,
            reason="Training complete",
            triggered_by="trainer",
        )

        record = lifecycle.get_record("model_v42")
        assert record.history[-1]["reason"] == "Training complete"
        assert record.history[-1]["triggered_by"] == "trainer"

    def test_transition_invalid_raises(self) -> None:
        """Invalid transition should raise."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.TRAINING)

        with pytest.raises(InvalidTransitionError):
            lifecycle.transition("model_v42", ModelState.PRODUCTION)

    def test_transition_force(self) -> None:
        """Force transition should skip validation."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.TRAINING)

        result = lifecycle.transition(
            "model_v42",
            ModelState.PRODUCTION,
            force=True,
        )

        assert result is True
        assert lifecycle.get_state("model_v42") == ModelState.PRODUCTION

    def test_transition_nonexistent_raises(self) -> None:
        """Transition on unknown model should raise."""
        lifecycle = ModelLifecycleStateMachine()

        with pytest.raises(ValueError, match="not registered"):
            lifecycle.transition("unknown", ModelState.TRAINED)

    def test_transition_chain(self) -> None:
        """Multiple transitions should be tracked in history."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42")

        lifecycle.transition("model_v42", ModelState.TRAINED)
        lifecycle.transition("model_v42", ModelState.EVALUATING)
        lifecycle.transition("model_v42", ModelState.EVALUATED)
        lifecycle.transition("model_v42", ModelState.STAGING)
        lifecycle.transition("model_v42", ModelState.PRODUCTION)

        record = lifecycle.get_record("model_v42")
        assert record.current_state == ModelState.PRODUCTION
        assert len(record.history) == 5


# --- Rollback Tests ---


class TestRollback:
    """Tests for rollback functionality."""

    def test_rollback_success(self) -> None:
        """Rollback should move current model and restore old."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v41", initial_state=ModelState.ARCHIVED)
        lifecycle.register_model("model_v42", initial_state=ModelState.PRODUCTION)

        result = lifecycle.rollback(
            current_model_id="model_v42",
            rollback_to_model_id="model_v41",
            reason="Performance regression",
        )

        assert result is True
        assert lifecycle.get_state("model_v42") == ModelState.ROLLED_BACK
        assert lifecycle.get_state("model_v41") == ModelState.PRODUCTION

    def test_rollback_nonexistent_current(self) -> None:
        """Rollback should handle missing current model."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v41", initial_state=ModelState.ARCHIVED)

        result = lifecycle.rollback(
            current_model_id="unknown",
            rollback_to_model_id="model_v41",
        )

        # Should still restore old model
        assert result is True
        assert lifecycle.get_state("model_v41") == ModelState.PRODUCTION

    def test_rollback_nonexistent_target(self) -> None:
        """Rollback should handle missing target model."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.PRODUCTION)

        result = lifecycle.rollback(
            current_model_id="model_v42",
            rollback_to_model_id="unknown",
        )

        assert result is False


# --- Listener Tests ---


class TestTransitionListeners:
    """Tests for transition listeners."""

    def test_add_listener(self) -> None:
        """Listener should be added."""
        lifecycle = ModelLifecycleStateMachine()

        listener = MagicMock()
        lifecycle.add_transition_listener(listener)

        assert listener in lifecycle._transition_listeners

    def test_listener_called_on_transition(self) -> None:
        """Listener should be called when transition occurs."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42")

        listener = MagicMock()
        lifecycle.add_transition_listener(listener)

        lifecycle.transition("model_v42", ModelState.TRAINED, reason="Done")

        listener.assert_called_once_with(
            "model_v42",
            ModelState.TRAINING,
            ModelState.TRAINED,
            "Done",
        )

    def test_listener_error_handled(self) -> None:
        """Listener error should not break transition."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42")

        def bad_listener(*args):
            raise RuntimeError("Listener error")

        lifecycle.add_transition_listener(bad_listener)

        # Should not raise
        lifecycle.transition("model_v42", ModelState.TRAINED)
        assert lifecycle.get_state("model_v42") == ModelState.TRAINED


# --- Model Collection Tests ---


class TestModelCollections:
    """Tests for model collection methods."""

    def test_get_models_in_state(self) -> None:
        """get_models_in_state should return matching models."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v41", initial_state=ModelState.ARCHIVED)
        lifecycle.register_model("model_v42", initial_state=ModelState.STAGING)
        lifecycle.register_model("model_v43", initial_state=ModelState.STAGING)

        staging_models = lifecycle.get_models_in_state(ModelState.STAGING)
        assert len(staging_models) == 2
        assert "model_v42" in staging_models
        assert "model_v43" in staging_models

    def test_get_production_model(self) -> None:
        """get_production_model should return production model."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.PRODUCTION)

        prod = lifecycle.get_production_model()
        assert prod == "model_v42"

    def test_get_production_model_none(self) -> None:
        """get_production_model should return None if no production."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.STAGING)

        assert lifecycle.get_production_model() is None

    def test_get_staging_models(self) -> None:
        """get_staging_models should return staging models."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v41", initial_state=ModelState.STAGING)
        lifecycle.register_model("model_v42", initial_state=ModelState.STAGING)

        staging = lifecycle.get_staging_models()
        assert len(staging) == 2

    def test_get_all_models(self) -> None:
        """get_all_models should return all tracked models."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v41")
        lifecycle.register_model("model_v42")

        all_models = lifecycle.get_all_models()
        assert len(all_models) == 2
        assert "model_v41" in all_models
        assert "model_v42" in all_models


# --- History Tests ---


class TestModelHistory:
    """Tests for model history."""

    def test_get_model_history(self) -> None:
        """get_model_history should return transition history."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42")
        lifecycle.transition("model_v42", ModelState.TRAINED)
        lifecycle.transition("model_v42", ModelState.EVALUATING)

        history = lifecycle.get_model_history("model_v42")
        assert len(history) == 2

    def test_get_model_history_limit(self) -> None:
        """get_model_history should respect limit."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42")

        # Make many transitions via force
        for _ in range(10):
            lifecycle.transition("model_v42", ModelState.TRAINED, force=True)
            lifecycle.transition("model_v42", ModelState.TRAINING, force=True)

        history = lifecycle.get_model_history("model_v42", limit=5)
        assert len(history) == 5

    def test_get_model_history_nonexistent(self) -> None:
        """get_model_history should return empty for unknown model."""
        lifecycle = ModelLifecycleStateMachine()
        assert lifecycle.get_model_history("unknown") == []


# --- Stats Tests ---


class TestStats:
    """Tests for lifecycle statistics."""

    def test_get_stats(self) -> None:
        """get_stats should return statistics."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v41", initial_state=ModelState.ARCHIVED)
        lifecycle.register_model("model_v42", initial_state=ModelState.PRODUCTION)
        lifecycle.register_model("model_v43", initial_state=ModelState.TRAINING)
        lifecycle.transition("model_v43", ModelState.TRAINED)

        stats = lifecycle.get_stats()

        assert stats["total_models"] == 3
        assert stats["state_counts"]["archived"] == 1
        assert stats["state_counts"]["production"] == 1
        assert stats["state_counts"]["trained"] == 1
        assert stats["total_transitions"] == 1
        assert stats["production_model"] == "model_v42"


# --- Thread Safety Tests ---


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self) -> None:
        """Concurrent registration should be safe."""
        lifecycle = ModelLifecycleStateMachine()
        errors = []

        def register(i: int) -> None:
            try:
                lifecycle.register_model(f"model_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(lifecycle._models) == 10

    def test_concurrent_transitions(self) -> None:
        """Concurrent transitions should be safe."""
        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("model_v42", initial_state=ModelState.STAGING)
        errors = []

        def transition_to_shadow() -> None:
            try:
                lifecycle.transition("model_v42", ModelState.SHADOW, force=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=transition_to_shadow) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors, model should be in shadow state
        assert len(errors) == 0
        assert lifecycle.get_state("model_v42") == ModelState.SHADOW


# --- Singleton Tests ---


class TestSingleton:
    """Tests for global singleton."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_model_lifecycle()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_model_lifecycle()

    def test_get_model_lifecycle_creates(self) -> None:
        """get_model_lifecycle should create instance."""
        lifecycle = get_model_lifecycle()
        assert lifecycle is not None
        assert isinstance(lifecycle, ModelLifecycleStateMachine)

    def test_get_model_lifecycle_same_instance(self) -> None:
        """get_model_lifecycle should return same instance."""
        lc1 = get_model_lifecycle()
        lc2 = get_model_lifecycle()
        assert lc1 is lc2

    def test_reset_model_lifecycle(self) -> None:
        """reset_model_lifecycle should clear singleton."""
        lc1 = get_model_lifecycle()
        lc1.register_model("test")
        reset_model_lifecycle()
        lc2 = get_model_lifecycle()

        assert lc1 is not lc2
        assert lc2.get_state("test") is None


# --- PromotionControllerIntegration Tests ---


class TestPromotionControllerIntegration:
    """Tests for PromotionControllerIntegration."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_model_lifecycle()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_model_lifecycle()

    def test_integration_init(self) -> None:
        """Integration should initialize with lifecycle."""
        integration = PromotionControllerIntegration()
        assert integration._lifecycle is not None

    def test_integration_custom_lifecycle(self) -> None:
        """Integration should accept custom lifecycle."""
        lifecycle = ModelLifecycleStateMachine()
        integration = PromotionControllerIntegration(lifecycle=lifecycle)
        assert integration._lifecycle is lifecycle

    def test_wire_promotion_controller(self) -> None:
        """wire_promotion_controller should wrap execute_promotion."""
        lifecycle = ModelLifecycleStateMachine()
        integration = PromotionControllerIntegration(lifecycle=lifecycle)

        # Create mock controller
        controller = MagicMock()
        original_execute = MagicMock(return_value=True)
        controller.execute_promotion = original_execute

        integration.wire_promotion_controller(controller)

        # execute_promotion should be wrapped
        assert controller.execute_promotion is not original_execute

    def test_wired_controller_updates_state(self) -> None:
        """Wired controller should update model state on promotion."""
        lifecycle = ModelLifecycleStateMachine()
        integration = PromotionControllerIntegration(lifecycle=lifecycle)

        # Create mock controller
        mock_controller = MagicMock()

        # Store original execute to simulate real behavior
        def real_execute(decision, dry_run=False):
            return True

        mock_controller.execute_promotion = real_execute

        integration.wire_promotion_controller(mock_controller)

        # Patch the PromotionType import inside _update_state_from_decision
        with patch.object(
            integration,
            "_update_state_from_decision",
            wraps=integration._update_state_from_decision,
        ) as mock_update:
            # Create a decision-like object
            mock_decision = MagicMock()
            mock_decision.model_id = "model_v42"

            # Call the wrapped execute_promotion
            result = mock_controller.execute_promotion(mock_decision)

            # Verify execute returned True and update was called
            assert result is True
            mock_update.assert_called_once_with(mock_decision)

    def test_update_state_from_decision_staging(self) -> None:
        """_update_state_from_decision should update state for staging promotion."""
        lifecycle = ModelLifecycleStateMachine()
        integration = PromotionControllerIntegration(lifecycle=lifecycle)

        # Create mock decision with staging promotion type
        mock_decision = MagicMock()
        mock_decision.model_id = "model_v42"
        mock_decision.reason = "Passed tests"

        # Patch the PromotionType import
        mock_promotion_type = MagicMock()
        mock_promotion_type.STAGING = mock_decision.promotion_type

        with patch(
            "app.training.promotion_controller.PromotionType",
            mock_promotion_type,
            create=True,
        ):
            # Call update method - it will use force=True
            integration._update_state_from_decision(mock_decision)

        # Model should be registered (with default TRAINED state)
        record = lifecycle.get_record("model_v42")
        assert record is not None
