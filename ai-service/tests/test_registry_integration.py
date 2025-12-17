"""End-to-end integration tests for model registry and related components.

Tests the full workflow from model registration through promotion,
rollback, and CMA-ES integration.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from app.training.model_registry import (
    ModelRegistry,
    ModelStage,
    ModelType,
    ModelMetrics,
    TrainingConfig,
)
from app.training.cmaes_registry_integration import (
    register_cmaes_result,
    get_best_heuristic_model,
)
from app.training.rollback_manager import (
    RollbackManager,
    RollbackThresholds,
)


class TestModelLifecycleIntegration:
    """Test complete model lifecycle from development to production to archive."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"
        self.registry = ModelRegistry(self.registry_dir)

        # Create test model file
        self.model_path = Path(self.temp_dir) / "model.pt"
        self.model_path.write_bytes(b"test model weights" * 100)

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_promotion_workflow(self):
        """Test model going through all stages."""
        # Register in development
        model_id, v1 = self.registry.register_model(
            name="Test Model",
            model_path=self.model_path,
            model_type=ModelType.POLICY_VALUE,
            metrics=ModelMetrics(elo=1400, games_played=50),
            initial_stage=ModelStage.DEVELOPMENT,
        )

        model = self.registry.get_model(model_id, v1)
        assert model.stage == ModelStage.DEVELOPMENT

        # Promote to staging
        self.registry.promote(model_id, v1, ModelStage.STAGING, reason="Passed dev tests")
        model = self.registry.get_model(model_id, v1)
        assert model.stage == ModelStage.STAGING

        # Promote to production
        self.registry.promote(model_id, v1, ModelStage.PRODUCTION, reason="Passed staging eval")
        model = self.registry.get_model(model_id, v1)
        assert model.stage == ModelStage.PRODUCTION

        # Verify it's the production model
        prod = self.registry.get_production_model()
        assert prod.model_id == model_id
        assert prod.version == v1

        # Archive (retire from production)
        self.registry.promote(model_id, v1, ModelStage.ARCHIVED, reason="Replaced by newer model")
        model = self.registry.get_model(model_id, v1)
        assert model.stage == ModelStage.ARCHIVED

    def test_production_replacement_flow(self):
        """Test that promoting a new model archives the old production model."""
        # Create and promote first model to production
        model_path1 = Path(self.temp_dir) / "model1.pt"
        model_path1.write_bytes(b"model 1 weights" * 100)

        m1_id, m1_v = self.registry.register_model(
            name="Model 1",
            model_path=model_path1,
            metrics=ModelMetrics(elo=1450),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="test_model",
        )
        self.registry.promote(m1_id, m1_v, ModelStage.STAGING)
        self.registry.promote(m1_id, m1_v, ModelStage.PRODUCTION)

        # Verify model 1 is in production
        prod = self.registry.get_production_model()
        assert prod.version == m1_v

        # Create and promote second model
        model_path2 = Path(self.temp_dir) / "model2.pt"
        model_path2.write_bytes(b"model 2 weights" * 100)

        m2_id, m2_v = self.registry.register_model(
            name="Model 2",
            model_path=model_path2,
            metrics=ModelMetrics(elo=1500),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="test_model",  # Same model ID, new version
        )
        self.registry.promote(m2_id, m2_v, ModelStage.STAGING)
        self.registry.promote(m2_id, m2_v, ModelStage.PRODUCTION)

        # Model 2 should now be production
        prod = self.registry.get_production_model()
        assert prod.version == m2_v

        # Model 1 should be archived
        old_model = self.registry.get_model(m1_id, m1_v)
        assert old_model.stage == ModelStage.ARCHIVED

    def test_metrics_update_through_lifecycle(self):
        """Test that metrics can be updated at any stage."""
        model_id, version = self.registry.register_model(
            name="Metric Test Model",
            model_path=self.model_path,
            metrics=ModelMetrics(elo=1400, games_played=0),
        )

        # Update metrics in development
        self.registry.update_metrics(model_id, version, ModelMetrics(
            elo=1420, games_played=50, win_rate=0.52
        ))

        self.registry.promote(model_id, version, ModelStage.STAGING)

        # Update metrics in staging
        self.registry.update_metrics(model_id, version, ModelMetrics(
            elo=1450, games_played=150, win_rate=0.55
        ))

        self.registry.promote(model_id, version, ModelStage.PRODUCTION)

        # Update metrics in production
        self.registry.update_metrics(model_id, version, ModelMetrics(
            elo=1480, games_played=500, win_rate=0.58
        ))

        # Verify final metrics
        model = self.registry.get_model(model_id, version)
        assert model.metrics.elo == 1480
        assert model.metrics.games_played == 500
        assert model.metrics.win_rate == 0.58


class TestRollbackIntegration:
    """Test rollback functionality with model registry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"
        self.registry = ModelRegistry(self.registry_dir)
        self.rollback_manager = RollbackManager(
            self.registry,
            thresholds=RollbackThresholds(
                elo_drop_threshold=30,
                min_games_for_evaluation=10,
            ),
            history_path=Path(self.temp_dir) / "rollback_history.json",
        )

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rollback_to_previous_version(self):
        """Test rolling back from current production to previous version."""
        # Create and promote first model to production
        model_path1 = Path(self.temp_dir) / "model1.pt"
        model_path1.write_bytes(b"model 1 weights" * 100)

        m1_id, m1_v = self.registry.register_model(
            name="Model",
            model_path=model_path1,
            metrics=ModelMetrics(elo=1500, games_played=100),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="rollback_test",
        )
        self.registry.promote(m1_id, m1_v, ModelStage.STAGING)
        self.registry.promote(m1_id, m1_v, ModelStage.PRODUCTION)

        # Create and promote second model (archives first)
        model_path2 = Path(self.temp_dir) / "model2.pt"
        model_path2.write_bytes(b"model 2 weights" * 100)

        m2_id, m2_v = self.registry.register_model(
            name="Model",
            model_path=model_path2,
            metrics=ModelMetrics(elo=1450, games_played=50),  # Worse!
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="rollback_test",
        )
        self.registry.promote(m2_id, m2_v, ModelStage.STAGING)
        self.registry.promote(m2_id, m2_v, ModelStage.PRODUCTION)

        # Verify current state
        prod = self.registry.get_production_model()
        assert prod.version == m2_v
        old = self.registry.get_model(m1_id, m1_v)
        assert old.stage == ModelStage.ARCHIVED

        # Perform rollback
        result = self.rollback_manager.rollback_model(
            "rollback_test",
            to_version=m1_v,
            reason="Performance regression",
        )

        assert result["success"]
        assert result["to_version"] == m1_v

        # Verify model 1 is back in production
        prod = self.registry.get_production_model()
        assert prod.version == m1_v
        assert prod.stage == ModelStage.PRODUCTION

        # Model 2 should be archived
        new_m2 = self.registry.get_model(m2_id, m2_v)
        assert new_m2.stage == ModelStage.ARCHIVED

    def test_performance_check_triggers_rollback(self):
        """Test that performance degradation detection works."""
        # Create model and set baseline
        model_path = Path(self.temp_dir) / "model.pt"
        model_path.write_bytes(b"model weights" * 100)

        model_id, version = self.registry.register_model(
            name="Perf Test",
            model_path=model_path,
            metrics=ModelMetrics(elo=1500, games_played=100),
            initial_stage=ModelStage.DEVELOPMENT,
            model_id="perf_test",
        )
        self.registry.promote(model_id, version, ModelStage.STAGING)
        self.registry.promote(model_id, version, ModelStage.PRODUCTION)

        # Set baseline
        self.rollback_manager.set_baseline(model_id, {
            "elo": 1500,
            "games_played": 100,
        })

        # Check with good performance
        is_degraded, reason = self.rollback_manager.check_performance(
            model_id,
            {"elo": 1490, "games_played": 150},
        )
        assert not is_degraded

        # Check with degraded performance
        is_degraded, reason = self.rollback_manager.check_performance(
            model_id,
            {"elo": 1460, "games_played": 150},  # 40 point drop
        )
        assert is_degraded
        assert "Elo dropped" in reason


class TestCMAESRegistryIntegration:
    """Test CMA-ES and model registry integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cmaes_registers_heuristic_model(self):
        """Test that CMA-ES results are properly registered."""
        weights_path = Path(self.temp_dir) / "cmaes_weights.json"
        with open(weights_path, "w") as f:
            json.dump({
                "weights": {"material": 1.0, "position": 0.5},
                "fitness": 0.75,
                "generation": 50,
            }, f)

        model_id, version = register_cmaes_result(
            weights_path=weights_path,
            board_type="square8",
            num_players=2,
            fitness=0.75,
            generation=50,
            cmaes_config={
                "population_size": 20,
                "sigma": 0.5,
                "generations": 50,
            },
            registry_dir=self.registry_dir,
            auto_promote=False,
        )

        assert model_id == "heuristic_square8_2p"
        assert version == 1

        # Verify model type
        registry = ModelRegistry(self.registry_dir)
        model = registry.get_model(model_id, version)
        assert model.model_type == ModelType.HEURISTIC

    def test_cmaes_auto_promotes_on_improvement(self):
        """Test auto-promotion when fitness improves significantly."""
        # First run
        weights1 = Path(self.temp_dir) / "weights1.json"
        with open(weights1, "w") as f:
            json.dump({"weights": {}, "fitness": 0.60}, f)

        register_cmaes_result(
            weights_path=weights1,
            board_type="square8",
            num_players=2,
            fitness=0.60,
            generation=25,
            registry_dir=self.registry_dir,
            auto_promote=False,
        )

        # Second run with better fitness
        weights2 = Path(self.temp_dir) / "weights2.json"
        with open(weights2, "w") as f:
            json.dump({"weights": {}, "fitness": 0.75}, f)

        model_id, version = register_cmaes_result(
            weights_path=weights2,
            board_type="square8",
            num_players=2,
            fitness=0.75,
            generation=50,
            registry_dir=self.registry_dir,
            auto_promote=True,
            min_fitness_improvement=0.05,
        )

        # Should be auto-promoted to staging
        registry = ModelRegistry(self.registry_dir)
        model = registry.get_model(model_id, version)
        assert model.stage == ModelStage.STAGING

    def test_get_best_heuristic_model(self):
        """Test retrieving best heuristic model."""
        # Register multiple versions
        for i, fitness in enumerate([0.55, 0.70, 0.65]):
            weights = Path(self.temp_dir) / f"weights_{i}.json"
            with open(weights, "w") as f:
                json.dump({"weights": {}, "fitness": fitness}, f)

            register_cmaes_result(
                weights_path=weights,
                board_type="square8",
                num_players=2,
                fitness=fitness,
                generation=i * 25,
                registry_dir=self.registry_dir,
                auto_promote=False,
            )

        # Get best model
        best = get_best_heuristic_model(
            board_type="square8",
            num_players=2,
            registry_dir=self.registry_dir,
        )

        assert best is not None
        assert best["metrics"]["win_rate"] == 0.70


class TestStageTransitionHistory:
    """Test stage transition history tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"
        self.registry = ModelRegistry(self.registry_dir)

        self.model_path = Path(self.temp_dir) / "model.pt"
        self.model_path.write_bytes(b"test model" * 100)

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_history_tracks_all_transitions(self):
        """Test that all stage transitions are recorded."""
        model_id, version = self.registry.register_model(
            name="History Test",
            model_path=self.model_path,
            initial_stage=ModelStage.DEVELOPMENT,
        )

        # Go through stages
        self.registry.promote(model_id, version, ModelStage.STAGING, "Eval start")
        self.registry.promote(model_id, version, ModelStage.PRODUCTION, "Eval passed")
        self.registry.promote(model_id, version, ModelStage.ARCHIVED, "Replaced")

        # Check history
        history = self.registry.get_stage_history(model_id, version)

        assert len(history) == 4  # Initial + 3 transitions
        assert history[0]["to_stage"] == "development"
        assert history[1]["to_stage"] == "staging"
        assert history[2]["to_stage"] == "production"
        assert history[3]["to_stage"] == "archived"

        # Check reasons are recorded
        assert "Eval start" in history[1]["reason"]
        assert "Eval passed" in history[2]["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
