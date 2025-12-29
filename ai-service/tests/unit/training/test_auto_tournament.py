"""Unit tests for app.training.auto_tournament module.

Tests cover:
- RegisteredModel dataclass
- MatchResult dataclass
- TournamentResult dataclass
- ChallengerResult dataclass
- Statistical functions (binomial p-value, Elo calculations)
- AutoTournamentPipeline class
- Helper functions
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.training.auto_tournament import (
    AutoTournamentPipeline,
    ChallengerResult,
    MatchResult,
    RegisteredModel,
    TournamentResult,
    _binomial_coefficient,
    calculate_binomial_p_value,
    calculate_elo_change,
    expected_score,
    should_promote,
)
from app.training.model_versioning import LegacyCheckpointError, ModelMetadata


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metadata() -> ModelMetadata:
    """Create sample model metadata for testing."""
    return ModelMetadata(
        model_class="HexNeuralNet_v5",
        architecture_version="5.0.0",
        checksum="abc123def456",
        created_at=datetime.now(timezone.utc).isoformat(),
        training_info={"epochs": 100, "batch_size": 256},
    )


@pytest.fixture
def sample_metadata_2() -> ModelMetadata:
    """Create second sample metadata with different checksum."""
    return ModelMetadata(
        model_class="HexNeuralNet_v5",
        architecture_version="5.1.0",
        checksum="xyz789uvw012",
        created_at=datetime.now(timezone.utc).isoformat(),
        training_info={"epochs": 150, "batch_size": 512},
    )


@pytest.fixture
def temp_dirs():
    """Create temporary directories for models and results."""
    with tempfile.TemporaryDirectory() as models_dir:
        with tempfile.TemporaryDirectory() as results_dir:
            yield models_dir, results_dir


@pytest.fixture
def mock_model_file(temp_dirs) -> str:
    """Create a mock model file."""
    models_dir, _ = temp_dirs
    model_path = os.path.join(models_dir, "test_model.pth")
    # Create empty file
    with open(model_path, "wb") as f:
        f.write(b"mock model data")
    return model_path


@pytest.fixture
def mock_model_file_2(temp_dirs) -> str:
    """Create a second mock model file."""
    models_dir, _ = temp_dirs
    model_path = os.path.join(models_dir, "test_model_2.pth")
    with open(model_path, "wb") as f:
        f.write(b"mock model data 2")
    return model_path


# =============================================================================
# Tests for RegisteredModel
# =============================================================================


class TestRegisteredModel:
    """Tests for RegisteredModel dataclass."""

    def test_creation_with_defaults(self, sample_metadata):
        """Test creating RegisteredModel with default values."""
        model = RegisteredModel(
            model_id="test_model_123",
            model_path="/path/to/model.pth",
            metadata=sample_metadata,
        )

        assert model.model_id == "test_model_123"
        assert model.model_path == "/path/to/model.pth"
        assert model.elo_rating == 1500.0
        assert model.is_champion is False
        assert model.games_played == 0
        assert model.wins == 0
        assert model.losses == 0
        assert model.draws == 0
        # registered_at should be auto-set
        assert model.registered_at != ""

    def test_creation_with_custom_values(self, sample_metadata):
        """Test creating RegisteredModel with custom values."""
        model = RegisteredModel(
            model_id="champion_v1",
            model_path="/models/champion.pth",
            metadata=sample_metadata,
            elo_rating=1650.0,
            is_champion=True,
            games_played=100,
            wins=60,
            losses=30,
            draws=10,
        )

        assert model.elo_rating == 1650.0
        assert model.is_champion is True
        assert model.games_played == 100
        assert model.wins == 60
        assert model.losses == 30
        assert model.draws == 10

    def test_to_dict(self, sample_metadata):
        """Test serialization to dictionary."""
        model = RegisteredModel(
            model_id="test_123",
            model_path="/path/model.pth",
            metadata=sample_metadata,
            elo_rating=1600.0,
            games_played=50,
            wins=30,
            losses=15,
            draws=5,
        )

        data = model.to_dict()

        assert data["model_id"] == "test_123"
        assert data["model_path"] == "/path/model.pth"
        assert data["elo_rating"] == 1600.0
        assert data["games_played"] == 50
        assert "metadata" in data
        assert data["metadata"]["model_class"] == "HexNeuralNet_v5"

    def test_from_dict(self, sample_metadata):
        """Test deserialization from dictionary."""
        data = {
            "model_id": "restored_model",
            "model_path": "/path/restored.pth",
            "metadata": sample_metadata.to_dict(),
            "elo_rating": 1550.0,
            "registered_at": "2025-12-29T10:00:00+00:00",
            "is_champion": False,
            "games_played": 25,
            "wins": 15,
            "losses": 8,
            "draws": 2,
        }

        model = RegisteredModel.from_dict(data)

        assert model.model_id == "restored_model"
        assert model.elo_rating == 1550.0
        assert model.games_played == 25
        assert model.metadata.model_class == "HexNeuralNet_v5"

    def test_win_rate_with_games(self, sample_metadata):
        """Test win rate calculation with games played."""
        model = RegisteredModel(
            model_id="test",
            model_path="/path",
            metadata=sample_metadata,
            games_played=100,
            wins=65,
            losses=25,
            draws=10,
        )

        assert model.win_rate == 65.0

    def test_win_rate_zero_games(self, sample_metadata):
        """Test win rate returns 0 when no games played."""
        model = RegisteredModel(
            model_id="test",
            model_path="/path",
            metadata=sample_metadata,
            games_played=0,
        )

        assert model.win_rate == 0.0


# =============================================================================
# Tests for MatchResult
# =============================================================================


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_creation_with_winner(self):
        """Test creating MatchResult with a winner."""
        result = MatchResult(
            model_a_id="model_a",
            model_b_id="model_b",
            winner_id="model_a",
            victory_reason="territory",
            game_number=1,
        )

        assert result.model_a_id == "model_a"
        assert result.model_b_id == "model_b"
        assert result.winner_id == "model_a"
        assert result.victory_reason == "territory"
        assert result.game_number == 1
        assert result.played_at != ""

    def test_creation_with_draw(self):
        """Test creating MatchResult for a draw."""
        result = MatchResult(
            model_a_id="model_a",
            model_b_id="model_b",
            winner_id=None,
            victory_reason="draw",
            game_number=5,
        )

        assert result.winner_id is None
        assert result.victory_reason == "draw"


# =============================================================================
# Tests for TournamentResult
# =============================================================================


class TestTournamentResult:
    """Tests for TournamentResult dataclass."""

    def test_creation(self):
        """Test creating TournamentResult."""
        matches = [
            MatchResult("a", "b", "a", "territory", 1),
            MatchResult("a", "b", "b", "rings", 2),
        ]

        result = TournamentResult(
            tournament_id="test_tournament_001",
            participants=["model_a", "model_b"],
            matches=matches,
            final_elo_ratings={"model_a": 1520.0, "model_b": 1480.0},
            final_standings=[("model_a", 1520.0), ("model_b", 1480.0)],
            started_at="2025-12-29T10:00:00+00:00",
        )

        assert result.tournament_id == "test_tournament_001"
        assert len(result.participants) == 2
        assert len(result.matches) == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        matches = [
            MatchResult("a", "b", "a", "territory", 1),
        ]

        result = TournamentResult(
            tournament_id="test_001",
            participants=["a", "b"],
            matches=matches,
            final_elo_ratings={"a": 1510.0, "b": 1490.0},
            final_standings=[("a", 1510.0), ("b", 1490.0)],
            started_at="2025-12-29T10:00:00+00:00",
            finished_at="2025-12-29T11:00:00+00:00",
            victory_reasons={"territory": 1},
        )

        data = result.to_dict()

        assert data["tournament_id"] == "test_001"
        assert len(data["matches"]) == 1
        assert data["victory_reasons"]["territory"] == 1


# =============================================================================
# Tests for ChallengerResult
# =============================================================================


class TestChallengerResult:
    """Tests for ChallengerResult dataclass."""

    def test_creation(self):
        """Test creating ChallengerResult."""
        result = ChallengerResult(
            challenger_id="new_model",
            champion_id="champion_v1",
            challenger_wins=35,
            champion_wins=15,
            draws=0,
            total_games=50,
            challenger_win_rate=0.70,
            champion_win_rate=0.30,
            statistical_p_value=0.001,
            is_statistically_significant=True,
            challenger_final_elo=1600.0,
            champion_final_elo=1480.0,
            should_promote=True,
            victory_reasons={"territory": 35, "rings": 15},
        )

        assert result.challenger_id == "new_model"
        assert result.challenger_wins == 35
        assert result.should_promote is True
        assert result.evaluation_time != ""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ChallengerResult(
            challenger_id="challenger",
            champion_id="champion",
            challenger_wins=25,
            champion_wins=25,
            draws=0,
            total_games=50,
            challenger_win_rate=0.50,
            champion_win_rate=0.50,
            statistical_p_value=0.50,
            is_statistically_significant=False,
            challenger_final_elo=1500.0,
            champion_final_elo=1500.0,
            should_promote=False,
        )

        data = result.to_dict()

        assert data["challenger_id"] == "challenger"
        assert data["total_games"] == 50
        assert data["should_promote"] is False


# =============================================================================
# Tests for Statistical Functions
# =============================================================================


class TestBinomialCoefficient:
    """Tests for _binomial_coefficient function."""

    def test_basic_cases(self):
        """Test basic binomial coefficient calculations."""
        assert _binomial_coefficient(5, 0) == 1
        assert _binomial_coefficient(5, 5) == 1
        assert _binomial_coefficient(5, 1) == 5
        assert _binomial_coefficient(5, 2) == 10
        assert _binomial_coefficient(5, 3) == 10

    def test_symmetry(self):
        """Test that C(n,k) = C(n, n-k)."""
        assert _binomial_coefficient(10, 3) == _binomial_coefficient(10, 7)
        assert _binomial_coefficient(8, 2) == _binomial_coefficient(8, 6)

    def test_edge_cases(self):
        """Test edge cases."""
        assert _binomial_coefficient(0, 0) == 1
        assert _binomial_coefficient(5, -1) == 0
        assert _binomial_coefficient(5, 6) == 0


class TestBinomialPValue:
    """Tests for calculate_binomial_p_value function."""

    def test_zero_trials(self):
        """Test p-value is 1.0 with zero trials."""
        p_value = calculate_binomial_p_value(0, 0)
        assert p_value == 1.0

    def test_all_wins(self):
        """Test p-value when all trials are successes."""
        # 10 wins out of 10 under null p=0.5 is very unlikely
        p_value = calculate_binomial_p_value(10, 10, 0.5)
        assert p_value < 0.01

    def test_half_wins(self):
        """Test p-value when exactly half are wins."""
        # 5 wins out of 10 under null p=0.5
        p_value = calculate_binomial_p_value(5, 10, 0.5)
        # Should be around 0.5 (not significant)
        assert 0.4 < p_value < 0.7

    def test_significance_threshold(self):
        """Test that we can detect significant results."""
        # 35 wins out of 50 games (70% win rate)
        p_value = calculate_binomial_p_value(35, 50, 0.5)
        # Should be significant at p < 0.05
        assert p_value < 0.05


class TestEloChange:
    """Tests for calculate_elo_change function."""

    def test_equal_rating_win(self):
        """Test Elo change when equal-rated player wins."""
        new_a, new_b = calculate_elo_change(1500.0, 1500.0, 1.0, k_factor=32.0)

        # Winner should gain, loser should lose
        assert new_a > 1500.0
        assert new_b < 1500.0
        # Changes should be symmetric
        assert abs((new_a - 1500.0) + (new_b - 1500.0)) < 0.01

    def test_equal_rating_draw(self):
        """Test Elo change when equal-rated players draw."""
        new_a, new_b = calculate_elo_change(1500.0, 1500.0, 0.5, k_factor=32.0)

        # Ratings should remain almost unchanged
        assert abs(new_a - 1500.0) < 0.01
        assert abs(new_b - 1500.0) < 0.01

    def test_underdog_win(self):
        """Test Elo change when lower-rated player wins."""
        # Lower rated (1400) beats higher rated (1600)
        new_a, new_b = calculate_elo_change(1400.0, 1600.0, 1.0, k_factor=32.0)

        # Underdog should gain more than in equal matchup
        assert new_a > 1400.0 + 16  # More than half of k-factor
        assert new_b < 1600.0 - 16

    def test_favorite_win(self):
        """Test Elo change when higher-rated player wins."""
        # Higher rated (1600) beats lower rated (1400)
        new_a, new_b = calculate_elo_change(1600.0, 1400.0, 1.0, k_factor=32.0)

        # Favorite should gain less than in equal matchup
        assert 1600.0 < new_a < 1616.0  # Less than half of k-factor


class TestExpectedScore:
    """Tests for expected_score function."""

    def test_equal_ratings(self):
        """Test expected score for equal ratings."""
        score = expected_score(1500.0, 1500.0)
        assert abs(score - 0.5) < 0.001

    def test_higher_rated_favorite(self):
        """Test that higher rated player is favorite."""
        score = expected_score(1600.0, 1400.0)
        assert score > 0.5
        # 200 point difference should give ~75% expected
        assert 0.7 < score < 0.8

    def test_lower_rated_underdog(self):
        """Test that lower rated player is underdog."""
        score = expected_score(1400.0, 1600.0)
        assert score < 0.5


# =============================================================================
# Tests for AutoTournamentPipeline
# =============================================================================


class TestAutoTournamentPipelineInit:
    """Tests for AutoTournamentPipeline initialization."""

    def test_init_creates_directories(self, temp_dirs):
        """Test that init creates necessary directories."""
        models_dir, results_dir = temp_dirs
        # Remove directories to test creation
        os.rmdir(models_dir)
        os.rmdir(results_dir)

        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        assert os.path.exists(models_dir)
        assert os.path.exists(results_dir)
        assert pipeline.registry_file == os.path.join(
            results_dir, "model_registry.json"
        )

    def test_init_with_custom_registry(self, temp_dirs):
        """Test initialization with custom registry file."""
        models_dir, results_dir = temp_dirs
        custom_registry = os.path.join(results_dir, "custom_registry.json")

        pipeline = AutoTournamentPipeline(
            models_dir, results_dir, registry_file=custom_registry
        )

        assert pipeline.registry_file == custom_registry

    def test_loads_existing_registry(self, temp_dirs, sample_metadata):
        """Test that existing registry is loaded on init."""
        models_dir, results_dir = temp_dirs
        registry_file = os.path.join(results_dir, "model_registry.json")

        # Create a pre-existing registry
        registry_data = {
            "models": {
                "existing_model": {
                    "model_id": "existing_model",
                    "model_path": "/path/to/model.pth",
                    "metadata": sample_metadata.to_dict(),
                    "elo_rating": 1600.0,
                    "registered_at": "2025-12-29T10:00:00+00:00",
                    "is_champion": True,
                    "games_played": 100,
                    "wins": 60,
                    "losses": 30,
                    "draws": 10,
                }
            }
        }

        with open(registry_file, "w") as f:
            json.dump(registry_data, f)

        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        assert "existing_model" in pipeline._models
        assert pipeline._models["existing_model"].elo_rating == 1600.0


class TestAutoTournamentPipelineRegistration:
    """Tests for model registration."""

    def test_register_model_with_metadata(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test registering a model with provided metadata."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id = pipeline.register_model(mock_model_file, metadata=sample_metadata)

        assert model_id in pipeline._models
        model = pipeline._models[model_id]
        assert model.metadata.model_class == "HexNeuralNet_v5"
        # First model should be champion
        assert model.is_champion is True

    def test_register_second_model_not_champion(
        self, temp_dirs, mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test that second registered model is not champion."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)
        model_id_2 = pipeline.register_model(
            mock_model_file_2, metadata=sample_metadata_2
        )

        assert pipeline._models[model_id_2].is_champion is False

    def test_register_model_with_custom_elo(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test registering with custom initial Elo."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id = pipeline.register_model(
            mock_model_file, metadata=sample_metadata, initial_elo=1700.0
        )

        assert pipeline._models[model_id].elo_rating == 1700.0

    def test_register_duplicate_checksum_returns_existing(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test that registering same checksum returns existing model."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id_1 = pipeline.register_model(mock_model_file, metadata=sample_metadata)
        model_id_2 = pipeline.register_model(mock_model_file, metadata=sample_metadata)

        assert model_id_1 == model_id_2
        assert len(pipeline._models) == 1

    def test_register_nonexistent_file_raises(self, temp_dirs, sample_metadata):
        """Test that registering non-existent file raises error."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        with pytest.raises(FileNotFoundError):
            pipeline.register_model("/nonexistent/model.pth", metadata=sample_metadata)

    def test_register_without_metadata_raises_for_legacy(
        self, temp_dirs, mock_model_file
    ):
        """Test that registering without metadata on legacy checkpoint raises."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        with patch.object(
            pipeline._version_manager,
            "get_metadata",
            side_effect=LegacyCheckpointError("/path/model.pth"),
        ):
            with pytest.raises(LegacyCheckpointError):
                pipeline.register_model(mock_model_file)


class TestAutoTournamentPipelineQueries:
    """Tests for model query methods."""

    def test_get_champion(
        self, temp_dirs, mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test getting the current champion."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id_1 = pipeline.register_model(mock_model_file, metadata=sample_metadata)
        pipeline.register_model(mock_model_file_2, metadata=sample_metadata_2)

        champion = pipeline.get_champion()

        assert champion is not None
        assert champion.model_id == model_id_1
        assert champion.is_champion is True

    def test_get_champion_none_when_empty(self, temp_dirs):
        """Test get_champion returns None when no models registered."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        assert pipeline.get_champion() is None

    def test_get_model(self, temp_dirs, mock_model_file, sample_metadata):
        """Test getting a specific model by ID."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id = pipeline.register_model(mock_model_file, metadata=sample_metadata)
        model = pipeline.get_model(model_id)

        assert model is not None
        assert model.model_id == model_id

    def test_get_model_not_found(self, temp_dirs):
        """Test get_model returns None for unknown ID."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        assert pipeline.get_model("nonexistent") is None

    def test_list_models_sorted_by_elo(
        self, temp_dirs, mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test list_models returns models sorted by Elo."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        # Register with different Elos
        model_id_1 = pipeline.register_model(
            mock_model_file, metadata=sample_metadata, initial_elo=1400.0
        )
        model_id_2 = pipeline.register_model(
            mock_model_file_2, metadata=sample_metadata_2, initial_elo=1600.0
        )

        models = pipeline.list_models()

        assert len(models) == 2
        assert models[0].model_id == model_id_2  # Higher Elo first
        assert models[1].model_id == model_id_1


class TestAutoTournamentPipelineTournament:
    """Tests for tournament execution."""

    def test_run_tournament_requires_two_participants(self, temp_dirs):
        """Test that tournament requires at least 2 participants."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        with pytest.raises(ValueError, match="Need at least 2 participants"):
            pipeline.run_tournament()

    @patch("app.training.auto_tournament.Tournament")
    def test_run_tournament_basic(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test basic tournament execution."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        # Register models
        pipeline.register_model(mock_model_file, metadata=sample_metadata)
        pipeline.register_model(mock_model_file_2, metadata=sample_metadata_2)

        # Mock tournament run
        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1520.0, "B": 1480.0}
        mock_tournament.victory_reasons = {"territory": 3, "rings": 2}
        mock_tournament.run.return_value = {"A": 3, "B": 2, "Draw": 0}
        mock_tournament_class.return_value = mock_tournament

        result = pipeline.run_tournament(games_per_match=5)

        assert result.tournament_id.startswith("tournament_")
        assert len(result.participants) == 2
        assert len(result.matches) == 5

    @patch("app.training.auto_tournament.Tournament")
    def test_run_tournament_updates_elo(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test that tournament updates Elo ratings."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id_1 = pipeline.register_model(
            mock_model_file, metadata=sample_metadata, initial_elo=1500.0
        )
        model_id_2 = pipeline.register_model(
            mock_model_file_2, metadata=sample_metadata_2, initial_elo=1500.0
        )

        # Use a real dict that gets modified by run()
        ratings_dict = {"A": 1500.0, "B": 1500.0}

        def run_side_effect():
            # Simulate tournament updating ratings after games
            ratings_dict["A"] = 1540.0
            ratings_dict["B"] = 1460.0
            return {"A": 7, "B": 3, "Draw": 0}

        mock_tournament = MagicMock()
        mock_tournament.ratings = ratings_dict
        mock_tournament.victory_reasons = {}
        mock_tournament.run.side_effect = run_side_effect
        mock_tournament_class.return_value = mock_tournament

        pipeline.run_tournament(games_per_match=10)

        assert pipeline._models[model_id_1].elo_rating == 1540.0
        assert pipeline._models[model_id_2].elo_rating == 1460.0


class TestAutoTournamentPipelineChallenger:
    """Tests for challenger evaluation."""

    def test_evaluate_challenger_no_champion_raises(self, temp_dirs, mock_model_file):
        """Test that evaluating challenger without champion raises."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        with pytest.raises(ValueError, match="No champion registered"):
            pipeline.evaluate_challenger(mock_model_file)

    def test_evaluate_challenger_nonexistent_raises(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test that evaluating nonexistent challenger raises."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        with pytest.raises(FileNotFoundError):
            pipeline.evaluate_challenger("/nonexistent/challenger.pth")

    @patch("app.training.auto_tournament.Tournament")
    def test_evaluate_challenger_decisive_victory(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test challenger evaluation with decisive victory."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        # Register champion
        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        # Use a real dict that gets modified by run()
        ratings_dict = {"A": 1500.0, "B": 1500.0}

        def run_side_effect():
            # Simulate tournament updating ratings - challenger wins decisively
            ratings_dict["A"] = 1600.0  # Challenger (A) gains rating
            ratings_dict["B"] = 1400.0  # Champion (B) loses rating
            return {"A": 35, "B": 15, "Draw": 0}

        mock_tournament = MagicMock()
        mock_tournament.ratings = ratings_dict
        mock_tournament.victory_reasons = {"territory": 35, "rings": 15}
        mock_tournament.run.side_effect = run_side_effect
        mock_tournament_class.return_value = mock_tournament

        result = pipeline.evaluate_challenger(
            mock_model_file_2,
            games=50,
            challenger_metadata=sample_metadata_2,
        )

        assert result.challenger_wins == 35
        assert result.champion_wins == 15
        assert result.challenger_win_rate == 0.7
        assert result.should_promote is True  # 70% > 55% and significant

    @patch("app.training.auto_tournament.Tournament")
    def test_evaluate_challenger_close_match(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test challenger evaluation with close match."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        # Mock close match
        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1510.0, "B": 1490.0}
        mock_tournament.victory_reasons = {}
        mock_tournament.run.return_value = {"A": 26, "B": 24, "Draw": 0}
        mock_tournament_class.return_value = mock_tournament

        result = pipeline.evaluate_challenger(
            mock_model_file_2,
            games=50,
            challenger_metadata=sample_metadata_2,
        )

        assert result.challenger_win_rate == 0.52
        assert result.should_promote is False  # 52% < 55%


class TestAutoTournamentPipelinePromotion:
    """Tests for champion promotion."""

    def test_promote_champion(
        self, temp_dirs, mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test promoting a model to champion."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id_1 = pipeline.register_model(mock_model_file, metadata=sample_metadata)
        model_id_2 = pipeline.register_model(
            mock_model_file_2, metadata=sample_metadata_2
        )

        # First model should be champion
        assert pipeline._models[model_id_1].is_champion is True
        assert pipeline._models[model_id_2].is_champion is False

        # Promote second model
        pipeline.promote_champion(model_id_2)

        assert pipeline._models[model_id_1].is_champion is False
        assert pipeline._models[model_id_2].is_champion is True

    def test_promote_nonexistent_raises(self, temp_dirs):
        """Test promoting non-existent model raises."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        with pytest.raises(ValueError, match="Model not found"):
            pipeline.promote_champion("nonexistent_model")

    def test_should_promote_criteria(self, temp_dirs):
        """Test _should_promote logic."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        # Should promote: high win rate, significant, higher elo
        assert pipeline._should_promote(
            challenger_win_rate=0.65,
            is_significant=True,
            challenger_elo=1550.0,
            champion_elo=1500.0,
        ) is True

        # Should not: win rate too low
        assert pipeline._should_promote(
            challenger_win_rate=0.50,
            is_significant=True,
            challenger_elo=1550.0,
            champion_elo=1500.0,
        ) is False

        # Should not: not significant
        assert pipeline._should_promote(
            challenger_win_rate=0.65,
            is_significant=False,
            challenger_elo=1550.0,
            champion_elo=1500.0,
        ) is False

        # Should not: lower elo
        assert pipeline._should_promote(
            challenger_win_rate=0.65,
            is_significant=True,
            challenger_elo=1490.0,
            champion_elo=1500.0,
        ) is False


class TestAutoTournamentPipelineRankings:
    """Tests for Elo rankings."""

    def test_get_elo_rankings(
        self, temp_dirs, mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test getting Elo rankings."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id_1 = pipeline.register_model(
            mock_model_file, metadata=sample_metadata, initial_elo=1400.0
        )
        model_id_2 = pipeline.register_model(
            mock_model_file_2, metadata=sample_metadata_2, initial_elo=1600.0
        )

        rankings = pipeline.get_elo_rankings()

        assert len(rankings) == 2
        assert rankings[0] == (model_id_2, 1600.0)
        assert rankings[1] == (model_id_1, 1400.0)


class TestAutoTournamentPipelineReports:
    """Tests for report generation."""

    def test_generate_report(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test generating markdown report."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        report = pipeline.generate_report()

        assert "# AI Tournament Performance Report" in report
        assert "## Current Champion" in report
        assert "## Elo Rankings" in report
        assert "HexNeuralNet_v5" in report

    def test_generate_report_no_champion(self, temp_dirs):
        """Test report generation with no champion."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        report = pipeline.generate_report()

        assert "*No champion registered*" in report

    def test_save_report(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test saving report to file."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        report_path = pipeline.save_report("test_report.md")

        assert os.path.exists(report_path)
        with open(report_path) as f:
            content = f.read()
        assert "# AI Tournament Performance Report" in content


# =============================================================================
# Tests for Helper Functions
# =============================================================================


class TestShouldPromoteHelper:
    """Tests for should_promote helper function."""

    def test_should_promote_true(self):
        """Test should_promote returns True when result indicates promotion."""
        result = ChallengerResult(
            challenger_id="challenger",
            champion_id="champion",
            challenger_wins=35,
            champion_wins=15,
            draws=0,
            total_games=50,
            challenger_win_rate=0.70,
            champion_win_rate=0.30,
            statistical_p_value=0.001,
            is_statistically_significant=True,
            challenger_final_elo=1600.0,
            champion_final_elo=1400.0,
            should_promote=True,
        )

        assert should_promote(result) is True

    def test_should_promote_false(self):
        """Test should_promote returns False when result indicates no promotion."""
        result = ChallengerResult(
            challenger_id="challenger",
            champion_id="champion",
            challenger_wins=25,
            champion_wins=25,
            draws=0,
            total_games=50,
            challenger_win_rate=0.50,
            champion_win_rate=0.50,
            statistical_p_value=0.50,
            is_statistically_significant=False,
            challenger_final_elo=1500.0,
            champion_final_elo=1500.0,
            should_promote=False,
        )

        assert should_promote(result) is False


# =============================================================================
# Tests for Registry Persistence
# =============================================================================


class TestRegistryPersistence:
    """Tests for registry save/load functionality."""

    def test_save_and_load_registry(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test that registry is properly saved and loaded."""
        models_dir, results_dir = temp_dirs

        # Create pipeline and register model
        pipeline1 = AutoTournamentPipeline(models_dir, results_dir)
        model_id = pipeline1.register_model(
            mock_model_file, metadata=sample_metadata, initial_elo=1550.0
        )

        # Create new pipeline instance (should load registry)
        pipeline2 = AutoTournamentPipeline(models_dir, results_dir)

        assert model_id in pipeline2._models
        assert pipeline2._models[model_id].elo_rating == 1550.0

    def test_registry_file_structure(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test that registry JSON has correct structure."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        with open(pipeline.registry_file) as f:
            data = json.load(f)

        assert "models" in data
        assert "updated_at" in data
        assert len(data["models"]) == 1


# =============================================================================
# Tests for Tournament Result Saving
# =============================================================================


class TestTournamentResultSaving:
    """Tests for tournament result persistence."""

    @patch("app.training.auto_tournament.Tournament")
    def test_saves_tournament_result(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test that tournament results are saved to file."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)
        pipeline.register_model(mock_model_file_2, metadata=sample_metadata_2)

        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1520.0, "B": 1480.0}
        mock_tournament.victory_reasons = {}
        mock_tournament.run.return_value = {"A": 3, "B": 2, "Draw": 0}
        mock_tournament_class.return_value = mock_tournament

        result = pipeline.run_tournament(games_per_match=5)

        # Check result file exists
        result_path = os.path.join(results_dir, f"{result.tournament_id}.json")
        assert os.path.exists(result_path)

        with open(result_path) as f:
            saved_data = json.load(f)

        assert saved_data["tournament_id"] == result.tournament_id
        assert len(saved_data["participants"]) == 2


# =============================================================================
# Additional Tests for Edge Cases and Error Conditions
# =============================================================================


class TestBinomialCoefficientAdvanced:
    """Additional tests for binomial coefficient edge cases."""

    def test_large_values(self):
        """Test binomial coefficient with larger values."""
        # C(20, 10) = 184756
        assert _binomial_coefficient(20, 10) == 184756
        # C(15, 5) = 3003
        assert _binomial_coefficient(15, 5) == 3003

    def test_pascal_identity(self):
        """Test Pascal's identity: C(n,k) = C(n-1,k-1) + C(n-1,k)."""
        for n in range(5, 15):
            for k in range(1, n):
                assert _binomial_coefficient(n, k) == (
                    _binomial_coefficient(n - 1, k - 1) +
                    _binomial_coefficient(n - 1, k)
                )


class TestBinomialPValueAdvanced:
    """Additional tests for binomial p-value edge cases."""

    def test_single_trial_win(self):
        """Test p-value with single trial that is a win."""
        # 1 win in 1 trial under p=0.5 gives p=0.5
        p_value = calculate_binomial_p_value(1, 1, 0.5)
        assert abs(p_value - 0.5) < 0.001

    def test_single_trial_loss(self):
        """Test p-value with single trial that is a loss."""
        # 0 wins in 1 trial under p=0.5 gives p=1.0
        p_value = calculate_binomial_p_value(0, 1, 0.5)
        assert p_value == 1.0

    def test_extreme_null_probability(self):
        """Test p-value with extreme null probabilities."""
        # 5 wins out of 10 with p=0.1 (very unlikely)
        p_value = calculate_binomial_p_value(5, 10, 0.1)
        assert p_value < 0.001

        # 5 wins out of 10 with p=0.9 (expected to have more)
        p_value = calculate_binomial_p_value(5, 10, 0.9)
        assert p_value > 0.95


class TestEloChangeAdvanced:
    """Additional tests for Elo change calculations."""

    def test_extreme_rating_difference(self):
        """Test Elo change with extreme rating difference."""
        # Very large rating difference
        new_a, new_b = calculate_elo_change(2000.0, 1000.0, 0.0, k_factor=32.0)
        # Lower rated upset - large gain
        assert new_b > 1000.0 + 30  # Huge gain for 1000-point upset

    def test_zero_k_factor(self):
        """Test Elo change with zero k-factor (no rating change)."""
        new_a, new_b = calculate_elo_change(1500.0, 1500.0, 1.0, k_factor=0.0)
        assert new_a == 1500.0
        assert new_b == 1500.0

    def test_small_k_factor(self):
        """Test Elo change with small k-factor."""
        new_a, new_b = calculate_elo_change(1500.0, 1500.0, 1.0, k_factor=1.0)
        # Changes should be very small (0.5 each)
        assert abs(new_a - 1500.5) < 0.01
        assert abs(new_b - 1499.5) < 0.01


class TestExpectedScoreAdvanced:
    """Additional tests for expected score calculations."""

    def test_very_large_difference(self):
        """Test expected score with 1000+ point difference."""
        # 1000 point difference
        exp = expected_score(2500.0, 1500.0)
        assert exp > 0.99

        exp = expected_score(1500.0, 2500.0)
        assert exp < 0.01

    def test_moderate_difference(self):
        """Test expected score with moderate rating difference."""
        # 100 point difference
        exp = expected_score(1600.0, 1500.0)
        assert 0.6 < exp < 0.7  # About 64%


class TestRegisteredModelAdvanced:
    """Additional tests for RegisteredModel edge cases."""

    def test_serialization_roundtrip(self, sample_metadata):
        """Test complete serialization roundtrip."""
        original = RegisteredModel(
            model_id="roundtrip_test",
            model_path="/path/to/model.pth",
            metadata=sample_metadata,
            elo_rating=1623.5,
            registered_at="2025-12-29T10:30:00+00:00",
            is_champion=True,
            games_played=150,
            wins=90,
            losses=50,
            draws=10,
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = RegisteredModel.from_dict(data)

        assert restored.model_id == original.model_id
        assert restored.model_path == original.model_path
        assert restored.elo_rating == original.elo_rating
        assert restored.is_champion == original.is_champion
        assert restored.games_played == original.games_played
        assert restored.wins == original.wins
        assert restored.losses == original.losses
        assert restored.draws == original.draws

    def test_win_rate_all_draws(self, sample_metadata):
        """Test win rate when all games are draws."""
        model = RegisteredModel(
            model_id="draws_test",
            model_path="/path",
            metadata=sample_metadata,
            games_played=100,
            wins=0,
            losses=0,
            draws=100,
        )
        assert model.win_rate == 0.0


class TestMatchResultAdvanced:
    """Additional tests for MatchResult edge cases."""

    def test_custom_played_at(self):
        """Test MatchResult with custom played_at time."""
        custom_time = "2025-01-01T12:30:00+00:00"
        result = MatchResult(
            model_a_id="a",
            model_b_id="b",
            winner_id="a",
            victory_reason="territory",
            game_number=5,
            played_at=custom_time,
        )
        assert result.played_at == custom_time


class TestTournamentResultAdvanced:
    """Additional tests for TournamentResult edge cases."""

    def test_empty_matches(self):
        """Test TournamentResult with no matches."""
        result = TournamentResult(
            tournament_id="empty_001",
            participants=["a", "b"],
            matches=[],
            final_elo_ratings={"a": 1500.0, "b": 1500.0},
            final_standings=[("a", 1500.0), ("b", 1500.0)],
            started_at="2025-12-29T10:00:00+00:00",
        )
        data = result.to_dict()
        assert len(data["matches"]) == 0

    def test_many_participants(self):
        """Test TournamentResult with many participants."""
        participants = [f"model_{i}" for i in range(10)]
        result = TournamentResult(
            tournament_id="many_participants",
            participants=participants,
            matches=[],
            final_elo_ratings={p: 1500.0 + i * 10 for i, p in enumerate(participants)},
            final_standings=[(p, 1500.0 + (9 - i) * 10) for i, p in enumerate(participants)],
            started_at="2025-12-29T10:00:00+00:00",
        )
        assert len(result.participants) == 10


class TestChallengerResultAdvanced:
    """Additional tests for ChallengerResult edge cases."""

    def test_all_draws_challenge(self):
        """Test ChallengerResult when all games are draws."""
        result = ChallengerResult(
            challenger_id="challenger",
            champion_id="champion",
            challenger_wins=0,
            champion_wins=0,
            draws=50,
            total_games=50,
            challenger_win_rate=0.5,  # Convention for all draws
            champion_win_rate=0.5,
            statistical_p_value=1.0,
            is_statistically_significant=False,
            challenger_final_elo=1500.0,
            champion_final_elo=1500.0,
            should_promote=False,
        )
        assert result.draws == 50
        assert not result.should_promote


class TestAutoTournamentPipelineAdvanced:
    """Additional advanced tests for AutoTournamentPipeline."""

    def test_register_model_relative_path(self, temp_dirs, sample_metadata):
        """Test registering model with relative path."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        # Create model in models_dir
        model_name = "relative_model.pth"
        model_path = os.path.join(models_dir, model_name)
        with open(model_path, "wb") as f:
            f.write(b"model data")

        # Register with relative path (just filename)
        model_id = pipeline.register_model(model_name, metadata=sample_metadata)

        assert model_id in pipeline._models
        assert pipeline._models[model_id].model_path == model_path

    def test_corrupted_registry_graceful_handling(self, temp_dirs):
        """Test graceful handling of corrupted registry."""
        models_dir, results_dir = temp_dirs
        registry_file = os.path.join(results_dir, "model_registry.json")

        # Write invalid JSON
        with open(registry_file, "w") as f:
            f.write("{ invalid json }")

        # Should not raise, just log warning
        pipeline = AutoTournamentPipeline(models_dir, results_dir)
        assert len(pipeline._models) == 0

    def test_registry_with_malformed_model_data(self, temp_dirs, sample_metadata):
        """Test handling of registry with malformed model data."""
        models_dir, results_dir = temp_dirs
        registry_file = os.path.join(results_dir, "model_registry.json")

        # Write registry with partial/malformed data
        with open(registry_file, "w") as f:
            json.dump({
                "models": {
                    "bad_model": {
                        "model_id": "bad_model",
                        # Missing required fields
                    }
                }
            }, f)

        # Should handle gracefully
        with pytest.raises(Exception):  # Will raise due to missing fields
            AutoTournamentPipeline(models_dir, results_dir)

    @patch("app.training.auto_tournament.Tournament")
    def test_tournament_with_many_games(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test tournament with many games per match."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)
        pipeline.register_model(mock_model_file_2, metadata=sample_metadata_2)

        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1550.0, "B": 1450.0}
        mock_tournament.victory_reasons = {"territory": 40, "rings": 55, "unknown": 5}
        mock_tournament.run.return_value = {"A": 60, "B": 35, "Draw": 5}
        mock_tournament_class.return_value = mock_tournament

        result = pipeline.run_tournament(games_per_match=100)

        assert len(result.matches) == 100
        assert result.victory_reasons["territory"] == 40

    @patch("app.training.auto_tournament.Tournament")
    def test_tournament_updates_statistics(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test that tournament updates game statistics."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id_1 = pipeline.register_model(mock_model_file, metadata=sample_metadata)
        model_id_2 = pipeline.register_model(mock_model_file_2, metadata=sample_metadata_2)

        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1520.0, "B": 1480.0}
        mock_tournament.victory_reasons = {}
        mock_tournament.run.return_value = {"A": 6, "B": 3, "Draw": 1}
        mock_tournament_class.return_value = mock_tournament

        pipeline.run_tournament(games_per_match=10)

        # Check statistics were updated
        model_1 = pipeline._models[model_id_1]
        model_2 = pipeline._models[model_id_2]

        assert model_1.games_played == 10
        assert model_2.games_played == 10
        # Note: wins/losses depend on which model is "A" vs "B"

    @patch("app.training.auto_tournament.Tournament")
    def test_evaluate_challenger_updates_statistics(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test challenger evaluation updates statistics."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        champion_id = pipeline.register_model(mock_model_file, metadata=sample_metadata)

        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1510.0, "B": 1490.0}
        mock_tournament.victory_reasons = {}
        mock_tournament.run.return_value = {"A": 5, "B": 4, "Draw": 1}
        mock_tournament_class.return_value = mock_tournament

        result = pipeline.evaluate_challenger(
            mock_model_file_2,
            games=10,
            challenger_metadata=sample_metadata_2,
        )

        # Champion should have updated statistics
        champion = pipeline._models[champion_id]
        assert champion.games_played == 10

    def test_report_with_tournament_history(
        self, temp_dirs, mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test report includes tournament history."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        model_id_1 = pipeline.register_model(mock_model_file, metadata=sample_metadata)
        model_id_2 = pipeline.register_model(mock_model_file_2, metadata=sample_metadata_2)

        # Add tournament history manually
        pipeline._tournament_history.append(
            TournamentResult(
                tournament_id="history_tournament_001",
                participants=[model_id_1, model_id_2],
                matches=[],
                final_elo_ratings={model_id_1: 1520.0, model_id_2: 1480.0},
                final_standings=[(model_id_1, 1520.0), (model_id_2, 1480.0)],
                started_at="2025-12-28T10:00:00+00:00",
                finished_at="2025-12-28T11:00:00+00:00",
                victory_reasons={"territory": 5, "rings": 3},
            )
        )

        report = pipeline.generate_report()

        assert "## Tournament History" in report
        assert "history_tournament_001" in report

    def test_report_victory_types_aggregation(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test report aggregates victory types from all tournaments."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        # Add multiple tournaments with victory reasons
        for i in range(3):
            pipeline._tournament_history.append(
                TournamentResult(
                    tournament_id=f"agg_tournament_{i}",
                    participants=["a"],
                    matches=[],
                    final_elo_ratings={"a": 1500.0},
                    final_standings=[("a", 1500.0)],
                    started_at="2025-12-28T10:00:00+00:00",
                    victory_reasons={"territory": 10, "ring_elimination": 5},
                )
            )

        report = pipeline.generate_report()

        assert "## Victory Types" in report

    def test_save_report_auto_filename(
        self, temp_dirs, mock_model_file, sample_metadata
    ):
        """Test save_report generates timestamped filename."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        # Save without filename
        report_path = pipeline.save_report()

        assert os.path.exists(report_path)
        assert report_path.endswith(".md")
        assert "report_" in os.path.basename(report_path)


class TestAutoTournamentPipelineSpecificParticipants:
    """Tests for tournament with specific participant list."""

    @patch("app.training.auto_tournament.Tournament")
    def test_tournament_with_subset_of_models(
        self, mock_tournament_class, temp_dirs,
        sample_metadata
    ):
        """Test running tournament with only some registered models."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        # Create 4 models
        model_ids = []
        for i in range(4):
            path = os.path.join(models_dir, f"model_{i}.pth")
            with open(path, "wb") as f:
                f.write(f"model {i}".encode())
            meta = ModelMetadata(
                model_class="TestNet",
                architecture_version="1.0.0",
                checksum=f"checksum{i}" * 8,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            model_ids.append(pipeline.register_model(path, metadata=meta))

        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1510.0, "B": 1490.0}
        mock_tournament.victory_reasons = {}
        mock_tournament.run.return_value = {"A": 3, "B": 2, "Draw": 0}
        mock_tournament_class.return_value = mock_tournament

        # Run tournament with only first 2 models
        result = pipeline.run_tournament(
            participants=[model_ids[0], model_ids[1]],
            games_per_match=5,
        )

        assert len(result.participants) == 2
        assert model_ids[0] in result.participants
        assert model_ids[1] in result.participants
        assert model_ids[2] not in result.participants
        assert model_ids[3] not in result.participants

    @patch("app.training.auto_tournament.Tournament")
    def test_tournament_filters_invalid_participants(
        self, mock_tournament_class, temp_dirs, sample_metadata
    ):
        """Test that invalid participant IDs are filtered out."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        # Create 2 models
        path1 = os.path.join(models_dir, "model_1.pth")
        path2 = os.path.join(models_dir, "model_2.pth")
        for path in [path1, path2]:
            with open(path, "wb") as f:
                f.write(b"model")

        meta1 = ModelMetadata(
            model_class="TestNet",
            architecture_version="1.0.0",
            checksum="checksum1" * 8,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        meta2 = ModelMetadata(
            model_class="TestNet",
            architecture_version="1.0.0",
            checksum="checksum2" * 8,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        id1 = pipeline.register_model(path1, metadata=meta1)
        id2 = pipeline.register_model(path2, metadata=meta2)

        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1510.0, "B": 1490.0}
        mock_tournament.victory_reasons = {}
        mock_tournament.run.return_value = {"A": 3, "B": 2, "Draw": 0}
        mock_tournament_class.return_value = mock_tournament

        # Include an invalid participant ID
        result = pipeline.run_tournament(
            participants=[id1, id2, "nonexistent_model"],
            games_per_match=5,
        )

        # Should only include valid participants
        assert len(result.participants) == 2
        assert "nonexistent_model" not in result.participants


class TestAutoTournamentPipelineConstants:
    """Tests for pipeline constants and thresholds."""

    def test_default_constants(self, temp_dirs):
        """Test that default constants are set correctly."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        # Check that constants are defined
        assert hasattr(pipeline, "WIN_RATE_THRESHOLD")
        assert hasattr(pipeline, "PROMOTION_SIGNIFICANCE_LEVEL")
        assert hasattr(pipeline, "DEFAULT_ELO")
        assert hasattr(pipeline, "ELO_K")

        # Check reasonable values
        assert 0.5 <= pipeline.WIN_RATE_THRESHOLD <= 0.7
        assert 0.01 <= pipeline.PROMOTION_SIGNIFICANCE_LEVEL <= 0.1
        assert pipeline.DEFAULT_ELO == 1500.0
        assert pipeline.ELO_K == 32.0


class TestChallengerEvaluationEdgeCases:
    """Edge case tests for challenger evaluation."""

    @patch("app.training.auto_tournament.Tournament")
    def test_challenger_with_all_draws_decision(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test challenger evaluation when all games are draws."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1500.0, "B": 1500.0}
        mock_tournament.victory_reasons = {}
        mock_tournament.run.return_value = {"A": 0, "B": 0, "Draw": 50}
        mock_tournament_class.return_value = mock_tournament

        result = pipeline.evaluate_challenger(
            mock_model_file_2,
            games=50,
            challenger_metadata=sample_metadata_2,
        )

        # All draws means 50% win rate convention
        assert result.challenger_win_rate == 0.5
        assert result.champion_win_rate == 0.5
        # Should not promote without decisive wins
        assert not result.should_promote

    @patch("app.training.auto_tournament.Tournament")
    def test_challenger_saves_result_to_file(
        self, mock_tournament_class, temp_dirs,
        mock_model_file, mock_model_file_2,
        sample_metadata, sample_metadata_2
    ):
        """Test that challenger evaluation saves result to file."""
        models_dir, results_dir = temp_dirs
        pipeline = AutoTournamentPipeline(models_dir, results_dir)

        pipeline.register_model(mock_model_file, metadata=sample_metadata)

        mock_tournament = MagicMock()
        mock_tournament.ratings = {"A": 1520.0, "B": 1480.0}
        mock_tournament.victory_reasons = {}
        mock_tournament.run.return_value = {"A": 30, "B": 20, "Draw": 0}
        mock_tournament_class.return_value = mock_tournament

        result = pipeline.evaluate_challenger(
            mock_model_file_2,
            games=50,
            challenger_metadata=sample_metadata_2,
        )

        # Check that result file was created
        result_files = [f for f in os.listdir(results_dir) if f.startswith("challenge_")]
        assert len(result_files) >= 1

        # Verify file content
        result_path = os.path.join(results_dir, result_files[0])
        with open(result_path) as f:
            saved_data = json.load(f)

        assert saved_data["challenger_wins"] == 30
        assert saved_data["champion_wins"] == 20
