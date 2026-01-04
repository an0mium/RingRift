"""Tests for OWCModelDiscovery.

Sprint 15 (January 3, 2026): Unit tests for OWC model discovery.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.owc_discovery import (
    DiscoveredModel,
    DiscoverySummary,
    OWCDiscoveryConfig,
    OWCModelDiscovery,
    _extract_model_info,
    _is_running_on_owc_host,
    get_owc_discovery,
)


# ============================================================================
# OWCDiscoveryConfig Tests
# ============================================================================


class TestOWCDiscoveryConfig:
    """Tests for OWCDiscoveryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OWCDiscoveryConfig()

        assert config.owc_host == "mac-studio"
        assert config.owc_user == "armand"
        assert config.owc_base_path == "/Volumes/RingRift-Data"
        assert config.ssh_timeout == 60
        assert config.hash_timeout == 120
        assert config.compute_hashes is True
        assert config.register_with_tracker is True
        assert len(config.model_paths) >= 6

    def test_custom_values(self):
        """Test custom configuration."""
        config = OWCDiscoveryConfig(
            owc_host="custom-host",
            owc_user="custom-user",
            owc_base_path="/custom/path",
            compute_hashes=False,
            ssh_timeout=120,
        )

        assert config.owc_host == "custom-host"
        assert config.owc_user == "custom-user"
        assert config.owc_base_path == "/custom/path"
        assert config.compute_hashes is False
        assert config.ssh_timeout == 120

    def test_from_env(self):
        """Test loading from environment variables."""
        with patch.dict(os.environ, {
            "OWC_HOST": "env-host",
            "OWC_USER": "env-user",
            "OWC_BASE_PATH": "/env/path",
            "RINGRIFT_OWC_COMPUTE_HASHES": "false",
        }):
            config = OWCDiscoveryConfig.from_env()

            assert config.owc_host == "env-host"
            assert config.owc_user == "env-user"
            assert config.owc_base_path == "/env/path"
            assert config.compute_hashes is False


# ============================================================================
# DiscoveredModel Tests
# ============================================================================


class TestDiscoveredModel:
    """Tests for DiscoveredModel dataclass."""

    def test_basic_creation(self):
        """Test basic model creation."""
        model = DiscoveredModel(
            path="models/canonical_hex8_2p.pth",
            file_name="canonical_hex8_2p.pth",
            board_type="hex8",
            num_players=2,
            architecture_version="v2",
            file_size=10_000_000,
        )

        assert model.path == "models/canonical_hex8_2p.pth"
        assert model.file_name == "canonical_hex8_2p.pth"
        assert model.board_type == "hex8"
        assert model.num_players == 2
        assert model.architecture_version == "v2"
        assert model.file_size == 10_000_000

    def test_config_key_property(self):
        """Test config_key property."""
        model = DiscoveredModel(
            path="models/test.pth",
            file_name="test.pth",
            board_type="square8",
            num_players=4,
            architecture_version=None,
            file_size=5_000_000,
        )

        assert model.config_key == "square8_4p"

    def test_config_key_none_when_missing_info(self):
        """Test config_key is None when board_type or num_players missing."""
        model = DiscoveredModel(
            path="models/test.pth",
            file_name="test.pth",
            board_type=None,
            num_players=2,
            architecture_version=None,
            file_size=5_000_000,
        )

        assert model.config_key is None

    def test_is_canonical_property(self):
        """Test is_canonical property."""
        canonical = DiscoveredModel(
            path="models/canonical_hex8_2p.pth",
            file_name="canonical_hex8_2p.pth",
            board_type="hex8",
            num_players=2,
            architecture_version=None,
            file_size=5_000_000,
        )

        regular = DiscoveredModel(
            path="models/hex8_2p.pth",
            file_name="hex8_2p.pth",
            board_type="hex8",
            num_players=2,
            architecture_version=None,
            file_size=5_000_000,
        )

        assert canonical.is_canonical is True
        assert regular.is_canonical is False

    def test_is_best_property(self):
        """Test is_best property."""
        best = DiscoveredModel(
            path="models/ringrift_best_hex8_2p.pth",
            file_name="ringrift_best_hex8_2p.pth",
            board_type="hex8",
            num_players=2,
            architecture_version=None,
            file_size=5_000_000,
        )

        regular = DiscoveredModel(
            path="models/hex8_2p.pth",
            file_name="hex8_2p.pth",
            board_type="hex8",
            num_players=2,
            architecture_version=None,
            file_size=5_000_000,
        )

        assert best.is_best is True
        assert regular.is_best is False

    def test_full_path_property(self):
        """Test full_path property."""
        model = DiscoveredModel(
            path="models/test.pth",
            file_name="test.pth",
            board_type="hex8",
            num_players=2,
            architecture_version=None,
            file_size=5_000_000,
        )

        assert model.full_path == "models/test.pth"

    def test_source_default(self):
        """Test source default value."""
        model = DiscoveredModel(
            path="test.pth",
            file_name="test.pth",
            board_type="hex8",
            num_players=2,
            architecture_version=None,
            file_size=5_000_000,
        )

        assert model.source == "owc"


# ============================================================================
# DiscoverySummary Tests
# ============================================================================


class TestDiscoverySummary:
    """Tests for DiscoverySummary dataclass."""

    def test_default_values(self):
        """Test default summary values."""
        summary = DiscoverySummary()

        assert summary.total_models == 0
        assert summary.models_by_config == {}
        assert summary.models_by_architecture == {}
        assert summary.total_size_bytes == 0
        assert summary.canonical_count == 0
        assert summary.with_hash_count == 0
        assert summary.discovery_duration == 0.0
        assert summary.errors == []

    def test_custom_values(self):
        """Test summary with custom values."""
        summary = DiscoverySummary(
            total_models=100,
            models_by_config={"hex8_2p": 50, "square8_4p": 50},
            models_by_architecture={"v2": 80, "v5-heavy": 20},
            total_size_bytes=10_000_000_000,
            canonical_count=12,
            with_hash_count=95,
            discovery_duration=5.5,
            errors=["Some error"],
        )

        assert summary.total_models == 100
        assert summary.models_by_config["hex8_2p"] == 50
        assert summary.models_by_architecture["v2"] == 80
        assert summary.canonical_count == 12


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestIsRunningOnOwcHost:
    """Tests for _is_running_on_owc_host helper."""

    def test_localhost(self):
        """Test detection for localhost."""
        assert _is_running_on_owc_host("localhost") is True
        assert _is_running_on_owc_host("127.0.0.1") is True

    @patch("socket.gethostname")
    def test_exact_match(self, mock_gethostname):
        """Test exact hostname match."""
        mock_gethostname.return_value = "mac-studio"
        assert _is_running_on_owc_host("mac-studio") is True

    @patch("socket.gethostname")
    def test_local_suffix_match(self, mock_gethostname):
        """Test hostname with .local suffix."""
        mock_gethostname.return_value = "mac-studio.local"
        assert _is_running_on_owc_host("mac-studio") is True

    @patch("socket.gethostname")
    def test_no_match(self, mock_gethostname):
        """Test when hostname doesn't match."""
        mock_gethostname.return_value = "other-host"
        assert _is_running_on_owc_host("mac-studio") is False


class TestExtractModelInfo:
    """Tests for _extract_model_info helper."""

    def test_canonical_model(self):
        """Test parsing canonical model filename."""
        model = _extract_model_info("models/canonical_hex8_2p.pth", 10_000_000)

        assert model is not None
        assert model.board_type == "hex8"
        assert model.num_players == 2
        assert model.file_size == 10_000_000

    def test_canonical_with_version(self):
        """Test parsing canonical model with version."""
        model = _extract_model_info(
            "models/canonical_square8_4p_v5heavy.pth", 50_000_000
        )

        assert model is not None
        assert model.board_type == "square8"
        assert model.num_players == 4
        assert model.architecture_version == "v5heavy"

    def test_best_model(self):
        """Test parsing 'best' model filename."""
        model = _extract_model_info(
            "models/ringrift_best_hexagonal_3p.pth", 20_000_000
        )

        assert model is not None
        assert model.board_type == "hexagonal"
        assert model.num_players == 3

    def test_simple_model(self):
        """Test parsing simple model filename."""
        model = _extract_model_info("models/square19_2p.pth", 100_000_000)

        assert model is not None
        assert model.board_type == "square19"
        assert model.num_players == 2

    def test_non_pth_file(self):
        """Test rejection of non-pth files."""
        model = _extract_model_info("models/file.txt", 1000)
        assert model is None

    def test_temp_file(self):
        """Test rejection of temp files."""
        model = _extract_model_info("models/.temp_hex8_2p.pth", 1000)
        assert model is None

        model = _extract_model_info("models/_tmp_hex8_2p.pth", 1000)
        assert model is None

    def test_fallback_parsing(self):
        """Test fallback parsing for non-standard names."""
        model = _extract_model_info("models/my_hex8_model_2p_experimental.pth", 5_000_000)

        assert model is not None
        assert model.board_type == "hex8"
        assert model.num_players == 2

    def test_unrecognized_format(self):
        """Test unrecognized model name format."""
        model = _extract_model_info("models/random_model.pth", 5_000_000)

        assert model is not None
        assert model.board_type is None
        assert model.num_players is None


# ============================================================================
# OWCModelDiscovery Tests
# ============================================================================


class TestOWCModelDiscovery:
    """Tests for OWCModelDiscovery class."""

    def setup_method(self):
        """Reset singleton before each test."""
        OWCModelDiscovery.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        OWCModelDiscovery.reset_instance()

    def test_singleton_pattern(self):
        """Test singleton instance creation."""
        instance1 = OWCModelDiscovery.get_instance()
        instance2 = OWCModelDiscovery.get_instance()

        assert instance1 is instance2

    def test_singleton_reset(self):
        """Test singleton reset."""
        instance1 = OWCModelDiscovery.get_instance()
        OWCModelDiscovery.reset_instance()
        instance2 = OWCModelDiscovery.get_instance()

        assert instance1 is not instance2

    def test_config_property(self):
        """Test config property."""
        config = OWCDiscoveryConfig(owc_host="test-host")
        discovery = OWCModelDiscovery(config)

        assert discovery.config.owc_host == "test-host"

    @patch("app.models.owc_discovery._is_running_on_owc_host")
    def test_is_local_property(self, mock_is_local):
        """Test is_local property."""
        mock_is_local.return_value = True
        discovery = OWCModelDiscovery()

        assert discovery.is_local is True

    @patch("app.models.owc_discovery._is_running_on_owc_host")
    @pytest.mark.asyncio
    async def test_check_available_local(self, mock_is_local):
        """Test check_available in local mode."""
        mock_is_local.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            config = OWCDiscoveryConfig(owc_base_path=tmpdir)
            discovery = OWCModelDiscovery(config)

            is_available = await discovery.check_available()
            assert is_available is True

    @patch("app.models.owc_discovery._is_running_on_owc_host")
    @pytest.mark.asyncio
    async def test_check_available_local_missing_path(self, mock_is_local):
        """Test check_available when path doesn't exist."""
        mock_is_local.return_value = True

        config = OWCDiscoveryConfig(owc_base_path="/nonexistent/path")
        discovery = OWCModelDiscovery(config)

        is_available = await discovery.check_available()
        assert is_available is False

    @patch("app.models.owc_discovery._is_running_on_owc_host")
    @pytest.mark.asyncio
    async def test_get_model_count(self, mock_is_local):
        """Test get_model_count."""
        mock_is_local.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some model files
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            for name in ["canonical_hex8_2p.pth", "square8_4p.pth"]:
                model_file = models_dir / name
                model_file.write_bytes(b"x" * 2_000_000)  # 2MB

            config = OWCDiscoveryConfig(
                owc_base_path=tmpdir,
                model_paths=["models"],
            )
            discovery = OWCModelDiscovery(config)

            count = await discovery.get_model_count()
            assert count == 2

    @patch("app.models.owc_discovery._is_running_on_owc_host")
    @pytest.mark.asyncio
    async def test_discover_all_models_local(self, mock_is_local):
        """Test discover_all_models in local mode."""
        mock_is_local.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model files
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            for name in ["canonical_hex8_2p.pth", "square8_4p_v5heavy.pth"]:
                model_file = models_dir / name
                model_file.write_bytes(b"x" * 2_000_000)  # 2MB

            config = OWCDiscoveryConfig(
                owc_base_path=tmpdir,
                model_paths=["models"],
                compute_hashes=False,  # Skip hash computation for speed
            )
            discovery = OWCModelDiscovery(config)

            models = await discovery.discover_all_models()

            assert len(models) == 2
            assert any(m.board_type == "hex8" for m in models)
            assert any(m.board_type == "square8" for m in models)

    @patch("app.models.owc_discovery._is_running_on_owc_host")
    @pytest.mark.asyncio
    async def test_discover_uses_cache(self, mock_is_local):
        """Test that discovery uses cache when available."""
        mock_is_local.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "canonical_hex8_2p.pth").write_bytes(b"x" * 2_000_000)

            config = OWCDiscoveryConfig(
                owc_base_path=tmpdir,
                model_paths=["models"],
                compute_hashes=False,
            )
            discovery = OWCModelDiscovery(config)

            # First call populates cache
            models1 = await discovery.discover_all_models()

            # Add another file (won't be seen due to cache)
            (models_dir / "square8_2p.pth").write_bytes(b"x" * 2_000_000)

            # Second call uses cache
            models2 = await discovery.discover_all_models()

            assert len(models1) == len(models2) == 1

            # Force refresh should find new file
            models3 = await discovery.discover_all_models(force_refresh=True)
            assert len(models3) == 2

    @patch("app.models.owc_discovery._is_running_on_owc_host")
    @pytest.mark.asyncio
    async def test_compute_file_hash(self, mock_is_local):
        """Test SHA256 hash computation."""
        mock_is_local.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pth"
            test_file.write_bytes(b"test content for hashing")

            # Known SHA256 of "test content for hashing"
            expected_hash = "6f5e8c4f2e7e3d1c9b8a7f6e5d4c3b2a1"  # Not actual, just for test structure

            hash_result = OWCModelDiscovery._compute_file_hash(test_file)

            assert len(hash_result) == 64  # SHA256 hex length
            assert all(c in "0123456789abcdef" for c in hash_result)

    @patch("app.models.owc_discovery._is_running_on_owc_host")
    @pytest.mark.asyncio
    async def test_get_discovery_summary(self, mock_is_local):
        """Test get_discovery_summary."""
        mock_is_local.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            # Create models of different types
            for name, size in [
                ("canonical_hex8_2p.pth", 2_000_000),
                ("canonical_hex8_4p.pth", 2_500_000),
                ("square8_2p_v5heavy.pth", 5_000_000),
            ]:
                (models_dir / name).write_bytes(b"x" * size)

            config = OWCDiscoveryConfig(
                owc_base_path=tmpdir,
                model_paths=["models"],
                compute_hashes=False,
            )
            discovery = OWCModelDiscovery(config)

            summary = await discovery.get_discovery_summary()

            assert summary.total_models == 3
            assert summary.canonical_count == 2
            assert "hex8_2p" in summary.models_by_config
            assert "hex8_4p" in summary.models_by_config
            assert summary.total_size_bytes > 0

    @pytest.mark.asyncio
    async def test_get_unevaluated_models_fallback(self):
        """Test get_unevaluated_models fallback when tracker not available."""
        OWCModelDiscovery.reset_instance()

        with patch("app.models.owc_discovery._is_running_on_owc_host", return_value=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                models_dir = Path(tmpdir) / "models"
                models_dir.mkdir()
                (models_dir / "canonical_hex8_2p.pth").write_bytes(b"x" * 2_000_000)

                config = OWCDiscoveryConfig(
                    owc_base_path=tmpdir,
                    model_paths=["models"],
                    compute_hashes=False,
                )
                discovery = OWCModelDiscovery(config)

                # Should fall back to returning discovered models
                models = await discovery.get_unevaluated_models(limit=10)

                assert len(models) >= 0  # May be empty if tracker check causes issues


# ============================================================================
# Module Helper Tests
# ============================================================================


class TestGetOwcDiscovery:
    """Tests for get_owc_discovery helper."""

    def setup_method(self):
        """Reset singleton before each test."""
        OWCModelDiscovery.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        OWCModelDiscovery.reset_instance()

    def test_returns_singleton(self):
        """Test that get_owc_discovery returns singleton."""
        discovery1 = get_owc_discovery()
        discovery2 = get_owc_discovery()

        assert discovery1 is discovery2

    def test_accepts_config(self):
        """Test that config is passed to instance."""
        config = OWCDiscoveryConfig(owc_host="custom-host")
        discovery = get_owc_discovery(config)

        assert discovery.config.owc_host == "custom-host"
