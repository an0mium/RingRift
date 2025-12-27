"""Tests for app.coordination.daemon_factory module.

Tests the centralized daemon factory for lazy loading:
- DaemonSpec dataclass
- DaemonFactory class
- Registry building
- Daemon creation and caching

December 2025 - Phase 1.2 architecture cleanup.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.coordination.daemon_factory import (
    DaemonFactory,
    DaemonSpec,
    get_daemon_factory,
    reset_daemon_factory,
)
from app.coordination.daemon_types import DaemonType


class TestDaemonSpec:
    """Tests for DaemonSpec dataclass."""

    def test_basic_spec(self):
        """Should create spec with required fields."""
        spec = DaemonSpec(
            import_path="app.coordination.auto_sync_daemon",
            class_name="AutoSyncDaemon",
        )
        assert spec.import_path == "app.coordination.auto_sync_daemon"
        assert spec.class_name == "AutoSyncDaemon"
        assert spec.factory_fn is None
        assert spec.singleton is True

    def test_spec_with_factory(self):
        """Should create spec with factory function."""
        spec = DaemonSpec(
            import_path="app.coordination.event_router",
            class_name="EventRouter",
            factory_fn="get_router",
        )
        assert spec.factory_fn == "get_router"

    def test_spec_non_singleton(self):
        """Should create non-singleton spec."""
        spec = DaemonSpec(
            import_path="app.coordination.auto_sync_daemon",
            class_name="AutoSyncDaemon",
            singleton=False,
        )
        assert spec.singleton is False


class TestDaemonFactoryInit:
    """Tests for DaemonFactory initialization."""

    def setup_method(self):
        """Reset factory before each test."""
        reset_daemon_factory()

    def teardown_method(self):
        """Reset factory after each test."""
        reset_daemon_factory()

    def test_get_singleton(self):
        """Should return singleton factory."""
        factory1 = get_daemon_factory()
        factory2 = get_daemon_factory()
        assert factory1 is factory2

    def test_reset_factory(self):
        """Should reset singleton."""
        factory1 = get_daemon_factory()
        reset_daemon_factory()
        factory2 = get_daemon_factory()
        assert factory1 is not factory2


class TestDaemonFactoryRegistry:
    """Tests for factory registry building."""

    def setup_method(self):
        """Reset factory before each test."""
        reset_daemon_factory()

    def teardown_method(self):
        """Reset factory after each test."""
        reset_daemon_factory()

    def test_list_registered(self):
        """Should list all registered daemon types."""
        factory = get_daemon_factory()
        registered = factory.list_registered()

        assert len(registered) > 0
        assert "AUTO_SYNC" in registered
        assert "CLUSTER_MONITOR" in registered
        assert "EVENT_ROUTER" in registered

    def test_get_spec_by_enum(self):
        """Should get spec by DaemonType enum."""
        factory = get_daemon_factory()
        spec = factory.get_spec(DaemonType.AUTO_SYNC)

        assert spec is not None
        assert spec.class_name == "AutoSyncDaemon"
        assert spec.import_path == "app.coordination.auto_sync_daemon"

    def test_get_spec_by_string(self):
        """Should get spec by string name."""
        factory = get_daemon_factory()
        spec = factory.get_spec("AUTO_SYNC")

        assert spec is not None
        assert spec.class_name == "AutoSyncDaemon"

    def test_get_unknown_spec(self):
        """Should return None for unknown type."""
        factory = get_daemon_factory()
        spec = factory.get_spec("UNKNOWN_DAEMON_TYPE")
        assert spec is None


class TestDaemonFactoryCreate:
    """Tests for daemon creation."""

    def setup_method(self):
        """Reset factory before each test."""
        reset_daemon_factory()

    def teardown_method(self):
        """Reset factory after each test."""
        reset_daemon_factory()

    def test_create_unknown_type_raises(self):
        """Should raise ValueError for unknown type."""
        factory = get_daemon_factory()

        with pytest.raises(ValueError, match="Unknown daemon type"):
            factory.create("UNKNOWN_DAEMON_TYPE")

    def test_create_with_mock_import(self):
        """Should create daemon via importlib."""
        factory = get_daemon_factory()

        # Register a test daemon
        factory.register(
            "TEST_DAEMON",
            DaemonSpec(
                import_path="unittest.mock",
                class_name="MagicMock",
            ),
        )

        daemon = factory.create("TEST_DAEMON")
        assert daemon is not None
        assert isinstance(daemon, MagicMock)

    def test_singleton_caching(self):
        """Should cache singleton instances."""
        factory = get_daemon_factory()

        # Register a test daemon
        factory.register(
            "TEST_DAEMON",
            DaemonSpec(
                import_path="unittest.mock",
                class_name="MagicMock",
            ),
        )

        daemon1 = factory.create("TEST_DAEMON")
        daemon2 = factory.create("TEST_DAEMON")
        assert daemon1 is daemon2

    def test_force_new_bypasses_cache(self):
        """Should create new instance with force_new."""
        factory = get_daemon_factory()

        # Register a test daemon
        factory.register(
            "TEST_DAEMON",
            DaemonSpec(
                import_path="unittest.mock",
                class_name="MagicMock",
            ),
        )

        daemon1 = factory.create("TEST_DAEMON")
        daemon2 = factory.create("TEST_DAEMON", force_new=True)
        assert daemon1 is not daemon2

    def test_non_singleton_always_new(self):
        """Should always create new instance for non-singletons."""
        factory = get_daemon_factory()

        # Register a non-singleton test daemon
        factory.register(
            "TEST_DAEMON",
            DaemonSpec(
                import_path="unittest.mock",
                class_name="MagicMock",
                singleton=False,
            ),
        )

        daemon1 = factory.create("TEST_DAEMON")
        daemon2 = factory.create("TEST_DAEMON")
        assert daemon1 is not daemon2

    def test_clear_cache(self):
        """Should clear singleton cache."""
        factory = get_daemon_factory()

        # Register and create a test daemon
        factory.register(
            "TEST_DAEMON",
            DaemonSpec(
                import_path="unittest.mock",
                class_name="MagicMock",
            ),
        )

        daemon1 = factory.create("TEST_DAEMON")
        factory.clear_cache()
        daemon2 = factory.create("TEST_DAEMON")
        assert daemon1 is not daemon2


class TestDaemonFactoryWithFactory:
    """Tests for factory function usage."""

    def setup_method(self):
        """Reset factory before each test."""
        reset_daemon_factory()

    def teardown_method(self):
        """Reset factory after each test."""
        reset_daemon_factory()

    def test_create_with_factory_fn(self):
        """Should use factory function when specified."""
        factory = get_daemon_factory()

        # Create a mock module with a factory function
        mock_module = MagicMock()
        mock_daemon = MagicMock()
        mock_module.get_test_daemon = MagicMock(return_value=mock_daemon)

        with patch("importlib.import_module", return_value=mock_module):
            factory.register(
                "TEST_DAEMON",
                DaemonSpec(
                    import_path="fake.module",
                    class_name="TestDaemon",
                    factory_fn="get_test_daemon",
                ),
            )

            daemon = factory.create("TEST_DAEMON")
            assert daemon is mock_daemon
            mock_module.get_test_daemon.assert_called_once()

    def test_fallback_to_class_if_factory_missing(self):
        """Should fall back to class if factory function not found."""
        factory = get_daemon_factory()

        # Create a mock module without the factory function
        mock_module = MagicMock()
        del mock_module.get_missing_factory  # Ensure it doesn't exist
        mock_daemon_class = MagicMock()
        mock_module.TestDaemon = mock_daemon_class

        with patch("importlib.import_module", return_value=mock_module):
            factory.register(
                "TEST_DAEMON",
                DaemonSpec(
                    import_path="fake.module",
                    class_name="TestDaemon",
                    factory_fn="get_missing_factory",
                ),
            )

            factory.create("TEST_DAEMON")
            mock_daemon_class.assert_called_once()


class TestDaemonFactoryRegistration:
    """Tests for dynamic daemon registration."""

    def setup_method(self):
        """Reset factory before each test."""
        reset_daemon_factory()

    def teardown_method(self):
        """Reset factory after each test."""
        reset_daemon_factory()

    def test_register_new_daemon(self):
        """Should register new daemon type."""
        factory = get_daemon_factory()

        spec = DaemonSpec(
            import_path="test.module",
            class_name="TestDaemon",
        )

        factory.register("CUSTOM_DAEMON", spec)
        retrieved = factory.get_spec("CUSTOM_DAEMON")
        assert retrieved is spec

    def test_register_overwrites_existing(self):
        """Should overwrite existing registration."""
        factory = get_daemon_factory()

        spec1 = DaemonSpec(
            import_path="test.module1",
            class_name="TestDaemon1",
        )
        spec2 = DaemonSpec(
            import_path="test.module2",
            class_name="TestDaemon2",
        )

        factory.register("TEST_DAEMON", spec1)
        factory.register("TEST_DAEMON", spec2)

        retrieved = factory.get_spec("TEST_DAEMON")
        assert retrieved is spec2
