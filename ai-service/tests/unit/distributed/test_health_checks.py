"""Tests for health checking system.

These tests verify:
1. ComponentHealth and HealthSummary dataclasses
2. HealthChecker component checks
3. Resource monitoring
4. Health report formatting
5. Health-to-recovery integration
"""

import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.distributed.health_checks import (
    ComponentHealth,
    HealthChecker,
    HealthSummary,
    format_health_report,
    get_health_summary,
)


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_create_healthy_component(self):
        """ComponentHealth should be created with healthy=True."""
        health = ComponentHealth(
            name="test",
            healthy=True,
            status="ok",
            message="All good",
        )
        assert health.name == "test"
        assert health.healthy is True
        assert health.status == "ok"
        assert health.message == "All good"

    def test_create_unhealthy_component(self):
        """ComponentHealth should be created with healthy=False."""
        health = ComponentHealth(
            name="test",
            healthy=False,
            status="error",
            message="Something wrong",
        )
        assert health.healthy is False
        assert health.status == "error"

    def test_default_values(self):
        """ComponentHealth should have sensible defaults."""
        health = ComponentHealth(
            name="test",
            healthy=True,
            status="ok",
        )
        assert health.message == ""
        assert health.last_activity is None
        assert health.details == {}

    def test_with_details(self):
        """ComponentHealth should store arbitrary details."""
        health = ComponentHealth(
            name="test",
            healthy=True,
            status="ok",
            details={"count": 42, "rate": 0.95},
        )
        assert health.details["count"] == 42
        assert health.details["rate"] == 0.95

    def test_with_last_activity(self):
        """ComponentHealth should store last_activity timestamp."""
        ts = time.time()
        health = ComponentHealth(
            name="test",
            healthy=True,
            status="ok",
            last_activity=ts,
        )
        assert health.last_activity == ts


class TestHealthSummary:
    """Tests for HealthSummary dataclass."""

    def test_healthy_summary(self):
        """HealthSummary should be healthy when no issues."""
        components = [
            ComponentHealth(name="a", healthy=True, status="ok"),
            ComponentHealth(name="b", healthy=True, status="ok"),
        ]
        summary = HealthSummary(
            healthy=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            components=components,
        )
        assert summary.healthy is True
        assert len(summary.issues) == 0
        assert len(summary.warnings) == 0

    def test_unhealthy_summary(self):
        """HealthSummary should be unhealthy when issues exist."""
        summary = HealthSummary(
            healthy=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            components=[],
            issues=["Database connection failed"],
        )
        assert summary.healthy is False
        assert len(summary.issues) == 1

    def test_component_status_property(self):
        """component_status should return status dict."""
        components = [
            ComponentHealth(name="sync", healthy=True, status="ok"),
            ComponentHealth(name="train", healthy=False, status="error"),
        ]
        summary = HealthSummary(
            healthy=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            components=components,
        )
        status_map = summary.component_status
        assert status_map["sync"] == "ok"
        assert status_map["train"] == "error"


class TestHealthCheckerDataSync:
    """Tests for HealthChecker.check_data_sync."""

    def test_data_sync_missing_database(self):
        """check_data_sync should return error when database missing."""
        checker = HealthChecker(merged_db_path=Path("/nonexistent/path.db"))
        health = checker.check_data_sync()

        assert health.name == "data_sync"
        assert health.healthy is False
        assert health.status == "error"
        assert "not found" in health.message.lower()

    def test_data_sync_healthy_database(self):
        """check_data_sync should return ok with valid database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            # Create a minimal valid database
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE games (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO games (id) VALUES (1)")
            conn.execute("INSERT INTO games (id) VALUES (2)")
            conn.commit()
            conn.close()

            checker = HealthChecker(merged_db_path=db_path)
            health = checker.check_data_sync()

            assert health.name == "data_sync"
            assert health.healthy is True
            assert health.status == "ok"
            assert "2 games" in health.message
        finally:
            db_path.unlink(missing_ok=True)

    def test_data_sync_stale_database(self):
        """check_data_sync should return warning when database is stale."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            # Create database
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE games (id INTEGER PRIMARY KEY)")
            conn.commit()
            conn.close()

            # Make it appear old by modifying mtime
            old_time = time.time() - 7200  # 2 hours ago
            import os
            os.utime(db_path, (old_time, old_time))

            checker = HealthChecker(merged_db_path=db_path)
            checker.DATA_SYNC_STALE_THRESHOLD = 3600  # 1 hour
            health = checker.check_data_sync()

            assert health.status == "warning"
            assert "stale" in health.message.lower()
        finally:
            db_path.unlink(missing_ok=True)


class TestHealthCheckerTraining:
    """Tests for HealthChecker.check_training."""

    def test_training_no_runs_directory(self):
        """check_training should return ok when no runs directory exists."""
        with patch.object(HealthChecker, "__init__", lambda x, **kwargs: None):
            checker = HealthChecker()
            checker.merged_db_path = Path("/nonexistent")
            checker.elo_db_path = Path("/nonexistent")
            checker.coordinator_db_path = Path("/nonexistent")
            checker.state_path = Path("/nonexistent")

        # Mock the runs directory to not exist
        with patch("app.distributed.health_checks.AI_SERVICE_ROOT", Path("/nonexistent")):
            with patch.object(Path, "exists", return_value=False):
                health = checker.check_training()
                assert health.healthy is True
                assert "no training runs" in health.message.lower()


class TestHealthCheckerResources:
    """Tests for HealthChecker.check_resources."""

    def test_resources_healthy(self):
        """check_resources should return ok when resources are fine."""
        # Mock psutil to return healthy values
        mock_mem = MagicMock()
        mock_mem.percent = 50.0
        mock_mem.available = 8 * (1024**3)  # 8 GB

        mock_disk = MagicMock()
        mock_disk.percent = 40.0
        mock_disk.free = 100 * (1024**3)  # 100 GB

        with patch("app.distributed.health_checks.psutil.virtual_memory", return_value=mock_mem):
            with patch("app.distributed.health_checks.psutil.disk_usage", return_value=mock_disk):
                with patch("app.distributed.health_checks.psutil.cpu_percent", return_value=30.0):
                    checker = HealthChecker()
                    health = checker.check_resources()

                    assert health.name == "resources"
                    assert health.healthy is True
                    assert health.status == "ok"

    def test_resources_memory_warning(self):
        """check_resources should warn on high memory usage."""
        mock_mem = MagicMock()
        mock_mem.percent = 75.0  # Above warning threshold
        mock_mem.available = 2 * (1024**3)

        mock_disk = MagicMock()
        mock_disk.percent = 40.0
        mock_disk.free = 100 * (1024**3)

        with patch("app.distributed.health_checks.psutil.virtual_memory", return_value=mock_mem):
            with patch("app.distributed.health_checks.psutil.disk_usage", return_value=mock_disk):
                with patch("app.distributed.health_checks.psutil.cpu_percent", return_value=30.0):
                    checker = HealthChecker()
                    health = checker.check_resources()

                    assert health.healthy is False
                    assert "memory" in health.message.lower()

    def test_resources_disk_critical(self):
        """check_resources should error on critical disk usage."""
        mock_mem = MagicMock()
        mock_mem.percent = 50.0
        mock_mem.available = 8 * (1024**3)

        mock_disk = MagicMock()
        mock_disk.percent = 85.0  # Above critical threshold
        mock_disk.free = 10 * (1024**3)

        with patch("app.distributed.health_checks.psutil.virtual_memory", return_value=mock_mem):
            with patch("app.distributed.health_checks.psutil.disk_usage", return_value=mock_disk):
                with patch("app.distributed.health_checks.psutil.cpu_percent", return_value=30.0):
                    checker = HealthChecker()
                    health = checker.check_resources()

                    assert health.healthy is False
                    assert "disk" in health.message.lower()

    def test_resources_multiple_issues(self):
        """check_resources should report multiple issues."""
        mock_mem = MagicMock()
        mock_mem.percent = 85.0  # Critical
        mock_mem.available = 1 * (1024**3)

        mock_disk = MagicMock()
        mock_disk.percent = 75.0  # Critical
        mock_disk.free = 20 * (1024**3)

        with patch("app.distributed.health_checks.psutil.virtual_memory", return_value=mock_mem):
            with patch("app.distributed.health_checks.psutil.disk_usage", return_value=mock_disk):
                with patch("app.distributed.health_checks.psutil.cpu_percent", return_value=85.0):
                    checker = HealthChecker()
                    health = checker.check_resources()

                    assert health.healthy is False
                    assert health.status == "error"  # Multiple issues = error
                    assert "memory" in health.message.lower()
                    assert "disk" in health.message.lower()


class TestHealthCheckerCheckAll:
    """Tests for HealthChecker.check_all."""

    def test_check_all_returns_summary(self):
        """check_all should return a HealthSummary."""
        # Mock all individual checks
        mock_health = ComponentHealth(name="mock", healthy=True, status="ok")

        with patch.object(HealthChecker, "check_data_sync", return_value=mock_health):
            with patch.object(HealthChecker, "check_training", return_value=mock_health):
                with patch.object(HealthChecker, "check_evaluation", return_value=mock_health):
                    with patch.object(HealthChecker, "check_coordinator", return_value=mock_health):
                        with patch.object(HealthChecker, "check_coordinators", return_value=mock_health):
                            with patch.object(HealthChecker, "check_resources", return_value=mock_health):
                                checker = HealthChecker()
                                summary = checker.check_all()

                                assert isinstance(summary, HealthSummary)
                                assert summary.healthy is True
                                assert len(summary.components) == 6

    def test_check_all_unhealthy_with_errors(self):
        """check_all should be unhealthy when any component has errors."""
        ok_health = ComponentHealth(name="ok", healthy=True, status="ok")
        error_health = ComponentHealth(
            name="error", healthy=False, status="error", message="Failed"
        )

        with patch.object(HealthChecker, "check_data_sync", return_value=error_health):
            with patch.object(HealthChecker, "check_training", return_value=ok_health):
                with patch.object(HealthChecker, "check_evaluation", return_value=ok_health):
                    with patch.object(HealthChecker, "check_coordinator", return_value=ok_health):
                        with patch.object(HealthChecker, "check_coordinators", return_value=ok_health):
                            with patch.object(HealthChecker, "check_resources", return_value=ok_health):
                                checker = HealthChecker()
                                summary = checker.check_all()

                                assert summary.healthy is False
                                assert len(summary.issues) == 1
                                assert "error" in summary.issues[0].lower()

    def test_check_all_collects_warnings(self):
        """check_all should collect warnings from components."""
        ok_health = ComponentHealth(name="ok", healthy=True, status="ok")
        warn_health = ComponentHealth(
            name="warn", healthy=False, status="warning", message="Stale data"
        )

        with patch.object(HealthChecker, "check_data_sync", return_value=warn_health):
            with patch.object(HealthChecker, "check_training", return_value=ok_health):
                with patch.object(HealthChecker, "check_evaluation", return_value=ok_health):
                    with patch.object(HealthChecker, "check_coordinator", return_value=ok_health):
                        with patch.object(HealthChecker, "check_coordinators", return_value=ok_health):
                            with patch.object(HealthChecker, "check_resources", return_value=ok_health):
                                checker = HealthChecker()
                                summary = checker.check_all()

                                # Warnings don't make summary unhealthy (only errors do)
                                assert summary.healthy is True
                                assert len(summary.warnings) == 1


class TestFormatHealthReport:
    """Tests for format_health_report function."""

    def test_format_healthy_report(self):
        """format_health_report should format healthy summary."""
        components = [
            ComponentHealth(name="sync", healthy=True, status="ok", message="5 games"),
            ComponentHealth(name="train", healthy=True, status="ok", message="Running"),
        ]
        summary = HealthSummary(
            healthy=True,
            timestamp="2024-01-01T00:00:00Z",
            components=components,
        )

        report = format_health_report(summary)

        assert "HEALTH CHECK REPORT" in report
        assert "HEALTHY" in report
        assert "sync" in report.lower()
        assert "train" in report.lower()
        assert "✓" in report

    def test_format_unhealthy_report(self):
        """format_health_report should format unhealthy summary."""
        components = [
            ComponentHealth(name="sync", healthy=False, status="error", message="Failed"),
        ]
        summary = HealthSummary(
            healthy=False,
            timestamp="2024-01-01T00:00:00Z",
            components=components,
            issues=["[sync] Failed"],
        )

        report = format_health_report(summary)

        assert "UNHEALTHY" in report
        assert "ISSUES:" in report
        assert "✗" in report

    def test_format_warning_report(self):
        """format_health_report should show warnings."""
        components = [
            ComponentHealth(name="sync", healthy=False, status="warning", message="Stale"),
        ]
        summary = HealthSummary(
            healthy=True,
            timestamp="2024-01-01T00:00:00Z",
            components=components,
            warnings=["[sync] Stale"],
        )

        report = format_health_report(summary)

        assert "WARNINGS:" in report
        assert "⚠" in report

    def test_format_with_last_activity(self):
        """format_health_report should show last activity time."""
        now = time.time()
        components = [
            ComponentHealth(
                name="sync",
                healthy=True,
                status="ok",
                message="OK",
                last_activity=now - 300,  # 5 minutes ago
            ),
        ]
        summary = HealthSummary(
            healthy=True,
            timestamp="2024-01-01T00:00:00Z",
            components=components,
        )

        report = format_health_report(summary)

        assert "Last activity:" in report
        assert "5 minutes" in report


class TestGetHealthSummary:
    """Tests for get_health_summary convenience function."""

    def test_get_health_summary_returns_summary(self):
        """get_health_summary should return HealthSummary."""
        mock_health = ComponentHealth(name="mock", healthy=True, status="ok")

        with patch.object(HealthChecker, "check_data_sync", return_value=mock_health):
            with patch.object(HealthChecker, "check_training", return_value=mock_health):
                with patch.object(HealthChecker, "check_evaluation", return_value=mock_health):
                    with patch.object(HealthChecker, "check_coordinator", return_value=mock_health):
                        with patch.object(HealthChecker, "check_coordinators", return_value=mock_health):
                            with patch.object(HealthChecker, "check_resources", return_value=mock_health):
                                summary = get_health_summary()

                                assert isinstance(summary, HealthSummary)


class TestHealthCheckerCoordinator:
    """Tests for HealthChecker.check_coordinator."""

    def test_coordinator_no_database(self):
        """check_coordinator should return ok when no database (standalone mode)."""
        checker = HealthChecker()
        checker.coordinator_db_path = Path("/nonexistent/path.db")

        health = checker.check_coordinator()

        assert health.healthy is True
        assert health.status == "ok"
        assert "standalone" in health.message.lower()

    def test_coordinator_with_database(self):
        """check_coordinator should check task registry."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            # Create task registry database
            conn = sqlite3.connect(str(db_path))
            conn.execute("""
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("INSERT INTO tasks (name) VALUES ('task1')")
            conn.execute("INSERT INTO tasks (name) VALUES ('task2')")
            conn.commit()
            conn.close()

            checker = HealthChecker()
            checker.coordinator_db_path = db_path

            health = checker.check_coordinator()

            assert health.healthy is True
            assert health.status == "ok"
            assert "2 active tasks" in health.message
        finally:
            db_path.unlink(missing_ok=True)


class TestHealthCheckerEvaluation:
    """Tests for HealthChecker.check_evaluation."""

    def test_evaluation_no_database(self):
        """check_evaluation should warn when no Elo database."""
        checker = HealthChecker()
        checker.elo_db_path = Path("/nonexistent/path.db")

        health = checker.check_evaluation()

        assert health.status == "warning"
        assert "not found" in health.message.lower()

    def test_evaluation_with_matches(self):
        """check_evaluation should count matches."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            # Create Elo database
            conn = sqlite3.connect(str(db_path))
            conn.execute("""
                CREATE TABLE match_history (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("INSERT INTO match_history DEFAULT VALUES")
            conn.execute("INSERT INTO match_history DEFAULT VALUES")
            conn.execute("INSERT INTO match_history DEFAULT VALUES")
            conn.commit()
            conn.close()

            checker = HealthChecker()
            checker.elo_db_path = db_path

            health = checker.check_evaluation()

            assert health.healthy is True
            assert health.status == "ok"
            assert "3 matches" in health.message
        finally:
            db_path.unlink(missing_ok=True)

    def test_evaluation_no_matches(self):
        """check_evaluation should handle empty match history."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("""
                CREATE TABLE match_history (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()

            checker = HealthChecker()
            checker.elo_db_path = db_path

            health = checker.check_evaluation()

            assert health.healthy is True
            assert "no evaluations" in health.message.lower()
        finally:
            db_path.unlink(missing_ok=True)
