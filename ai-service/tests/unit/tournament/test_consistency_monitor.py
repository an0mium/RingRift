"""Tests for consistency monitor."""

import pytest

from app.tournament.consistency_monitor import (
    ConsistencyMonitor,
    ConsistencyReport,
    InvariantCheck,
)


class TestInvariantCheck:
    """Tests for InvariantCheck dataclass."""

    def test_invariant_check_creation(self):
        """Test invariant check creation."""
        check = InvariantCheck(
            name="test_check",
            passed=True,
            message="All good",
            severity="info",
        )

        assert check.name == "test_check"
        assert check.passed is True
        assert check.message == "All good"
        assert check.severity == "info"

    def test_failed_check(self):
        """Test failed invariant check."""
        check = InvariantCheck(
            name="failed_check",
            passed=False,
            message="Something wrong",
            severity="error",
        )

        assert check.passed is False
        assert check.severity == "error"


class TestConsistencyReport:
    """Tests for ConsistencyReport dataclass."""

    def test_report_creation(self):
        """Test consistency report creation."""
        report = ConsistencyReport(
            board_type="square8",
            num_players=2,
        )

        assert report.board_type == "square8"
        assert report.num_players == 2
        assert report.overall_healthy is True
        assert len(report.checks) == 0

    def test_report_with_checks(self):
        """Test consistency report with checks."""
        checks = [
            InvariantCheck(name="check1", passed=True, message="OK", severity="info"),
            InvariantCheck(name="check2", passed=False, message="Fail", severity="error"),
        ]
        report = ConsistencyReport(
            board_type="square8",
            num_players=2,
            checks=checks,
            overall_healthy=False,
        )

        assert len(report.checks) == 2
        assert report.overall_healthy is False
        assert len(report.errors) == 1
        assert len(report.warnings) == 0


class TestConsistencyMonitor:
    """Tests for ConsistencyMonitor."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = ConsistencyMonitor(
            board_type="square8",
            num_players=2,
        )

        assert monitor.board_type == "square8"
        assert monitor.num_players == 2

    @pytest.mark.parametrize("board_type", ["square8", "square19", "hexagonal"])
    def test_different_board_types(self, board_type):
        """Test monitor works with different board types."""
        monitor = ConsistencyMonitor(
            board_type=board_type,
            num_players=2,
        )

        assert monitor.board_type == board_type

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_different_player_counts(self, num_players):
        """Test monitor works with different player counts."""
        monitor = ConsistencyMonitor(
            board_type="square8",
            num_players=num_players,
        )

        assert monitor.num_players == num_players

    def test_run_all_checks_returns_report(self):
        """Test run_all_checks returns a ConsistencyReport."""
        monitor = ConsistencyMonitor(
            board_type="square8",
            num_players=2,
        )

        report = monitor.run_all_checks()

        assert isinstance(report, ConsistencyReport)
        assert report.board_type == "square8"
        assert report.num_players == 2

    def test_check_nn_ranking_consistency(self):
        """Test NN ranking consistency check."""
        monitor = ConsistencyMonitor(
            board_type="square8",
            num_players=2,
        )

        check = monitor.check_nn_ranking_consistency()

        assert isinstance(check, InvariantCheck)
        assert check.name == "NN Ranking Consistency"

    def test_check_algorithm_ranking_stability(self):
        """Test algorithm ranking stability check."""
        monitor = ConsistencyMonitor(
            board_type="square8",
            num_players=2,
        )

        check = monitor.check_algorithm_ranking_stability()

        assert isinstance(check, InvariantCheck)
        assert check.name == "Algorithm Ranking Stability"

    def test_check_elo_transitivity(self):
        """Test ELO transitivity check."""
        monitor = ConsistencyMonitor(
            board_type="square8",
            num_players=2,
        )

        check = monitor.check_elo_transitivity()

        assert isinstance(check, InvariantCheck)
        assert check.name == "Elo Transitivity"
