"""Tests for game balance analysis.

Tests for app/analysis/game_balance.py covering:
- GameBalanceAnalyzer initialization
- WinRateStats computation with edge cases
- Balance issue detection
- Statistical calculations
- Empty database handling
- Error conditions
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy import stats

from app.analysis.game_balance import (
    BalanceIssue,
    BalanceReport,
    GameBalanceAnalyzer,
    GameLengthStats,
    WinRateStats,
    analyze_game_balance,
)


class TestWinRateStats:
    """Tests for WinRateStats dataclass."""

    def test_advantage_calculation(self):
        """Should calculate advantage as difference from expected rate."""
        stats = WinRateStats(
            wins=60,
            losses=40,
            draws=0,
            total_games=100,
            win_rate=0.6,
            confidence_interval=(0.5, 0.7),
            expected_rate=0.5,
            is_significant=True,
        )
        assert abs(stats.advantage - 0.1) < 1e-10

    def test_negative_advantage(self):
        """Should handle negative advantage (disadvantage)."""
        stats = WinRateStats(
            wins=40,
            losses=60,
            draws=0,
            total_games=100,
            win_rate=0.4,
            confidence_interval=(0.3, 0.5),
            expected_rate=0.5,
            is_significant=True,
        )
        assert abs(stats.advantage - (-0.1)) < 1e-10

    def test_zero_advantage(self):
        """Should handle zero advantage (perfectly balanced)."""
        stats = WinRateStats(
            wins=50,
            losses=50,
            draws=0,
            total_games=100,
            win_rate=0.5,
            confidence_interval=(0.4, 0.6),
            expected_rate=0.5,
            is_significant=False,
        )
        assert stats.advantage == 0.0


class TestGameBalanceAnalyzer:
    """Tests for GameBalanceAnalyzer class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
        temp_file.close()
        db_path = Path(temp_file.name)

        # Create database schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                winner INTEGER,
                move_history TEXT,
                status TEXT,
                created_at TEXT,
                completed_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        db_path.unlink()

    @pytest.fixture
    def analyzer(self, temp_db):
        """Create analyzer with temporary database."""
        return GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)

    def test_initialization(self, temp_db):
        """Should initialize with database path and optional filters."""
        analyzer = GameBalanceAnalyzer(temp_db, board_type="hex8", num_players=4)
        assert analyzer.db_path == temp_db
        assert analyzer.board_type == "hex8"
        assert analyzer.num_players == 4

    def test_initialization_without_filters(self, temp_db):
        """Should initialize without board type or player filters."""
        analyzer = GameBalanceAnalyzer(temp_db)
        assert analyzer.db_path == temp_db
        assert analyzer.board_type is None
        assert analyzer.num_players is None

    def test_empty_database(self, analyzer):
        """Should handle empty database gracefully."""
        report = analyzer.analyze()
        assert report.total_games == 0
        assert report.is_balanced is True
        assert report.summary == "No games available for analysis."
        assert len(report.balance_issues) == 0
        assert report.draw_rate == 0.0

    def test_calculate_win_rate_stats_zero_games(self, analyzer):
        """Should handle zero games edge case."""
        stats = analyzer._calculate_win_rate_stats(0, 0, 0, 0.5)
        assert stats.total_games == 0
        assert stats.win_rate == 0.0
        assert stats.confidence_interval == (0.0, 0.0)
        assert stats.is_significant is False

    def test_calculate_win_rate_stats_all_wins(self, analyzer):
        """Should handle all wins edge case."""
        stats = analyzer._calculate_win_rate_stats(100, 0, 0, 0.5)
        assert stats.total_games == 100
        assert stats.win_rate == 1.0
        assert stats.wins == 100
        assert stats.losses == 0
        assert stats.draws == 0

    def test_calculate_win_rate_stats_all_draws(self, analyzer):
        """Should handle all draws edge case."""
        stats = analyzer._calculate_win_rate_stats(0, 0, 100, 0.5)
        assert stats.total_games == 100
        assert stats.win_rate == 0.0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.draws == 100

    def test_calculate_win_rate_stats_confidence_interval(self, analyzer):
        """Should calculate confidence interval correctly."""
        stats = analyzer._calculate_win_rate_stats(60, 40, 0, 0.5)
        # Wilson score interval should be within [0, 1]
        assert 0 <= stats.confidence_interval[0] <= 1
        assert 0 <= stats.confidence_interval[1] <= 1
        # Lower bound should be less than upper bound
        assert stats.confidence_interval[0] <= stats.confidence_interval[1]
        # Win rate should be within CI
        assert stats.confidence_interval[0] <= stats.win_rate <= stats.confidence_interval[1]

    def test_calculate_win_rate_stats_significance_test(self, analyzer):
        """Should perform statistical significance test."""
        # Large deviation should be significant
        stats_significant = analyzer._calculate_win_rate_stats(70, 30, 0, 0.5)
        assert stats_significant.is_significant == True

        # Small sample or no deviation should not be significant
        stats_not_significant = analyzer._calculate_win_rate_stats(5, 5, 0, 0.5)
        assert stats_not_significant.is_significant == False

    def test_calculate_game_length_stats_empty(self, analyzer):
        """Should handle empty game list."""
        stats = analyzer._calculate_game_length_stats([])
        assert stats.mean == 0
        assert stats.median == 0
        assert stats.std == 0
        assert stats.min == 0
        assert stats.max == 0
        assert all(v == 0 for v in stats.percentiles.values())

    def test_calculate_game_length_stats(self, temp_db):
        """Should calculate game length statistics correctly."""
        # Insert test games with move histories
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(10):
            move_history = json.dumps([{"move": j} for j in range(i * 10 + 10)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, 0, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db)
        games = analyzer._load_games()
        stats = analyzer._calculate_game_length_stats(games)

        assert stats.mean > 0
        assert stats.median > 0
        assert stats.std >= 0
        assert stats.min <= stats.median <= stats.max
        assert stats.percentiles[50] == stats.median

    def test_calculate_game_length_stats_invalid_json(self, temp_db):
        """Should handle invalid JSON in move history."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("game-1", "square8", 2, 0, "invalid-json", "completed", "2025-01-01", "2025-01-01"),
        )
        cursor.execute(
            "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("game-2", "square8", 2, 0, json.dumps([{"move": 1}]), "completed", "2025-01-01", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db)
        games = analyzer._load_games()
        stats = analyzer._calculate_game_length_stats(games)

        # Should skip invalid JSON and only count valid game
        assert stats.mean == 1.0
        assert stats.min == 1
        assert stats.max == 1

    def test_analyze_balanced_game(self, temp_db):
        """Should detect balanced game with equal win distribution."""
        # Insert 100 games with 50/50 win distribution
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(100):
            winner = i % 2  # Alternating winners
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, winner, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)
        report = analyzer.analyze()

        assert report.total_games == 100
        assert report.board_type == "square8"
        assert report.num_players == 2
        assert abs(report.first_player_advantage) < 0.05  # Should be near zero
        assert report.is_balanced is True
        # With perfectly balanced wins, should have no major issues
        major_issues = [i for i in report.balance_issues if i.severity == "major"]
        assert len(major_issues) == 0

    def test_analyze_first_player_advantage(self, temp_db):
        """Should detect first-player advantage."""
        # Insert 100 games where player 0 wins 70% of the time
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(100):
            winner = 0 if i < 70 else 1
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, winner, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)
        report = analyzer.analyze()

        assert report.first_player_advantage > 0.15  # Should be 0.2 (70% - 50%)
        assert report.is_balanced is False
        # Should have first-player advantage issue
        fp_issues = [i for i in report.balance_issues if i.category == "first_player"]
        assert len(fp_issues) > 0
        assert fp_issues[0].severity in ["moderate", "major"]

    def test_analyze_high_draw_rate(self, temp_db):
        """Should detect high draw rate."""
        # Insert 100 games with 40% draws
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(100):
            winner = None if i < 40 else i % 2
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, winner, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)
        report = analyzer.analyze()

        assert report.draw_rate == 0.4
        # Should have draw rate issue
        draw_issues = [i for i in report.balance_issues if i.category == "draw_rate"]
        assert len(draw_issues) > 0

    def test_analyze_game_length_variance(self, temp_db):
        """Should detect high variance in game length."""
        # Insert games with highly variable lengths
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        lengths = [10, 20, 100, 200, 300]  # High variance
        for i, length in enumerate(lengths * 20):  # 100 games total
            move_history = json.dumps([{"move": j} for j in range(length)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, i % 2, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)
        report = analyzer.analyze()

        # Check if length variance issue is detected
        length_issues = [i for i in report.balance_issues if i.category == "game_length"]
        assert len(length_issues) > 0

    def test_analyze_multiplayer(self, temp_db):
        """Should analyze 4-player games correctly."""
        # Insert 100 4-player games
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(100):
            winner = i % 4  # Rotate through players
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 4, winner, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=4)
        report = analyzer.analyze()

        assert report.num_players == 4
        assert len(report.player_win_rates) == 4
        # Each player should have ~25% win rate
        for player, stats in report.player_win_rates.items():
            assert abs(stats.win_rate - 0.25) < 0.05
            assert stats.expected_rate == 0.25

    def test_find_balance_issues(self, temp_db):
        """Should find and filter balance issues."""
        # Insert games with first-player advantage
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(100):
            winner = 0 if i < 65 else 1
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, winner, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)

        # Find all issues
        all_issues = analyzer.find_balance_issues()
        assert len(all_issues) > 0

        # Filter by severity
        major_issues = analyzer.find_balance_issues(severity_filter="major")
        moderate_issues = analyzer.find_balance_issues(severity_filter="moderate")
        minor_issues = analyzer.find_balance_issues(severity_filter="minor")

        assert len(all_issues) >= len(major_issues)
        assert len(all_issues) >= len(moderate_issues)
        assert len(all_issues) >= len(minor_issues)

    def test_generate_report(self, temp_db):
        """Should generate and optionally save report."""
        # Insert test games
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(50):
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, i % 2, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)

        # Generate without saving
        report_text = analyzer.generate_report()
        assert "GAME BALANCE ANALYSIS REPORT" in report_text
        assert "square8" in report_text
        assert "50 games" in report_text

    def test_generate_report_saves_to_file(self, temp_db, tmp_path):
        """Should save report to file when output_path provided."""
        # Insert test games
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(50):
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, i % 2, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)
        output_path = tmp_path / "balance_report.txt"

        report_text = analyzer.generate_report(output_path=output_path)

        assert output_path.exists()
        saved_text = output_path.read_text()
        assert saved_text == report_text

    def test_track_balance_over_time(self, temp_db):
        """Should track balance metrics over time."""
        # Insert 200 games chronologically
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(200):
            winner = 0 if i < 120 else 1  # First player advantage in early games
            move_history = json.dumps([{"move": j} for j in range(30)])
            created_at = f"2025-01-{(i // 10) + 1:02d}"
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, winner, move_history, "completed", created_at, created_at),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)
        snapshots = analyzer.track_balance_over_time(window_size=50, step_size=50)

        assert len(snapshots) > 0
        for snapshot in snapshots:
            assert "first_player_advantage" in snapshot
            assert "draw_rate" in snapshot
            assert "games_in_window" in snapshot
            assert snapshot["games_in_window"] == 50

    def test_analyze_all_configs(self, temp_db):
        """Should analyze multiple configurations."""
        # Insert games for different configs
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        configs = [
            ("square8", 2),
            ("square8", 3),
            ("square19", 2),
        ]
        for board_type, num_players in configs:
            for i in range(30):
                move_history = json.dumps([{"move": j} for j in range(20)])
                cursor.execute(
                    "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"game-{board_type}-{num_players}-{i}",
                        board_type,
                        num_players,
                        i % num_players,
                        move_history,
                        "completed",
                        "2025-01-01",
                        "2025-01-01",
                    ),
                )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db)
        cross_analysis = analyzer.analyze_all_configs()

        assert len(cross_analysis.per_config_stats) > 0
        # Check that we have some configs analyzed
        assert any("square8_2p" in key for key in cross_analysis.per_config_stats.keys())

    def test_load_games_with_filters(self, temp_db):
        """Should load games with board type and player filters."""
        # Insert games for multiple configs
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(30):
            move_history = json.dumps([{"move": j} for j in range(20)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-hex-{i}", "hex8", 2, i % 2, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-sq-{i}", "square8", 4, i % 4, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="hex8", num_players=2)
        games = analyzer._load_games()

        assert len(games) == 30
        assert all(g["board_type"] == "hex8" for g in games)
        assert all(g["num_players"] == 2 for g in games)

    def test_load_games_with_limit(self, temp_db):
        """Should respect limit parameter."""
        # Insert 100 games
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(100):
            move_history = json.dumps([{"move": j} for j in range(20)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, i % 2, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db)
        games = analyzer._load_games(limit=50)

        assert len(games) == 50


class TestAnalyzeGameBalanceFunction:
    """Tests for the convenience function analyze_game_balance."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
        temp_file.close()
        db_path = Path(temp_file.name)

        # Create database schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                winner INTEGER,
                move_history TEXT,
                status TEXT,
                created_at TEXT,
                completed_at TEXT
            )
        """)
        # Insert test games
        for i in range(50):
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, i % 2, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        db_path.unlink()

    def test_convenience_function(self, temp_db):
        """Should run analysis using convenience function."""
        report = analyze_game_balance(temp_db, board_type="square8", num_players=2)
        assert isinstance(report, BalanceReport)
        assert report.total_games == 50
        assert report.board_type == "square8"
        assert report.num_players == 2

    def test_convenience_function_with_output(self, temp_db, tmp_path):
        """Should save report when output_path provided."""
        output_path = tmp_path / "report.txt"
        report = analyze_game_balance(
            temp_db, board_type="square8", num_players=2, output_path=output_path
        )
        assert isinstance(report, BalanceReport)
        assert output_path.exists()


class TestBalanceIssueDetection:
    """Tests for specific balance issue detection logic."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
        temp_file.close()
        db_path = Path(temp_file.name)

        # Create database schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER,
                winner INTEGER,
                move_history TEXT,
                status TEXT,
                created_at TEXT,
                completed_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        db_path.unlink()

    def test_minor_first_player_advantage(self, temp_db):
        """Should detect minor first-player advantage (5-10% deviation)."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        # 55% win rate for player 0
        for i in range(100):
            winner = 0 if i < 55 else 1
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, winner, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)
        report = analyzer.analyze()

        fp_issues = [i for i in report.balance_issues if i.category == "first_player"]
        assert len(fp_issues) > 0
        assert fp_issues[0].severity == "minor"

    def test_moderate_first_player_advantage(self, temp_db):
        """Should detect moderate first-player advantage (10-15% deviation)."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        # 62% win rate for player 0
        for i in range(100):
            winner = 0 if i < 62 else 1
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, winner, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)
        report = analyzer.analyze()

        fp_issues = [i for i in report.balance_issues if i.category == "first_player"]
        assert len(fp_issues) > 0
        assert fp_issues[0].severity == "moderate"

    def test_major_first_player_advantage(self, temp_db):
        """Should detect major first-player advantage (>15% deviation)."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        # 70% win rate for player 0
        for i in range(100):
            winner = 0 if i < 70 else 1
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 2, winner, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=2)
        report = analyzer.analyze()

        fp_issues = [i for i in report.balance_issues if i.category == "first_player"]
        assert len(fp_issues) > 0
        assert fp_issues[0].severity == "major"

    def test_player_position_imbalance(self, temp_db):
        """Should detect imbalance in non-first player positions."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        # 4-player game where player 2 has significant advantage
        for i in range(100):
            # Player 2 wins 45%, others split remaining 55%
            if i < 45:
                winner = 2
            else:
                winner = (i - 45) % 3
                if winner >= 2:
                    winner += 1  # Skip player 2
            move_history = json.dumps([{"move": j} for j in range(30)])
            cursor.execute(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"game-{i}", "square8", 4, winner, move_history, "completed", "2025-01-01", "2025-01-01"),
            )
        conn.commit()
        conn.close()

        analyzer = GameBalanceAnalyzer(temp_db, board_type="square8", num_players=4)
        report = analyzer.analyze()

        position_issues = [i for i in report.balance_issues if i.category == "player_position"]
        # Player 2 with 45% vs expected 25% should trigger position imbalance
        assert len(position_issues) > 0
