"""Tests for PersistentEvaluationQueue.

Sprint 13 Session 4 (January 3, 2026): Part of model evaluation automation.

Comprehensive test suite covering:
- EvaluationRequest dataclass
- QueueStats dataclass
- RequestStatus constants
- PersistentEvaluationQueue functionality
- Factory functions and singleton pattern
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.coordination.evaluation_queue import (
    DEFAULT_QUEUE_PATH,
    DEFAULT_STUCK_TIMEOUT,
    EvaluationRequest,
    PersistentEvaluationQueue,
    QueueStats,
    RequestStatus,
    STUCK_TIMEOUT_SECONDS,
    get_evaluation_queue,
    reset_evaluation_queue,
)


# =============================================================================
# RequestStatus Tests
# =============================================================================


class TestRequestStatus:
    """Tests for RequestStatus constants."""

    def test_status_values(self):
        """Status values are correct strings."""
        assert RequestStatus.PENDING == "pending"
        assert RequestStatus.RUNNING == "running"
        assert RequestStatus.COMPLETED == "completed"
        assert RequestStatus.FAILED == "failed"


# =============================================================================
# QueueStats Tests
# =============================================================================


class TestQueueStats:
    """Tests for QueueStats dataclass."""

    def test_defaults(self):
        """Default statistics are zero."""
        stats = QueueStats()
        assert stats.pending_count == 0
        assert stats.running_count == 0
        assert stats.completed_count == 0
        assert stats.failed_count == 0
        assert stats.stuck_recoveries == 0
        assert stats.duplicate_requests_skipped == 0

    def test_custom_values(self):
        """Can set custom statistics."""
        stats = QueueStats(
            pending_count=10,
            running_count=2,
            completed_count=100,
            failed_count=5,
            stuck_recoveries=3,
            duplicate_requests_skipped=15,
        )
        assert stats.pending_count == 10
        assert stats.running_count == 2
        assert stats.completed_count == 100
        assert stats.failed_count == 5
        assert stats.stuck_recoveries == 3
        assert stats.duplicate_requests_skipped == 15


# =============================================================================
# EvaluationRequest Tests
# =============================================================================


class TestEvaluationRequest:
    """Tests for EvaluationRequest dataclass."""

    def _make_request(self, **kwargs) -> EvaluationRequest:
        """Create a request with defaults."""
        defaults = {
            "request_id": "test-id",
            "model_path": "/path/to/model.pth",
            "board_type": "hex8",
            "num_players": 2,
            "config_key": "hex8_2p",
            "status": RequestStatus.PENDING,
            "priority": 50,
            "created_at": time.time(),
            "started_at": 0.0,
            "completed_at": 0.0,
            "attempts": 0,
            "max_attempts": 3,
            "error": "",
            "result_elo": None,
            "source": "training",
        }
        defaults.update(kwargs)
        return EvaluationRequest(**defaults)

    def test_basic_creation(self):
        """Can create a request with all fields."""
        request = self._make_request()
        assert request.request_id == "test-id"
        assert request.model_path == "/path/to/model.pth"
        assert request.board_type == "hex8"
        assert request.num_players == 2

    def test_is_stuck_pending(self):
        """Pending requests are never stuck."""
        request = self._make_request(status=RequestStatus.PENDING)
        assert request.is_stuck is False

    def test_is_stuck_completed(self):
        """Completed requests are never stuck."""
        request = self._make_request(status=RequestStatus.COMPLETED)
        assert request.is_stuck is False

    def test_is_stuck_running_not_long_enough(self):
        """Running request that hasn't exceeded timeout is not stuck."""
        request = self._make_request(
            status=RequestStatus.RUNNING,
            started_at=time.time() - 100,  # 100 seconds ago
            board_type="hex8",  # 1 hour timeout
        )
        assert request.is_stuck is False

    def test_is_stuck_running_too_long(self):
        """Running request that exceeded timeout is stuck."""
        request = self._make_request(
            status=RequestStatus.RUNNING,
            started_at=time.time() - 4000,  # Over 1 hour ago
            board_type="hex8",  # 1 hour timeout
        )
        assert request.is_stuck is True

    def test_is_stuck_uses_board_timeout(self):
        """Stuck detection uses board-specific timeout."""
        # square19 has 3 hour timeout
        request = self._make_request(
            status=RequestStatus.RUNNING,
            started_at=time.time() - 8000,  # ~2.2 hours ago
            board_type="square19",
        )
        assert request.is_stuck is False  # Not over 3 hours

    def test_to_dict(self):
        """Can convert request to dictionary."""
        request = self._make_request()
        d = request.to_dict()
        assert d["request_id"] == "test-id"
        assert d["model_path"] == "/path/to/model.pth"
        assert d["board_type"] == "hex8"
        assert d["num_players"] == 2
        assert d["status"] == RequestStatus.PENDING


# =============================================================================
# PersistentEvaluationQueue Tests
# =============================================================================


class TestPersistentEvaluationQueue:
    """Tests for PersistentEvaluationQueue."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def queue(self, temp_db):
        """Create a queue with temporary database."""
        return PersistentEvaluationQueue(db_path=temp_db)

    def test_init_creates_database(self, temp_db):
        """Initializing queue creates database file."""
        queue = PersistentEvaluationQueue(db_path=temp_db)
        assert temp_db.exists()

    def test_init_creates_table(self, temp_db):
        """Initializing queue creates table with correct schema."""
        queue = PersistentEvaluationQueue(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='evaluation_requests'"
            )
            assert cursor.fetchone() is not None

    def test_init_creates_indexes(self, temp_db):
        """Initializing queue creates indexes."""
        queue = PersistentEvaluationQueue(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = {row[0] for row in cursor.fetchall()}
            assert "idx_status_priority" in indexes
            assert "idx_started_at" in indexes
            assert "idx_model_path" in indexes

    def test_add_request_basic(self, queue):
        """Can add a basic evaluation request."""
        request_id = queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
        )
        assert request_id is not None

    def test_add_request_with_priority(self, queue):
        """Can add request with custom priority."""
        request_id = queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
            priority=100,
        )

        # Verify priority was set
        request = queue.claim_next()
        assert request.priority == 100

    def test_add_request_duplicate_pending(self, queue):
        """Adding duplicate pending request returns existing ID."""
        id1 = queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
            priority=50,
        )
        id2 = queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
            priority=50,
        )
        assert id1 == id2

    def test_add_request_duplicate_higher_priority(self, queue):
        """Adding duplicate with higher priority updates it."""
        id1 = queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
            priority=50,
        )
        id2 = queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
            priority=100,  # Higher priority
        )
        assert id1 == id2

        # Verify priority was updated
        request = queue.claim_next()
        assert request.priority == 100

    def test_claim_next_empty_queue(self, queue):
        """claim_next returns None for empty queue."""
        assert queue.claim_next() is None

    def test_claim_next_returns_request(self, queue):
        """claim_next returns request and marks it running."""
        queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
        )

        request = queue.claim_next()
        assert request is not None
        assert request.status == RequestStatus.RUNNING

    def test_claim_next_priority_order(self, queue):
        """claim_next returns highest priority request first."""
        queue.add_request(
            model_path="/path/low.pth",
            board_type="hex8",
            num_players=2,
            priority=25,
        )
        queue.add_request(
            model_path="/path/high.pth",
            board_type="hex8",
            num_players=2,
            priority=100,
        )
        queue.add_request(
            model_path="/path/medium.pth",
            board_type="hex8",
            num_players=2,
            priority=50,
        )

        request = queue.claim_next()
        assert request.model_path == "/path/high.pth"

    def test_complete_marks_completed(self, queue):
        """complete marks request as completed with Elo."""
        request_id = queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
        )
        request = queue.claim_next()

        queue.complete(request.request_id, elo=1450.0)

        # Verify status
        assert queue.get_pending_count() == 0

    def test_fail_marks_failed(self, queue):
        """fail marks request as failed with error."""
        request_id = queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
        )
        request = queue.claim_next()

        queue.fail(request.request_id, "GPU OOM")

        # Verify status (if not retrying)
        # Request should go back to pending or failed
        request = queue.get_request(request.request_id)
        assert request.error == "GPU OOM"

    def test_get_stuck_evaluations(self, queue):
        """get_stuck_evaluations returns stuck requests."""
        # Add and claim a request
        queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
        )
        request = queue.claim_next()

        # Not stuck yet (just claimed)
        stuck = queue.get_stuck_evaluations()
        assert len(stuck) == 0

        # Manually set started_at to long ago
        with queue._get_connection() as conn:
            conn.execute(
                "UPDATE evaluation_requests SET started_at = ? WHERE request_id = ?",
                (time.time() - 5000, request.request_id),
            )
            conn.commit()

        stuck = queue.get_stuck_evaluations()
        assert len(stuck) == 1
        assert stuck[0].request_id == request.request_id

    def test_reset_stuck(self, queue):
        """reset_stuck resets stuck request to pending."""
        # Add and claim a request
        queue.add_request(
            model_path="/path/to/model.pth",
            board_type="hex8",
            num_players=2,
        )
        request = queue.claim_next()

        # Reset it
        queue.reset_stuck(request.request_id)

        # Should be claimable again
        request2 = queue.claim_next()
        assert request2.request_id == request.request_id
        # attempts increases from claim + reset
        assert request2.attempts >= 1

    def test_pending_count_after_operations(self, queue):
        """Pending count reflects queue state correctly."""
        # Add requests
        queue.add_request(model_path="/path/a.pth", board_type="hex8", num_players=2)
        queue.add_request(model_path="/path/b.pth", board_type="hex8", num_players=2)
        queue.add_request(model_path="/path/c.pth", board_type="hex8", num_players=2)

        # Verify 3 pending
        assert queue.get_pending_count() == 3

        # Claim one
        request = queue.claim_next()

        # Verify 2 pending
        assert queue.get_pending_count() == 2

        # Complete one
        queue.complete(request.request_id, elo=1400.0)

        # Still 2 pending (completed doesn't change pending count)
        assert queue.get_pending_count() == 2

    def test_health_check(self, queue):
        """health_check returns HealthCheckResult."""
        result = queue.health_check()

        assert result is not None
        assert hasattr(result, "healthy")
        assert result.healthy is True

    def test_get_pending_count(self, queue):
        """get_pending_count returns correct count."""
        assert queue.get_pending_count() == 0

        queue.add_request(model_path="/path/a.pth", board_type="hex8", num_players=2)
        queue.add_request(model_path="/path/b.pth", board_type="hex8", num_players=2)

        assert queue.get_pending_count() == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory and singleton functions."""

    def test_get_evaluation_queue_returns_queue(self):
        """get_evaluation_queue returns a PersistentEvaluationQueue."""
        reset_evaluation_queue()  # Ensure clean state
        queue = get_evaluation_queue()
        assert isinstance(queue, PersistentEvaluationQueue)

    def test_get_evaluation_queue_singleton(self):
        """get_evaluation_queue returns same instance."""
        reset_evaluation_queue()
        queue1 = get_evaluation_queue()
        queue2 = get_evaluation_queue()
        assert queue1 is queue2

    def test_reset_evaluation_queue(self):
        """reset_evaluation_queue clears singleton."""
        reset_evaluation_queue()
        queue1 = get_evaluation_queue()
        reset_evaluation_queue()
        queue2 = get_evaluation_queue()
        assert queue1 is not queue2


# =============================================================================
# Stuck Timeout Constants Tests
# =============================================================================


class TestStuckTimeouts:
    """Tests for stuck timeout constants."""

    def test_stuck_timeout_hex8(self):
        """hex8 has 1 hour timeout."""
        assert STUCK_TIMEOUT_SECONDS["hex8"] == 3600

    def test_stuck_timeout_square8(self):
        """square8 has 2 hour timeout."""
        assert STUCK_TIMEOUT_SECONDS["square8"] == 7200

    def test_stuck_timeout_square19(self):
        """square19 has 3 hour timeout."""
        assert STUCK_TIMEOUT_SECONDS["square19"] == 10800

    def test_stuck_timeout_hexagonal(self):
        """hexagonal has 4 hour timeout."""
        assert STUCK_TIMEOUT_SECONDS["hexagonal"] == 14400

    def test_default_stuck_timeout(self):
        """Default timeout is 2 hours."""
        assert DEFAULT_STUCK_TIMEOUT == 7200
