"""Work Queue Backend Abstraction (Jan 2, 2026).

Provides Strategy pattern for work queue backends, eliminating conditional
Raft/SQLite code paths in WorkQueue class.

Architecture:
    WorkQueue
        └── WorkQueueBackend (abstract)
                ├── SQLiteBackend - Local persistence
                └── RaftBackend - Cluster-wide via Raft, falls back to SQLite

Usage:
    from app.coordination.work_queue_backends import (
        create_backend,
        WorkQueueBackend,
        BackendType,
    )

    # Create backend (auto-selects Raft if available)
    backend = create_backend()

    # Operations are backend-agnostic
    backend.add_item(work_id, work_data)
    claimed = backend.claim_item(work_id, node_id)
    stats = backend.get_stats()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkItem

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Work queue backend types."""

    RAFT = "raft"
    SQLITE = "sqlite"


@dataclass
class BackendResult:
    """Result from a backend operation.

    Attributes:
        success: Whether operation succeeded
        data: Optional result data (e.g., claimed work item)
        error: Optional error message if failed
        fallback_used: True if Raft failed and SQLite was used
    """

    success: bool
    data: Any = None
    error: str | None = None
    fallback_used: bool = False


class WorkQueueBackend(ABC):
    """Abstract base class for work queue backends.

    Defines the data-layer interface for work queue operations.
    Implementations handle persistence (SQLite) or replication (Raft).

    The WorkQueue class handles business logic (events, stats, caching)
    while backends handle the data layer.
    """

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type identifier."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is currently operational."""
        ...

    @abstractmethod
    def add_item(self, work_id: str, work_data: dict[str, Any]) -> BackendResult:
        """Add a work item to the queue.

        Args:
            work_id: Unique identifier for the work
            work_data: Serialized work item data

        Returns:
            BackendResult with success status
        """
        ...

    @abstractmethod
    def claim_item(
        self,
        work_id: str,
        node_id: str,
        claimed_at: float | None = None,
    ) -> BackendResult:
        """Attempt to claim a specific work item.

        Args:
            work_id: Work item to claim
            node_id: Node claiming the work
            claimed_at: Timestamp of claim (defaults to now)

        Returns:
            BackendResult with success=True if claimed
        """
        ...

    def claim_items_batch(
        self,
        work_ids: list[str],
        node_id: str,
        claimed_at: float | None = None,
    ) -> BackendResult:
        """Attempt to claim multiple work items in a single transaction.

        Session 17.50 (Jan 30, 2026): Added to reduce database overhead when
        batch claiming work. Default implementation calls claim_item() for each,
        but subclasses can override for optimized single-transaction behavior.

        Args:
            work_ids: Work item IDs to claim
            node_id: Node claiming the work
            claimed_at: Timestamp of claim (defaults to now)

        Returns:
            BackendResult with data containing list of successfully claimed work_ids
        """
        # Default implementation: claim one at a time
        claimed = []
        for work_id in work_ids:
            result = self.claim_item(work_id, node_id, claimed_at)
            if result.success:
                claimed.append(work_id)
        return BackendResult(
            success=len(claimed) > 0,
            data={"claimed_ids": claimed, "node_id": node_id},
        )

    @abstractmethod
    def get_pending_items(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get pending work items for claim consideration.

        Args:
            limit: Maximum items to return

        Returns:
            List of work item dicts, sorted by priority (highest first)
        """
        ...

    @abstractmethod
    def start_item(self, work_id: str, started_at: float | None = None) -> BackendResult:
        """Mark work as started (running).

        Args:
            work_id: Work item to start
            started_at: Timestamp (defaults to now)

        Returns:
            BackendResult with success status
        """
        ...

    @abstractmethod
    def complete_item(
        self,
        work_id: str,
        result: dict[str, Any] | None = None,
        completed_at: float | None = None,
    ) -> BackendResult:
        """Mark work as completed.

        Args:
            work_id: Work item to complete
            result: Optional result data
            completed_at: Timestamp (defaults to now)

        Returns:
            BackendResult with success status
        """
        ...

    @abstractmethod
    def fail_item(
        self,
        work_id: str,
        error: str,
        permanent: bool,
        completed_at: float | None = None,
    ) -> BackendResult:
        """Mark work as failed.

        Args:
            work_id: Work item that failed
            error: Error message
            permanent: If True, no more retries
            completed_at: Timestamp for permanent failures

        Returns:
            BackendResult with success status
        """
        ...

    @abstractmethod
    def get_item(self, work_id: str) -> dict[str, Any] | None:
        """Get a specific work item.

        Args:
            work_id: Work item ID

        Returns:
            Work item dict or None if not found
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dict with counts by status, total items, etc.
        """
        ...

    @abstractmethod
    def update_item_config(self, work_id: str, config: dict[str, Any]) -> BackendResult:
        """Update a work item's config.

        Used for clearing stale target_node, updating config fields.

        Args:
            work_id: Work item to update
            config: New config dict

        Returns:
            BackendResult with success status
        """
        ...


class SQLiteBackend(WorkQueueBackend):
    """SQLite-based work queue backend.

    Provides local persistence using SQLite database.
    This is the fallback backend when Raft is unavailable.
    """

    def __init__(
        self,
        db_path: Path,
        get_connection: Callable[[], sqlite3.Connection],
        readonly_mode: bool = False,
    ):
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database
            get_connection: Factory function to get database connections
            readonly_mode: If True, skip write operations
        """
        self._db_path = db_path
        self._get_connection = get_connection
        self._readonly_mode = readonly_mode

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SQLITE

    @property
    def is_available(self) -> bool:
        return self._db_path.exists() and not self._readonly_mode

    def add_item(self, work_id: str, work_data: dict[str, Any]) -> BackendResult:
        if self._readonly_mode:
            return BackendResult(success=False, error="Backend is readonly")

        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO work_items
                    (work_id, work_type, priority, config, created_at, claimed_at,
                     started_at, completed_at, status, claimed_by, attempts,
                     max_attempts, timeout_seconds, result, error, depends_on)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        work_id,
                        work_data.get("work_type", "selfplay"),
                        work_data.get("priority", 50),
                        json.dumps(work_data.get("config", {})),
                        work_data.get("created_at", time.time()),
                        work_data.get("claimed_at", 0.0),
                        work_data.get("started_at", 0.0),
                        work_data.get("completed_at", 0.0),
                        work_data.get("status", "pending"),
                        work_data.get("claimed_by", ""),
                        work_data.get("attempts", 0),
                        work_data.get("max_attempts", 3),
                        work_data.get("timeout_seconds", 3600.0),
                        json.dumps(work_data.get("result", {})),
                        work_data.get("error", ""),
                        json.dumps(work_data.get("depends_on", [])),
                    ),
                )
                conn.commit()
                return BackendResult(success=True, data=work_id)
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to add item {work_id}: {e}")
            return BackendResult(success=False, error=str(e))

    def claim_item(
        self,
        work_id: str,
        node_id: str,
        claimed_at: float | None = None,
    ) -> BackendResult:
        if self._readonly_mode:
            return BackendResult(success=False, error="Backend is readonly")

        claimed_at = claimed_at or time.time()
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Atomic claim: only update if still pending
                cursor.execute(
                    """
                    UPDATE work_items
                    SET status = 'claimed', claimed_by = ?, claimed_at = ?, attempts = attempts + 1
                    WHERE work_id = ? AND status = 'pending'
                    """,
                    (node_id, claimed_at, work_id),
                )
                conn.commit()
                if cursor.rowcount > 0:
                    return BackendResult(success=True, data={"work_id": work_id, "node_id": node_id})
                return BackendResult(success=False, error="Item not pending or not found")
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to claim {work_id}: {e}")
            return BackendResult(success=False, error=str(e))

    def claim_items_batch(
        self,
        work_ids: list[str],
        node_id: str,
        claimed_at: float | None = None,
    ) -> BackendResult:
        """Claim multiple items in a single database transaction.

        Session 17.50 (Jan 30, 2026): Optimized batch claiming to reduce
        database round-trips. Instead of N transactions for N items, this
        uses a single transaction with one UPDATE per item.

        Args:
            work_ids: Work item IDs to claim
            node_id: Node claiming the work
            claimed_at: Timestamp of claim (defaults to now)

        Returns:
            BackendResult with claimed_ids in data
        """
        if self._readonly_mode:
            return BackendResult(success=False, error="Backend is readonly")

        if not work_ids:
            return BackendResult(success=False, data={"claimed_ids": []})

        claimed_at = claimed_at or time.time()
        claimed_ids = []

        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Use a single transaction for all claims
                for work_id in work_ids:
                    cursor.execute(
                        """
                        UPDATE work_items
                        SET status = 'claimed', claimed_by = ?, claimed_at = ?, attempts = attempts + 1
                        WHERE work_id = ? AND status = 'pending'
                        """,
                        (node_id, claimed_at, work_id),
                    )
                    if cursor.rowcount > 0:
                        claimed_ids.append(work_id)

                # Commit all claims at once
                conn.commit()
                return BackendResult(
                    success=len(claimed_ids) > 0,
                    data={"claimed_ids": claimed_ids, "node_id": node_id},
                )
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to batch claim {len(work_ids)} items: {e}")
            return BackendResult(success=False, error=str(e), data={"claimed_ids": []})

    def get_pending_items(self, limit: int = 100) -> list[dict[str, Any]]:
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM work_items
                    WHERE status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to get pending items: {e}")
            return []

    def start_item(self, work_id: str, started_at: float | None = None) -> BackendResult:
        if self._readonly_mode:
            return BackendResult(success=False, error="Backend is readonly")

        started_at = started_at or time.time()
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE work_items SET status = 'running', started_at = ?
                    WHERE work_id = ? AND status = 'claimed'
                    """,
                    (started_at, work_id),
                )
                conn.commit()
                return BackendResult(success=cursor.rowcount > 0)
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to start {work_id}: {e}")
            return BackendResult(success=False, error=str(e))

    def complete_item(
        self,
        work_id: str,
        result: dict[str, Any] | None = None,
        completed_at: float | None = None,
    ) -> BackendResult:
        if self._readonly_mode:
            return BackendResult(success=False, error="Backend is readonly")

        completed_at = completed_at or time.time()
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE work_items
                    SET status = 'completed', completed_at = ?, result = ?
                    WHERE work_id = ? AND status IN ('claimed', 'running')
                    """,
                    (completed_at, json.dumps(result or {}), work_id),
                )
                conn.commit()
                return BackendResult(success=cursor.rowcount > 0)
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to complete {work_id}: {e}")
            return BackendResult(success=False, error=str(e))

    def fail_item(
        self,
        work_id: str,
        error: str,
        permanent: bool,
        completed_at: float | None = None,
    ) -> BackendResult:
        if self._readonly_mode:
            return BackendResult(success=False, error="Backend is readonly")

        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if permanent:
                    completed_at = completed_at or time.time()
                    cursor.execute(
                        """
                        UPDATE work_items
                        SET status = 'failed', completed_at = ?, error = ?
                        WHERE work_id = ?
                        """,
                        (completed_at, error, work_id),
                    )
                else:
                    # Reset for retry
                    cursor.execute(
                        """
                        UPDATE work_items
                        SET status = 'pending', claimed_by = '', claimed_at = 0.0, error = ?
                        WHERE work_id = ?
                        """,
                        (error, work_id),
                    )
                conn.commit()
                return BackendResult(success=cursor.rowcount > 0, data={"permanent": permanent})
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to fail {work_id}: {e}")
            return BackendResult(success=False, error=str(e))

    def get_item(self, work_id: str) -> dict[str, Any] | None:
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM work_items WHERE work_id = ?", (work_id,))
                row = cursor.fetchone()
                if row:
                    data = dict(row)
                    # Parse JSON fields
                    if data.get("config"):
                        data["config"] = json.loads(data["config"])
                    if data.get("result"):
                        data["result"] = json.loads(data["result"])
                    if data.get("depends_on"):
                        data["depends_on"] = json.loads(data["depends_on"])
                    return data
                return None
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to get {work_id}: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT status, COUNT(*) as count FROM work_items
                    GROUP BY status
                    """
                )
                by_status = {row[0]: row[1] for row in cursor.fetchall()}

                cursor.execute("SELECT COUNT(*) FROM work_items")
                total = cursor.fetchone()[0]

                return {
                    "total": total,
                    "pending": by_status.get("pending", 0),
                    "claimed": by_status.get("claimed", 0),
                    "running": by_status.get("running", 0),
                    "completed": by_status.get("completed", 0),
                    "failed": by_status.get("failed", 0),
                    "backend": "sqlite",
                }
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to get stats: {e}")
            return {"backend": "sqlite", "error": str(e)}

    def update_item_config(self, work_id: str, config: dict[str, Any]) -> BackendResult:
        if self._readonly_mode:
            return BackendResult(success=False, error="Backend is readonly")

        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE work_items SET config = ? WHERE work_id = ?",
                    (json.dumps(config), work_id),
                )
                conn.commit()
                return BackendResult(success=cursor.rowcount > 0)
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"[SQLiteBackend] Failed to update config for {work_id}: {e}")
            return BackendResult(success=False, error=str(e))


class RaftBackend(WorkQueueBackend):
    """Raft-based work queue backend with SQLite fallback.

    Provides cluster-wide consistency via Raft consensus.
    Automatically falls back to SQLite if Raft becomes unavailable.
    """

    def __init__(
        self,
        sqlite_backend: SQLiteBackend,
        get_raft_queue: Callable[[], Any],
    ):
        """Initialize Raft backend.

        Args:
            sqlite_backend: SQLite backend to use as fallback
            get_raft_queue: Factory function to get Raft work queue
        """
        self._sqlite_backend = sqlite_backend
        self._get_raft_queue = get_raft_queue
        self._fallback_active = False

    @property
    def backend_type(self) -> BackendType:
        if self._fallback_active:
            return BackendType.SQLITE
        return BackendType.RAFT

    @property
    def is_available(self) -> bool:
        raft_wq = self._get_raft_queue()
        if raft_wq is not None:
            self._fallback_active = False
            return True
        # Raft unavailable, using fallback
        self._fallback_active = True
        return self._sqlite_backend.is_available

    def _with_fallback(
        self,
        raft_operation: Callable[[Any], BackendResult],
        sqlite_fallback: Callable[[], BackendResult],
        operation_name: str,
    ) -> BackendResult:
        """Execute Raft operation with automatic SQLite fallback.

        Args:
            raft_operation: Function taking raft_wq, returning BackendResult
            sqlite_fallback: Fallback function returning BackendResult
            operation_name: Name for logging

        Returns:
            BackendResult from either Raft or SQLite
        """
        raft_wq = self._get_raft_queue()
        if raft_wq is None:
            if not self._fallback_active:
                logger.warning(f"[RaftBackend] Raft unavailable for {operation_name}, using SQLite")
                self._fallback_active = True
            result = sqlite_fallback()
            result.fallback_used = True
            return result

        self._fallback_active = False
        try:
            return raft_operation(raft_wq)
        except Exception as e:
            logger.warning(f"[RaftBackend] {operation_name} failed: {e}, falling back to SQLite")
            self._fallback_active = True
            result = sqlite_fallback()
            result.fallback_used = True
            return result

    def add_item(self, work_id: str, work_data: dict[str, Any]) -> BackendResult:
        def raft_op(raft_wq: Any) -> BackendResult:
            success = raft_wq.add_work(work_id, work_data)
            return BackendResult(success=success, data=work_id if success else None)

        def sqlite_op() -> BackendResult:
            return self._sqlite_backend.add_item(work_id, work_data)

        return self._with_fallback(raft_op, sqlite_op, f"add_item({work_id})")

    def claim_item(
        self,
        work_id: str,
        node_id: str,
        claimed_at: float | None = None,
    ) -> BackendResult:
        def raft_op(raft_wq: Any) -> BackendResult:
            success = raft_wq.claim_work(work_id, node_id)
            if success:
                return BackendResult(success=True, data={"work_id": work_id, "node_id": node_id})
            return BackendResult(success=False, error="Claim failed (race or not pending)")

        def sqlite_op() -> BackendResult:
            return self._sqlite_backend.claim_item(work_id, node_id, claimed_at)

        return self._with_fallback(raft_op, sqlite_op, f"claim_item({work_id})")

    def claim_items_batch(
        self,
        work_ids: list[str],
        node_id: str,
        claimed_at: float | None = None,
    ) -> BackendResult:
        """Batch claim items with Raft fallback to SQLite.

        Session 17.50 (Jan 30, 2026): Added batch claiming support.
        Raft doesn't have native batch claim, so we iterate and claim each.
        Falls back to SQLite batch claim on Raft failure.

        Args:
            work_ids: Work item IDs to claim
            node_id: Node claiming the work
            claimed_at: Timestamp of claim (defaults to now)

        Returns:
            BackendResult with claimed_ids in data
        """
        def raft_op(raft_wq: Any) -> BackendResult:
            claimed_ids = []
            for work_id in work_ids:
                try:
                    if raft_wq.claim_work(work_id, node_id):
                        claimed_ids.append(work_id)
                except Exception as e:
                    logger.debug(f"[RaftBackend] Batch claim {work_id} failed: {e}")
            return BackendResult(
                success=len(claimed_ids) > 0,
                data={"claimed_ids": claimed_ids, "node_id": node_id},
            )

        def sqlite_op() -> BackendResult:
            return self._sqlite_backend.claim_items_batch(work_ids, node_id, claimed_at)

        return self._with_fallback(raft_op, sqlite_op, f"claim_items_batch({len(work_ids)} items)")

    def get_pending_items(self, limit: int = 100) -> list[dict[str, Any]]:
        raft_wq = self._get_raft_queue()
        if raft_wq is None:
            return self._sqlite_backend.get_pending_items(limit)

        try:
            return raft_wq.get_pending_work(limit=limit) or []
        except Exception as e:
            logger.warning(f"[RaftBackend] get_pending_items failed: {e}")
            return self._sqlite_backend.get_pending_items(limit)

    def start_item(self, work_id: str, started_at: float | None = None) -> BackendResult:
        def raft_op(raft_wq: Any) -> BackendResult:
            success = raft_wq.start_work(work_id)
            return BackendResult(success=success)

        def sqlite_op() -> BackendResult:
            return self._sqlite_backend.start_item(work_id, started_at)

        return self._with_fallback(raft_op, sqlite_op, f"start_item({work_id})")

    def complete_item(
        self,
        work_id: str,
        result: dict[str, Any] | None = None,
        completed_at: float | None = None,
    ) -> BackendResult:
        def raft_op(raft_wq: Any) -> BackendResult:
            success = raft_wq.complete_work(work_id, result)
            return BackendResult(success=success)

        def sqlite_op() -> BackendResult:
            return self._sqlite_backend.complete_item(work_id, result, completed_at)

        return self._with_fallback(raft_op, sqlite_op, f"complete_item({work_id})")

    def fail_item(
        self,
        work_id: str,
        error: str,
        permanent: bool,
        completed_at: float | None = None,
    ) -> BackendResult:
        def raft_op(raft_wq: Any) -> BackendResult:
            success = raft_wq.fail_work(work_id, error)
            return BackendResult(success=success, data={"permanent": permanent})

        def sqlite_op() -> BackendResult:
            return self._sqlite_backend.fail_item(work_id, error, permanent, completed_at)

        return self._with_fallback(raft_op, sqlite_op, f"fail_item({work_id})")

    def get_item(self, work_id: str) -> dict[str, Any] | None:
        raft_wq = self._get_raft_queue()
        if raft_wq is None:
            return self._sqlite_backend.get_item(work_id)

        try:
            return raft_wq.get_work(work_id)
        except Exception as e:
            logger.warning(f"[RaftBackend] get_item({work_id}) failed: {e}")
            return self._sqlite_backend.get_item(work_id)

    def get_stats(self) -> dict[str, Any]:
        raft_wq = self._get_raft_queue()
        if raft_wq is None:
            stats = self._sqlite_backend.get_stats()
            stats["fallback_active"] = True
            return stats

        try:
            raft_stats = raft_wq.get_queue_stats()
            return {
                "total": raft_stats.get("total", 0),
                "pending": raft_stats.get("pending", 0),
                "claimed": raft_stats.get("claimed", 0),
                "running": raft_stats.get("running", 0),
                "completed": raft_stats.get("completed", 0),
                "failed": raft_stats.get("failed", 0),
                "backend": "raft",
                "is_leader": raft_stats.get("is_leader", False),
                "leader_address": raft_stats.get("leader_address"),
                "is_ready": raft_stats.get("is_ready", False),
            }
        except Exception as e:
            logger.warning(f"[RaftBackend] get_stats failed: {e}")
            stats = self._sqlite_backend.get_stats()
            stats["fallback_active"] = True
            return stats

    def update_item_config(self, work_id: str, config: dict[str, Any]) -> BackendResult:
        # Raft doesn't have a direct update_config, use SQLite
        # This is acceptable since config updates are rare (stale target_node cleanup)
        return self._sqlite_backend.update_item_config(work_id, config)


def create_backend(
    db_path: Path,
    get_connection: Callable[[], sqlite3.Connection],
    use_raft: bool = True,
    readonly_mode: bool = False,
) -> WorkQueueBackend:
    """Factory function to create appropriate backend.

    Args:
        db_path: Path to SQLite database
        get_connection: Function to get database connections
        use_raft: Whether to try Raft backend first
        readonly_mode: If True, backend is read-only

    Returns:
        Configured WorkQueueBackend instance
    """
    # Always create SQLite backend (used as fallback)
    sqlite_backend = SQLiteBackend(
        db_path=db_path,
        get_connection=get_connection,
        readonly_mode=readonly_mode,
    )

    if not use_raft:
        logger.debug("[create_backend] Using SQLite backend (Raft disabled)")
        return sqlite_backend

    # Import here to avoid circular dependency
    from app.coordination.work_queue import get_raft_work_queue

    # Try to use Raft
    raft_wq = get_raft_work_queue()
    if raft_wq is not None:
        logger.info("[create_backend] Using Raft backend with SQLite fallback")
        return RaftBackend(
            sqlite_backend=sqlite_backend,
            get_raft_queue=get_raft_work_queue,
        )

    logger.debug("[create_backend] Raft unavailable, using SQLite backend")
    return sqlite_backend


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "BackendType",
    "BackendResult",
    "WorkQueueBackend",
    "SQLiteBackend",
    "RaftBackend",
    "create_backend",
]
