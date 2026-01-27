"""Base query builder for P2P orchestrator.

Phase 3.2 Code Quality Cleanup: Provides generic query infrastructure
with thread-safe access, error handling, and common patterns.

This consolidates the repetitive try-except-return-dict pattern found
in 70+ _get_* methods in p2p_orchestrator.py.

Example usage:
    class PeerQueryBuilder(BaseQueryBuilder):
        def alive(self) -> QueryResult[list]:
            return self.filter(lambda p: p.is_alive())

        def summary(self) -> SummaryResult:
            return self.aggregate(
                key_fn=lambda p: p.node_id,
                value_fn=lambda p: {"alive": p.is_alive(), "gpu": p.gpu_type}
            )
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@dataclass
class QueryResult(Generic[T]):
    """Result wrapper for query operations.

    Provides consistent error handling and metadata for all queries.

    Attributes:
        data: The query result data.
        success: Whether the query succeeded.
        error: Error message if query failed.
        duration_ms: Query execution time in milliseconds.
        count: Number of items in result (if applicable).
    """

    data: T
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0
    count: int = 0

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success

    def unwrap(self) -> T:
        """Get data or raise if failed."""
        if not self.success:
            raise RuntimeError(self.error or "Query failed")
        return self.data

    def unwrap_or(self, default: T) -> T:
        """Get data or return default if failed."""
        return self.data if self.success else default


@dataclass
class SummaryResult:
    """Result wrapper for summary/aggregation operations.

    Provides consistent structure for API response summaries.

    Attributes:
        counts: Aggregate counts (e.g., total, healthy, failed).
        details: Per-item details keyed by identifier.
        metadata: Additional metadata about the summary.
        success: Whether the aggregation succeeded.
        error: Error message if aggregation failed.
    """

    counts: Dict[str, int] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "counts": self.counts,
            "details": self.details,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        if not self.success:
            result["error"] = self.error
        return result


class BaseQueryBuilder(Generic[T]):
    """Base class for thread-safe query operations.

    Provides common patterns for filtering, mapping, and aggregating
    collections with consistent error handling and timing.

    Subclasses should:
    1. Set self._items to the collection to query
    2. Set self._lock to the threading lock protecting the collection
    3. Implement domain-specific query methods using filter/map/aggregate

    Example:
        class PeerQueryBuilder(BaseQueryBuilder):
            def __init__(self, peers: dict, lock: threading.RLock):
                super().__init__()
                self._items = peers
                self._lock = lock

            def alive(self) -> QueryResult[list]:
                return self.filter(lambda p: p.is_alive())
    """

    def __init__(self):
        """Initialize the query builder."""
        self._items: Union[Dict[str, T], List[T]] = {}
        self._lock: Optional[threading.RLock] = None
        self._default_timeout: float = 5.0  # seconds

    def _get_items_iter(self) -> Iterator[T]:
        """Get iterator over items (dict values or list items)."""
        if isinstance(self._items, dict):
            return iter(self._items.values())
        return iter(self._items)

    def _get_items_with_keys(self) -> Iterator[tuple[str, T]]:
        """Get iterator over (key, item) pairs."""
        if isinstance(self._items, dict):
            return iter(self._items.items())
        return ((str(i), item) for i, item in enumerate(self._items))

    def filter(
        self,
        predicate: Callable[[T], bool],
        *,
        default: Optional[List[T]] = None,
    ) -> QueryResult[List[T]]:
        """Filter items matching predicate with thread-safe access.

        Args:
            predicate: Function that returns True for items to include.
            default: Default value if query fails (default: empty list).

        Returns:
            QueryResult containing filtered items.
        """
        start = time.perf_counter()
        default = default if default is not None else []

        try:
            results = []
            if self._lock:
                with self._lock:
                    for item in self._get_items_iter():
                        try:
                            if predicate(item):
                                results.append(item)
                        except Exception as e:
                            logger.debug(f"Predicate error for item: {e}")
                            continue
            else:
                for item in self._get_items_iter():
                    try:
                        if predicate(item):
                            results.append(item)
                    except Exception as e:
                        logger.debug(f"Predicate error for item: {e}")
                        continue

            duration = (time.perf_counter() - start) * 1000
            return QueryResult(
                data=results,
                success=True,
                duration_ms=duration,
                count=len(results),
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"Filter query failed: {e}")
            return QueryResult(
                data=default,
                success=False,
                error=str(e),
                duration_ms=duration,
                count=0,
            )

    def map(
        self,
        mapper: Callable[[T], V],
        *,
        predicate: Optional[Callable[[T], bool]] = None,
        default: Optional[List[V]] = None,
    ) -> QueryResult[List[V]]:
        """Map items through a transformation with optional filtering.

        Args:
            mapper: Function to transform each item.
            predicate: Optional filter to apply before mapping.
            default: Default value if query fails (default: empty list).

        Returns:
            QueryResult containing mapped items.
        """
        start = time.perf_counter()
        default = default if default is not None else []

        try:
            results = []
            if self._lock:
                with self._lock:
                    for item in self._get_items_iter():
                        try:
                            if predicate is None or predicate(item):
                                results.append(mapper(item))
                        except Exception as e:
                            logger.debug(f"Mapper error for item: {e}")
                            continue
            else:
                for item in self._get_items_iter():
                    try:
                        if predicate is None or predicate(item):
                            results.append(mapper(item))
                    except Exception as e:
                        logger.debug(f"Mapper error for item: {e}")
                        continue

            duration = (time.perf_counter() - start) * 1000
            return QueryResult(
                data=results,
                success=True,
                duration_ms=duration,
                count=len(results),
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"Map query failed: {e}")
            return QueryResult(
                data=default,
                success=False,
                error=str(e),
                duration_ms=duration,
                count=0,
            )

    def aggregate(
        self,
        key_fn: Callable[[T], K],
        value_fn: Callable[[T], V],
        *,
        predicate: Optional[Callable[[T], bool]] = None,
        default: Optional[Dict[K, V]] = None,
    ) -> QueryResult[Dict[K, V]]:
        """Aggregate items into a dictionary.

        Args:
            key_fn: Function to extract key from each item.
            value_fn: Function to extract value from each item.
            predicate: Optional filter to apply before aggregation.
            default: Default value if query fails (default: empty dict).

        Returns:
            QueryResult containing aggregated dictionary.
        """
        start = time.perf_counter()
        default = default if default is not None else {}

        try:
            result: Dict[K, V] = {}
            if self._lock:
                with self._lock:
                    for item in self._get_items_iter():
                        try:
                            if predicate is None or predicate(item):
                                key = key_fn(item)
                                result[key] = value_fn(item)
                        except Exception as e:
                            logger.debug(f"Aggregation error for item: {e}")
                            continue
            else:
                for item in self._get_items_iter():
                    try:
                        if predicate is None or predicate(item):
                            key = key_fn(item)
                            result[key] = value_fn(item)
                    except Exception as e:
                        logger.debug(f"Aggregation error for item: {e}")
                        continue

            duration = (time.perf_counter() - start) * 1000
            return QueryResult(
                data=result,
                success=True,
                duration_ms=duration,
                count=len(result),
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"Aggregate query failed: {e}")
            return QueryResult(
                data=default,
                success=False,
                error=str(e),
                duration_ms=duration,
                count=0,
            )

    def summarize(
        self,
        count_fn: Callable[[T], Dict[str, bool]],
        detail_fn: Optional[Callable[[T], Dict[str, Any]]] = None,
        key_fn: Optional[Callable[[T], str]] = None,
    ) -> SummaryResult:
        """Build a summary with counts and optional per-item details.

        This is the most common pattern for _get_*_summary() methods.

        Args:
            count_fn: Function returning {counter_name: should_increment} dict.
                     Example: lambda p: {"alive": p.is_alive(), "gpu": bool(p.gpu_type)}
            detail_fn: Optional function to extract details for each item.
            key_fn: Optional function to extract key for details dict.

        Returns:
            SummaryResult with counts and details.

        Example:
            summary = builder.summarize(
                count_fn=lambda p: {
                    "total": True,
                    "alive": p.is_alive(),
                    "gpu": bool(p.gpu_type),
                },
                detail_fn=lambda p: {"status": "alive" if p.is_alive() else "dead"},
                key_fn=lambda p: p.node_id,
            )
        """
        start = time.perf_counter()
        counts: Dict[str, int] = {}
        details: Dict[str, Any] = {}

        try:
            if self._lock:
                with self._lock:
                    for item in self._get_items_iter():
                        try:
                            # Update counts
                            for counter_name, should_count in count_fn(item).items():
                                if should_count:
                                    counts[counter_name] = counts.get(counter_name, 0) + 1

                            # Collect details if requested
                            if detail_fn is not None and key_fn is not None:
                                key = key_fn(item)
                                details[key] = detail_fn(item)
                        except Exception as e:
                            logger.debug(f"Summarize error for item: {e}")
                            continue
            else:
                for item in self._get_items_iter():
                    try:
                        for counter_name, should_count in count_fn(item).items():
                            if should_count:
                                counts[counter_name] = counts.get(counter_name, 0) + 1

                        if detail_fn is not None and key_fn is not None:
                            key = key_fn(item)
                            details[key] = detail_fn(item)
                    except Exception as e:
                        logger.debug(f"Summarize error for item: {e}")
                        continue

            duration = (time.perf_counter() - start) * 1000
            return SummaryResult(
                counts=counts,
                details=details,
                metadata={"duration_ms": duration, "item_count": sum(1 for _ in [])},
                success=True,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"Summarize query failed: {e}")
            return SummaryResult(
                success=False,
                error=str(e),
                metadata={"duration_ms": duration},
            )

    def first(
        self,
        predicate: Callable[[T], bool],
        *,
        default: Optional[T] = None,
    ) -> QueryResult[Optional[T]]:
        """Get first item matching predicate.

        Args:
            predicate: Function that returns True for matching item.
            default: Default value if no match found.

        Returns:
            QueryResult containing first matching item or default.
        """
        start = time.perf_counter()

        try:
            result = None
            if self._lock:
                with self._lock:
                    for item in self._get_items_iter():
                        try:
                            if predicate(item):
                                result = item
                                break
                        except Exception as e:
                            logger.debug(f"Predicate error for item: {e}")
                            continue
            else:
                for item in self._get_items_iter():
                    try:
                        if predicate(item):
                            result = item
                            break
                    except Exception as e:
                        logger.debug(f"Predicate error for item: {e}")
                        continue

            duration = (time.perf_counter() - start) * 1000
            return QueryResult(
                data=result if result is not None else default,
                success=True,
                duration_ms=duration,
                count=1 if result is not None else 0,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"First query failed: {e}")
            return QueryResult(
                data=default,
                success=False,
                error=str(e),
                duration_ms=duration,
                count=0,
            )

    def count(
        self,
        predicate: Optional[Callable[[T], bool]] = None,
    ) -> QueryResult[int]:
        """Count items, optionally matching predicate.

        Args:
            predicate: Optional filter function.

        Returns:
            QueryResult containing count.
        """
        start = time.perf_counter()

        try:
            count = 0
            if self._lock:
                with self._lock:
                    for item in self._get_items_iter():
                        try:
                            if predicate is None or predicate(item):
                                count += 1
                        except Exception as e:
                            logger.debug(f"Predicate error for item: {e}")
                            continue
            else:
                for item in self._get_items_iter():
                    try:
                        if predicate is None or predicate(item):
                            count += 1
                    except Exception as e:
                        logger.debug(f"Predicate error for item: {e}")
                        continue

            duration = (time.perf_counter() - start) * 1000
            return QueryResult(
                data=count,
                success=True,
                duration_ms=duration,
                count=count,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"Count query failed: {e}")
            return QueryResult(
                data=0,
                success=False,
                error=str(e),
                duration_ms=duration,
                count=0,
            )

    def safe_get(
        self,
        key: str,
        default: Optional[T] = None,
    ) -> QueryResult[Optional[T]]:
        """Safely get item by key (for dict-based collections).

        Args:
            key: Key to look up.
            default: Default value if key not found.

        Returns:
            QueryResult containing item or default.
        """
        start = time.perf_counter()

        try:
            if not isinstance(self._items, dict):
                raise TypeError("safe_get only works with dict-based collections")

            result = None
            if self._lock:
                with self._lock:
                    result = self._items.get(key, default)
            else:
                result = self._items.get(key, default)

            duration = (time.perf_counter() - start) * 1000
            return QueryResult(
                data=result,
                success=True,
                duration_ms=duration,
                count=1 if result is not None and result != default else 0,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"Safe get failed: {e}")
            return QueryResult(
                data=default,
                success=False,
                error=str(e),
                duration_ms=duration,
                count=0,
            )
