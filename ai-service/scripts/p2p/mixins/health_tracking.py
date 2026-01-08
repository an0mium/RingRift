"""Health Tracking Mixin for P2P Components.

January 2026 - Sprint 17: Consolidated health tracking patterns from gossip protocol
and P2P loops into a reusable mixin.

Problem Solved: Duplicate health tracking code existed in:
- GossipHealthTracker (gossip_protocol.py) - 200+ LOC
- HttpServerHealthLoop (http_server_health_loop.py) - ~50 LOC
- LeaderProbeLoop (leader_probe_loop.py) - ~60 LOC
- PeerRecoveryLoop (peer_recovery_loop.py) - ~40 LOC

This mixin consolidates:
- Per-entity success/failure tracking with thread safety
- Exponential backoff with jitter for failed entities
- Health score calculation (0.0-1.0)
- Staleness detection
- Configurable thresholds

Usage:
    from scripts.p2p.mixins import HealthTrackingMixin, HealthTrackingConfig

    class MyLoop(BaseLoop, HealthTrackingMixin):
        def __init__(self):
            super().__init__(loop_name="my_loop")
            self.init_health_tracking(HealthTrackingConfig(
                failure_threshold=5,
                stale_threshold_seconds=300.0,
            ))

        async def _run_iteration(self) -> None:
            for entity_id in self._get_entities_to_check():
                try:
                    await self._check_entity(entity_id)
                    self.record_entity_success(entity_id)
                except Exception:
                    should_emit, count = self.record_entity_failure(entity_id)
                    if should_emit:
                        self._emit_entity_unhealthy(entity_id, count)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "HealthTrackingMixin",
    "HealthTrackingConfig",
    "EntityHealthSummary",
    "EntityHealthState",
]


@dataclass
class HealthTrackingConfig:
    """Configuration for health tracking behavior.

    Attributes:
        failure_threshold: Consecutive failures before marking entity as unhealthy
        stale_threshold_seconds: Seconds without success before marking as stale
        backoff_base_seconds: Base delay for exponential backoff
        backoff_multiplier: Multiplier for exponential backoff
        backoff_max_seconds: Maximum backoff delay
        backoff_jitter_factor: Jitter range (+/- percentage)
        auto_reset_on_success: Whether to reset failure count on success
    """

    failure_threshold: int = 5
    stale_threshold_seconds: float = 300.0  # 5 minutes
    backoff_base_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    backoff_max_seconds: float = 8.0
    backoff_jitter_factor: float = 0.25  # +/- 25%
    auto_reset_on_success: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.stale_threshold_seconds <= 0:
            raise ValueError("stale_threshold_seconds must be > 0")
        if self.backoff_base_seconds <= 0:
            raise ValueError("backoff_base_seconds must be > 0")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError("backoff_max_seconds must be >= backoff_base_seconds")
        if not 0.0 <= self.backoff_jitter_factor <= 0.5:
            raise ValueError("backoff_jitter_factor must be between 0.0 and 0.5")


@dataclass
class EntityHealthState:
    """Health state for a single tracked entity.

    Thread-safe: All fields are immutable or copied.
    """

    entity_id: str
    failure_count: int = 0
    last_success_time: float | None = None
    last_failure_time: float | None = None
    is_suspect: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    @property
    def seconds_since_success(self) -> float | None:
        """Seconds since last success, or None if never succeeded."""
        if self.last_success_time is None:
            return None
        return time.time() - self.last_success_time

    @property
    def seconds_since_failure(self) -> float | None:
        """Seconds since last failure, or None if never failed."""
        if self.last_failure_time is None:
            return None
        return time.time() - self.last_failure_time

    def is_stale(self, threshold_seconds: float) -> bool:
        """Check if entity is stale (no success in threshold time)."""
        if self.last_success_time is None:
            return True  # Never succeeded = stale
        return (time.time() - self.last_success_time) > threshold_seconds


@dataclass
class EntityHealthSummary:
    """Summary of health across all tracked entities.

    Thread-safe: All fields are copied from internal state.
    """

    total_entities: int = 0
    healthy_count: int = 0
    suspect_count: int = 0
    stale_count: int = 0
    failure_counts: dict[str, int] = field(default_factory=dict)
    suspect_entities: list[str] = field(default_factory=list)
    stale_entities: list[str] = field(default_factory=list)
    failure_threshold: int = 5

    @property
    def unhealthy_ratio(self) -> float:
        """Ratio of unhealthy (suspect or stale) entities."""
        if self.total_entities == 0:
            return 0.0
        unhealthy = len(set(self.suspect_entities) | set(self.stale_entities))
        return unhealthy / self.total_entities

    @property
    def health_score(self) -> float:
        """Calculate health score from 0.0 (unhealthy) to 1.0 (healthy).

        Score is based on:
        - 50% weight: Ratio of healthy entities (not suspect or stale)
        - 50% weight: Average failure rate across entities

        This matches the calculation in GossipHealthSummary for consistency.
        """
        if self.total_entities == 0:
            return 1.0  # No entities tracked = healthy by default

        # Healthy entity ratio
        unhealthy = set(self.suspect_entities) | set(self.stale_entities)
        healthy_ratio = 1.0 - (len(unhealthy) / self.total_entities)

        # Average failure rate (capped at threshold)
        if self.failure_counts:
            avg_failures = sum(self.failure_counts.values()) / len(self.failure_counts)
            failure_ratio = 1.0 - min(avg_failures / self.failure_threshold, 1.0)
        else:
            failure_ratio = 1.0

        return 0.5 * healthy_ratio + 0.5 * failure_ratio

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_entities": self.total_entities,
            "healthy_count": self.healthy_count,
            "suspect_count": self.suspect_count,
            "stale_count": self.stale_count,
            "health_score": round(self.health_score, 3),
            "unhealthy_ratio": round(self.unhealthy_ratio, 3),
            "suspect_entities": self.suspect_entities.copy(),
            "stale_entities": self.stale_entities.copy(),
        }


class HealthTrackingMixin:
    """Mixin providing per-entity health tracking with backoff and scoring.

    This mixin consolidates common health tracking patterns from:
    - GossipHealthTracker (per-peer failure counting, backoff)
    - HttpServerHealthLoop (last_success_time tracking)
    - LeaderProbeLoop (consecutive failure counting)
    - PeerRecoveryLoop (probe failure recording)

    Features:
    - Thread-safe access via RLock
    - Per-entity success/failure tracking
    - Exponential backoff with jitter for retries
    - Health score calculation (0.0-1.0)
    - Staleness detection
    - Configurable thresholds

    Usage:
        class MyHealthTracker(HealthTrackingMixin):
            def __init__(self):
                self.init_health_tracking(HealthTrackingConfig())

            def check_peer(self, peer_id: str) -> None:
                if self.should_skip_entity(peer_id):
                    return  # In backoff period

                try:
                    self._probe_peer(peer_id)
                    self.record_entity_success(peer_id)
                except Exception as e:
                    should_emit, count = self.record_entity_failure(peer_id, str(e))
                    if should_emit:
                        self._emit_suspect_event(peer_id, count)
    """

    # Internal state (protected by _health_lock)
    _health_config: HealthTrackingConfig
    _health_lock: RLock
    _failure_counts: dict[str, int]
    _last_success: dict[str, float]
    _last_failure: dict[str, float]
    _suspect_emitted: set[str]
    _failure_reasons: dict[str, list[str]]
    _health_tracking_initialized: bool

    def init_health_tracking(
        self,
        config: HealthTrackingConfig | None = None,
    ) -> None:
        """Initialize health tracking state.

        Must be called in __init__ before using any health tracking methods.

        Args:
            config: Configuration for health tracking. Uses defaults if None.
        """
        self._health_config = config or HealthTrackingConfig()
        self._health_lock = RLock()
        self._failure_counts = {}
        self._last_success = {}
        self._last_failure = {}
        self._suspect_emitted = set()
        self._failure_reasons = {}
        self._health_tracking_initialized = True

    def _ensure_health_initialized(self) -> None:
        """Ensure health tracking was initialized."""
        if not getattr(self, "_health_tracking_initialized", False):
            raise RuntimeError(
                "Health tracking not initialized. Call init_health_tracking() first."
            )

    def record_entity_failure(
        self,
        entity_id: str,
        reason: str | None = None,
    ) -> tuple[bool, int]:
        """Record a failure for an entity.

        Thread-safe: Uses _health_lock to protect shared state.

        Args:
            entity_id: Identifier for the entity that failed
            reason: Optional reason for the failure (for debugging)

        Returns:
            Tuple of (should_emit_suspect, failure_count)
            - should_emit_suspect: True if this failure crossed the threshold
              (only True once per failure streak until success)
            - failure_count: Current consecutive failure count
        """
        self._ensure_health_initialized()

        with self._health_lock:
            self._failure_counts[entity_id] = self._failure_counts.get(entity_id, 0) + 1
            self._last_failure[entity_id] = time.time()
            count = self._failure_counts[entity_id]

            # Track failure reasons (keep last 5)
            if reason:
                if entity_id not in self._failure_reasons:
                    self._failure_reasons[entity_id] = []
                self._failure_reasons[entity_id].append(reason)
                self._failure_reasons[entity_id] = self._failure_reasons[entity_id][-5:]

            # Check if we should emit suspect (only once per streak)
            should_emit = (
                count >= self._health_config.failure_threshold
                and entity_id not in self._suspect_emitted
            )

            if should_emit:
                self._suspect_emitted.add(entity_id)
                logger.warning(
                    f"Entity {entity_id} marked as suspect after {count} failures"
                )

            return should_emit, count

    def record_entity_success(self, entity_id: str) -> None:
        """Record a success for an entity.

        Thread-safe: Uses _health_lock to protect shared state.

        Args:
            entity_id: Identifier for the entity that succeeded
        """
        self._ensure_health_initialized()

        with self._health_lock:
            self._last_success[entity_id] = time.time()

            if self._health_config.auto_reset_on_success:
                # Reset failure tracking on success
                prev_count = self._failure_counts.get(entity_id, 0)
                if prev_count > 0:
                    logger.debug(
                        f"Entity {entity_id} recovered after {prev_count} failures"
                    )
                self._failure_counts.pop(entity_id, None)
                self._suspect_emitted.discard(entity_id)
                self._failure_reasons.pop(entity_id, None)

    def get_entity_failure_count(self, entity_id: str) -> int:
        """Get current failure count for an entity.

        Thread-safe.

        Args:
            entity_id: Entity identifier

        Returns:
            Current consecutive failure count (0 if no failures)
        """
        self._ensure_health_initialized()

        with self._health_lock:
            return self._failure_counts.get(entity_id, 0)

    def get_backoff_seconds(self, entity_id: str) -> float:
        """Calculate exponential backoff delay with jitter for an entity.

        Thread-safe.

        The backoff formula is:
            base * multiplier^(failures-1), capped at max
            Then multiplied by random jitter factor

        Args:
            entity_id: Entity identifier

        Returns:
            Backoff delay in seconds (0.0 if no failures)
        """
        self._ensure_health_initialized()

        with self._health_lock:
            failure_count = self._failure_counts.get(entity_id, 0)

        if failure_count == 0:
            return 0.0

        config = self._health_config

        # Exponential backoff: base * multiplier^(failures-1)
        base_backoff = config.backoff_base_seconds * (
            config.backoff_multiplier ** (failure_count - 1)
        )
        base_backoff = min(base_backoff, config.backoff_max_seconds)

        # Apply jitter to prevent thundering herd
        jitter_multiplier = random.uniform(
            1 - config.backoff_jitter_factor,
            1 + config.backoff_jitter_factor,
        )

        return base_backoff * jitter_multiplier

    def should_skip_entity(self, entity_id: str) -> bool:
        """Check if an entity should be skipped due to backoff.

        Thread-safe.

        Args:
            entity_id: Entity identifier

        Returns:
            True if entity is in backoff period, False if OK to check
        """
        self._ensure_health_initialized()

        with self._health_lock:
            last_failure = self._last_failure.get(entity_id)

        if last_failure is None:
            return False  # No failures, OK to check

        backoff = self.get_backoff_seconds(entity_id)
        elapsed = time.time() - last_failure

        return elapsed < backoff

    def is_entity_suspect(self, entity_id: str) -> bool:
        """Check if an entity is currently marked as suspect.

        Thread-safe.

        Args:
            entity_id: Entity identifier

        Returns:
            True if entity has crossed failure threshold
        """
        self._ensure_health_initialized()

        with self._health_lock:
            return entity_id in self._suspect_emitted

    def is_entity_stale(self, entity_id: str) -> bool:
        """Check if an entity is stale (no success in threshold time).

        Thread-safe.

        Args:
            entity_id: Entity identifier

        Returns:
            True if entity hasn't succeeded within stale threshold
        """
        self._ensure_health_initialized()

        with self._health_lock:
            last_success = self._last_success.get(entity_id)

        if last_success is None:
            return True  # Never succeeded = stale

        threshold = self._health_config.stale_threshold_seconds
        return (time.time() - last_success) > threshold

    def get_entity_state(self, entity_id: str) -> EntityHealthState:
        """Get detailed health state for an entity.

        Thread-safe: Returns a copy of the state.

        Args:
            entity_id: Entity identifier

        Returns:
            EntityHealthState with current health information
        """
        self._ensure_health_initialized()

        with self._health_lock:
            return EntityHealthState(
                entity_id=entity_id,
                failure_count=self._failure_counts.get(entity_id, 0),
                last_success_time=self._last_success.get(entity_id),
                last_failure_time=self._last_failure.get(entity_id),
                is_suspect=entity_id in self._suspect_emitted,
                failure_reasons=list(self._failure_reasons.get(entity_id, [])),
            )

    def get_health_summary(self) -> EntityHealthSummary:
        """Get summary of health across all tracked entities.

        Thread-safe: Returns a copy of the summary.

        Returns:
            EntityHealthSummary with aggregate health information
        """
        self._ensure_health_initialized()

        with self._health_lock:
            all_entities = set(self._failure_counts.keys()) | set(self._last_success.keys())
            stale_threshold = self._health_config.stale_threshold_seconds
            now = time.time()

            # Calculate stale entities
            stale_entities = []
            for entity_id in all_entities:
                last_success = self._last_success.get(entity_id)
                if last_success is None or (now - last_success) > stale_threshold:
                    stale_entities.append(entity_id)

            # Count healthy (not suspect and not stale)
            suspect_entities = list(self._suspect_emitted)
            unhealthy = set(suspect_entities) | set(stale_entities)
            healthy_count = len(all_entities) - len(unhealthy)

            return EntityHealthSummary(
                total_entities=len(all_entities),
                healthy_count=healthy_count,
                suspect_count=len(suspect_entities),
                stale_count=len(stale_entities),
                failure_counts=dict(self._failure_counts),
                suspect_entities=suspect_entities,
                stale_entities=stale_entities,
                failure_threshold=self._health_config.failure_threshold,
            )

    def clear_entity_health(self, entity_id: str) -> None:
        """Clear all health tracking data for an entity.

        Thread-safe.

        Args:
            entity_id: Entity identifier to clear
        """
        self._ensure_health_initialized()

        with self._health_lock:
            self._failure_counts.pop(entity_id, None)
            self._last_success.pop(entity_id, None)
            self._last_failure.pop(entity_id, None)
            self._suspect_emitted.discard(entity_id)
            self._failure_reasons.pop(entity_id, None)

    def clear_all_health_tracking(self) -> None:
        """Clear all health tracking data.

        Thread-safe.
        """
        self._ensure_health_initialized()

        with self._health_lock:
            self._failure_counts.clear()
            self._last_success.clear()
            self._last_failure.clear()
            self._suspect_emitted.clear()
            self._failure_reasons.clear()

    def get_tracked_entities(self) -> list[str]:
        """Get list of all tracked entity IDs.

        Thread-safe.

        Returns:
            List of entity IDs being tracked
        """
        self._ensure_health_initialized()

        with self._health_lock:
            return list(set(self._failure_counts.keys()) | set(self._last_success.keys()))

    def get_suspect_entities(self) -> list[str]:
        """Get list of entities currently marked as suspect.

        Thread-safe.

        Returns:
            List of suspect entity IDs
        """
        self._ensure_health_initialized()

        with self._health_lock:
            return list(self._suspect_emitted)

    def reset_entity_failures(self, entity_id: str) -> None:
        """Reset failure count for an entity without recording success.

        Useful for manual recovery or circuit breaker reset.
        Thread-safe.

        Args:
            entity_id: Entity identifier
        """
        self._ensure_health_initialized()

        with self._health_lock:
            self._failure_counts.pop(entity_id, None)
            self._suspect_emitted.discard(entity_id)
            self._failure_reasons.pop(entity_id, None)
            # Note: We keep last_success and last_failure for history

    def get_last_success_time(self, entity_id: str) -> float | None:
        """Get timestamp of last success for an entity.

        Thread-safe.

        Args:
            entity_id: Entity identifier

        Returns:
            Unix timestamp of last success, or None if never succeeded
        """
        self._ensure_health_initialized()

        with self._health_lock:
            return self._last_success.get(entity_id)

    def get_seconds_since_success(self, entity_id: str) -> float | None:
        """Get seconds since last success for an entity.

        Thread-safe.

        Args:
            entity_id: Entity identifier

        Returns:
            Seconds since last success, or None if never succeeded
        """
        last_success = self.get_last_success_time(entity_id)
        if last_success is None:
            return None
        return time.time() - last_success

    def health_tracking_status(self) -> dict[str, Any]:
        """Get status dict for health endpoint.

        Thread-safe.

        Returns:
            Dictionary with health tracking status for JSON serialization
        """
        summary = self.get_health_summary()
        return {
            "health_score": round(summary.health_score, 3),
            "total_tracked": summary.total_entities,
            "healthy_count": summary.healthy_count,
            "suspect_count": summary.suspect_count,
            "stale_count": summary.stale_count,
            "suspect_entities": summary.suspect_entities[:10],  # Limit for response size
            "stale_entities": summary.stale_entities[:10],
            "config": {
                "failure_threshold": self._health_config.failure_threshold,
                "stale_threshold_seconds": self._health_config.stale_threshold_seconds,
                "backoff_max_seconds": self._health_config.backoff_max_seconds,
            },
        }
