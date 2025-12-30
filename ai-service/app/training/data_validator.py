"""Data validation orchestrator for RingRift training.

December 2025: Extracted from train.py to improve modularity.

This module provides the DataValidator class which orchestrates all pre-training
data validation including freshness, structure, content, and checksums.

Usage:
    from app.training.data_validator import DataValidator, DataValidationConfig

    config = DataValidationConfig(
        skip_freshness_check=False,
        max_data_age_hours=1.0,
    )
    validator = DataValidator(config)
    result = validator.validate_all(
        data_paths=["data/training/sq8_2p.npz"],
        board_type="square8",
        num_players=2,
    )
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.training.train_validation import (
    DataValidationResult,
    FreshnessResult,
    ValidationResult,
    StructureValidationResult,
    validate_training_data_freshness,
    validate_training_data_files,
    validate_data_checksums,
    validate_npz_structure_files,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class DataValidationConfig:
    """Configuration for data validation.

    Controls which validation checks are run and their thresholds.
    """

    # Freshness check
    skip_freshness_check: bool = False
    max_data_age_hours: float = 1.0
    allow_stale_data: bool = False

    # Stale fallback (for 48-hour autonomous operation)
    disable_stale_fallback: bool = False
    max_sync_failures: int = 5
    max_sync_duration: float = 2700.0  # 45 minutes

    # Structure validation
    validate_structure: bool = True
    require_policy: bool = True

    # Content validation
    validate_content: bool = True
    fail_on_invalid: bool = False

    # Checksum validation
    validate_checksums: bool = True
    fail_on_checksum_mismatch: bool = False
    checksum_size_limit_mb: float = 500.0

    # Event emission
    emit_events: bool = True

    @classmethod
    def from_resolved(cls, resolved: Any) -> "DataValidationConfig":
        """Create config from ResolvedConfig.

        Args:
            resolved: ResolvedConfig with validation settings

        Returns:
            DataValidationConfig instance
        """
        return cls(
            skip_freshness_check=getattr(resolved, "skip_freshness_check", False),
            max_data_age_hours=getattr(resolved, "max_data_age_hours", 1.0),
            allow_stale_data=getattr(resolved, "allow_stale_data", False),
            disable_stale_fallback=getattr(resolved, "disable_stale_fallback", False),
            max_sync_failures=getattr(resolved, "max_sync_failures", 5),
            max_sync_duration=getattr(resolved, "max_sync_duration", 2700.0),
            validate_content=getattr(resolved, "validate_data", True),
            fail_on_invalid=getattr(resolved, "fail_on_invalid_data", False),
        )


@dataclass
class StaleFallbackState:
    """State for stale fallback tracking.

    Used to determine if training should proceed with stale data
    after repeated sync failures or timeouts.
    """

    sync_failures: int = 0
    sync_start_time: float = field(default_factory=time.time)
    fallback_triggered: bool = False
    fallback_reason: str = ""


class DataValidator:
    """Orchestrates all pre-training data validation.

    Runs validation checks in order:
    1. Data freshness check (with stale fallback logic)
    2. NPZ structure validation
    3. Data content validation
    4. Checksum verification

    Example:
        validator = DataValidator(DataValidationConfig())
        result = validator.validate_all(
            data_paths=["data/training/sq8_2p.npz"],
            board_type="square8",
            num_players=2,
        )
        if not result.all_valid:
            logger.error(f"Validation failed: {result.errors}")
    """

    def __init__(self, config: DataValidationConfig | None = None):
        """Initialize the validator.

        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or DataValidationConfig()
        self._stale_fallback_state = StaleFallbackState()

    def validate_all(
        self,
        data_paths: list[str],
        board_type: str,
        num_players: int,
        distributed: bool = False,
        is_main_process: bool = True,
    ) -> DataValidationResult:
        """Run all validation checks and return aggregated result.

        Args:
            data_paths: Paths to training data files
            board_type: Board type (e.g., "square8")
            num_players: Number of players
            distributed: Whether running in distributed mode
            is_main_process: Whether this is the main process

        Returns:
            DataValidationResult with all validation results
        """
        errors: list[str] = []
        freshness_result = None
        structure_results = None
        file_validations = None
        checksum_results = None

        # Phase 1: Freshness check
        if not self.config.skip_freshness_check:
            try:
                freshness_result = self._check_freshness(
                    board_type=board_type,
                    num_players=num_players,
                    is_main_process=is_main_process,
                )
            except ValueError as e:
                errors.append(f"Freshness check failed: {e}")
                if not self.config.allow_stale_data:
                    return DataValidationResult(
                        all_valid=False,
                        freshness=freshness_result,
                        errors=errors,
                    )

        # Filter to existing paths
        valid_paths = [p for p in data_paths if p and os.path.exists(p)]
        if not valid_paths:
            errors.append("No valid data paths found")
            return DataValidationResult(
                all_valid=False,
                freshness=freshness_result,
                errors=errors,
            )

        # Phase 2: Structure validation
        if self.config.validate_structure:
            try:
                structure_results = validate_npz_structure_files(
                    data_paths=valid_paths,
                    require_policy=self.config.require_policy,
                    fail_on_invalid=self.config.fail_on_invalid,
                )

                # Check for structure errors
                for path, result in structure_results.items():
                    if not result.valid:
                        errors.extend(result.errors)

            except ValueError as e:
                errors.append(f"Structure validation failed: {e}")
                if self.config.fail_on_invalid:
                    return DataValidationResult(
                        all_valid=False,
                        freshness=freshness_result,
                        structure_results=structure_results,
                        errors=errors,
                    )

        # Phase 3: Content validation
        if self.config.validate_content:
            try:
                file_validations = validate_training_data_files(
                    data_paths=valid_paths,
                    fail_on_invalid=self.config.fail_on_invalid,
                )

                for result in file_validations:
                    if not result.valid:
                        errors.extend(result.issues)

            except ValueError as e:
                errors.append(f"Content validation failed: {e}")
                if self.config.fail_on_invalid:
                    return DataValidationResult(
                        all_valid=False,
                        freshness=freshness_result,
                        structure_results=structure_results,
                        file_validations=file_validations,
                        errors=errors,
                    )

        # Phase 4: Checksum verification
        if self.config.validate_checksums:
            # Only validate checksums for files under size limit
            paths_to_check = self._filter_by_size(valid_paths)

            if paths_to_check:
                try:
                    checksum_results = validate_data_checksums(
                        data_paths=paths_to_check,
                        fail_on_mismatch=self.config.fail_on_checksum_mismatch,
                    )

                    for path, (valid, check_errors) in checksum_results.items():
                        if not valid:
                            errors.extend(check_errors)

                except ValueError as e:
                    errors.append(f"Checksum verification failed: {e}")
                    if self.config.fail_on_checksum_mismatch:
                        return DataValidationResult(
                            all_valid=False,
                            freshness=freshness_result,
                            structure_results=structure_results,
                            file_validations=file_validations,
                            checksum_results=checksum_results,
                            errors=errors,
                        )

        # Determine overall validity
        all_valid = len(errors) == 0

        return DataValidationResult(
            all_valid=all_valid,
            freshness=freshness_result,
            structure_results=structure_results,
            file_validations=file_validations,
            checksum_results=checksum_results,
            errors=errors if errors else None,
        )

    def _check_freshness(
        self,
        board_type: str,
        num_players: int,
        is_main_process: bool = True,
    ) -> FreshnessResult:
        """Check data freshness with stale fallback logic.

        Args:
            board_type: Board type
            num_players: Number of players
            is_main_process: Whether this is the main process

        Returns:
            FreshnessResult with freshness status

        Raises:
            ValueError: If data is stale and no fallback is available
        """
        try:
            return validate_training_data_freshness(
                board_type=board_type,
                num_players=num_players,
                max_age_hours=self.config.max_data_age_hours,
                allow_stale=self.config.allow_stale_data,
                emit_events=self.config.emit_events,
            )
        except ValueError:
            # Check if stale fallback should be triggered
            if self._should_allow_stale_fallback(board_type, num_players):
                config_key = f"{board_type}_{num_players}p"
                logger.warning(
                    f"[{config_key}] Stale fallback triggered: "
                    f"{self._stale_fallback_state.fallback_reason}"
                )
                return FreshnessResult(
                    is_fresh=False,
                    data_age_hours=-1.0,  # Unknown
                    games_available=0,
                    message=f"Stale fallback: {self._stale_fallback_state.fallback_reason}",
                )
            raise

    def _should_allow_stale_fallback(
        self,
        board_type: str,
        num_players: int,
    ) -> bool:
        """Check if stale fallback should be allowed.

        Stale fallback is allowed when:
        1. Not explicitly disabled
        2. Either: sync failures exceed threshold OR sync duration exceeds timeout

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            True if stale fallback should be allowed
        """
        if self.config.disable_stale_fallback:
            return False

        state = self._stale_fallback_state

        # Check sync failures
        if state.sync_failures >= self.config.max_sync_failures:
            state.fallback_triggered = True
            state.fallback_reason = (
                f"sync_failures ({state.sync_failures}) >= "
                f"max ({self.config.max_sync_failures})"
            )
            return True

        # Check sync duration
        duration = time.time() - state.sync_start_time
        if duration >= self.config.max_sync_duration:
            state.fallback_triggered = True
            state.fallback_reason = (
                f"sync_duration ({duration:.0f}s) >= "
                f"max ({self.config.max_sync_duration:.0f}s)"
            )
            return True

        return False

    def record_sync_failure(self) -> None:
        """Record a sync failure for stale fallback tracking."""
        self._stale_fallback_state.sync_failures += 1
        logger.debug(
            f"Sync failure recorded: {self._stale_fallback_state.sync_failures} total"
        )

    def reset_sync_state(self) -> None:
        """Reset sync state (e.g., after successful sync)."""
        self._stale_fallback_state = StaleFallbackState()

    def _filter_by_size(self, paths: list[str]) -> list[str]:
        """Filter paths to only those under size limit.

        Args:
            paths: List of file paths

        Returns:
            Filtered list of paths under size limit
        """
        size_limit_bytes = self.config.checksum_size_limit_mb * 1024 * 1024
        filtered = []

        for path in paths:
            try:
                size = os.path.getsize(path)
                if size <= size_limit_bytes:
                    filtered.append(path)
                else:
                    logger.debug(
                        f"Skipping checksum for large file: {path} "
                        f"({size / 1024 / 1024:.1f}MB > {self.config.checksum_size_limit_mb}MB)"
                    )
            except OSError:
                filtered.append(path)  # Include on error, let checksum handle it

        return filtered

    def emit_blocked_event(
        self,
        config_key: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Emit TRAINING_BLOCKED_BY_QUALITY event.

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            reason: Reason for blocking
            **kwargs: Additional event payload fields
        """
        if not self.config.emit_events:
            return

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus:
                payload = {
                    "config_key": config_key,
                    "reason": reason,
                    **kwargs,
                }
                bus.emit(DataEventType.TRAINING_BLOCKED_BY_QUALITY, payload)
                logger.info(f"Emitted TRAINING_BLOCKED_BY_QUALITY: {config_key} ({reason})")
        except Exception as e:
            logger.debug(f"Failed to emit training blocked event: {e}")


# =============================================================================
# Factory functions
# =============================================================================


def create_data_validator(
    skip_freshness_check: bool = False,
    max_data_age_hours: float = 1.0,
    allow_stale_data: bool = False,
    validate_data: bool = True,
    fail_on_invalid_data: bool = False,
    **kwargs: Any,
) -> DataValidator:
    """Create a DataValidator with the specified settings.

    Args:
        skip_freshness_check: Skip freshness validation
        max_data_age_hours: Maximum data age threshold
        allow_stale_data: Allow training with stale data
        validate_data: Enable content validation
        fail_on_invalid_data: Fail on validation errors
        **kwargs: Additional config parameters

    Returns:
        Configured DataValidator instance
    """
    config = DataValidationConfig(
        skip_freshness_check=skip_freshness_check,
        max_data_age_hours=max_data_age_hours,
        allow_stale_data=allow_stale_data,
        validate_content=validate_data,
        fail_on_invalid=fail_on_invalid_data,
        **{k: v for k, v in kwargs.items() if hasattr(DataValidationConfig, k)},
    )
    return DataValidator(config)


def validate_training_data(
    data_paths: list[str],
    board_type: str,
    num_players: int,
    skip_freshness_check: bool = False,
    max_data_age_hours: float = 1.0,
    allow_stale_data: bool = False,
    validate_data: bool = True,
    fail_on_invalid: bool = False,
) -> DataValidationResult:
    """Convenience function to validate training data.

    Args:
        data_paths: Paths to training data files
        board_type: Board type
        num_players: Number of players
        skip_freshness_check: Skip freshness check
        max_data_age_hours: Maximum data age
        allow_stale_data: Allow stale data
        validate_data: Enable content validation
        fail_on_invalid: Fail on validation errors

    Returns:
        DataValidationResult with all validation results
    """
    validator = create_data_validator(
        skip_freshness_check=skip_freshness_check,
        max_data_age_hours=max_data_age_hours,
        allow_stale_data=allow_stale_data,
        validate_data=validate_data,
        fail_on_invalid_data=fail_on_invalid,
    )
    return validator.validate_all(
        data_paths=data_paths,
        board_type=board_type,
        num_players=num_players,
    )
