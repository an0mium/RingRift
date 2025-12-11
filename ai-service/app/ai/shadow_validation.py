"""Shadow Validation: GPU vs CPU Parity Verification.

This module provides runtime validation of GPU-generated moves against the
canonical CPU rules engine. It's the safety net for Phase 2 of the GPU pipeline.

Architecture (per GPU_PIPELINE_ROADMAP.md Section 7):
- GPU generates moves in parallel (fast, approximate)
- Shadow validator samples X% of moves
- Validates sampled moves against CPU rules engine
- Logs divergence statistics
- Halts if divergence exceeds threshold

Usage:
    validator = ShadowValidator(sample_rate=0.05, threshold=0.001)

    # During selfplay
    gpu_moves = generate_moves_batch(batch_state)
    validator.validate_batch(gpu_moves, cpu_game_states)

    # Check stats
    print(validator.get_report())

Configuration:
    SHADOW_SAMPLE_RATE: 0.05 (5% of moves validated)
    DIVERGENCE_THRESHOLD: 0.001 (0.1% max divergence before halt)

See Also:
    - docs/GPU_PIPELINE_ROADMAP.md Section 7 (Phase 2 Architecture)
    - docs/GPU_ARCHITECTURE_SIMPLIFICATION.md Section 2.4 (Evaluation Discrepancy)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import GameState, Move, Position
    from app.game_engine import GameEngine
    from .gpu_parallel_games import BatchGameState

from app.models import MoveType

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default validation rates
DEFAULT_SAMPLE_RATE = 0.05  # 5% of moves validated
DEFAULT_DIVERGENCE_THRESHOLD = 0.001  # 0.1% max divergence

# Logging thresholds
WARN_DIVERGENCE_RATE = 0.0001  # Log warning at 0.01% divergence
LOG_EVERY_N_VALIDATIONS = 100  # Log stats every N validations


class DivergenceType(Enum):
    """Types of move divergence between GPU and CPU."""
    MISSING_IN_GPU = "missing_in_gpu"  # CPU has move, GPU doesn't
    EXTRA_IN_GPU = "extra_in_gpu"  # GPU has move, CPU doesn't
    MOVE_COUNT_MISMATCH = "move_count_mismatch"  # Different number of moves
    MOVE_DETAILS_MISMATCH = "move_details_mismatch"  # Same count, different moves


@dataclass
class DivergenceRecord:
    """Record of a single divergence event."""
    timestamp: float
    game_index: int
    move_number: int
    divergence_type: DivergenceType
    cpu_move_count: int
    gpu_move_count: int
    missing_moves: List[str]  # Moves in CPU but not GPU
    extra_moves: List[str]  # Moves in GPU but not CPU
    game_state_hash: Optional[str] = None  # For debugging


@dataclass
class ValidationStats:
    """Aggregate validation statistics."""
    total_validations: int = 0
    total_divergences: int = 0
    divergence_by_type: Dict[DivergenceType, int] = field(default_factory=dict)

    # Per-move-type stats
    placement_validations: int = 0
    placement_divergences: int = 0
    movement_validations: int = 0
    movement_divergences: int = 0
    capture_validations: int = 0
    capture_divergences: int = 0
    recovery_validations: int = 0
    recovery_divergences: int = 0

    # Timing
    total_validation_time_ms: float = 0.0

    @property
    def divergence_rate(self) -> float:
        """Current divergence rate."""
        if self.total_validations == 0:
            return 0.0
        return self.total_divergences / self.total_validations

    @property
    def avg_validation_time_ms(self) -> float:
        """Average time per validation."""
        if self.total_validations == 0:
            return 0.0
        return self.total_validation_time_ms / self.total_validations


# =============================================================================
# Shadow Validator
# =============================================================================


class ShadowValidator:
    """Validates GPU-generated moves against CPU rules engine.

    This is the primary safety mechanism for Phase 2 GPU acceleration.
    It ensures GPU move generation doesn't silently diverge from canonical rules.

    Thread-safety: Not thread-safe. Use one validator per thread/process.

    Attributes:
        sample_rate: Fraction of moves to validate (0.0-1.0)
        threshold: Maximum allowed divergence rate before raising error
        stats: Aggregate validation statistics
        divergence_log: Recent divergence records for debugging
    """

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
        max_divergence_log: int = 100,
        halt_on_threshold: bool = True,
    ):
        """Initialize shadow validator.

        Args:
            sample_rate: Fraction of moves to validate (0.0-1.0)
            threshold: Maximum divergence rate before halting
            max_divergence_log: Maximum divergence records to keep
            halt_on_threshold: If True, raise error when threshold exceeded
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.max_divergence_log = max_divergence_log
        self.halt_on_threshold = halt_on_threshold

        self.stats = ValidationStats()
        self.divergence_log: List[DivergenceRecord] = []

        self._rng = random.Random()  # Dedicated RNG for sampling

        logger.info(
            f"ShadowValidator initialized: sample_rate={sample_rate:.1%}, "
            f"threshold={threshold:.4%}, halt_on_threshold={halt_on_threshold}"
        )

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible sampling."""
        self._rng.seed(seed)

    def should_validate(self) -> bool:
        """Determine if this move should be validated (probabilistic sampling)."""
        return self._rng.random() < self.sample_rate

    def validate_placement_moves(
        self,
        gpu_positions: List[Tuple[int, int]],
        game_state: "GameState",
        player: int,
    ) -> bool:
        """Validate GPU placement move generation against CPU.

        Args:
            gpu_positions: List of (row, col) positions from GPU
            game_state: CPU GameState for validation
            player: Player whose moves to validate

        Returns:
            True if validation passed, False if divergence detected
        """
        if not self.should_validate():
            return True

        start_time = time.perf_counter()

        # Import here to avoid circular imports
        from app.game_engine import GameEngine

        # Get CPU moves
        cpu_moves = GameEngine.get_valid_moves(game_state, player)
        cpu_placement_moves = [
            m for m in cpu_moves
            if m.type == MoveType.PLACE_RING
        ]

        # Convert to comparable format (Position uses x, y fields)
        cpu_positions = set((m.to.x, m.to.y) for m in cpu_placement_moves)
        gpu_positions_set = set(gpu_positions)

        # Compare
        missing = cpu_positions - gpu_positions_set
        extra = gpu_positions_set - cpu_positions

        self.stats.placement_validations += 1
        self.stats.total_validations += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_validation_time_ms += elapsed_ms

        if missing or extra:
            self._record_divergence(
                game_index=0,
                move_number=game_state.move_count,
                divergence_type=DivergenceType.MOVE_DETAILS_MISMATCH,
                cpu_count=len(cpu_positions),
                gpu_count=len(gpu_positions_set),
                missing=[f"({r},{c})" for r, c in missing],
                extra=[f"({r},{c})" for r, c in extra],
            )
            self.stats.placement_divergences += 1
            return False

        return True

    def validate_movement_moves(
        self,
        gpu_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        game_state: "GameState",
        player: int,
    ) -> bool:
        """Validate GPU movement move generation against CPU.

        Args:
            gpu_moves: List of ((from_row, from_col), (to_row, to_col)) from GPU
            game_state: CPU GameState for validation
            player: Player whose moves to validate

        Returns:
            True if validation passed, False if divergence detected
        """
        if not self.should_validate():
            return True

        start_time = time.perf_counter()

        from app.game_engine import GameEngine

        cpu_moves = GameEngine.get_valid_moves(game_state, player)
        cpu_movement_moves = [
            m for m in cpu_moves
            if m.type == MoveType.MOVE_STACK
        ]

        cpu_move_set = set(
            ((m.from_pos.x, m.from_pos.y), (m.to.x, m.to.y))
            for m in cpu_movement_moves
        )
        gpu_move_set = set(gpu_moves)

        self.stats.movement_validations += 1
        self.stats.total_validations += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_validation_time_ms += elapsed_ms

        missing = cpu_move_set - gpu_move_set
        extra = gpu_move_set - cpu_move_set

        if missing or extra:
            self._record_divergence(
                game_index=0,
                move_number=game_state.move_count,
                divergence_type=DivergenceType.MOVE_DETAILS_MISMATCH,
                cpu_count=len(cpu_move_set),
                gpu_count=len(gpu_move_set),
                missing=[f"{f}->{t}" for f, t in missing],
                extra=[f"{f}->{t}" for f, t in extra],
            )
            self.stats.movement_divergences += 1
            return False

        return True

    def validate_capture_moves(
        self,
        gpu_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        game_state: "GameState",
        player: int,
    ) -> bool:
        """Validate GPU capture move generation against CPU.

        Args:
            gpu_moves: List of ((from_row, from_col), (to_row, to_col)) from GPU
            game_state: CPU GameState for validation
            player: Player whose moves to validate

        Returns:
            True if validation passed, False if divergence detected
        """
        if not self.should_validate():
            return True

        start_time = time.perf_counter()

        from app.game_engine import GameEngine

        cpu_moves = GameEngine.get_valid_moves(game_state, player)
        cpu_capture_moves = [
            m for m in cpu_moves
            if m.type == MoveType.OVERTAKING_CAPTURE
        ]

        cpu_move_set = set(
            ((m.from_pos.x, m.from_pos.y), (m.to.x, m.to.y))
            for m in cpu_capture_moves
        )
        gpu_move_set = set(gpu_moves)

        self.stats.capture_validations += 1
        self.stats.total_validations += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_validation_time_ms += elapsed_ms

        missing = cpu_move_set - gpu_move_set
        extra = gpu_move_set - cpu_move_set

        if missing or extra:
            self._record_divergence(
                game_index=0,
                move_number=game_state.move_count,
                divergence_type=DivergenceType.MOVE_DETAILS_MISMATCH,
                cpu_count=len(cpu_move_set),
                gpu_count=len(gpu_move_set),
                missing=[f"{f}->{t}" for f, t in missing],
                extra=[f"{f}->{t}" for f, t in extra],
            )
            self.stats.capture_divergences += 1
            return False

        return True

    def validate_recovery_moves(
        self,
        gpu_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        game_state: "GameState",
        player: int,
    ) -> bool:
        """Validate GPU recovery move generation against CPU.

        Args:
            gpu_moves: List of ((from_row, from_col), (to_row, to_col)) from GPU
            game_state: CPU GameState for validation
            player: Player whose moves to validate

        Returns:
            True if validation passed, False if divergence detected
        """
        if not self.should_validate():
            return True

        start_time = time.perf_counter()

        from app.game_engine import GameEngine

        cpu_moves = GameEngine.get_valid_moves(game_state, player)
        cpu_recovery_moves = [
            m for m in cpu_moves
            if m.type == MoveType.RECOVERY_SLIDE
        ]

        cpu_move_set = set(
            ((m.from_pos.x, m.from_pos.y), (m.to.x, m.to.y))
            for m in cpu_recovery_moves
        )
        gpu_move_set = set(gpu_moves)

        self.stats.recovery_validations += 1
        self.stats.total_validations += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_validation_time_ms += elapsed_ms

        missing = cpu_move_set - gpu_move_set
        extra = gpu_move_set - cpu_move_set

        if missing or extra:
            self._record_divergence(
                game_index=0,
                move_number=game_state.move_count,
                divergence_type=DivergenceType.MOVE_DETAILS_MISMATCH,
                cpu_count=len(cpu_move_set),
                gpu_count=len(gpu_move_set),
                missing=[f"{f}->{t}" for f, t in missing],
                extra=[f"{f}->{t}" for f, t in extra],
            )
            self.stats.recovery_divergences += 1
            return False

        return True

    def validate_all_moves(
        self,
        gpu_moves: List[Dict[str, Any]],
        game_state: "GameState",
        player: int,
    ) -> bool:
        """Validate all GPU-generated moves against CPU.

        Args:
            gpu_moves: List of move dicts with 'type' and position info
            game_state: CPU GameState for validation
            player: Player whose moves to validate

        Returns:
            True if validation passed, False if divergence detected
        """
        if not self.should_validate():
            return True

        start_time = time.perf_counter()

        from app.game_engine import GameEngine

        cpu_moves = GameEngine.get_valid_moves(game_state)
        cpu_moves_for_player = [m for m in cpu_moves if m.player == player]

        # Convert to comparable format
        def move_key(m) -> str:
            if hasattr(m, 'type'):
                # CPU Move object
                from_str = f"({m.from_pos.row},{m.from_pos.col})" if m.from_pos else "None"
                to_str = f"({m.to.row},{m.to.col})" if m.to else "None"
                return f"{m.type}:{from_str}->{to_str}"
            else:
                # GPU move dict
                from_str = f"({m.get('from_row', -1)},{m.get('from_col', -1)})"
                to_str = f"({m.get('to_row', -1)},{m.get('to_col', -1)})"
                return f"{m.get('type', 'unknown')}:{from_str}->{to_str}"

        cpu_move_set = set(move_key(m) for m in cpu_moves_for_player)
        gpu_move_set = set(move_key(m) for m in gpu_moves)

        self.stats.total_validations += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_validation_time_ms += elapsed_ms

        missing = cpu_move_set - gpu_move_set
        extra = gpu_move_set - cpu_move_set

        if missing or extra:
            self._record_divergence(
                game_index=0,
                move_number=game_state.move_count,
                divergence_type=DivergenceType.MOVE_DETAILS_MISMATCH,
                cpu_count=len(cpu_move_set),
                gpu_count=len(gpu_move_set),
                missing=list(missing)[:10],  # Limit for logging
                extra=list(extra)[:10],
            )
            return False

        return True

    def _record_divergence(
        self,
        game_index: int,
        move_number: int,
        divergence_type: DivergenceType,
        cpu_count: int,
        gpu_count: int,
        missing: List[str],
        extra: List[str],
    ) -> None:
        """Record a divergence event."""
        self.stats.total_divergences += 1

        # Update type-specific count
        if divergence_type not in self.stats.divergence_by_type:
            self.stats.divergence_by_type[divergence_type] = 0
        self.stats.divergence_by_type[divergence_type] += 1

        # Log the divergence
        record = DivergenceRecord(
            timestamp=time.time(),
            game_index=game_index,
            move_number=move_number,
            divergence_type=divergence_type,
            cpu_move_count=cpu_count,
            gpu_move_count=gpu_count,
            missing_moves=missing,
            extra_moves=extra,
        )

        self.divergence_log.append(record)
        if len(self.divergence_log) > self.max_divergence_log:
            self.divergence_log.pop(0)

        # Log warning
        logger.warning(
            f"GPU/CPU divergence at game {game_index}, move {move_number}: "
            f"CPU={cpu_count}, GPU={gpu_count}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )

        # Check threshold
        if self.halt_on_threshold and self.stats.divergence_rate > self.threshold:
            error_msg = (
                f"GPU divergence rate {self.stats.divergence_rate:.4%} exceeds "
                f"threshold {self.threshold:.4%}. Halting to prevent training corruption."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Periodic stats logging
        if self.stats.total_validations % LOG_EVERY_N_VALIDATIONS == 0:
            self._log_stats()

    def _log_stats(self) -> None:
        """Log current validation statistics."""
        logger.info(
            f"Shadow validation stats: {self.stats.total_validations} validated, "
            f"{self.stats.total_divergences} divergences ({self.stats.divergence_rate:.4%}), "
            f"avg time: {self.stats.avg_validation_time_ms:.2f}ms"
        )

    def get_report(self) -> Dict[str, Any]:
        """Get a detailed validation report."""
        return {
            "total_validations": self.stats.total_validations,
            "total_divergences": self.stats.total_divergences,
            "divergence_rate": self.stats.divergence_rate,
            "threshold": self.threshold,
            "status": "PASS" if self.stats.divergence_rate <= self.threshold else "FAIL",
            "by_move_type": {
                "placement": {
                    "validations": self.stats.placement_validations,
                    "divergences": self.stats.placement_divergences,
                },
                "movement": {
                    "validations": self.stats.movement_validations,
                    "divergences": self.stats.movement_divergences,
                },
                "capture": {
                    "validations": self.stats.capture_validations,
                    "divergences": self.stats.capture_divergences,
                },
                "recovery": {
                    "validations": self.stats.recovery_validations,
                    "divergences": self.stats.recovery_divergences,
                },
            },
            "timing": {
                "total_ms": self.stats.total_validation_time_ms,
                "avg_ms": self.stats.avg_validation_time_ms,
            },
            "recent_divergences": [
                {
                    "game": r.game_index,
                    "move": r.move_number,
                    "type": r.divergence_type.value,
                    "cpu_count": r.cpu_move_count,
                    "gpu_count": r.gpu_move_count,
                }
                for r in self.divergence_log[-5:]
            ],
        }

    def reset_stats(self) -> None:
        """Reset all validation statistics."""
        self.stats = ValidationStats()
        self.divergence_log.clear()
        logger.info("Shadow validator stats reset")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_shadow_validator(
    sample_rate: Optional[float] = None,
    threshold: Optional[float] = None,
    enabled: bool = True,
) -> Optional[ShadowValidator]:
    """Create a shadow validator with sensible defaults.

    Args:
        sample_rate: Override default sample rate
        threshold: Override default threshold
        enabled: If False, returns None (disabled validation)

    Returns:
        ShadowValidator if enabled, None otherwise
    """
    if not enabled:
        return None

    return ShadowValidator(
        sample_rate=sample_rate or DEFAULT_SAMPLE_RATE,
        threshold=threshold or DEFAULT_DIVERGENCE_THRESHOLD,
    )


def validate_batch_moves(
    validator: Optional[ShadowValidator],
    gpu_moves_by_game: List[List[Dict[str, Any]]],
    cpu_game_states: List["GameState"],
) -> bool:
    """Convenience function to validate a batch of moves.

    Args:
        validator: Shadow validator (if None, validation skipped)
        gpu_moves_by_game: GPU moves indexed by game index
        cpu_game_states: Corresponding CPU game states

    Returns:
        True if all validations passed (or validation disabled)
    """
    if validator is None:
        return True

    all_passed = True
    for game_idx, (gpu_moves, cpu_state) in enumerate(zip(gpu_moves_by_game, cpu_game_states)):
        player = cpu_state.current_player
        if not validator.validate_all_moves(gpu_moves, cpu_state, player):
            all_passed = False

    return all_passed
