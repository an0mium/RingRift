"""Data freshness and signal fetching mixin for SelfplayScheduler.

Sprint 17.9+ (Feb 2026): Extracted from selfplay_scheduler.py to reduce file size.

Provides methods for fetching external data needed for priority calculation:
- Data freshness (hours since last export)
- Elo velocities per config
- Current Elo ratings (4-layer fallback)
- Feedback loop signals
- Curriculum weights

These methods are async because they may query external services (databases,
singletons) and are called in parallel during _update_priorities().
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

from app.coordination.budget_calculator import parse_config_key
from app.coordination.priority_calculator import ALL_CONFIGS

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FreshnessFetcherMixin:
    """Mixin providing async data fetching for priority calculation.

    Expects the host class to have:
    - _training_freshness: Optional training freshness tracker
    - _config_priorities: dict of ConfigPriority
    """

    async def _get_data_freshness(self) -> dict[str, float]:
        """Get data freshness (hours since last export) per config.

        Returns:
            Dict mapping config_key to hours since last export
        """
        # Import here to access module-level constant without circular import
        from app.coordination.selfplay_scheduler import MAX_STALENESS_HOURS

        result: dict[str, float] = {}

        try:
            # Try using TrainingFreshness if available
            if self._training_freshness is None:
                try:
                    from app.coordination.training_freshness import get_training_freshness
                    self._training_freshness = get_training_freshness()
                except ImportError:
                    pass

            if self._training_freshness:
                for config_key in ALL_CONFIGS:
                    staleness = await self._training_freshness.get_staleness(config_key)
                    result[config_key] = staleness
            else:
                # Fallback: check NPZ file modification times
                from pathlib import Path
                now = time.time()

                for config_key in ALL_CONFIGS:
                    board_type, num_players = parse_config_key(config_key)
                    npz_path = Path(f"data/training/{board_type}_{num_players}p.npz")

                    if npz_path.exists():
                        mtime = npz_path.stat().st_mtime
                        result[config_key] = (now - mtime) / 3600.0
                    else:
                        result[config_key] = MAX_STALENESS_HOURS

        except (OSError, ValueError, IndexError) as e:
            logger.warning(f"[SelfplayScheduler] Error getting freshness (using defaults): {e}")

        return result

    async def _get_elo_velocities(self) -> dict[str, float]:
        """Get ELO improvement velocities per config.

        Returns:
            Dict mapping config_key to ELO points per day
        """
        result: dict[str, float] = {}

        try:
            # Try using QueuePopulator's ConfigTarget if available
            from app.coordination.unified_queue_populator import get_queue_populator

            populator = get_queue_populator()
            if populator:
                for config_key, target in populator._targets.items():
                    result[config_key] = target.elo_velocity
        except ImportError:
            pass
        except (AttributeError, KeyError) as e:
            logger.warning(f"[SelfplayScheduler] Error getting ELO velocities (using defaults): {e}")

        return result

    async def _get_current_elos(self) -> dict[str, float]:
        """Get current Elo ratings per config.

        Dec 29, 2025: Added for adaptive Gumbel budget scaling.
        Jan 2026: Prefers composite Elo entries (with harness tracking) over legacy.

        Uses 4-layer fallback:
        1. Composite Elo (harness-tracked, highest quality)
        2. QueuePopulator (fast, in-memory)
        3. Legacy EloService (any participant)
        4. Default 1500

        Returns:
            Dict mapping config_key to current Elo rating
        """
        result: dict[str, float] = {}

        # Layer 1: Prefer composite Elo entries (Jan 2026)
        # These have harness_type and simulation_count, representing quality evaluations
        try:
            from app.training.elo_service import get_elo_service
            import sqlite3

            elo_service = get_elo_service()
            if hasattr(elo_service, '_db_path'):
                conn = sqlite3.connect(elo_service._db_path)
                # Query composite entries with highest budget (b800, b1600)
                # Format: canonical_hex8_2p:gumbel_mcts:b800
                for config_key in ALL_CONFIGS:
                    board_type, num_players = parse_config_key(config_key)
                    cur = conn.execute("""
                        SELECT participant_id, rating, games_played, simulation_count
                        FROM elo_ratings
                        WHERE board_type = ? AND num_players = ?
                          AND participant_id LIKE ?
                          AND participant_id NOT LIKE '%heuristic%'
                          AND participant_id NOT LIKE '%random%'
                          AND participant_id NOT LIKE '%baseline%'
                          AND games_played >= 10
                        ORDER BY simulation_count DESC NULLS LAST, rating DESC
                        LIMIT 1
                    """, (board_type, num_players, f"canonical_{config_key}:%"))
                    row = cur.fetchone()
                    if row:
                        result[config_key] = row[1]  # rating
                        logger.debug(
                            f"[SelfplayScheduler] Using composite Elo for {config_key}: "
                            f"{row[1]:.0f} ({row[0]}, {row[2]} games)"
                        )
                conn.close()
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Composite Elo query failed: {e}")

        # Layer 2: Try QueuePopulator's ConfigTarget (fastest, in-memory)
        try:
            from app.coordination.unified_queue_populator import get_queue_populator

            populator = get_queue_populator()
            if populator:
                for config_key, target in populator._targets.items():
                    if config_key not in result and target.current_best_elo > 0:
                        result[config_key] = target.current_best_elo
        except ImportError:
            pass
        except (AttributeError, KeyError) as e:
            logger.debug(f"[SelfplayScheduler] QueuePopulator unavailable: {e}")

        # Layer 3: Fallback to EloService database for missing configs
        missing_configs = [c for c in ALL_CONFIGS if c not in result]
        if missing_configs:
            try:
                from app.training.elo_service import get_elo_service

                elo_service = get_elo_service()
                for config_key in missing_configs:
                    # Parse config_key: "hex8_2p" -> board_type="hex8", num_players=2
                    board_type, num_players = parse_config_key(config_key)
                    # Get leaderboard and take top NN model's Elo
                    # Use limit=10 to skip heuristic/random baselines at top
                    leaderboard = elo_service.get_leaderboard(
                        board_type=board_type,
                        num_players=num_players,
                        limit=10,
                        min_games=5,  # Require some confidence
                    )
                    for entry in leaderboard:
                        pid = entry.participant_id.lower()
                        if any(x in pid for x in ("heuristic", "random", "baseline")):
                            continue
                        result[config_key] = entry.rating
                        logger.debug(
                            f"[SelfplayScheduler] Got Elo from EloService for {config_key}: "
                            f"{entry.rating:.0f} ({entry.participant_id})"
                        )
                        break
            except ImportError:
                logger.debug("[SelfplayScheduler] EloService not available")
            except Exception as e:
                logger.debug(f"[SelfplayScheduler] EloService query failed: {e}")

        # Layer 4: Default 1500 for any still-missing configs
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 1500.0  # Default starting Elo

        return result

    async def _get_feedback_signals(self) -> dict[str, dict[str, Any]]:
        """Get feedback loop signals per config.

        Returns:
            Dict mapping config_key to feedback data
        """
        result: dict[str, dict[str, Any]] = {}

        try:
            from app.coordination.feedback_loop_controller import get_feedback_loop_controller

            controller = get_feedback_loop_controller()
            if controller:
                for config_key in ALL_CONFIGS:
                    state = controller._get_or_create_state(config_key)
                    result[config_key] = {
                        "exploration_boost": state.current_exploration_boost,
                        "training_pending": state.current_training_intensity == "accelerated",
                    }
        except ImportError:
            pass
        except (AttributeError, KeyError) as e:
            logger.warning(f"[SelfplayScheduler] Error getting feedback signals (using defaults): {e}")

        return result

    async def _get_curriculum_weights(self) -> dict[str, float]:
        """Get curriculum weights per config.

        December 2025 - Phase 2C.3: Wire curriculum weights into priority calculation.
        Higher curriculum weight = config needs more training data.

        Returns:
            Dict mapping config_key to curriculum weight (default 1.0)
        """
        result: dict[str, float] = {}

        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            if feedback:
                # Use get_curriculum_weights() which computes weights from metrics
                # (win rates, Elo trends, weak opponent data) instead of accessing
                # _current_weights directly which may be empty initially
                computed_weights = feedback.get_curriculum_weights()
                if computed_weights:
                    result = computed_weights
                else:
                    # Fall back to manually tracked weights if metrics unavailable
                    result = dict(feedback._current_weights)

        except ImportError:
            logger.debug("[SelfplayScheduler] curriculum_feedback not available")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting curriculum weights: {e}")

        # Ensure all configs have a default weight
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 1.0

        # December 2025: Partial normalization of curriculum weights
        # Session 17.42: Changed from full normalization to 50% partial to preserve differentials
        # Full normalization was flattening weights (e.g., 1.4 -> 1.23), reducing curriculum influence
        # Now we apply only 50% of the scaling adjustment to preserve weight differentials
        if result:
            total = sum(result.values())
            target_sum = len(result)  # Average of 1.0
            if total > 0 and abs(total - target_sum) > 0.01:
                raw_scale = target_sum / total
                # Apply only 50% of the scaling to preserve differentials
                # e.g., if raw_scale=0.88, partial_scale=0.94 (keeps half the differential)
                scale = 1.0 + (raw_scale - 1.0) * 0.5
                result = {k: v * scale for k, v in result.items()}
                logger.debug(
                    f"[SelfplayScheduler] Partial normalized curriculum weights: "
                    f"raw_scale={raw_scale:.3f}, applied_scale={scale:.3f}, "
                    f"sum={sum(result.values()):.2f}"
                )

        return result
