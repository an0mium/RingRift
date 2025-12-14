"""Prometheus metrics for the RingRift AI service.

This module centralises counters and histograms so that /ai/move and
related endpoints can record lightweight telemetry without each handler
having to manage its own metric instances. The metrics are intentionally
minimal but labeled so they can be filtered by AI type and difficulty in
local/dev Prometheus setups.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import Counter, Gauge, Histogram


AI_MOVE_REQUESTS: Final[Counter] = Counter(
    "ai_move_requests_total",
    (
        "Total number of /ai/move requests, labeled by ai_type, "
        "difficulty and outcome."
    ),
    labelnames=("ai_type", "difficulty", "outcome"),
)

AI_MOVE_LATENCY: Final[Histogram] = Histogram(
    "ai_move_latency_seconds",
    (
        "Latency of /ai/move requests in seconds, labeled by ai_type "
        "and difficulty."
    ),
    labelnames=("ai_type", "difficulty"),
    # Buckets chosen to cover sub-100ms up to several seconds while keeping
    # the set small enough for local/dev use. These can be refined later if
    # we deploy a dedicated metrics stack.
    buckets=(
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
    ),
)


PYTHON_INVARIANT_VIOLATIONS: Final[Counter] = Counter(
    "ringrift_python_invariant_violations_total",
    (
        "Total number of Python self-play invariant violations observed in "
        "run_self_play_soak, labeled by high-level invariant_id and "
        "low-level violation type."
    ),
    labelnames=("invariant_id", "type"),
)

AI_INSTANCE_CACHE_LOOKUPS: Final[Counter] = Counter(
    "ai_instance_cache_lookups_total",
    "Total AI instance cache lookups, labeled by ai_type and outcome.",
    labelnames=("ai_type", "outcome"),
)

AI_INSTANCE_CACHE_SIZE: Final[Gauge] = Gauge(
    "ai_instance_cache_size",
    "Current number of cached AI instances in this process.",
)

# Game outcome metrics
GAME_OUTCOMES: Final[Counter] = Counter(
    "ringrift_game_outcomes_total",
    "Total game outcomes from selfplay, labeled by board_type, num_players, and outcome.",
    labelnames=("board_type", "num_players", "outcome"),
)

GAMES_COMPLETED: Final[Counter] = Counter(
    "ringrift_games_completed_total",
    "Total completed selfplay games, labeled by board_type and num_players.",
    labelnames=("board_type", "num_players"),
)

GAMES_MOVES_TOTAL: Final[Counter] = Counter(
    "ringrift_games_moves_total",
    "Total moves across all selfplay games, labeled by board_type and num_players.",
    labelnames=("board_type", "num_players"),
)

GAME_DURATION_SECONDS: Final[Histogram] = Histogram(
    "ringrift_game_duration_seconds",
    "Duration of selfplay games in seconds.",
    labelnames=("board_type", "num_players"),
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

WIN_RATE_BY_PLAYER: Final[Gauge] = Gauge(
    "ringrift_win_rate_by_player",
    "Win rate per player position (0-indexed), updated periodically.",
    labelnames=("board_type", "num_players", "player_position"),
)

DRAW_RATE: Final[Gauge] = Gauge(
    "ringrift_draw_rate",
    "Draw rate for selfplay games, updated periodically.",
    labelnames=("board_type", "num_players"),
)

# Pre-initialize one labeled time series for the core /ai/move metrics so the
# /metrics endpoint exposes histogram buckets even before the first request.
# This keeps smoke tests and local Prometheus setups stable.
#
# Note: we intentionally do NOT call .observe() / .inc() here; creating the
# labeled child is sufficient to emit zero-valued samples.
AI_MOVE_REQUESTS.labels("init", "0", "init")  # type: ignore[arg-type]
AI_MOVE_LATENCY.labels("init", "0")  # type: ignore[arg-type]


def observe_ai_move_start(ai_type: str, difficulty: int) -> tuple[str, str]:
    """Prepare metric label values for a new /ai/move request.

    This helper just normalises difficulty into a string label; callers are
    expected to pass the returned labels into the Counter/Histogram as
    needed. It exists mainly to keep the label-shape logic in one place.
    """

    return ai_type, str(difficulty)


def record_game_outcome(
    board_type: str,
    num_players: int,
    winner: int | None,  # None for draw, player index (0-based) for win
    move_count: int,
    duration_seconds: float,
) -> None:
    """Record metrics for a completed selfplay game.

    Args:
        board_type: Board type (e.g., 'square8', 'hexagonal')
        num_players: Number of players (2, 3, or 4)
        winner: Player index (0-based) who won, or None for draw
        move_count: Total moves in the game
        duration_seconds: Game duration in seconds
    """
    np_str = str(num_players)

    # Record game completion
    GAMES_COMPLETED.labels(board_type, np_str).inc()

    # Record outcome
    if winner is None:
        GAME_OUTCOMES.labels(board_type, np_str, "draw").inc()
    else:
        GAME_OUTCOMES.labels(board_type, np_str, f"player_{winner}_win").inc()

    # Record moves
    GAMES_MOVES_TOTAL.labels(board_type, np_str).inc(move_count)

    # Record duration
    GAME_DURATION_SECONDS.labels(board_type, np_str).observe(duration_seconds)


__all__ = [
    "AI_MOVE_REQUESTS",
    "AI_MOVE_LATENCY",
    "AI_INSTANCE_CACHE_LOOKUPS",
    "AI_INSTANCE_CACHE_SIZE",
    "PYTHON_INVARIANT_VIOLATIONS",
    "GAME_OUTCOMES",
    "GAMES_COMPLETED",
    "GAMES_MOVES_TOTAL",
    "GAME_DURATION_SECONDS",
    "WIN_RATE_BY_PLAYER",
    "DRAW_RATE",
    "observe_ai_move_start",
    "record_game_outcome",
]
