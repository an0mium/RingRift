"""Tier-aware evaluation and gating profiles for RingRift AI difficulty tiers.

This module defines small, data-driven configuration objects describing
how candidate AIs should be evaluated for promotion at each difficulty
tier. The intent is to keep all thresholds and opponent mixes in one
place so that tuning does not require code changes.

The initial configuration focuses on square8, 2-player games and
provides representative tiers (D2, D4, D6, D8). Additional tiers can
be added by extending the TIER_EVAL_CONFIGS mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

from app.models import BoardType, AIType

TierRole = Literal["baseline", "previous_tier", "peer", "other"]


@dataclass(frozen=True)
class TierOpponentConfig:
    """Configuration for a single opponent used in tier evaluation."""

    id: str
    description: str
    difficulty: int
    ai_type: Optional[AIType] = None
    role: TierRole = "baseline"
    weight: float = 1.0
    games: Optional[int] = None


@dataclass(frozen=True)
class TierEvaluationConfig:
    """Evaluation profile for a single difficulty tier."""

    tier_name: str
    display_name: str
    board_type: BoardType
    num_players: int
    num_games: int
    candidate_difficulty: int
    time_budget_ms: Optional[int]
    opponents: List[TierOpponentConfig] = field(default_factory=list)
    min_win_rate_vs_baseline: Optional[float] = None
    max_regression_vs_previous_tier: Optional[float] = None
    description: str = ""


TIER_EVAL_CONFIGS: Dict[str, TierEvaluationConfig] = {}


def _build_default_configs() -> Dict[str, TierEvaluationConfig]:
    """Return the built-in tier evaluation profiles.

    The defaults are intentionally modest and primarily intended as a
    starting point for calibration. They can be overridden or replaced
    in higher-level tooling by constructing new TierEvaluationConfig
    instances.
    """
    return {
        "D2": TierEvaluationConfig(
            tier_name="D2",
            display_name="D2 – easy heuristic (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=200,
            candidate_difficulty=2,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
            ],
            min_win_rate_vs_baseline=0.6,
            max_regression_vs_previous_tier=None,
            description=(
                "Sanity-check that a difficulty-2 candidate clearly "
                "outperforms pure random play on square8, 2-player."
            ),
        ),
        "D4": TierEvaluationConfig(
            tier_name="D4",
            display_name="D4 – mid minimax (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=4,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d2",
                    description=(
                        "Previous tier reference using canonical "
                        "difficulty 2 profile"
                    ),
                    difficulty=2,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.7,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Candidate should solidly beat random and show no major "
                "regression relative to the canonical difficulty-2 tier."
            ),
        ),
        "D6": TierEvaluationConfig(
            tier_name="D6",
            display_name="D6 – high minimax (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=6,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="heuristic_d2",
                    description=(
                        "Baseline heuristic profile at canonical "
                        "difficulty 2"
                    ),
                    difficulty=2,
                    ai_type=AIType.HEURISTIC,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d4",
                    description=(
                        "Previous tier reference using canonical "
                        "difficulty 4 profile"
                    ),
                    difficulty=4,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.75,
            max_regression_vs_previous_tier=0.05,
            description=(
                "High-difficulty minimax tier expected to dominate random "
                "and baseline heuristic opponents and at least match the "
                "canonical difficulty-4 profile."
            ),
        ),
        "D8": TierEvaluationConfig(
            tier_name="D8",
            display_name="D8 – strong MCTS (square8, 2p)",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=8,
            time_budget_ms=None,
            opponents=[
                TierOpponentConfig(
                    id="random_d1",
                    description="Random baseline (canonical difficulty 1)",
                    difficulty=1,
                    ai_type=AIType.RANDOM,
                    role="baseline",
                ),
                TierOpponentConfig(
                    id="tier_d6",
                    description=(
                        "Previous tier reference using canonical "
                        "difficulty 6 profile"
                    ),
                    difficulty=6,
                    ai_type=None,
                    role="previous_tier",
                ),
            ],
            min_win_rate_vs_baseline=0.8,
            max_regression_vs_previous_tier=0.05,
            description=(
                "Strong MCTS tier intended to comfortably beat random and "
                "not regress catastrophically relative to the canonical "
                "difficulty-6 profile."
            ),
        ),
    }
 
 
TIER_EVAL_CONFIGS = _build_default_configs()
 

def get_tier_config(tier_name: str) -> TierEvaluationConfig:
    """Return the TierEvaluationConfig for the given tier name.

    The lookup is case-insensitive and expects identifiers such as "D2",
    "d4", "D6", etc.
    """
    key = tier_name.upper()
    try:
        return TIER_EVAL_CONFIGS[key]
    except KeyError as exc:
        available = ", ".join(sorted(TIER_EVAL_CONFIGS.keys()))
        raise KeyError(
            f"Unknown tier '{tier_name}'. Available tiers: {available}"
        ) from exc


@dataclass(frozen=True)
class HeuristicTierSpec:
    """Minimal heuristic-tier specification for eval-pool based evaluation.

    This is intentionally small and square8-focused; higher level tooling can
    extend it over time as needed. All identifiers are strings so that CMA-ES
    and other tuning jobs can feed in weight-profile ids or parameter-set
    hashes without changing the harness.
    """

    id: str
    name: str
    board_type: BoardType
    num_players: int
    eval_pool_id: str
    num_games: int
    candidate_profile_id: str
    baseline_profile_id: str
    description: str = ""


# Square8-focused heuristic tiers for eval-pool based evaluation. These are
# deliberately conservative and can be extended in future waves. The default
# tiers assume the canonical v1 balanced heuristic weights; CMA-ES or GA jobs
# can point candidate_profile_id at alternative entries in
# HEURISTIC_WEIGHT_PROFILES without needing code changes.
HEURISTIC_TIER_SPECS: List[HeuristicTierSpec] = [
    HeuristicTierSpec(
        id="sq8_heuristic_baseline_v1",
        name="Square8 – heuristic_v1 vs baseline_v1 (eval pool v1)",
        board_type=BoardType.SQUARE8,
        num_players=2,
        eval_pool_id="v1",
        num_games=64,
        candidate_profile_id="baseline_v1_balanced",
        baseline_profile_id="baseline_v1_balanced",
        description=(
            "Sanity-check tier evaluating the canonical v1 balanced "
            "heuristic weights on the Square8 v1 eval pool."
        ),
    ),
]