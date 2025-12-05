"""Canonical difficulty ladder configuration for RingRift.

This module defines a small, serialisable mapping from logical difficulty
levels (1–10) to concrete AI configurations for specific board types and
player counts. It is the single source of truth for the production
difficulty ladder used by:

* The FastAPI service (/ai/move) when constructing AIs for live games.
* Tier evaluation and gating scripts when calibrating candidate models.

The initial slice focuses on square8, 2-player D2/D4/D6/D8 tiers.
Additional boards or player counts can be added incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.models import AIType, BoardType


LadderKey = Tuple[int, BoardType, int]


@dataclass(frozen=True)
class LadderTierConfig:
    """Canonical assignment for a single difficulty tier.

    The (difficulty, board_type, num_players) triple identifies the tier.
    The remaining fields capture how that tier is realised in production.
    All fields are intentionally JSON-serialisable so that CI and offline
    tooling can persist and diff ladder assignments.
    """

    difficulty: int
    board_type: BoardType
    num_players: int
    ai_type: AIType
    model_id: Optional[str]
    heuristic_profile_id: Optional[str]
    randomness: float
    think_time_ms: int
    notes: str = ""


def _build_default_square8_two_player_configs() -> Dict[
    LadderKey, LadderTierConfig
]:
    """Return the built-in ladder assignments for square8 2-player tiers.

    These defaults mirror the canonical difficulty profiles in ``app.main``
    for D2/D4/D6/D8 while threading through an explicit model / profile id
    so that calibration and promotion tooling can reason about assignments.

    Strength differences between tiers are primarily expressed via
    ``ai_type``, ``randomness`` and ``think_time_ms``. For the current
    heuristic-driven tree-search stack the same 2-player heuristic weights
    (``heuristic_v1_2p``) are shared across tiers.
    """
    return {
        # D2 – easy heuristic baseline on square8, 2-player.
        (2, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.HEURISTIC,
            # Optimised 2-player heuristic weights; also exposed via
            # trained_heuristic_profiles.json and HeuristicAI.
            model_id="heuristic_v1_2p",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.3,
            think_time_ms=200,
            notes=(
                "Easy square8 2p tier backed by v1 2-player heuristic "
                "weights. Intended as the first non-random production "
                "difficulty."
            ),
        ),
        # D4 – mid-tier minimax on square8, 2-player.
        (4, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MINIMAX,
            # Version tag for the search configuration / persona.
            model_id="v1-minimax-4",
            # Minimax continues to use the 2-player heuristic weights
            # for static evaluation.
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.1,
            think_time_ms=2100,
            notes="Mid square8 2p tier using v1 minimax configuration.",
        ),
        # D6 – high minimax on square8, 2-player.
        (6, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=6,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-6",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.02,
            think_time_ms=4800,
            notes="High square8 2p tier with extended minimax search budget.",
        ),
        # D8 – strong MCTS on square8, 2-player.
        (8, BoardType.SQUARE8, 2): LadderTierConfig(
            difficulty=8,
            board_type=BoardType.SQUARE8,
            num_players=2,
            ai_type=AIType.MCTS,
            model_id="v1-mcts-8",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.0,
            think_time_ms=9600,
            notes="Strong square8 2p tier using v1 MCTS configuration.",
        ),
    }


def _build_default_square19_two_player_configs() -> Dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for square19 2-player tiers.

    These entries are currently experimental and primarily intended for
    evaluation/gating and multi-board smoke tests. They reuse the canonical
    square19 2-player heuristic weights where available.
    """
    return {
        # D2 – easy heuristic baseline on square19, 2-player.
        (2, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.HEURISTIC,
            # Optimised square19 2-player heuristic; see heuristic_weights.
            model_id="heuristic_v1_square19_2p",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.3,
            # Slightly larger think-time budget than square8 to account
            # for the larger board while remaining practical.
            think_time_ms=250,
            notes=(
                "Experimental square19 2p D2 tier backed by square19-tuned "
                "heuristic weights. Intended for gating and tooling smoke "
                "tests rather than end-user gameplay calibration."
            ),
        ),
        # D4 – mid-tier minimax on square19, 2-player.
        (4, BoardType.SQUARE19, 2): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.SQUARE19,
            num_players=2,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-square19-4",
            heuristic_profile_id="heuristic_v1_square19_2p",
            randomness=0.1,
            # Higher search budget than D2; still conservative for tests.
            think_time_ms=1500,
            notes=(
                "Experimental square19 2p D4 tier using minimax with "
                "square19-tuned heuristic evaluation. Calibrated mainly "
                "for tier evaluation and ladder experiments."
            ),
        ),
    }


def _build_default_hex_two_player_configs() -> Dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for hexagonal 2-player tiers.

    These entries currently reuse the canonical 2-player heuristic profile and
    should be treated as experimental until dedicated hex weights are wired in.
    """
    return {
        # D2 – easy heuristic on hexagonal, 2-player.
        (2, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.HEURISTIC,
            model_id="heuristic_v1_2p",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.3,
            think_time_ms=250,
            notes=(
                "Experimental hexagonal 2p D2 tier reusing the canonical 2p "
                "heuristic weights. Intended for multi-board tier evaluation "
                "and smoke testing."
            ),
        ),
        # D4 – mid minimax on hexagonal, 2-player.
        (4, BoardType.HEXAGONAL, 2): LadderTierConfig(
            difficulty=4,
            board_type=BoardType.HEXAGONAL,
            num_players=2,
            ai_type=AIType.MINIMAX,
            model_id="v1-minimax-hex-4",
            heuristic_profile_id="heuristic_v1_2p",
            randomness=0.1,
            think_time_ms=1500,
            notes=(
                "Experimental hexagonal 2p D4 tier using minimax with the "
                "canonical 2p heuristic profile. Primarily used for gating "
                "and tooling smoke tests."
            ),
        ),
    }


def _build_default_square8_three_player_configs() -> Dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for square8 3-player tiers.

    These tiers reuse the 3-player-optimised heuristic profile and are
    intentionally conservative. They primarily serve as entrypoints for
    multiplayer tier evaluation rather than full production calibration.
    """
    return {
        (2, BoardType.SQUARE8, 3): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.SQUARE8,
            num_players=3,
            ai_type=AIType.HEURISTIC,
            model_id="heuristic_v1_3p",
            heuristic_profile_id="heuristic_v1_3p",
            randomness=0.3,
            think_time_ms=250,
            notes=(
                "Experimental square8 3p D2 tier backed by the CMA-ES "
                "optimised 3-player heuristic profile. Intended for "
                "multiplayer evaluation smoke tests."
            ),
        ),
    }


def _build_default_square8_four_player_configs() -> Dict[
    LadderKey, LadderTierConfig
]:
    """Return ladder assignments for square8 4-player tiers.

    These tiers reuse the 4-player-optimised heuristic profile and are
    intended primarily for smoke tests and early ladder experiments.
    """
    return {
        (2, BoardType.SQUARE8, 4): LadderTierConfig(
            difficulty=2,
            board_type=BoardType.SQUARE8,
            num_players=4,
            ai_type=AIType.HEURISTIC,
            model_id="heuristic_v1_4p",
            heuristic_profile_id="heuristic_v1_4p",
            randomness=0.3,
            think_time_ms=300,
            notes=(
                "Experimental square8 4p D2 tier backed by the CMA-ES "
                "optimised 4-player heuristic profile. Intended for "
                "4-player evaluation and ladder smoke tests."
            ),
        ),
    }


_LADDER_TIER_CONFIGS: Dict[LadderKey, LadderTierConfig] = {
    **_build_default_square8_two_player_configs(),
    **_build_default_square19_two_player_configs(),
    **_build_default_hex_two_player_configs(),
    **_build_default_square8_three_player_configs(),
    **_build_default_square8_four_player_configs(),
}


def get_ladder_tier_config(
    difficulty: int,
    board_type: BoardType,
    num_players: int,
) -> LadderTierConfig:
    """Return the LadderTierConfig for a given (difficulty, board, players).

    The lookup is exact on all three fields. Callers that want to fall back
    to legacy difficulty logic should catch KeyError and handle it
    explicitly.
    """
    key: LadderKey = (difficulty, board_type, num_players)
    try:
        return _LADDER_TIER_CONFIGS[key]
    except KeyError as exc:  # pragma: no cover - defensive error path
        available = ", ".join(
            sorted(
                f"(difficulty={d}, board_type={bt.value}, num_players={n})"
                for (d, bt, n) in _LADDER_TIER_CONFIGS.keys()
            )
        )
        raise KeyError(
            f"No ladder tier configured for difficulty={difficulty}, "
            f"board_type={board_type!r}, num_players={num_players}. "
            f"Available tiers: {available}"
        ) from exc


def list_ladder_tiers(
    board_type: Optional[BoardType] = None,
    num_players: Optional[int] = None,
) -> List[LadderTierConfig]:
    """Return all configured LadderTierConfig entries, optionally filtered."""
    configs = list(_LADDER_TIER_CONFIGS.values())
    if board_type is not None:
        configs = [c for c in configs if c.board_type == board_type]
    if num_players is not None:
        configs = [c for c in configs if c.num_players == num_players]
    # Deterministic ordering for debugging / tests.
    configs.sort(
        key=lambda c: (c.board_type.value, c.num_players, c.difficulty)
    )
    return configs