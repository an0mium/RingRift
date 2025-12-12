from __future__ import annotations

from app.training.crossboard_strength import (
    inversion_count,
    normalise_tier_name,
    rank_map,
    spearman_rank_correlation,
    summarize_crossboard_tier_strength,
)


def test_normalise_tier_name_accepts_numeric_and_prefixed() -> None:
    assert normalise_tier_name("D6") == "D6"
    assert normalise_tier_name("d6") == "D6"
    assert normalise_tier_name("6") == "D6"


def test_spearman_rank_correlation_matches_expected_extremes() -> None:
    # Strongest -> weakest order.
    order_a = ["D3", "D2", "D1"]
    order_b = ["D3", "D2", "D1"]
    order_c = ["D1", "D2", "D3"]

    r_a = rank_map(order_a)
    r_b = rank_map(order_b)
    r_c = rank_map(order_c)

    assert spearman_rank_correlation(r_a, r_b) == 1.0
    assert spearman_rank_correlation(r_a, r_c) == -1.0


def test_inversion_count_detects_non_monotone_elos() -> None:
    assert inversion_count({"D1": 1500.0, "D2": 1600.0, "D3": 1700.0}) == 0
    assert inversion_count({"D1": 1700.0, "D2": 1600.0, "D3": 1500.0}) == 3


def test_summarize_crossboard_tier_strength_shapes() -> None:
    summary = summarize_crossboard_tier_strength(
        {
            "square8": {"D1": 1000.0, "D2": 1100.0},
            "square19": {"D1": 900.0, "D2": 1200.0},
        }
    )

    assert summary["boards"] == ["square19", "square8"] or summary["boards"] == ["square8", "square19"]
    assert summary["common_tiers"] == ["D1", "D2"]
    assert isinstance(summary["pairwise_rank_correlation"], list)
    assert set(summary["inversion_counts"].keys()) == {"square8", "square19"}
    assert set(summary["tier_rank_std"].keys()) == {"D1", "D2"}

