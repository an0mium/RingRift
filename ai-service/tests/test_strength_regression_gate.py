from __future__ import annotations

from scripts import run_strength_regression_gate as gate  # type: ignore[import]


def test_compute_gate_without_significance_uses_raw_win_rate() -> None:
    threshold = gate.GateThreshold(
        min_win_rate=0.55,
        require_significance=False,
        confidence=0.95,
    )
    summary = gate._compute_gate(wins=6, losses=4, draws=0, threshold=threshold)
    assert summary["win_rate"] == 0.6
    assert summary["passes"] is True


def test_compute_gate_with_significance_requires_ci_lower_bound() -> None:
    threshold = gate.GateThreshold(
        min_win_rate=0.55,
        require_significance=True,
        confidence=0.95,
    )
    # 6/10 wins meets raw win-rate threshold but should fail the significance
    # gate because the Wilson lower bound remains below 0.55 for n=10.
    summary = gate._compute_gate(wins=6, losses=4, draws=0, threshold=threshold)
    assert summary["win_rate"] == 0.6
    assert summary["win_rate_ci_low"] is not None
    assert summary["win_rate_ci_low"] < 0.55
    assert summary["passes"] is False


def test_compute_gate_all_draws_fails_significance_gate() -> None:
    threshold = gate.GateThreshold(
        min_win_rate=0.52,
        require_significance=True,
        confidence=0.95,
    )
    summary = gate._compute_gate(wins=0, losses=0, draws=10, threshold=threshold)
    assert summary["decisive_games"] == 0
    assert summary["win_rate"] == 0.5
    assert summary["passes"] is False

