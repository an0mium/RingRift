from scripts.run_crossboard_tier_gating import combine_gate_reports


def _full_gate(final_decision: str = "promote"):
    return {
        "tier": "D6",
        "board": "square8",
        "num_players": 2,
        "candidate_id": "cand",
        "evaluation": {"overall_pass": True},
        "perf": {"overall_pass": True},
        "final_decision": final_decision,
    }


def test_combined_promotes_when_parity_skipped_and_tier_promotes():
    combined = combine_gate_reports(None, _full_gate("promote"))
    assert combined["parity_gate"]["run"] is False
    assert combined["final_decision"] == "promote"


def test_combined_rejects_when_parity_fails():
    parity_report = {"gate": {"overall_pass": False}}
    combined = combine_gate_reports(parity_report, _full_gate("promote"))
    assert combined["parity_gate"]["run"] is True
    assert combined["parity_gate"]["overall_pass"] is False
    assert combined["final_decision"] == "reject"


def test_combined_rejects_when_tier_rejects_even_if_parity_passes():
    parity_report = {"gate": {"overall_pass": True}}
    combined = combine_gate_reports(parity_report, _full_gate("reject"))
    assert combined["parity_gate"]["overall_pass"] is True
    assert combined["final_decision"] == "reject"

