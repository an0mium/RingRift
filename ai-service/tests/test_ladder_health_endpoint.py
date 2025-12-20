"""Smoke tests for the /internal/ladder/health endpoint."""

from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient

# Ensure app/ is importable when running tests directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.main import app  # type: ignore

client = TestClient(app)


def test_ladder_health_endpoint_returns_tiers() -> None:
    resp = client.get("/internal/ladder/health")
    assert resp.status_code == 200
    payload = resp.json()

    assert "summary" in payload
    assert "tiers" in payload
    assert isinstance(payload["tiers"], list)
    assert payload["summary"]["tiers"] == len(payload["tiers"])

    tiers = payload["tiers"]
    assert any(
        t.get("board_type") == "square8"
        and t.get("num_players") == 2
        and t.get("difficulty") == 2
        for t in tiers
    )


def test_ladder_health_endpoint_filters_to_single_tier() -> None:
    resp = client.get(
        "/internal/ladder/health",
        params={"board_type": "square8", "num_players": 2, "difficulty": 6},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["summary"]["tiers"] == 1
    tier = payload["tiers"][0]
    assert tier["board_type"] == "square8"
    assert tier["num_players"] == 2
    assert tier["difficulty"] == 6


def test_ladder_health_endpoint_rejects_invalid_board_type() -> None:
    resp = client.get("/internal/ladder/health", params={"board_type": "not-a-board"})
    assert resp.status_code == 400

