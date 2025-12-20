"""Smoke tests for the Prometheus /metrics endpoint."""

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


def test_metrics_endpoint_exposes_ai_move_metrics() -> None:
    """The /metrics endpoint should expose AI move counters and histograms."""
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text

    # Counter and histogram names as registered in app.metrics.
    assert "ai_move_requests_total" in body
    assert "ai_move_latency_seconds_bucket" in body
