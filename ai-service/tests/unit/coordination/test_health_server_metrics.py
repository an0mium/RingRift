from __future__ import annotations

from app.coordination.daemon_manager import DaemonManager


def test_render_metrics_contains_core_fields() -> None:
    DaemonManager.reset_instance()
    manager = DaemonManager.get_instance()

    metrics = manager.render_metrics()

    assert "daemon_count" in metrics
    assert "daemon_health_score" in metrics
    assert "event_router_events_routed_total" in metrics
