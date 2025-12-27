import time
from pathlib import Path

from app.coordination import cluster_data_sync


def test_get_sync_targets_filters_and_sorts(monkeypatch) -> None:
    now = time.time()

    def fake_status():
        return {
            "peers": {
                "node-a": {
                    "disk_free_gb": 200,
                    "last_heartbeat": now,
                    "host": "10.0.0.1",
                },
                "node-b": {
                    "disk_free_gb": 40,  # below threshold
                    "last_heartbeat": now,
                    "host": "10.0.0.2",
                },
                "node-c": {
                    "disk_free_gb": 150,
                    "last_heartbeat": now - 400,  # stale
                    "host": "10.0.0.3",
                },
                "node-d": {
                    "disk_free_gb": 120,
                    "last_heartbeat": now,
                    "host": "10.0.0.4",
                },
                "node-skip": {
                    "disk_free_gb": 180,
                    "last_heartbeat": now,
                    "host": "10.0.0.5",
                },
            }
        }

    class FakePolicy:
        min_disk_free_gb = 50

        def should_exclude(self, node_id: str) -> bool:
            return node_id == "node-skip"

    monkeypatch.setattr(cluster_data_sync, "get_p2p_status", fake_status)
    monkeypatch.setattr(cluster_data_sync, "get_exclusion_policy", lambda: FakePolicy())

    targets = cluster_data_sync.get_sync_targets()
    node_ids = [t.node_id for t in targets]

    assert node_ids == ["node-a", "node-d"]


def test_discover_local_databases_filters_and_prioritizes(monkeypatch, tmp_path) -> None:
    fake_file = tmp_path / "app" / "coordination" / "cluster_data_sync.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("# placeholder")

    data_dir = tmp_path / "data" / "games"
    data_dir.mkdir(parents=True)

    high_priority = data_dir / "canonical_square8_2p.db"
    high_priority.write_bytes(b"0" * 2048)
    other_priority = data_dir / "selfplay_hex8_2p.db"
    other_priority.write_bytes(b"0" * 2048)
    non_priority = data_dir / "canonical_square19_2p.db"
    non_priority.write_bytes(b"0" * 2048)
    too_small = data_dir / "canonical_small.db"
    too_small.write_bytes(b"0" * 10)

    monkeypatch.setattr(cluster_data_sync, "__file__", str(fake_file))

    databases = cluster_data_sync.discover_local_databases()
    names = [Path(db).name for db in databases]

    assert "canonical_small.db" not in names
    assert names[-1] == "canonical_square19_2p.db"
