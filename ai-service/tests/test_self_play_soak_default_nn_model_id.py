from __future__ import annotations

from pathlib import Path

from app.models import BoardType
from scripts import run_self_play_soak as soak


def test_resolve_default_nn_model_id_prefers_best_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(soak, "ROOT", str(tmp_path))

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Improvement-loop checkpoint naming convention.
    (models_dir / "square8_2p_best.pth").write_bytes(b"ok")

    resolved = soak._resolve_default_nn_model_id(BoardType.SQUARE8, 2)
    assert resolved == "square8_2p_best"

