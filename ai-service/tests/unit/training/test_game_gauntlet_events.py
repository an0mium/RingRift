from types import SimpleNamespace

import app.coordination.event_router as event_router
from app.models import BoardType
from app.training import game_gauntlet


class _StubBus:
    def __init__(self) -> None:
        self.events = []

    def publish_sync(self, event):
        self.events.append(event)
        return event


def test_evaluation_progress_payload_includes_config_key_and_board_type(monkeypatch):
    stub_bus = _StubBus()
    monkeypatch.setattr(event_router, "get_event_bus", lambda: stub_bus)

    monkeypatch.setattr(game_gauntlet, "create_neural_ai", lambda *args, **kwargs: object())
    monkeypatch.setattr(game_gauntlet, "create_baseline_ai", lambda *args, **kwargs: object())

    def fake_play_single_game(**kwargs):
        return SimpleNamespace(
            candidate_won=True,
            winner=1,
            victory_reason="test",
            move_count=1,
        )

    monkeypatch.setattr(game_gauntlet, "play_single_game", fake_play_single_game)

    game_gauntlet._evaluate_single_opponent(
        baseline=game_gauntlet.BaselineOpponent.RANDOM,
        model_path="dummy",
        board_type=BoardType.SQUARE19,
        games_per_opponent=1,
        num_players=2,
        verbose=False,
        model_getter=None,
        model_type="cnn",
        early_stopping=False,
        early_stopping_confidence=0.95,
        early_stopping_min_games=1,
    )

    assert stub_bus.events, "Expected evaluation progress events to be published"
    payload = stub_bus.events[0].payload
    assert payload["config_key"] == "square19_2p"
    assert payload["board_type"] == "square19"
