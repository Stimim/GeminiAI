from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from msflags import server


def test_game_category_is_resolved_once_and_generation_is_preserved(monkeypatch):
    concrete = "r32x3-bs8k-sp12.mcts100-c1.5"
    monkeypatch.setattr(server.registry, "resolve", lambda agent_id: concrete if agent_id == "r32x3.mcts100-c1.5" else agent_id)
    monkeypatch.setattr(server.registry, "available", lambda agent_id: agent_id == concrete)
    calls = []
    monkeypatch.setattr(server, "make_agent", lambda name, sims, rng: calls.append((name, sims)) or SimpleNamespace())
    try:
        for selected in ("r32x3.mcts100-c1.5", concrete):
            result = server.new_game(server.NewGame(players=["human", selected], sims=7, seed=1))
            assert result["players"] == ["human", concrete]
            assert calls[-1] == (concrete, 7)
            server.sessions.pop(result["id"])
    finally:
        server.sessions.clear()


def test_game_unavailable_generation_is_rejected(monkeypatch):
    monkeypatch.setattr(server.registry, "resolve", lambda agent_id: agent_id)
    monkeypatch.setattr(server.registry, "available", lambda agent_id: False)
    with pytest.raises(HTTPException) as error:
        server.new_game(server.NewGame(players=["human", "unavailable"]))
    assert error.value.status_code == 400


def test_catalog_ratings_are_per_generation_not_category():
    entry = {"id": "category", "generations": [{"id": "old"}, {"id": "new"}]}
    result = server.with_ratings(entry, {"old": {"games": 20, "rating": 1040, "uncertainty": 80}})
    assert result["games"] == 0
    assert "elo" not in result
    assert result["generations"][0] == {"id": "old", "games": 20, "elo": 1040, "uncertainty": 80}
    assert result["generations"][1] == {"id": "new", "games": 0}
    assert "games" not in entry["generations"][0]


def test_hint_budget_overrides_game_budget(monkeypatch):
    session = SimpleNamespace(game=SimpleNamespace(public=lambda: "public"), sims=100, lock=server.threading.RLock())
    monkeypatch.setattr(server, "get_session", lambda game_id: session)
    monkeypatch.setattr(server, "agent_names", lambda: ["mcts-category"])
    calls = []
    def make_agent(name, sims, rng):
        calls.append(sims)
        return SimpleNamespace(info=lambda public: {"budget": sims})
    monkeypatch.setattr(server, "make_agent", make_agent)
    assert server.hint("game", "mcts-category", sims=7) == {"budget": 7}
    assert server.hint("game", "mcts-category") == {"budget": 100}
    assert calls == [7, 100]
