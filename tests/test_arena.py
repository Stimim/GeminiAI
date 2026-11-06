import json
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from msflags import arena
from msflags.game import Config, UNKNOWN


CATALOG = [{"id": agent_id, "name": agent_id.title(), "available": True}
           for agent_id in ("solver", "strong", "weak", "unplayed")]


def outcome(a, b, winner):
    return {"seats": [a, b], "winner": winner}


def by_id(result):
    return {row["id"]: row for row in result["ratings"]}


def test_ratings_anchor_and_joint_order():
    # Strong has never played solver: its strength must propagate through weak.
    records = ([outcome("weak", "solver", "weak")] * 30
               + [outcome("weak", "solver", "solver")] * 10
               + [outcome("strong", "weak", "strong")] * 30
               + [outcome("strong", "weak", "weak")] * 10)
    result = arena.fit_ratings(records, CATALOG)
    rows = by_id(result)
    assert result["anchor"] == "solver"
    assert result["total_games"] == 80
    assert "rules" not in result
    assert rows["solver"]["rating"] == 1000.0
    assert rows["solver"]["uncertainty"] == 0.0
    assert rows["strong"]["rating"] > rows["weak"]["rating"] > 1000.0
    assert rows["weak"]["games"] == 80
    assert rows["strong"]["games"] == 40
    assert rows["unplayed"]["rating"] == 1000.0
    assert rows["unplayed"]["uncertainty"] == pytest.approx(1.96 * 800)
    assert 0 < rows["weak"]["uncertainty"] < rows["unplayed"]["uncertainty"]


def test_ratings_finite_for_separation_and_disconnected_components():
    records = ([outcome("strong", "solver", "strong")] * 200
               + [outcome("retired", "weak", "retired")] * 40)
    rows = by_id(arena.fit_ratings(records, CATALOG))
    assert "retired" in rows
    for row in rows.values():
        assert math.isfinite(row["rating"])
        assert math.isfinite(row["uncertainty"])
        assert row["uncertainty"] >= 0
    assert rows["strong"]["rating"] > 1000
    assert rows["retired"]["rating"] > 1000 > rows["weak"]["rating"]
    assert rows["retired"]["rating"] + rows["weak"]["rating"] == pytest.approx(2000)
    assert rows["retired"]["uncertainty"] == pytest.approx(rows["weak"]["uncertainty"])


def test_ratings_invariant_to_seat_and_record_order():
    records = ([outcome("strong", "solver", "strong")] * 8
               + [outcome("strong", "solver", "solver")] * 3
               + [outcome("weak", "strong", "weak")] * 2)
    swapped = [{"seats": list(reversed(record["seats"])), "winner": record["winner"]}
               for record in reversed(records)]
    original = by_id(arena.fit_ratings(records, CATALOG))
    changed = by_id(arena.fit_ratings(swapped, CATALOG))
    assert original == changed


def test_ratings_outcome_reversal_reflects_around_anchor():
    records = ([outcome("strong", "solver", "strong")] * 13
               + [outcome("strong", "solver", "solver")] * 4
               + [outcome("weak", "strong", "weak")] * 6)
    reversed_outcomes = [outcome(*record["seats"],
                                 next(agent_id for agent_id in record["seats"]
                                      if agent_id != record["winner"]))
                         for record in records]
    original = by_id(arena.fit_ratings(records, CATALOG))
    changed = by_id(arena.fit_ratings(reversed_outcomes, CATALOG))
    for agent_id, row in original.items():
        assert row["rating"] + changed[agent_id]["rating"] == pytest.approx(2000)
        assert row["uncertainty"] == pytest.approx(changed[agent_id]["uncertainty"])


def test_balanced_ratings_and_prior_only_catalog():
    records = [outcome("strong", "solver", "strong"), outcome("solver", "strong", "solver")]
    rows = by_id(arena.fit_ratings(records, CATALOG))
    assert rows["strong"]["rating"] == 1000
    assert rows["strong"]["games"] == rows["solver"]["games"] == 2
    empty = by_id(arena.fit_ratings([], CATALOG))
    assert all(row["rating"] == 1000 for row in empty.values())
    assert all(row["games"] == 0 for row in empty.values())
    anchor_only = arena.fit_ratings([], [])
    assert anchor_only["ratings"] == [empty["solver"] | {"name": "Solver"}]


def test_ratings_include_games_appended_after_manager_creation(tmp_path):
    path = tmp_path / "games.jsonl"
    manager = arena.ArenaManager(path)
    assert manager.ratings()["total_games"] == 0
    path.write_text(json.dumps(outcome("random", "solver", "solver")) + "\n")
    result = manager.ratings()
    assert result["total_games"] == 1
    assert by_id(result)["random"]["games"] == 1
    assert by_id(result)["random"]["rating"] < 1000


def test_ledger_read_preserves_records_without_migration(tmp_path):
    path = tmp_path / "games.jsonl"
    original = json.dumps(outcome("random", "solver", "random")) + "\n"
    path.write_text(original)
    manager = arena.ArenaManager(path)
    assert manager.ratings()["total_games"] == 1
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]
    assert by_id(manager.ratings())["random"]["rating"] > 1000


@pytest.fixture
def fake_registry(monkeypatch):
    def create(agent_id, sims=100, rng=None):
        return SimpleNamespace(act=lambda public: int(rng.choice(np.flatnonzero(public.board == UNKNOWN))))

    fake = SimpleNamespace(
        catalog=lambda: [{"id": agent_id, "name": agent_id, "available": agent_id != "offline"}
                         for agent_id in ("random", "solver", "offline")],
        available=lambda agent_id: agent_id in ("random", "solver"),
        create=create,
    )
    def resolve(agent_id):
        if not fake.available(agent_id):
            raise ValueError("unknown or unavailable agent")
        return agent_id
    fake.resolve = resolve
    fake.entries = fake.catalog
    monkeypatch.setattr(arena, "registry", fake)
    return fake


def finish(manager):
    manager._thread.join(timeout=5)
    assert not manager._thread.is_alive(), "arena worker did not finish"


def test_match_persists_alternating_seats_and_reloads_ratings(tmp_path, fake_registry):
    path = tmp_path / "arena" / "games.jsonl"
    manager = arena.ArenaManager(path, cfg=Config(2, 3, 3))
    match = manager.start("random", "solver", 4, seed=7)
    finish(manager)
    result = manager.status(match["id"])
    assert result["status"] == "completed"
    assert result["completed"] == result["games"] == 4
    assert result["error"] is None
    assert "rules" not in result
    assert "rules" not in result["current"]
    assert result["current"]["bombs"] == [1, 1]
    assert result["current"]["winner"] in (0, 1)
    assert result["current"]["moves"] > 0
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == result["results"]
    assert [record["seats"] for record in records] == [["random", "solver"], ["solver", "random"]] * 2
    assert [record["seed"] for record in records] == [7, 8, 9, 10]
    for record in records:
        assert "rules" not in record
        assert record["winner"] in record["seats"]
        assert record["scores"][record["seats"].index(record["winner"])] == 2
        assert record["seconds"] >= 0
        assert record["time"].endswith("+00:00")
    assert arena.ArenaManager(path).ratings() == manager.ratings()
    another = manager.start("solver", "random", 1)
    finish(manager)
    assert manager.status(another["id"])["status"] == "completed"
    assert len(path.read_text().splitlines()) == 5


def test_bomb_action_is_published_and_persisted(tmp_path, fake_registry):
    fake_registry.create = lambda *args, **kwargs: SimpleNamespace(act=lambda public: 6)
    manager = arena.ArenaManager(tmp_path / "games.jsonl", cfg=Config(2, 3, 3))
    match = manager.start("random", "solver", 1)
    finish(manager)
    result = manager.status(match["id"])
    assert result["status"] == "completed"
    current = result["current"]
    assert current["bombs"] == [0, 1]
    assert current["last_move"]["action"] == 6
    assert current["last_move"]["kind"] == "bomb"
    assert current["scores"] == [3, 0]
    assert result["results"][0]["winner"] == "random"
    assert "rules" not in result["results"][0]


@pytest.mark.parametrize("games", [0, -1, 1001, 1.5, True, "2"])
def test_invalid_game_counts(tmp_path, fake_registry, games):
    manager = arena.ArenaManager(tmp_path / "games.jsonl")
    with pytest.raises(ValueError):
        manager.start("random", "solver", games)
    assert not manager.path.exists()


@pytest.mark.parametrize("a,b", [("random", "random"), ("missing", "solver"), ("random", "offline")])
def test_invalid_agents_and_unknown_match(tmp_path, fake_registry, a, b):
    manager = arena.ArenaManager(tmp_path / "games.jsonl")
    with pytest.raises(ValueError):
        manager.start(a, b, 1)
    with pytest.raises(KeyError):
        manager.status("missing")


def test_busy_and_live_snapshot_are_thread_safe(tmp_path, fake_registry):
    entered = threading.Event()
    release = threading.Event()

    def act(public):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release agent")
        return int(np.flatnonzero(public.board == UNKNOWN)[0])

    fake_registry.create = lambda *args, **kwargs: SimpleNamespace(act=act)
    manager = arena.ArenaManager(tmp_path / "games.jsonl", cfg=Config(2, 3, 3))
    match = manager.start("random", "solver", 2)
    try:
        assert entered.wait(timeout=5)
        snapshot = manager.status(match["id"])
        assert snapshot["status"] == "running"
        assert snapshot["current"]["moves"] == 0
        assert snapshot["current"]["board"] == [-1] * 6
        with pytest.raises(ValueError, match="busy"):
            manager.start("solver", "random", 1)
        snapshot["current"]["board"][0] = 99
        snapshot["results"].append({})
        assert manager.status(match["id"])["current"]["board"][0] == -1
        assert manager.status(match["id"])["results"] == []
    finally:
        release.set()
        finish(manager)
    assert manager.status(match["id"])["status"] == "completed"


def test_agent_failure_keeps_snapshot_and_releases_busy_state(tmp_path, fake_registry):
    fake_registry.create = lambda *args, **kwargs: SimpleNamespace(act=lambda public: -1)
    manager = arena.ArenaManager(tmp_path / "games.jsonl", cfg=Config(2, 3, 3))
    first = manager.start("random", "solver", 1)
    finish(manager)
    result = manager.status(first["id"])
    assert result["status"] == "failed"
    assert "illegal action" in result["error"]
    assert result["completed"] == 0
    assert result["results"] == []
    assert result["current"]["moves"] == 0
    assert not manager.path.exists()
    second = manager.start("random", "solver", 1)
    finish(manager)
    assert second["id"] != first["id"]


def test_move_snapshots_publish_every_move_under_lock(tmp_path, fake_registry):
    snapshots = []

    class ObservedArena(arena.ArenaManager):
        def _snapshot(self, match, game, seats, moves, last_move):
            # The base helper relies on its caller holding this lock.
            assert self._lock._is_owned()
            super()._snapshot(match, game, seats, moves, last_move)
            snapshots.append(self.status(match["id"])["current"])

    manager = ObservedArena(tmp_path / "games.jsonl", cfg=Config(2, 3, 3))
    match = manager.start("random", "solver", 1)
    finish(manager)
    result = manager.status(match["id"])
    assert result["status"] == "completed"
    assert [snapshot["moves"] for snapshot in snapshots] == list(range(len(snapshots)))
    assert snapshots[0]["last_move"] is None
    for snapshot in snapshots[1:]:
        last = snapshot["last_move"]
        assert snapshot["board"][last["cell"]] != UNKNOWN
        assert last["cell"] in last["opened"]
    assert snapshots[-1] == result["current"]


def test_category_selection_persists_concrete_generations(tmp_path, fake_registry):
    fake_registry.resolve = lambda agent_id: {"category": "random"}.get(agent_id, agent_id)
    create = fake_registry.create
    requested = []
    def fixed_settings(agent_id, **kwargs):
        requested.append(kwargs)
        return create(agent_id, **kwargs)
    fake_registry.create = fixed_settings
    manager = arena.ArenaManager(tmp_path / "games.jsonl", cfg=Config(2, 3, 3))
    match = manager.start("category", "solver", 1)
    finish(manager)
    assert match["a"] == "random"
    assert manager.status(match["id"])["results"][0]["seats"] == ["random", "solver"]
    assert requested and all("sims" not in kwargs for kwargs in requested)
    with pytest.raises(ValueError, match="distinct"):
        manager.start("category", "random", 1)


@pytest.mark.parametrize("command", ["ratings", "roundrobin"])
def test_cli_uses_registry_configuration_and_persisted_manager(
        tmp_path, fake_registry, monkeypatch, capsys, command):
    configured = []
    fake_registry.configure = lambda **kwargs: configured.append(kwargs)
    manager = arena.ArenaManager(tmp_path / "games.jsonl", cfg=Config(2, 3, 3))
    monkeypatch.setattr(arena, "manager", manager)
    arguments = ["msflags.arena", command]
    if command == "roundrobin":
        arguments += ["--games", "2", "--agents", "random", "solver",
                      "--device", "cpu", "--seed", "17"]
    monkeypatch.setattr("sys.argv", arguments)
    arena.main()
    assert configured == [{"device": "cpu"}]
    output = capsys.readouterr().out
    if command == "roundrobin":
        first_line, ratings_json = output.split("\n", 1)
        match = json.loads(first_line)
        assert match["completed"] == 2
        assert [record["seed"] for record in match["results"]] == [17, 18]
    else:
        ratings_json = output
    assert json.loads(ratings_json)["anchor"] == "solver"
    assert json.loads(ratings_json)["total_games"] == (2 if command == "roundrobin" else 0)
