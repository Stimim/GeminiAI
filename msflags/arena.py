"""Background arena matches and persistent, solver-anchored Bradley–Terry ratings.

Completed games are appended to checkpoints/arena/games.jsonl before publication.
Ratings use a Gaussian 800-Elo prior around 1000 and marginal 95% normal
half-widths; the solver is fixed at 1000 Elo.
Run ``python -m msflags.arena roundrobin --games 20`` or
``python -m msflags.arena ratings``.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from . import registry
from .agents import MCTSAgent, NetAgent
from .mcts import Search
from .game import Config, Game


ELO_SCALE = 400.0 / math.log(10.0)
PRIOR_SD = 800.0


def fit_ratings(records: list[dict], catalog: list[dict]) -> dict:
    """Fit all outcomes, including retired catalog entries."""
    names = {entry["id"]: entry["name"] for entry in catalog}
    names.setdefault("solver", "Solver")
    for record in records:
        for agent_id in record["seats"]:
            names.setdefault(agent_id, agent_id)
    ids = sorted(agent_id for agent_id in names if agent_id != "solver")
    indices = {agent_id: i for i, agent_id in enumerate(ids)}
    counts = dict.fromkeys(names, 0)
    pairs: dict[tuple[str, str], list[float]] = {}
    for record in records:
        a, b = record["seats"]
        counts[a] += 1
        counts[b] += 1
        # Sorting makes seat swaps mathematically identical, not separate fits.
        a, b = sorted((a, b))
        pair = pairs.setdefault((a, b), [0.0, 0.0])
        pair[0] += 1.0
        pair[1] += 0.5 if record["winner"] is None else float(record["winner"] == a)

    design = np.zeros((len(pairs), len(ids)), dtype=np.float64)
    totals = np.empty(len(pairs), dtype=np.float64)
    wins = np.empty(len(pairs), dtype=np.float64)
    for row, ((a, b), (total, won)) in enumerate(sorted(pairs.items())):
        if a != "solver":
            design[row, indices[a]] += 1.0
        if b != "solver":
            design[row, indices[b]] -= 1.0
        totals[row], wins[row] = total, won
    precision = (ELO_SCALE / PRIOR_SD) ** 2
    theta = np.zeros(len(ids), dtype=np.float64)

    def objective(value):
        logits = design @ value
        return float(np.sum(totals * np.logaddexp(0.0, logits) - wins * logits)
                     + 0.5 * precision * np.dot(value, value))

    for _ in range(100):
        logits = design @ theta
        probabilities = np.exp(-np.logaddexp(0.0, -logits))
        gradient = design.T @ (totals * probabilities - wins) + precision * theta
        weights = totals * probabilities * (1.0 - probabilities)
        hessian = design.T @ (weights[:, None] * design) + precision * np.eye(len(ids))
        step = np.linalg.solve(hessian, gradient)
        if not len(step) or np.max(np.abs(step)) < 1e-10:
            break
        scale = 1.0
        current = objective(theta)
        while scale > 1e-10:
            candidate = theta - scale * step
            if objective(candidate) <= current:
                theta = candidate
                break
            scale *= 0.5
        else:
            break
    probabilities = np.exp(-np.logaddexp(0.0, -(design @ theta)))
    weights = totals * probabilities * (1.0 - probabilities)
    hessian = design.T @ (weights[:, None] * design) + precision * np.eye(len(ids))
    covariance = np.linalg.inv(hessian)
    rows = [{"id": "solver", "name": names["solver"], "rating": 1000.0,
             "uncertainty": 0.0, "games": counts["solver"]}]
    for agent_id, i in indices.items():
        rows.append({"id": agent_id, "name": names[agent_id],
                     "rating": float(1000.0 + ELO_SCALE * theta[i]),
                     "uncertainty": float(1.96 * ELO_SCALE * math.sqrt(covariance[i, i])),
                     "games": counts[agent_id]})
    rows.sort(key=lambda row: (-row["rating"], row["id"]))
    return {"ratings": rows, "anchor": "solver", "prior_sd": PRIOR_SD,
            "confidence": 0.95, "total_games": len(records),
            "method": "regularized Bradley–Terry; posterior Hessian intervals"}


class ArenaManager:
    def __init__(self, path: str | Path = "checkpoints/arena/games.jsonl", *, cfg: Config | None = None):
        self.path = Path(path)
        self.cfg = cfg or Config()
        self._lock = threading.RLock()
        self._matches: dict[str, dict] = {}
        self._active: str | None = None
        self._thread: threading.Thread | None = None
        self.batch_size = 1

    def start(self, a: str, b: str, games: int, seed: int = 42) -> dict:
        if isinstance(games, bool) or not isinstance(games, int) or not 1 <= games <= 1000:
            raise ValueError("games must be an integer from 1 to 1000")
        a, b = registry.resolve(a), registry.resolve(b)
        if a == b:
            raise ValueError("choose two distinct agents")
        if not registry.available(a) or not registry.available(b):
            raise ValueError("both agents must be available")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        with self._lock:
            if self._active is not None:
                raise ValueError("arena is busy")
            match_id = uuid.uuid4().hex
            match = {"id": match_id, "a": a, "b": b, "games": games, "seed": seed,
                     "status": "running", "completed": 0, "results": [],
                     "current": None, "error": None}
            self._matches[match_id] = match
            self._active = match_id
            self._thread = threading.Thread(target=self._run, args=(match_id,), daemon=True,
                                            name=f"arena-{match_id}")
            try:
                self._thread.start()
            except Exception as exc:
                match["status"], match["error"] = "failed", str(exc)
                self._active = None
                raise
            return copy.deepcopy(match)

    def status(self, match_id: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._matches[match_id])

    def _records(self) -> list[dict]:
        """Read the current ledger without modifying its historical records."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        return records

    def ratings(self) -> dict:
        with self._lock:
            # Read afresh so offline CLI results appear without a server restart.
            records = self._records()
        entries = registry.entries()
        result = fit_ratings(records, entries)
        metadata = {entry["id"]: entry for entry in entries}
        for row in result["ratings"]:
            entry = metadata.get(row["id"], {})
            row.update({key: entry[key] for key in ("category_id", "generation") if key in entry})
        return result

    def _snapshot(self, match: dict, game: Game, seats: list[str], moves: int,
                  last_move: dict | None) -> None:
        # Called only with the manager lock; never expose mutable engine arrays.
        match["current"] = {
            "h": game.cfg.h, "w": game.cfg.w, "mines": game.cfg.mines,
            "bombs": list(game.bombs),
            "board": game.board.astype(int).tolist(), "scores": list(game.scores),
            "to_move": game.to_move, "winner": game.winner, "players": list(seats),
            "moves": moves, "last_move": last_move,
        }

    def _actions(self, active: list[dict]) -> dict[int, int]:
        """Batch neural leaves across offline games; basic agents keep independent RNGs."""
        from .train import run_searches
        actions = {}
        groups = {}
        for item in active:
            game = item["game"]
            agent = item["agents"][game.to_move]
            if isinstance(agent, (NetAgent, MCTSAgent)):
                groups.setdefault((type(agent), id(agent.ev)), []).append((item, agent))
            else:
                actions[item["index"]] = int(agent.act(game.public()))
        for (kind, _), group in groups.items():
            evaluator = group[0][1].ev
            if kind is NetAgent:
                policies, _, _ = evaluator([(item["game"].board, item["game"].scores, item["game"].to_move, item["game"].bombs) for item, _ in group], group[0][0]["game"].cfg)
                for (item, _), policy in zip(group, policies):
                    actions[item["index"]] = int(np.argmax(policy))
            else:
                searches = [Search(item["game"].public(), agent.sims, rng=agent.rng, **agent.search_kw) for item, agent in group]
                run_searches(searches, evaluator, group[0][0]["game"].cfg)
                for (item, _), search in zip(group, searches):
                    actions[item["index"]] = search.best_action()
        return actions

    def _run(self, match_id: str) -> None:
        match = self._matches[match_id]
        cfg = self.cfg
        try:
            for offset in range(0, match["games"], self.batch_size):
                active = []
                for index in range(offset, min(match["games"], offset + self.batch_size)):
                    seats = [match["a"], match["b"]] if index % 2 == 0 else [match["b"], match["a"]]
                    seed = match["seed"] + index
                    streams = np.random.SeedSequence(seed).spawn(3)
                    game = Game.random(cfg, np.random.default_rng(streams[0]))
                    agents = [registry.create(agent_id, rng=np.random.default_rng(streams[i + 1]))
                              for i, agent_id in enumerate(seats)]
                    active.append(dict(index=index, seed=seed, seats=seats, game=game, agents=agents, moves=0, started=time.perf_counter()))
                    with self._lock:
                        self._snapshot(match, game, seats, 0, None)
                while active:
                    actions = self._actions(active)
                    remaining = []
                    for item in active:
                        game, seats = item["game"], item["seats"]
                        player = game.to_move
                        action = actions[item["index"]]
                        legal = game.legal_mask()
                        if action < 0 or action >= len(legal) or not legal[action]:
                            raise ValueError(f"agent {seats[player]} selected illegal action {action}")
                        with self._lock:
                            result = game.step(action)
                            item["moves"] += 1
                            self._snapshot(match, game, seats, item["moves"],
                                           {"cell": action % cfg.n, "action": action, "player": player,
                                            "kind": result.kind, "hit": result.hit,
                                            "claimed": result.claimed, "opened": result.opened})
                        if not game.done:
                            remaining.append(item)
                            continue
                        record = {"match": match_id, "game": item["index"] + 1, "seats": seats,
                                  "winner": None if game.winner is None else seats[game.winner], "scores": list(game.scores),
                                  "seed": item["seed"], "time": datetime.now(timezone.utc).isoformat(),
                                  "seconds": time.perf_counter() - item["started"]}
                        with self._lock:
                            self.path.parent.mkdir(parents=True, exist_ok=True)
                            with self.path.open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                                stream.flush()
                                os.fsync(stream.fileno())
                            match["results"].append(record)
                            match["completed"] += 1
                    active = remaining
            with self._lock:
                match["results"].sort(key=lambda record: record["game"])
                match["status"] = "completed"
        except Exception as exc:
            with self._lock:
                match["status"], match["error"] = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._active = None


manager = ArenaManager()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    roundrobin = commands.add_parser("roundrobin")
    roundrobin.add_argument("--games", type=int, default=20, help="games per pair (1..1000)")
    roundrobin.add_argument("--agents", nargs="+")
    roundrobin.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    roundrobin.add_argument("--seed", type=int, default=42)
    commands.add_parser("ratings")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    registry.configure(device=getattr(args, "device", "cpu"))
    if args.command == "roundrobin":
        agents = args.agents or [entry["default_generation"] for entry in registry.catalog() if entry["available"]]
        if len(agents) < 2 or len(set(agents)) != len(agents):
            parser.error("choose at least two distinct agents")
        if not 1 <= args.games <= 1000 or args.seed < 0:
            parser.error("games must be 1..1000 and seed must be nonnegative")
        try:
            agents = [registry.resolve(agent_id) for agent_id in agents]
            if len(set(agents)) != len(agents) or any(not registry.available(agent_id) for agent_id in agents):
                raise ValueError("all agents must be distinct and available")
        except ValueError as exc:
            parser.error(str(exc))
        manager.batch_size = 32
        for index, (a, b) in enumerate(itertools.combinations(agents, 2)):
            match = manager.start(a, b, args.games, seed=args.seed + index * args.games)
            manager._thread.join()
            result = manager.status(match["id"])
            print(json.dumps(result), flush=True)
            if result["status"] == "failed":
                parser.exit(1, result["error"] + "\n")
    print(json.dumps(manager.ratings(), indent=2))


if __name__ == "__main__":
    main()
