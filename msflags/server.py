from __future__ import annotations

import argparse
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import registry
from .arena import manager
from .game import Config, Game

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="Minesweeper Flags")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@dataclass
class Session:
    game: Game
    players: list[str]
    sims: int
    agents: list[Any]
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_move: dict | None = None
    moves: int = 0


sessions: dict[str, Session] = {}
sessions_lock = threading.Lock()


class NewGame(BaseModel):
    players: list[str]
    sims: int = 200
    seed: int | None = None


class Move(BaseModel):
    cell: int
    bomb: bool = False


def configure(requested_device: str) -> None:
    registry.configure(device=requested_device)


def agent_names() -> list[str]:
    return [entry["id"] for entry in registry.entries() if entry["available"]] + [
        entry["id"] for entry in registry.catalog() if entry["available"]]


def make_agent(name: str, sims: int, rng: np.random.Generator):
    return registry.create(name, sims=sims, rng=rng)


def serialize(s: Session) -> dict:
    g = s.game
    out = {"h": g.cfg.h, "w": g.cfg.w, "mines": g.cfg.mines, "board": g.board.astype(int).tolist(),
           "scores": list(g.scores), "to_move": g.to_move, "winner": g.winner, "players": s.players,
           "last_move": s.last_move, "moves": s.moves, "bombs": list(g.bombs)}
    if g.done: out["mine_layout"] = g.mines.astype(int).tolist()
    return out


def get_session(game_id: str) -> Session:
    with sessions_lock: s = sessions.get(game_id)
    if s is None: raise HTTPException(404, "game not found")
    return s


@app.get("/")
def index(): return FileResponse(WEB / "index.html")

@app.get("/api/agents")
def agents():
    rows = {row["id"]: row for row in manager.ratings()["ratings"]}
    return {"agents": [with_ratings(category, rows) for category in registry.catalog()]}


def with_ratings(entry: dict, rows: dict) -> dict:
    result = dict(entry)
    if "generations" in result:
        result["generations"] = [with_ratings(generation, rows) for generation in result["generations"]]
    row = rows.get(result["id"], {})
    result["games"] = row.get("games", 0)
    if result["games"]:
        result.update(elo=row["rating"], uncertainty=row["uncertainty"])
    return result


@app.get("/api/agents/{agent_id}")
def agent_detail(agent_id: str):
    try:
        rows = {row["id"]: row for row in manager.ratings()["ratings"]}
        return with_ratings(registry.detail(agent_id), rows)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/agents.html")
def agents_page(): return FileResponse(WEB / "agents.html")


@app.get("/arena.html")
def arena_page(): return FileResponse(WEB / "arena.html")


class NewMatch(BaseModel):
    a: str
    b: str
    games: int = Field(default=20, strict=True)
    seed: int | None = Field(default=None, strict=True)


@app.post("/api/arena/matches")
def start_match(body: NewMatch):
    seed = body.seed if body.seed is not None else int(np.random.default_rng().integers(0, 2**31))
    try: return manager.start(body.a, body.b, body.games, seed)
    except ValueError as exc:
        raise HTTPException(409 if "busy" in str(exc).lower() else 400, str(exc)) from exc


@app.get("/api/arena/matches/{match_id}")
def match_status(match_id: str):
    try: return manager.status(match_id)
    except KeyError as exc: raise HTTPException(404, "match not found") from exc


@app.get("/api/ratings")
def ratings(): return manager.ratings()

@app.post("/api/games")
def new_game(body: NewGame):
    if len(body.players) != 2 or body.sims < 1: raise HTTPException(400, "players must contain two seats and sims must be positive")
    try:
        players = ["human" if player == "human" else registry.resolve(player) for player in body.players]
        if any(player != "human" and not registry.available(player) for player in players):
            raise ValueError("unknown or unavailable player")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rng = np.random.default_rng(body.seed)
    game = Game.random(Config(), rng)
    try:
        ags = [None if p == "human" else make_agent(p, body.sims, rng) for p in players]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    gid = uuid.uuid4().hex
    s = Session(game, players, body.sims, ags)
    with sessions_lock: sessions[gid] = s
    result = serialize(s); result["id"] = gid
    return result

@app.get("/api/games/{game_id}")
def state(game_id: str):
    s = get_session(game_id)
    with s.lock: return serialize(s)

@app.post("/api/games/{game_id}/move")
def move(game_id: str, body: Move):
    s = get_session(game_id)
    with s.lock:
        g = s.game
        if g.done: raise HTTPException(400, "game is over")
        if s.players[g.to_move] != "human": raise HTTPException(409, "not a human turn")
        if body.cell < 0 or body.cell >= g.cfg.n: raise HTTPException(400, "cell is out of bounds")
        action = body.cell + (g.cfg.n if body.bomb else 0)
        if not g.legal_mask()[action]:
            raise HTTPException(400, "bomb unavailable or area fully opened" if body.bomb else "cell is not unknown")
        player = g.to_move
        result = g.step(action); s.moves += 1
        s.last_move = {"cell": body.cell, "player": player, "kind": result.kind,
                       "hit": result.hit, "claimed": result.claimed, "opened": result.opened}
        return serialize(s)

@app.post("/api/games/{game_id}/ai")
def ai_move(game_id: str):
    s = get_session(game_id)
    with s.lock:
        g = s.game
        if g.done: raise HTTPException(400, "game is over")
        player = g.to_move; agent = s.agents[player]
        if agent is None: raise HTTPException(409, "not an AI turn")
        action = int(agent.act(g.public()))
        if action < 0 or action >= 2 * g.cfg.n or not g.legal_mask()[action]:
            raise HTTPException(500, "agent selected an illegal action")
        result = g.step(action); s.moves += 1
        s.last_move = {"cell": action % g.cfg.n, "player": player, "kind": result.kind,
                       "hit": result.hit, "claimed": result.claimed, "opened": result.opened}
        return serialize(s)

@app.get("/api/games/{game_id}/hint")
def hint(game_id: str, agent: str = "solver", sims: int | None = None):
    s = get_session(game_id)
    if sims is not None and sims < 1: raise HTTPException(400, "sims must be positive")
    with s.lock:
        if agent not in agent_names(): raise HTTPException(400, "hint agent unavailable")
        try:
            return make_agent(agent, s.sims if sims is None else sims, np.random.default_rng()).info(s.game.public())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    configure(args.device)
    import torch
    torch.set_num_threads(4)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__": main()
