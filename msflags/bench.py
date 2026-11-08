"""Parallel paired-layout benchmark for non-neural (and plug-in) agents.

Each layout is played twice with seats swapped. Agent specs look like
``solver``, ``random`` or ``expert:alpha=1,cap=0.9``; keyword values are parsed
as floats/ints. Example::

    uv run python -m msflags.bench expert solver --layouts 200 --workers 16
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import time

import numpy as np

from .game import Config, Game

AGENTS = {
    "random": "msflags.agents:RandomAgent",
    "solver": "msflags.agents:SolverAgent",
    "expert": "msflags.expert:ExpertAgent",
    "expert2": "msflags.expert:Expert2Agent",
    "deep": "msflags.deep:deep_agent",
}


def _value(text: str):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(text.lower(), text)


def make_agent(spec: str, rng: np.random.Generator):
    name, _, args = spec.partition(":")
    module, _, attr = AGENTS[name].partition(":")
    cls = getattr(importlib.import_module(module), attr)
    kwargs = {k: _value(v) for k, v in (kv.split("=", 1) for kv in args.split(",") if kv)}
    return cls(rng=rng, **kwargs)


def play_layout(job):
    """Play one layout in both seat orders; return per-game (a_won, a_score, b_score)."""
    a, b, seed, layout = job
    cfg = Config()
    out = []
    for swap in (0, 1):
        game = Game.random(cfg, np.random.default_rng(np.random.SeedSequence([seed, 0, layout])))
        specs = [a, b] if not swap else [b, a]
        agents = [make_agent(s, np.random.default_rng(np.random.SeedSequence([seed, 1, layout, swap, i])))
                  for i, s in enumerate(specs)]
        while not game.done:
            game.step(agents[game.to_move].act(game.public()))
        a_seat = swap
        out.append((int(game.winner == a_seat), game.scores[a_seat], game.scores[1 - a_seat]))
    return out


def run(a: str, b: str, layouts: int, seed: int, workers: int) -> dict:
    started = time.monotonic()
    jobs = [(a, b, seed, i) for i in range(layouts)]
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(play_layout, jobs, chunksize=1)
    pairs = np.array([[g[0] for g in r] for r in results], dtype=float).mean(1)
    margin = np.array([g[1] - g[2] for r in results for g in r], dtype=float)
    boot = np.random.default_rng(seed)
    means = pairs[boot.integers(len(pairs), size=(10000, len(pairs)))].mean(1)
    return dict(a=a, b=b, layouts=layouts, games=2 * layouts, win_rate=float(pairs.mean()),
                ci95=np.quantile(means, [0.025, 0.975]).round(4).tolist(),
                losses=int(round((1 - pairs).sum() * 2)), mean_margin=float(margin.mean()),
                seed=seed, seconds=round(time.monotonic() - started, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--layouts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args.a, args.b, args.layouts, args.seed, args.workers)), flush=True)


if __name__ == "__main__":
    main()
