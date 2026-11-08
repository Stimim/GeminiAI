"""Policy-rollout search: compare candidate moves by playing them out.

For each decision the agent samples layouts consistent with the public board
(exact posterior), plays every top-``candidates`` move on every layout, and
finishes each game with the greedy network policy for both sides. The move with
the best mean result wins. All candidates share the same layouts (common random
numbers), and all rollouts advance together in GPU batches. The rollout policy
sees only public states; sampled layouts only resolve what chance would reveal.
"""

from __future__ import annotations

import numpy as np

from .agents import Agent
from .expert import EPS
from .fastsolve import FastAnalysis, batch_probs
from .game import UNKNOWN, Game, PublicState


class RolloutAgent(Agent):
    name = "rollout"

    def __init__(self, evaluator, rng: np.random.Generator | None = None, *, candidates: int = 6,
                 rollouts: int = 32, margin_weight: float = 0.02, horizon: int = 400):
        self.ev = evaluator
        self.rng = rng or np.random.default_rng()
        self.candidates = candidates
        self.rollouts = rollouts
        self.margin_weight = margin_weight
        self.horizon = horizon

    def _greedy(self, games: list[Game], cfg) -> list[int]:
        states = [(g.board, g.scores, g.to_move, g.bombs) for g in games]
        probs = batch_probs(np.array([g.board for g in games], dtype=np.int8),
                            np.array([g.scores for g in games]), cfg,
                            seed=int(self.rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)))
        policy, _, _ = self.ev(states, cfg, probs)
        actions = []
        for g, pi, p in zip(games, policy, probs):
            sure = np.flatnonzero((g.board == UNKNOWN) & (p >= 1 - EPS))
            actions.append(int(sure[0]) if len(sure) else int(np.argmax(pi)))
        return actions

    def act(self, pub: PublicState) -> int:
        cfg = pub.cfg
        analysis = FastAnalysis(pub, rng=self.rng)
        sure = np.flatnonzero((pub.board == UNKNOWN) & (analysis.probs >= 1 - EPS))
        if len(sure):
            return int(sure[0])
        policy, _, _ = self.ev([(pub.board, pub.scores, pub.to_move, pub.bombs)], cfg, analysis.probs[None])
        legal = pub.legal_mask()
        prior = np.where(legal, policy[0], -1.0)
        cand = np.argsort(-prior)[: self.candidates]
        cand = cand[prior[cand] >= 0]
        if len(cand) == 1:
            return int(cand[0])
        layouts = analysis.sample_many(self.rollouts, self.rng)
        me = pub.to_move
        games, owner = [], []
        for ci, a in enumerate(cand):
            for layout in layouts:
                g = Game.determinize(pub, layout)
                g.step(int(a))
                games.append(g)
                owner.append(ci)
        live = [g for g in games if not g.done]
        steps = 0
        while live and steps < self.horizon:
            for g, a in zip(live, self._greedy(live, cfg)):
                g.step(a)
            live = [g for g in live if not g.done]
            steps += 1
        totals = np.zeros(len(cand))
        for g, ci in zip(games, owner):
            win = 1.0 if g.winner == me else -1.0
            totals[ci] += win + self.margin_weight * (g.scores[me] - g.scores[1 - me])
        return int(cand[int(np.argmax(totals))])


_NETS = {}


def rollout_agent(rng=None, *, net: str, device: str = "cuda", **kwargs):
    import torch
    from .deep import DeepEvaluator, load_net
    key = (net, device)
    if key not in _NETS:
        torch.set_num_threads(1)
        _NETS[key] = DeepEvaluator(load_net(net, device), device)
    return RolloutAgent(_NETS[key], rng, **kwargs)
