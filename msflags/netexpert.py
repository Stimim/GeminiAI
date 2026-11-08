"""Expert2 move selection with a single batched, distilled reply evaluator.

The one-ply shortlist and sampled transitions are unchanged: safe opens are
conditioned on missing a mine, mine hits use geometric guess value, and winning
bomb outcomes contribute the full mine count without querying the value net.
"""

from __future__ import annotations

import numpy as np

from .expert import EPS, ExpertAgent, guess_value
from .fastsolve import FastAnalysis
from .game import UNKNOWN, Game, PublicState


class NetExpert2Agent(ExpertAgent):
    """Rescore Expert's shortlist using predicted opponent reply advantages."""

    name = "netexpert2"

    def __init__(self, value_net, rng: np.random.Generator | None = None, *,
                 top: int = 8, samples2: int = 32, bomb_reserve: float = 3.0,
                 cap: float = 0.9, beta: float = 1.0, **kwargs):
        from .vnet import ValueEvaluator

        super().__init__(rng, bomb_reserve=bomb_reserve, cap=cap, **kwargs)
        self.top = top
        self.samples2 = samples2
        self.beta = beta
        self.ev = (value_net if isinstance(value_net, ValueEvaluator) else
                   ValueEvaluator(value_net, next(value_net.parameters()).device, rng=self.rng))

    def act(self, pub: PublicState) -> int:
        probs = self.analysis(pub, rng=self.rng).probs
        sure = np.flatnonzero((pub.board == UNKNOWN) & (probs >= 1 - EPS))
        if len(sure):
            return int(sure[0])
        actions, values, _ = self.scores(pub)
        if len(actions) <= 1:
            return int(actions[0])
        order = np.argsort(-values)[: self.top]
        layouts = FastAnalysis(pub, rng=self.rng).sample_many(self.samples2, self.rng)
        cfg = pub.cfg
        me = pub.to_move
        totals = np.zeros(len(order))
        states, owners = [], []
        for ci, i in enumerate(order):
            action = int(actions[i])
            for layout in layouts:
                game = Game.determinize(pub, layout)
                before = game.scores[me]
                game.step(action)
                gained = game.scores[me] - before
                if action < cfg.n:
                    if gained:
                        continue  # Mine hits, including wins, use guess_value below.
                else:
                    if game.done:
                        totals[ci] += cfg.mines
                        continue
                    totals[ci] += gained
                states.append((game.board, game.scores, game.to_move, game.bombs))
                owners.append(ci)
        if states:
            replies = self.ev(states, cfg, beta=self.beta)
            np.add.at(totals, owners, -replies)
        best_action, best_value = int(actions[order[0]]), -np.inf
        for ci, i in enumerate(order):
            action = int(actions[i])
            if action < cfg.n:
                safe = int((~layouts[:, action]).sum())
                value = (float(guess_value(float(probs[action]), self.cap))
                         + (totals[ci] / safe if safe else 0.0))
            else:
                value = totals[ci] / len(layouts) - self.bomb_reserve
            if value > best_value:
                best_action, best_value = action, value
        return best_action


_NETS = {}


def netexpert2_agent(rng=None, *, net: str, device: str = "cuda", **kwargs):
    """Load a value checkpoint once per process and construct a NetExpert2Agent."""
    import torch
    from .vnet import ValueEvaluator, load_net

    key = (net, device)
    if key not in _NETS:
        torch.set_num_threads(1)
        _NETS[key] = load_net(net, device)
    return NetExpert2Agent(ValueEvaluator(_NETS[key], device, rng=rng), rng, **kwargs)
