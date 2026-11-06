"""Players. Every agent sees only the public state."""

from __future__ import annotations

import numpy as np

from .game import UNKNOWN, PublicState, window_sum
from .mcts import Search
from .model import Evaluator
from .solver import Analysis


class Agent:
    name = "agent"

    def act(self, pub: PublicState) -> int:
        raise NotImplementedError

    def info(self, pub: PublicState) -> dict:
        """Per-cell diagnostics for the UI (optional)."""
        return {}


class RandomAgent(Agent):
    name = "random"

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def act(self, pub: PublicState) -> int:
        return int(self.rng.choice(np.flatnonzero(pub.legal_mask())))

    def info(self, pub: PublicState) -> dict:
        legal = pub.legal_mask()
        policy = legal.astype(float) / max(1, int(legal.sum()))
        unknown = pub.board == UNKNOWN
        mine = np.where(unknown, pub.remaining_mines / max(1, int(unknown.sum())), pub.board >= 9)
        return {"mine": mine.tolist(), "policy": policy[:pub.cfg.n].tolist(),
                "bomb_policy": policy[pub.cfg.n:].tolist(), "value": 0.0}


class SolverAgent(Agent):
    """Claim certain mines first; otherwise bomb windows worth at least three."""

    name = "solver"

    def __init__(self, rng: np.random.Generator | None = None, bomb_threshold: float = 3.0):
        self.rng = rng or np.random.default_rng()
        self.bomb_threshold = bomb_threshold

    def act(self, pub: PublicState) -> int:
        return int(self.rng.choice(self.best_actions(pub, Analysis(pub, rng=self.rng).probs)))

    def best_actions(self, pub: PublicState, probs: np.ndarray) -> np.ndarray:
        """All tied greedy actions, reusing supplied exact mine marginals."""
        p = np.where(pub.board == UNKNOWN, probs, -1.0)
        if p.max() < 1.0 - 1e-9 and pub.bombs[pub.to_move]:
            expected = window_sum(
                np.where(pub.board == UNKNOWN, p, 0.0).reshape(pub.cfg.h, pub.cfg.w),
            ).reshape(-1)
            expected = np.where(pub.legal_mask()[pub.cfg.n:], expected, -1.0)
            if expected.max() >= max(self.bomb_threshold, p.max()):
                best = np.flatnonzero(expected >= expected.max() - 1e-9)
                return pub.cfg.n + best
        return np.flatnonzero(p >= p.max() - 1e-9)

    def info(self, pub: PublicState) -> dict:
        probs = Analysis(pub, rng=self.rng).probs
        best = self.best_actions(pub, probs)
        policy = np.zeros(2 * pub.cfg.n)
        policy[best] = 1 / len(best)
        return {"mine": probs.tolist(), "policy": policy[:pub.cfg.n].tolist(),
                "bomb_policy": policy[pub.cfg.n:].tolist(), "value": 0.0}


class NetAgent(Agent):
    """Network policy only (no search)."""

    name = "net"

    def __init__(self, evaluator: Evaluator):
        self.ev = evaluator

    def act(self, pub: PublicState) -> int:
        policy, _, _ = self.ev([(pub.board, pub.scores, pub.to_move, pub.bombs)], pub.cfg)
        return int(np.argmax(policy[0]))

    def info(self, pub: PublicState) -> dict:
        policy, value, mine = self.ev([(pub.board, pub.scores, pub.to_move, pub.bombs)], pub.cfg)
        return {"policy": policy[0, :pub.cfg.n].tolist(),
                "bomb_policy": policy[0, pub.cfg.n:].tolist(),
                "value": float(value[0]), "mine": mine[0].tolist()}


class MCTSAgent(Agent):
    """Network + information-set MCTS."""

    name = "mcts"

    def __init__(self, evaluator: Evaluator, sims: int = 200, rng: np.random.Generator | None = None, **search_kw):
        self.ev = evaluator
        self.sims = sims
        self.rng = rng or np.random.default_rng()
        self.search_kw = search_kw
        self.last: Search | None = None

    def search(self, pub: PublicState) -> Search:
        s = Search(pub, self.sims, rng=self.rng, **self.search_kw)
        while (leaf := s.next_leaf()) is not None:
            policy, value, _ = self.ev([leaf], pub.cfg)
            s.provide(policy[0], float(value[0]))
        self.last = s
        return s

    def act(self, pub: PublicState) -> int:
        return self.search(pub).best_action()

    def info(self, pub: PublicState) -> dict:
        s = self.last if self.last is not None and self.last.pub is pub else self.search(pub)
        _, value, mine = self.ev([(pub.board, pub.scores, pub.to_move, pub.bombs)], pub.cfg)
        root = s.root
        policy = s.visit_policy()
        q = np.zeros_like(policy)
        q[root.actions] = np.where(root.N > 0, root.W / np.maximum(root.N, 1), 0.0)
        return {
            "policy": policy[:pub.cfg.n].tolist(), "q": q[:pub.cfg.n].tolist(),
            "bomb_policy": policy[pub.cfg.n:].tolist(), "bomb_q": q[pub.cfg.n:].tolist(),
            "value": float(value[0]), "mine": mine[0].tolist(),
        }
