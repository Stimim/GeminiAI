"""Information-set MCTS with PUCT (AlphaZero-style) for a hidden-mine game.

Each simulation samples a mine layout consistent with the public board (via
the exact solver), then walks a tree whose nodes are *public* states: the
child of (node, action) is keyed by the resulting public board, so different
layouts that produce different outcomes (mine / some number / flood region)
branch into different children while sharing action statistics at the parent.

The search is written as a resumable state machine (`next_leaf` / `provide`)
so many searches can share batched network evaluations.
"""

from __future__ import annotations

import math

import numpy as np

from .game import Game, PublicState
from .solver import Analysis


class Node:
    __slots__ = ("to_move", "actions", "prior", "N", "W", "children", "value")

    def __init__(self, to_move: int):
        self.to_move = to_move
        self.actions: np.ndarray | None = None  # legal actions; None = unexpanded
        self.prior: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.children: dict[tuple[int, bytes], Node] = {}
        self.value = 0.0  # network value for `to_move`

    def expand(self, legal: np.ndarray, policy: np.ndarray, value: float) -> None:
        self.actions = np.flatnonzero(legal)
        p = policy[self.actions].astype(np.float64)
        s = p.sum()
        self.prior = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
        self.N = np.zeros(len(self.actions))
        self.W = np.zeros(len(self.actions))
        self.value = float(value)


class Search:
    def __init__(
        self,
        pub: PublicState,
        sims: int,
        *,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.2,
        dirichlet_alpha: float | None = None,
        dirichlet_frac: float = 0.25,
        rng: np.random.Generator | None = None,
        root: Node | None = None,
    ):
        self.pub = pub
        self.sims = sims
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.rng = rng or np.random.default_rng()
        self.analysis = Analysis(pub, rng=self.rng)
        self.root = root if root is not None and root.actions is not None else Node(pub.to_move)
        if self.root.actions is not None:
            self._add_noise()
        self.done_sims = 0
        self._pending: tuple[Node, list[tuple[Node, int]], np.ndarray] | None = None

    @property
    def finished(self) -> bool:
        return self.done_sims >= self.sims

    def next_leaf(self) -> tuple[np.ndarray, list[int], int, list[int]] | None:
        """Run simulations until one needs a network evaluation.

        Returns (board, scores, to_move, bombs) to evaluate, or None when the
        simulation budget is exhausted.
        """
        assert self._pending is None, "provide() the previous evaluation first"
        while not self.finished:
            if self.root.actions is None:
                self._pending = (self.root, [], self.pub.legal_mask())
                return self.pub.board, self.pub.scores, self.pub.to_move, self.pub.bombs
            g = Game.determinize(self.pub, self.analysis.sample(self.rng))
            node, path = self.root, []
            while True:
                i = self._select(node)
                a = int(node.actions[i])
                path.append((node, i))
                g.step(a)
                key = (a, g.board.tobytes())
                child = node.children.get(key)
                if child is None:
                    child = node.children[key] = Node(g.to_move)
                node = child
                if g.done:
                    self._backup(path, 1.0 if g.winner == 0 else -1.0)
                    break
                if node.actions is None:
                    self._pending = (node, path, g.legal_mask())
                    return g.board, list(g.scores), g.to_move, list(g.bombs)
        return None

    def provide(self, policy: np.ndarray, value: float) -> None:
        """Feed the network output (full-board policy, value for the leaf's player to move)."""
        node, path, legal = self._pending
        self._pending = None
        node.expand(legal, policy, value)
        if node is self.root:
            self._add_noise()
        self._backup(path, value if node.to_move == 0 else -value)

    def visit_policy(self) -> np.ndarray:
        pi = np.zeros(2 * self.pub.cfg.n)
        if self.root.actions is not None:
            pi[self.root.actions] = self.root.N
            s = pi.sum()
            if s > 0:
                return pi / s
            pi[self.root.actions] = self.root.prior
        return pi

    def best_action(self) -> int:
        return int(np.argmax(self.visit_policy()))

    def child_root(self, action: int, board: np.ndarray) -> Node | None:
        """Subtree for the actually observed outcome (for tree reuse)."""
        return self.root.children.get((action, board.tobytes()))

    # -------------------------------------------------------------- internals
    def _select(self, node: Node) -> int:
        total = node.N.sum()
        fpu = node.value - self.fpu_reduction
        q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), fpu)
        u = self.c_puct * node.prior * math.sqrt(total + 1) / (1 + node.N)
        return int(np.argmax(q + u))

    def _backup(self, path: list[tuple[Node, int]], v0: float) -> None:
        """v0: value from player 0's perspective."""
        for node, i in path:
            node.N[i] += 1
            node.W[i] += v0 if node.to_move == 0 else -v0
        self.done_sims += 1

    def _add_noise(self) -> None:
        if self.dirichlet_alpha is None:
            return
        r = self.root
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(r.actions))
        r.prior = (1 - self.dirichlet_frac) * r.prior + self.dirichlet_frac * noise
