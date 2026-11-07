"""Information-set MCTS with PUCT or Gumbel sequential-halving root search.

Each simulation samples a mine layout consistent with the public board (via
the exact solver), then walks a tree whose nodes are *public* states: the
child of (node, action) is keyed by the resulting public board, so different
layouts that produce different outcomes (mine / some number / flood region)
branch into different children while sharing action statistics at the parent.

The resumable single-leaf and virtual-loss multi-leaf interfaces let independent
searches share batched network evaluations.
"""

from __future__ import annotations

import math

import numpy as np

from .game import Game, PublicState
from .solver import Analysis


class Node:
    __slots__ = ("to_move", "actions", "prior", "prior_argmax", "N", "W", "R", "children", "value", "public_key")

    def __init__(self, to_move: int):
        self.to_move = to_move
        self.actions: np.ndarray | None = None  # legal actions; None = unexpanded
        self.prior: np.ndarray | None = None
        self.prior_argmax: int | None = None  # clean network choice, before root noise
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.R: np.ndarray | None = None  # in-flight virtual visits, never real statistics
        self.children: dict[tuple[int, bytes], Node] = {}
        self.value = 0.0  # network value for `to_move`
        self.public_key = None

    def expand(self, legal: np.ndarray, policy: np.ndarray, value: float) -> None:
        self.actions = np.flatnonzero(legal)
        p = policy[self.actions].astype(np.float64)
        s = p.sum()
        self.prior = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
        self.prior_argmax = int(self.actions[np.argmax(self.prior)])
        self.N = np.zeros(len(self.actions))
        self.W = np.zeros(len(self.actions))
        self.R = np.zeros(len(self.actions))
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
        search: str = "puct",
        gumbel_k: int = 16,
        gumbel_noise: bool = True,
        gumbel_c_visit: float = 50.0,
        gumbel_c_scale: float = 0.1,
    ):
        self.pub = pub
        if search not in ("puct", "gumbel"):
            raise ValueError("search must be puct or gumbel")
        if sims < 1 or gumbel_k < 1:
            raise ValueError("sims and gumbel_k must be positive")
        self.search = search
        self.gumbel_k = gumbel_k
        self.gumbel_noise = gumbel_noise
        self.gumbel_c_visit = gumbel_c_visit
        self.gumbel_c_scale = gumbel_c_scale
        self._batch = []
        self._g_candidates = None
        self._g_schedule = []
        self._g_position = 0
        self._g_noise = None
        self.sims = sims
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.rng = rng or np.random.default_rng()
        self.analysis = Analysis(pub, rng=self.rng)
        self.root = root if root is not None and root.actions is not None else Node(pub.to_move)
        self.root.public_key = self.public_key(pub)
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
        while self.done_sims + len(self._batch) < self.sims:
            if self.root.actions is None:
                self._pending = (self.root, [], self.pub.legal_mask())
                return self.pub.board, self.pub.scores, self.pub.to_move, self.pub.bombs
            g = Game.determinize(self.pub, self.analysis.sample(self.rng))
            node, path = self.root, []
            while True:
                if node is self.root and self.search == "gumbel":
                    i = self._select_gumbel_root()
                    if i is None:  # sequential-halving boundary: await this batch
                        return None
                else:
                    i = self._select(node)
                a = int(node.actions[i])
                path.append((node, i))
                g.step(a)
                key = (a, g.board.tobytes())
                child = node.children.get(key)
                if child is None:
                    child = node.children[key] = Node(g.to_move)
                child.public_key = (g.board.tobytes(), tuple(g.scores), g.to_move, tuple(g.bombs))
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

    def next_leaves(self, count: int = 1) -> list[tuple[np.ndarray, list[int], int, list[int]]]:
        """Reserve up to count leaves; provide_batch must resolve all before another call.

        Separate reservation counters make virtual loss exactly reversible, without
        subtract/add roundoff in the accumulated real values. A duplicate unexpanded
        node ends the batch rather than evaluating/expanding the same node twice.
        """
        if count < 1:
            raise ValueError("leaf batch size must be positive")
        assert self._pending is None and not self._batch
        leaves = []
        reserved_nodes = set()
        while len(leaves) < count:
            leaf = self.next_leaf()
            if leaf is None:
                break
            pending = self._pending
            self._pending = None
            node, path, _ = pending
            if node in reserved_nodes:
                # This rollout was not issued, including its root allocation.
                if self.search == "gumbel" and path:
                    self._g_position -= 1
                break
            reserved_nodes.add(node)
            self._batch.append(pending)
            for parent, i in path:
                parent.R[i] += 1
            leaves.append(leaf)
            if node is self.root:
                break  # root must be expanded before selecting any edge
        return leaves

    def provide_batch(self, policies: np.ndarray, values: np.ndarray) -> None:
        """Resolve an ordered batch, reverting every reservation before backup."""
        if len(policies) != len(self._batch) or len(values) != len(self._batch):
            raise ValueError("one policy and value are required per pending leaf")
        batch, self._batch = self._batch, []
        for pending, policy, value in zip(batch, policies, values):
            for parent, i in pending[1]:
                parent.R[i] -= 1
            self._pending = pending
            self.provide(policy, float(value))

    def improved_policy(self) -> np.ndarray:
        """Completed-Q policy improvement target (visits for ordinary PUCT)."""
        if self.search != "gumbel" or self.root.actions is None:
            return self.visit_policy()
        logits = np.log(np.maximum(self.root.prior, 1e-300)) + self._sigma_q()
        weights = np.exp(logits - logits.max())
        pi = np.zeros(2 * self.pub.cfg.n)
        pi[self.root.actions] = weights / weights.sum()
        return pi

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
        if self.search == "gumbel" and self.root.actions is not None:
            self._init_gumbel()
            candidates = self._g_candidates
            scores = self._g_scores()
            return int(self.root.actions[candidates[np.argmax(scores[candidates])]])
        return int(np.argmax(self.visit_policy()))

    def child_root(self, action: int, board: np.ndarray) -> Node | None:
        """Subtree for the actually observed outcome (for tree reuse)."""
        return self.root.children.get((action, board.tobytes()))

    @staticmethod
    def public_key(pub: PublicState) -> tuple:
        return (pub.board.tobytes(), tuple(pub.scores), pub.to_move, tuple(pub.bombs))

    def matching_root(self, pub: PublicState) -> Node | None:
        """Find descendants using only observed board/scores/turn/bomb inventory."""
        key = self.public_key(pub)
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.public_key == key:
                return node
            stack.extend(node.children.values())
        return None

    # -------------------------------------------------------------- internals
    def _select(self, node: Node) -> int:
        visits = node.N + node.R
        total = visits.sum()
        fpu = node.value - self.fpu_reduction
        q = np.where(visits > 0, (node.W - node.R) / np.maximum(visits, 1), fpu)
        u = self.c_puct * node.prior * math.sqrt(total + 1) / (1 + visits)
        return int(np.argmax(q + u))

    def _sigma_q(self) -> np.ndarray:
        r = self.root
        visited = r.N > 0
        q = r.W / np.maximum(r.N, 1)
        mass = r.prior[visited].sum()
        mean = float(np.dot(r.prior[visited], q[visited]) / mass) if mass > 0 else r.value
        mixed = (r.value + r.N.sum() * mean) / (1 + r.N.sum())
        completed = np.where(visited, q, mixed)
        spread = completed.max() - completed.min()
        normalized = (completed - completed.min()) / max(spread, 1e-8)
        return (self.gumbel_c_visit + r.N.max(initial=0)) * self.gumbel_c_scale * normalized

    def _g_scores(self) -> np.ndarray:
        return self._g_noise + np.log(np.maximum(self.root.prior, 1e-300)) + self._sigma_q()

    def _init_gumbel(self) -> None:
        if self._g_candidates is not None:
            return
        self._g_noise = self.rng.gumbel(size=len(self.root.actions)) if self.gumbel_noise else np.zeros(len(self.root.actions))
        # Root expansion consumes one simulation, as it does in PUCT.
        budget = max(1, self.sims - self.done_sims)
        k = min(self.gumbel_k, len(self.root.actions), budget)
        while k > 1:
            m, cost = k, 0
            while m > 1:
                cost += m
                m //= 2
            if cost <= budget:
                break
            k -= 1
        logits = self._g_noise + np.log(np.maximum(self.root.prior, 1e-300))
        self._g_candidates = np.argsort(-logits, kind="stable")[:k]

    def _select_gumbel_root(self) -> int | None:
        self._init_gumbel()
        if self._g_position == len(self._g_schedule):
            if self._batch:
                return None
            if self._g_schedule and len(self._g_candidates) > 1:
                candidates = self._g_candidates
                order = np.argsort(-self._g_scores()[candidates], kind="stable")
                self._g_candidates = candidates[order[:max(1, len(candidates) // 2)]]
            remaining = self.sims - self.done_sims
            if remaining <= 0:
                return None
            m = len(self._g_candidates)
            phases = max(1, math.ceil(math.log2(m)))
            allocation = remaining if m == 1 else min(remaining, m * max(1, remaining // (phases * m)))
            self._g_schedule = [int(self._g_candidates[j % m]) for j in range(allocation)]
            self._g_position = 0
        i = self._g_schedule[self._g_position]
        self._g_position += 1
        return i

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
