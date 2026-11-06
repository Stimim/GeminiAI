"""Exact mine-probability solver and consistent-layout sampler.

Given the public board, every opened cell `i` gives a linear constraint
    sum(unknown neighbours of i) == count(i) - claimed neighbours of i.
Unknown cells touching a constraint form the *frontier*; it splits into
independent components. Each component is enumerated by backtracking,
recording, per mine count k: the number of solutions, per-cell mine
counts and a reservoir of sample solutions. Components are then combined
with the global mine count: the interior (non-frontier unknowns) holds the
remaining mines uniformly, giving weight C(interior, R - K).

If a component is too big to enumerate within `node_limit`, it falls back to
randomized backtracking samples (approximate; `Analysis.exact` is False).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .game import UNKNOWN, PublicState, neighbours


class _Overflow(Exception):
    pass


@dataclass
class _KStats:
    count: float = 0.0
    cell_counts: np.ndarray | None = None
    reservoir: list[int] = field(default_factory=list)
    seen: int = 0


@dataclass
class _Component:
    cells: list[int]
    stats: dict[int, _KStats]

    def poly(self) -> np.ndarray:
        p = np.zeros(max(self.stats) + 1 if self.stats else 1)
        for k, s in self.stats.items():
            p[k] = s.count
        return p


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


class Analysis:
    """Mine probabilities for every cell plus a consistent-layout sampler."""

    def __init__(
        self,
        pub: PublicState,
        *,
        node_limit: int = 50_000,
        reservoir: int = 64,
        fallback_samples: int = 48,
        rng: np.random.Generator | None = None,
    ):
        self.pub = pub
        self.rng = rng or np.random.default_rng()
        self.exact = True
        cfg = pub.cfg
        board = pub.board
        nb = neighbours(cfg)
        self.unknown = np.flatnonzero(board == UNKNOWN)
        self.R = pub.remaining_mines

        # Build constraints (deduplicated).
        cons: dict[tuple[int, ...], int] = {}
        for i in np.flatnonzero((board >= 0) & (board <= 8)):
            nbi = nb[i]
            unk = tuple(int(j) for j in nbi if board[j] == UNKNOWN)
            if not unk:
                continue
            need = int(board[i]) - int(np.count_nonzero(board[nbi] >= 9))
            if unk in cons and cons[unk] != need:
                raise ValueError("inconsistent board")
            cons[unk] = need
        self.constraints = [(list(k), v) for k, v in cons.items()]

        # Union-find the frontier into components.
        parent: dict[int, int] = {}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for cells, _ in self.constraints:
            for c in cells:
                parent.setdefault(c, c)
            r0 = find(cells[0])
            for c in cells[1:]:
                rc = find(c)
                if rc != r0:
                    parent[rc] = r0
        groups: dict[int, list[int]] = {}
        for c in parent:
            groups.setdefault(find(c), []).append(c)
        group_cons: dict[int, list[tuple[list[int], int]]] = {}
        for cs, need in self.constraints:
            group_cons.setdefault(find(cs[0]), []).append((cs, need))
        frontier = set(parent)
        self.interior = np.array([c for c in self.unknown if int(c) not in frontier], dtype=np.int64)
        self.I = len(self.interior)

        self.components: list[_Component] = []
        for root, cells in groups.items():
            comp_cons = group_cons[root]
            order = self._order(cells, comp_cons)
            try:
                stats = self._enumerate(order, comp_cons, node_limit, reservoir)
            except _Overflow:
                self.exact = False
                stats = self._sample_component(order, comp_cons, fallback_samples, node_limit)
            self.components.append(_Component(order, stats))

        self._combine()

    # ------------------------------------------------------------------ build
    @staticmethod
    def _order(cells: list[int], cons: list[tuple[list[int], int]]) -> list[int]:
        """BFS order so constraints close early (better pruning)."""
        adj: dict[int, set[int]] = {c: set() for c in cells}
        for cs, _ in cons:
            for a in cs:
                adj[a].update(cs)
        start = min(cells, key=lambda c: len(adj[c]))
        seen, order, queue = {start}, [], [start]
        while queue:
            a = queue.pop(0)
            order.append(a)
            for b in sorted(adj[a]):
                if b not in seen:
                    seen.add(b)
                    queue.append(b)
        return order

    @staticmethod
    def _prep(order: list[int], cons: list[tuple[list[int], int]]):
        pos = {c: i for i, c in enumerate(order)}
        cons_of: list[list[int]] = [[] for _ in order]
        need = []
        size = []
        for ci, (cs, nd) in enumerate(cons):
            need.append(nd)
            size.append(len(cs))
            for c in cs:
                cons_of[pos[c]].append(ci)
        return cons_of, need, size

    def _enumerate(self, order, cons, node_limit, reservoir) -> dict[int, _KStats]:
        m = len(order)
        cons_of, need, unassigned = self._prep(order, cons)
        mines_in = [0] * len(cons)
        assign = [0] * m
        stats: dict[int, _KStats] = {}
        rng = self.rng
        R = self.R
        nodes = 0

        def record(k: int) -> None:
            s = stats.get(k)
            if s is None:
                s = stats[k] = _KStats(cell_counts=np.zeros(m))
            s.count += 1
            s.cell_counts += assign
            mask = 0
            for i in range(m):
                if assign[i]:
                    mask |= 1 << i
            s.seen += 1
            if len(s.reservoir) < reservoir:
                s.reservoir.append(mask)
            else:
                j = int(rng.integers(s.seen))
                if j < reservoir:
                    s.reservoir[j] = mask

        def dfs(i: int, k: int) -> None:
            nonlocal nodes
            nodes += 1
            if nodes > node_limit:
                raise _Overflow
            if i == m:
                record(k)
                return
            cs = cons_of[i]
            for v in (0, 1):
                if v and k >= R:
                    break
                ok = True
                for c in cs:
                    unassigned[c] -= 1
                    mines_in[c] += v
                    if mines_in[c] > need[c] or mines_in[c] + unassigned[c] < need[c]:
                        ok = False
                if ok:
                    assign[i] = v
                    dfs(i + 1, k + v)
                    assign[i] = 0
                for c in cs:
                    unassigned[c] += 1
                    mines_in[c] -= v

        dfs(0, 0)
        if not stats:
            raise ValueError("no consistent layout")
        return stats

    def _sample_component(self, order, cons, n_samples, node_limit) -> dict[int, _KStats]:
        """Approximate component statistics from randomized backtracking."""
        m = len(order)
        cons_of, need, size = self._prep(order, cons)
        density = self.R / max(1, len(self.unknown))
        stats: dict[int, _KStats] = {}
        rng = self.rng
        for _ in range(n_samples):
            unassigned = list(size)
            mines_in = [0] * len(cons)
            assign = [0] * m
            nodes = 0

            def dfs(i: int, k: int) -> bool:
                nonlocal nodes
                nodes += 1
                if nodes > node_limit:
                    raise _Overflow
                if i == m:
                    return True
                cs = cons_of[i]
                first = 1 if rng.random() < density else 0
                for v in (first, 1 - first):
                    if v and k >= self.R:
                        continue
                    ok = True
                    for c in cs:
                        unassigned[c] -= 1
                        mines_in[c] += v
                        if mines_in[c] > need[c] or mines_in[c] + unassigned[c] < need[c]:
                            ok = False
                    if ok:
                        assign[i] = v
                        if dfs(i + 1, k + v):
                            return True
                        assign[i] = 0
                    for c in cs:
                        unassigned[c] += 1
                        mines_in[c] -= v
                return False

            try:
                if not dfs(0, 0):
                    continue
            except _Overflow:
                continue
            k = sum(assign)
            s = stats.get(k)
            if s is None:
                s = stats[k] = _KStats(cell_counts=np.zeros(m))
            s.count += 1
            s.cell_counts += assign
            s.reservoir.append(sum(1 << i for i in range(m) if assign[i]))
        if not stats:
            raise ValueError("could not sample a consistent layout")
        return stats

    # ---------------------------------------------------------------- combine
    def _combine(self) -> None:
        R, I = self.R, self.I
        polys = [c.poly() for c in self.components]
        maxK = sum(len(p) - 1 for p in polys)
        logw = np.array([_log_comb(I, R - K) for K in range(maxK + 1)])
        finite = np.isfinite(logw)
        if not finite.any():
            raise ValueError("no consistent mine count")
        wC = np.where(finite, np.exp(logw - logw[finite].max()), 0.0)
        self._wC = wC

        total = np.ones(1)
        for p in polys:
            total = np.convolve(total, p)
        Z = float(total @ wC[: len(total)])
        if Z <= 0:
            raise ValueError("no consistent layout")

        probs = np.zeros(self.pub.cfg.n)
        for j, comp in enumerate(self.components):
            others = np.ones(1)
            for i, p in enumerate(polys):
                if i != j:
                    others = np.convolve(others, p)
            acc = np.zeros(len(comp.cells))
            for k, s in comp.stats.items():
                g = float(others @ wC[k : k + len(others)])
                acc += s.cell_counts * g
            probs[comp.cells] = acc / Z
        if I:
            ks = np.arange(len(total))
            probs[self.interior] = float((total * wC[: len(total)] * (R - ks)).sum() / Z) / I
        self.probs = probs

        # Suffix convolutions for sequential sampling.
        suffix = [np.ones(1)]
        for p in reversed(polys):
            suffix.append(np.convolve(suffix[-1], p))
        self._suffix = list(reversed(suffix))  # _suffix[j] = conv(polys[j:])

    # --------------------------------------------------------------- sampling
    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """A full mine layout (flat bool) consistent with the public board."""
        rng = rng or self.rng
        cfg = self.pub.cfg
        mines = self.pub.board >= 9  # already claimed mines are mines
        wC = self._wC
        prefix = 0
        for j, comp in enumerate(self.components):
            rest = self._suffix[j + 1]
            ks = sorted(comp.stats)
            w = np.array(
                [comp.stats[k].count * float(rest @ wC[prefix + k : prefix + k + len(rest)]) for k in ks]
            )
            k = ks[int(rng.choice(len(ks), p=w / w.sum()))]
            res = comp.stats[k].reservoir
            mask = res[int(rng.integers(len(res)))]
            for i, c in enumerate(comp.cells):
                if mask >> i & 1:
                    mines[c] = True
            prefix += k
        need = self.R - prefix
        if need:
            mines[rng.choice(self.interior, need, replace=False)] = True
        assert int(mines.sum()) == cfg.mines
        return mines


def mine_probabilities(pub: PublicState, **kw) -> np.ndarray:
    return Analysis(pub, **kw).probs
