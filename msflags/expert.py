"""Information-aware expert agent.

Every decision uses only the public board. Certain mines are always claimed
first (they reveal nothing and keep the turn). Otherwise each candidate move is
scored as

    gain(move) - E[opponent advantage after the move passes the turn]

where gain is the geometric value p / (1 - p) of a guess (a hit keeps the turn
and, roughly, offers a similar guess again) or the expected mines of a bomb, and
the opponent's advantage is the number of mines the revealed clues make
certain plus ``alpha`` times the value of their best follow-up guess. The
distribution of revealed clues comes from layouts sampled consistently with the
public board, so the agent never reads hidden information.
"""

from __future__ import annotations

import numpy as np

from .agents import Agent
from .game import UNKNOWN, PublicState, neighbour_sum, neighbours, window_sum
from .solver import Analysis

EPS = 1e-9


def guess_value(p: np.ndarray | float, cap: float) -> np.ndarray | float:
    p = np.minimum(p, cap)
    return p / (1.0 - p)


class ExpertAgent(Agent):
    name = "expert"

    def __init__(self, rng: np.random.Generator | None = None, *, samples: int = 48,
                 alpha: float = 1.0, cap: float = 0.9, bomb_reserve: float = 1.5,
                 interior_candidates: int = 12, bomb_candidates: int = 4,
                 analysis=Analysis):
        self.rng = rng or np.random.default_rng()
        self.samples = samples
        self.alpha = alpha
        self.cap = cap
        self.bomb_reserve = bomb_reserve
        self.interior_candidates = interior_candidates
        self.bomb_candidates = bomb_candidates
        self.analysis = analysis
        self._cache: dict[bytes, tuple[int, float]] = {}

    # ----------------------------------------------------------- evaluation
    def _opponent_advantage(self, pub: PublicState, board: np.ndarray, scores, bombs) -> float:
        """Certain mines handed over plus alpha times their best guess value."""
        key = board.tobytes() + bytes(scores)
        hit = self._cache.get(key)
        if hit is None:
            nxt = PublicState(pub.cfg, board, list(scores), 1 - pub.to_move, None, list(bombs))
            probs = self.analysis(nxt, rng=self.rng).probs
            unknown = board == UNKNOWN
            sure = unknown & (probs >= 1 - EPS)
            rest = probs[unknown & ~sure]
            best = float(guess_value(rest.max(), self.cap)) if rest.size else 0.0
            hit = (int(sure.sum()), best)
            self._cache[key] = hit
        sure_count, best = hit
        return sure_count + self.alpha * best

    def _reveal(self, board: np.ndarray, counts: np.ndarray, cells) -> np.ndarray:
        nb = self._nb
        out = board.copy()
        stack = []
        for c in cells:
            if out[c] == UNKNOWN:
                out[c] = counts[c]
                if counts[c] == 0:
                    stack.append(c)
        while stack:
            c = stack.pop()
            for j in nb[c]:
                if out[j] == UNKNOWN:
                    out[j] = counts[j]
                    if counts[j] == 0:
                        stack.append(int(j))
        return out

    def scores(self, pub: PublicState) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
        """(candidate actions, their scores, mine probabilities)."""
        cfg = pub.cfg
        self._nb = neighbours(cfg)
        self._cache.clear()
        analysis = self.analysis(pub, rng=self.rng)
        probs = analysis.probs
        board = pub.board
        unknown = board == UNKNOWN
        layouts = np.array([analysis.sample(self.rng) for _ in range(self.samples)])
        counts = neighbour_sum(layouts.reshape(-1, cfg.h, cfg.w).astype(np.int8)).reshape(len(layouts), -1)
        me, opp = pub.to_move, 1 - pub.to_move

        frontier = np.zeros(cfg.n, dtype=bool)
        opened = (board >= 0) & (board <= 8)
        for i in np.flatnonzero(opened):
            frontier[self._nb[i]] = True
        frontier &= unknown
        cand = list(np.flatnonzero(frontier))
        interior = np.flatnonzero(unknown & ~frontier)
        if len(interior) > self.interior_candidates:
            interior = self.rng.choice(interior, self.interior_candidates, replace=False)
        cand += list(interior)

        actions, values = [], []
        for c in cand:
            p = float(probs[c])
            safe = ~layouts[:, c]
            if not safe.any():
                continue
            outcomes: dict[bytes, list] = {}
            for s in np.flatnonzero(safe):
                nb_board = self._reveal(board, counts[s], [c])
                key = nb_board.tobytes()
                if key in outcomes:
                    outcomes[key][1] += 1
                else:
                    outcomes[key] = [nb_board, 1]
            total = sum(w for _, w in outcomes.values())
            adv = sum(w * self._opponent_advantage(pub, b, pub.scores, pub.bombs)
                      for b, w in outcomes.values()) / total
            actions.append(int(c))
            values.append(float(guess_value(p, self.cap)) - adv)

        if pub.bombs[me]:
            legal_bomb = pub.legal_mask()[cfg.n:]
            expected = window_sum(np.where(unknown, probs, 0.0).reshape(cfg.h, cfg.w)).reshape(-1)
            expected = np.where(legal_bomb, expected, -1.0)
            order = np.argsort(-expected)[: self.bomb_candidates]
            for centre in order:
                if expected[centre] <= 0:
                    continue
                r, col = divmod(int(centre), cfg.w)
                cells = [rr * cfg.w + cc for rr in range(max(0, r - 2), min(cfg.h, r + 3))
                         for cc in range(max(0, col - 2), min(cfg.w, col + 3))]
                cells = np.array(cells)
                adv = 0.0
                for s in range(len(layouts)):
                    b = board.copy()
                    hits = cells[(b[cells] == UNKNOWN) & layouts[s, cells]]
                    b[hits] = 9 + me
                    sc = list(pub.scores)
                    sc[me] += len(hits)
                    b = self._reveal(b, counts[s], [int(x) for x in cells if b[x] == UNKNOWN])
                    bombs = list(pub.bombs)
                    bombs[me] -= 1
                    adv += self._opponent_advantage(pub, b, sc, bombs)
                adv /= len(layouts)
                actions.append(cfg.n + int(centre))
                values.append(float(expected[centre]) - adv - self.bomb_reserve)
        return np.array(actions), np.array(values), probs

    def act(self, pub: PublicState) -> int:
        probs = self.analysis(pub, rng=self.rng).probs
        sure = np.flatnonzero((pub.board == UNKNOWN) & (probs >= 1 - EPS))
        if len(sure):
            return int(sure[0])
        actions, values, _ = self.scores(pub)
        best = np.flatnonzero(values >= values.max() - 1e-9)
        return int(actions[self.rng.choice(best)])
