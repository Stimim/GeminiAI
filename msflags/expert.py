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
from .fastsolve import FastAnalysis, expert_scores_native

EPS = 1e-9


def guess_value(p: np.ndarray | float, cap: float) -> np.ndarray | float:
    p = np.minimum(p, cap)
    return p / (1.0 - p)


class ExpertAgent(Agent):
    name = "expert"

    def __init__(self, rng: np.random.Generator | None = None, *, samples: int = 48,
                 alpha: float = 1.0, cap: float = 0.9, bomb_reserve: float = 1.5,
                 interior_candidates: int = 12, bomb_candidates: int = 4,
                 analysis=FastAnalysis, native: bool = True):
        self.rng = rng or np.random.default_rng()
        self.samples = samples
        self.alpha = alpha
        self.cap = cap
        self.bomb_reserve = bomb_reserve
        self.interior_candidates = interior_candidates
        self.bomb_candidates = bomb_candidates
        self.analysis = analysis
        self.native = native
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
        layouts = analysis.sample_many(self.samples, self.rng)
        me = pub.to_move
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

        if self.native and self.analysis is FastAnalysis and len(layouts):
            centres = np.array([], dtype=np.int64)
            expected = np.zeros(cfg.n)
            if pub.bombs[me]:
                legal_bomb = pub.legal_mask()[cfg.n:]
                expected = window_sum(np.where(unknown, probs, 0.0).reshape(cfg.h, cfg.w)).reshape(-1)
                expected = np.where(legal_bomb, expected, -1.0)
                order = np.argsort(-expected)[: self.bomb_candidates]
                centres = order[expected[order] > 0]
            actions, values = expert_scores_native(
                pub, layouts, probs, cand, centres, expected[centres],
                alpha=self.alpha, cap=self.cap, bomb_reserve=self.bomb_reserve,
                rng=self.rng,
            )
            return actions, values, probs

        counts = neighbour_sum(layouts.reshape(-1, cfg.h, cfg.w).astype(np.int8)).reshape(len(layouts), -1)

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


def reply_advantage(pub_or_game, inner: ExpertAgent, beta: float = 1.0) -> float:
    """Value the player to move as certain mines plus beta times their best score.

    Accept a public state or game, without caching or changing its state. Analysis
    and one-ply scoring both consume ``inner.rng``. Reaching the winning score
    through certain mines is valued as the full mine count.
    """
    pub = pub_or_game if isinstance(pub_or_game, PublicState) else pub_or_game.public()
    cfg = pub.cfg
    probs = FastAnalysis(pub, rng=inner.rng).probs
    unknown = pub.board == UNKNOWN
    sure = np.flatnonzero(unknown & (probs >= 1 - EPS))
    player = pub.to_move
    if pub.scores[player] + len(sure) >= cfg.win_score:
        return float(cfg.mines)
    board = pub.board.copy()
    board[sure] = 9 + player
    scores = list(pub.scores)
    scores[player] += len(sure)
    after = PublicState(cfg, board, scores, player, None, list(pub.bombs))
    if (board == UNKNOWN).any():
        _, values, _ = inner.scores(after)
        best = float(values.max()) if len(values) else 0.0
    else:
        best = 0.0
    return len(sure) + beta * best


class Expert2Agent(ExpertAgent):
    """Two-ply expert: the opponent's reply is valued by the full one-ply expert.

    The top ``top`` one-ply candidates are re-scored as gain minus the expected
    ``certain mines + beta * best one-ply score`` of the opponent after each
    sampled outcome, so a move that leaves the opponent only bad options counts
    in our favour. Outcomes come from layouts consistent with the public board.
    """

    name = "expert2"

    def __init__(self, rng: np.random.Generator | None = None, *, top: int = 8, beta: float = 1.0,
                 samples2: int = 32, **kwargs):
        super().__init__(rng, **kwargs)
        self.top = top
        self.beta = beta
        self.samples2 = samples2
        self.inner = ExpertAgent(self.rng, **kwargs)
        self._cache2: dict[bytes, float] = {}

    def _reply_advantage(self, game) -> float:
        """Opponent to move in `game`: their certain mines plus beta times their best score."""
        key = game.board.tobytes() + bytes(game.scores) + bytes(game.bombs)
        hit = self._cache2.get(key)
        if hit is not None:
            return hit
        value = reply_advantage(game, self.inner, self.beta)
        self._cache2[key] = value
        return value

    def act(self, pub: PublicState) -> int:
        from .game import Game
        probs = self.analysis(pub, rng=self.rng).probs
        sure = np.flatnonzero((pub.board == UNKNOWN) & (probs >= 1 - EPS))
        if len(sure):
            return int(sure[0])
        actions, values, _ = self.scores(pub)
        if len(actions) <= 1:
            return int(actions[0])
        self._cache2.clear()
        order = np.argsort(-values)[: self.top]
        layouts = FastAnalysis(pub, rng=self.rng).sample_many(self.samples2, self.rng)
        cfg = pub.cfg
        me = pub.to_move
        best_action, best_value = int(actions[order[0]]), -np.inf
        for i in order:
            a = int(actions[i])
            total = 0.0
            for layout in layouts:
                g = Game.determinize(pub, layout)
                before = g.scores[me]
                g.step(a)
                gained = g.scores[me] - before
                if a < cfg.n:
                    if gained:
                        continue  # hit: accounted by the guess value below
                    total -= self._reply_advantage(g)
                else:
                    if g.done:
                        total += cfg.mines
                        continue
                    total += gained - self._reply_advantage(g)
            if a < cfg.n:
                p = float(probs[a])
                safe = int((~layouts[:, a]).sum())
                value = float(guess_value(p, self.cap)) + (total / safe if safe else 0.0)
            else:
                value = total / len(layouts) - self.bomb_reserve
            if value > best_value:
                best_action, best_value = a, value
        return best_action
