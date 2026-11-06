"""Minesweeper Flags (1-on-1 minesweeper) game engine.

Rules:
- Two players alternate turns on a board with a fixed number of hidden mines.
- On your turn you open one unopened cell.
  - Mine: you claim it (+1 score) and move again.
  - Safe: it shows its neighbour-mine count (zeros flood-fill) and the turn passes.
- Each player has one bomb: reveal a clipped 5×5 area, claim its mines, then pass.
- First player to claim more than half the mines wins (51 mines -> 26 wins).

Cell codes in the *public* board (what both players see):
  UNKNOWN = -1, 0..8 = opened safe cell with that count,
  CLAIMED0 = 9 / CLAIMED1 = 10 = mine claimed by player 0 / 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNKNOWN = -1
CLAIMED = (9, 10)


@dataclass(frozen=True)
class Config:
    h: int = 16
    w: int = 16
    mines: int = 51

    @property
    def bombs(self) -> int:
        return 1

    @property
    def n(self) -> int:
        return self.h * self.w

    @property
    def win_score(self) -> int:
        return self.mines // 2 + 1



def window_sum(grid: np.ndarray) -> np.ndarray:
    """Clipped 5×5 sums, including the centre."""
    h, w = grid.shape
    padded = np.pad(grid, 2)
    result = np.zeros_like(grid)
    for dr in range(5):
        for dc in range(5):
            result += padded[dr:dr + h, dc:dc + w]
    return result


@dataclass
class StepResult:
    kind: str
    hit: bool
    claimed: list[int]
    opened: list[int]


def neighbours(cfg: Config) -> list[np.ndarray]:
    """Flat neighbour indices for every cell (cached per config)."""
    key = (cfg.h, cfg.w)
    cached = _NEIGH_CACHE.get(key)
    if cached is None:
        cached = []
        for r in range(cfg.h):
            for c in range(cfg.w):
                nb = [
                    rr * cfg.w + cc
                    for rr in range(max(0, r - 1), min(cfg.h, r + 2))
                    for cc in range(max(0, c - 1), min(cfg.w, c + 2))
                    if (rr, cc) != (r, c)
                ]
                cached.append(np.array(nb, dtype=np.int64))
        _NEIGH_CACHE[key] = cached
    return cached


_NEIGH_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def neighbour_sum(grid: np.ndarray) -> np.ndarray:
    """Sum of the 8 neighbours of each cell of a 2-D array (zero padded)."""
    p = np.pad(grid, 1)
    h, w = grid.shape
    out = np.zeros_like(grid)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            if dr == 1 and dc == 1:
                continue
            out = out + p[dr : dr + h, dc : dc + w]
    return out


class Game:
    """Full (omniscient) state. Actions 0..N-1 open; N..2N-1 detonate."""

    __slots__ = ("cfg", "mines", "counts", "board", "scores", "bombs", "to_move", "winner", "nb")

    def __init__(self, cfg: Config, mines: np.ndarray, *, _init: bool = True):
        self.cfg = cfg
        self.nb = neighbours(cfg)
        self.mines = mines.astype(bool).reshape(-1)
        if not _init:
            return
        if int(self.mines.sum()) != cfg.mines:
            raise ValueError(f"layout has {int(self.mines.sum())} mines, expected {cfg.mines}")
        self.counts = neighbour_sum(self.mines.reshape(cfg.h, cfg.w).astype(np.int8)).reshape(-1)
        self.board = np.full(cfg.n, UNKNOWN, dtype=np.int8)
        self.scores = [0, 0]
        self.bombs = [cfg.bombs, cfg.bombs]
        self.to_move = 0
        self.winner: int | None = None

    @classmethod
    def random(cls, cfg: Config, rng: np.random.Generator) -> Game:
        mines = np.zeros(cfg.n, dtype=bool)
        mines[rng.choice(cfg.n, cfg.mines, replace=False)] = True
        return cls(cfg, mines)

    def clone(self) -> Game:
        g = Game.__new__(Game)
        g.cfg = self.cfg
        g.nb = self.nb
        g.mines = self.mines
        g.counts = self.counts
        g.board = self.board.copy()
        g.scores = list(self.scores)
        g.bombs = list(self.bombs)
        g.to_move = self.to_move
        g.winner = self.winner
        return g

    @classmethod
    def determinize(cls, pub: PublicState, mines: np.ndarray) -> Game:
        """A full game with `pub`'s public state and the given hidden layout.

        `mines` must be consistent with the public board.
        """
        g = cls(pub.cfg, mines, _init=False)
        g.counts = neighbour_sum(g.mines.reshape(pub.cfg.h, pub.cfg.w).astype(np.int8)).reshape(-1)
        g.board = pub.board.copy()
        g.scores = list(pub.scores)
        g.bombs = list(pub.bombs)
        g.to_move = pub.to_move
        g.winner = pub.winner
        return g

    @property
    def done(self) -> bool:
        return self.winner is not None

    def legal_mask(self) -> np.ndarray:
        return PublicState(self.cfg, self.board, self.scores, self.to_move, self.winner, self.bombs).legal_mask()

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask())

    def step(self, action: int) -> StepResult:
        """Open a cell or detonate a bomb; return newly revealed/claimed cells."""
        if self.winner is not None:
            raise ValueError("game is over")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 2 * self.cfg.n:
            raise ValueError("action is outside the action space")
        if not self.legal_mask()[action]:
            raise ValueError(f"illegal action {action}")
        p = self.to_move
        if action >= self.cfg.n:
            centre = action - self.cfg.n
            r, c = divmod(centre, self.cfg.w)
            cells = [rr * self.cfg.w + cc
                     for rr in range(max(0, r - 2), min(self.cfg.h, r + 3))
                     for cc in range(max(0, c - 2), min(self.cfg.w, c + 3))]
            self.bombs[p] -= 1
            claimed, opened = [], []
            for cell in cells:
                if self.board[cell] == UNKNOWN and self.mines[cell]:
                    self.board[cell] = CLAIMED[p]
                    claimed.append(cell)
            self.scores[p] += len(claimed)
            for cell in cells:
                if self.board[cell] == UNKNOWN:
                    opened.extend(self._flood(cell))
            if self.scores[p] >= self.cfg.win_score:
                self.winner = p
            else:
                self.to_move = 1 - p
            return StepResult("bomb", False, claimed, opened)
        if self.mines[action]:
            self.board[action] = CLAIMED[p]
            self.scores[p] += 1
            if self.scores[p] >= self.cfg.win_score:
                self.winner = p
            return StepResult("open", True, [int(action)], [int(action)])
        opened = self._flood(action)
        self.to_move = 1 - p
        return StepResult("open", False, [], opened)

    def _flood(self, start: int) -> list[int]:
        board, counts, nb = self.board, self.counts, self.nb
        opened = [start]
        board[start] = counts[start]
        stack = [start] if counts[start] == 0 else []
        while stack:
            cell = stack.pop()
            for j in nb[cell]:
                if board[j] == UNKNOWN:
                    board[j] = counts[j]
                    opened.append(int(j))
                    if counts[j] == 0:
                        stack.append(int(j))
        return opened

    def public(self) -> PublicState:
        return PublicState(self.cfg, self.board.copy(), list(self.scores), self.to_move, self.winner, list(self.bombs))


@dataclass
class PublicState:
    """Everything a player can see."""

    cfg: Config
    board: np.ndarray  # flat int8, codes as documented above
    scores: list[int]
    to_move: int
    winner: int | None = None
    bombs: list[int] | None = None

    def __post_init__(self):
        if self.bombs is None:
            self.bombs = [self.cfg.bombs, self.cfg.bombs]

    def legal_mask(self) -> np.ndarray:
        unknown = self.board == UNKNOWN
        bomb = np.zeros(self.cfg.n, dtype=bool)
        if self.bombs[self.to_move] and self.winner is None:
            bomb = (window_sum(unknown.reshape(self.cfg.h, self.cfg.w).astype(np.int8)) > 0).reshape(-1)
        mask = np.concatenate((unknown, bomb))
        if self.winner is not None:
            mask[:] = False
        return mask

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask())

    @property
    def remaining_mines(self) -> int:
        return self.cfg.mines - self.scores[0] - self.scores[1]


NUM_PLANES = 20


def encode(board: np.ndarray, scores: list[int], to_move: int, cfg: Config, bombs: list[int] | None = None) -> np.ndarray:
    """Encode a public state from the perspective of the player to move.

    Planes (float32, shape [NUM_PLANES, h, w]):
      0 unknown, 1 claimed by me, 2 claimed by opponent,
      3..11 opened with count 0..8,
      12 residual mines around an opened cell (count - claimed neighbours) / 8,
      13 unknown-neighbour count / 8,
      14 my score / win_score, 15 opponent score / win_score,
      16 remaining mines / unknown cells (density), 17 constant ones (marks the board for zero-padding),
      18 my remaining bomb, 19 opponent's remaining bomb.
    """
    h, w = cfg.h, cfg.w
    b = board.reshape(h, w)
    x = np.zeros((NUM_PLANES, h, w), dtype=np.float32)
    unknown = b == UNKNOWN
    claimed_any = b >= 9
    x[0] = unknown
    x[1] = b == CLAIMED[to_move]
    x[2] = b == CLAIMED[1 - to_move]
    opened = (b >= 0) & (b <= 8)
    for k in range(9):
        x[3 + k] = b == k
    claimed_nb = neighbour_sum(claimed_any.astype(np.float32))
    x[12] = np.where(opened, (b - claimed_nb) / 8.0, 0.0)
    x[13] = neighbour_sum(unknown.astype(np.float32)) / 8.0
    x[14] = scores[to_move] / cfg.win_score
    x[15] = scores[1 - to_move] / cfg.win_score
    n_unknown = max(1, int(unknown.sum()))
    x[16] = (cfg.mines - scores[0] - scores[1]) / n_unknown
    x[17] = 1.0
    remaining = bombs if bombs is not None else [cfg.bombs, cfg.bombs]
    x[18] = remaining[to_move]
    x[19] = remaining[1 - to_move]
    return x
