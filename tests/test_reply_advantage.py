"""Factoring reply evaluation preserves Expert2 arithmetic and random streams."""
import copy

import numpy as np
import pytest

from msflags.expert import EPS, Expert2Agent, ExpertAgent, reply_advantage
from msflags.fastsolve import FastAnalysis
from msflags.game import Config, Game, PublicState, UNKNOWN


class LegacyExpert2(Expert2Agent):
    """Independent copy of the private implementation before factoring."""

    def _reply_advantage(self, game):
        key = game.board.tobytes() + bytes(game.scores) + bytes(game.bombs)
        hit = self._cache2.get(key)
        if hit is not None:
            return hit
        pub = game.public()
        cfg = pub.cfg
        probs = FastAnalysis(pub, rng=self.rng).probs
        unknown = pub.board == UNKNOWN
        sure = np.flatnonzero(unknown & (probs >= 1 - EPS))
        player = pub.to_move
        if pub.scores[player] + len(sure) >= cfg.win_score:
            value = float(cfg.mines)
        else:
            board = pub.board.copy()
            board[sure] = 9 + player
            scores = list(pub.scores)
            scores[player] += len(sure)
            after = PublicState(cfg, board, scores, player, None, list(pub.bombs))
            if (board == UNKNOWN).any():
                _, values, _ = self.inner.scores(after)
                best = float(values.max()) if len(values) else 0.0
            else:
                best = 0.0
            value = len(sure) + self.beta * best
        self._cache2[key] = value
        return value


def position(kind):
    cfg = Config(1, 11, 5)
    mines = np.zeros(cfg.n, dtype=bool)
    mines[[0, 4, 6, 8, 10]] = True
    game = Game(cfg, mines)
    game.to_move = 1
    if kind != "unknown":
        # The revealed zero and its neighbours force mines 0 and 4, while
        # the other three mines remain ambiguous among cells 5 through 10.
        game.board[[1, 2, 3]] = game.counts[[1, 2, 3]]
    if kind == "threshold":
        game.board[6] = 10
        game.scores[1] = 1
        game.bombs[1] = 0
    return game


@pytest.mark.parametrize("kind,sure_count", [("unknown", 0), ("sure", 2), ("threshold", 2)])
@pytest.mark.parametrize("beta", [0.0, 0.65, 1.0])
@pytest.mark.parametrize("public", [False, True])
def test_reply_advantage_matches_old_private_method(kind, sure_count, beta, public):
    game = position(kind)
    original = game.clone()
    pub = game.public()
    probs = FastAnalysis(pub, rng=np.random.default_rng(3)).probs
    assert np.count_nonzero((pub.board == UNKNOWN) & (probs >= 1 - EPS)) == sure_count
    if kind == "threshold":
        assert game.scores[game.to_move] + sure_count == game.cfg.win_score
    else:
        assert game.scores[game.to_move] + sure_count < game.cfg.win_score

    rng = np.random.default_rng(71)
    options = dict(samples=12, native=False, interior_candidates=4, bomb_candidates=2)
    old = LegacyExpert2(copy.deepcopy(rng), beta=beta, **options)
    inner = ExpertAgent(copy.deepcopy(rng), **options)
    expected = old._reply_advantage(game)
    actual = reply_advantage(pub if public else game, inner, beta)

    assert actual == expected
    if kind == "threshold":
        assert actual == float(game.cfg.mines)
    np.testing.assert_array_equal(inner.rng.integers(2**63, size=20),
                                  old.rng.integers(2**63, size=20))
    np.testing.assert_array_equal(game.board, original.board)
    assert game.scores == original.scores
    assert game.bombs == original.bombs
    assert game.to_move == original.to_move
    np.testing.assert_array_equal(pub.board, original.board)
    assert pub.scores == original.scores
    assert pub.bombs == original.bombs


@pytest.mark.parametrize("kind", ["unknown", "sure", "threshold"])
def test_expert2_retains_reply_cache_and_rng(kind):
    game = position(kind)
    rng = np.random.default_rng(82)
    options = dict(beta=0.75, samples=12, native=False,
                   interior_candidates=4, bomb_candidates=2)
    old = LegacyExpert2(copy.deepcopy(rng), **options)
    current = Expert2Agent(copy.deepcopy(rng), **options)

    expected = old._reply_advantage(game)
    assert current._reply_advantage(game) == expected
    assert current._cache2 == old._cache2
    np.testing.assert_array_equal(current.rng.integers(2**63, size=20),
                                  old.rng.integers(2**63, size=20))

    before_cached_call = copy.deepcopy(current.rng)
    assert current._reply_advantage(game) == expected
    np.testing.assert_array_equal(current.rng.integers(2**63, size=20),
                                  before_cached_call.integers(2**63, size=20))
