import itertools

import numpy as np
import pytest

from msflags.game import UNKNOWN, Config, Game
from msflags.solver import Analysis


def brute_force_probs(pub):
    """Marginals over all layouts consistent with the public board."""
    cfg = pub.cfg
    claimed = pub.board >= 9
    unknown = np.flatnonzero(pub.board == UNKNOWN)
    opened = np.flatnonzero((pub.board >= 0) & (pub.board <= 8))
    acc = np.zeros(cfg.n)
    total = 0
    for combo in itertools.combinations(unknown, pub.remaining_mines):
        mines = claimed.copy()
        mines[list(combo)] = True
        g = Game(cfg, mines)
        if np.all(g.counts[opened] == pub.board[opened]):
            acc += mines
            total += 1
    return acc / total


def random_positions(cfg, seeds, moves):
    for seed in seeds:
        rng = np.random.default_rng(seed)
        g = Game.random(cfg, rng)
        for _ in range(moves):
            if g.done:
                break
            g.step(int(rng.choice(np.flatnonzero(g.board == UNKNOWN))))
        if not g.done:
            yield g


@pytest.mark.parametrize("moves", [2, 4, 7])
def test_probabilities_match_brute_force(moves):
    cfg = Config(5, 5, 6)
    for g in random_positions(cfg, range(8), moves):
        pub = g.public()
        a = Analysis(pub, rng=np.random.default_rng(0))
        assert a.exact
        unknown = pub.board == UNKNOWN
        np.testing.assert_allclose(a.probs[unknown], brute_force_probs(pub)[unknown], atol=1e-9)


def test_samples_are_consistent_and_unbiased():
    cfg = Config(5, 5, 6)
    g = next(random_positions(cfg, [3], 5))
    pub = g.public()
    rng = np.random.default_rng(1)
    a = Analysis(pub, rng=rng, reservoir=10_000)
    opened = np.flatnonzero((pub.board >= 0) & (pub.board <= 8))
    acc = np.zeros(cfg.n)
    n = 4000
    for _ in range(n):
        m = a.sample(rng)
        h = Game(cfg, m)
        assert np.all(h.counts[opened] == pub.board[opened])
        assert np.all(m[pub.board >= 9])
        acc += m
    unknown = pub.board == UNKNOWN
    np.testing.assert_allclose(acc[unknown] / n, a.probs[unknown], atol=0.04)


def test_claimed_mines_reduce_requirement():
    # 1x3 board: [1][?][?] with one mine. Claim cell 2 -> cell 1 must be safe... unless
    # the 1 is satisfied by the claimed one. Layout: mine at 1, so opening 0 shows 1.
    cfg = Config(1, 3, 1)
    g = Game(cfg, np.array([False, True, False]))
    g.step(0)  # safe, shows 1
    a = Analysis(g.public())
    assert a.probs[1] == pytest.approx(1.0)
    assert a.probs[2] == pytest.approx(0.0)


def test_fallback_on_node_limit_still_samples_consistent_layouts():
    cfg = Config(16, 16, 51)
    rng = np.random.default_rng(5)
    for g in random_positions(cfg, range(5), 12):
        pub = g.public()
        a = Analysis(pub, node_limit=50, rng=rng)
        opened = np.flatnonzero((pub.board >= 0) & (pub.board <= 8))
        m = a.sample(rng)
        assert np.all(Game(cfg, m).counts[opened] == pub.board[opened])
        assert np.all((a.probs >= -1e-9) & (a.probs <= 1 + 1e-9))


def test_exact_probabilities_and_samples_after_bomb():
    cfg = Config(4, 5, 5)
    mines = np.zeros(cfg.n, dtype=bool)
    mines[[0, 4, 9, 14, 19]] = True
    g = Game(cfg, mines)
    g.step(cfg.n)  # claims one mine and reveals clipped upper-left window
    pub = g.public()
    analysis = Analysis(pub, rng=np.random.default_rng(3))
    assert analysis.exact
    unknown = pub.board == UNKNOWN
    np.testing.assert_allclose(analysis.probs[unknown], brute_force_probs(pub)[unknown], atol=1e-9)
    for _ in range(20):
        sampled = Game(cfg, analysis.sample())
        opened = (pub.board >= 0) & (pub.board <= 8)
        np.testing.assert_array_equal(sampled.counts[opened], pub.board[opened])
        assert np.all(sampled.mines[pub.board >= 9])


def test_sampler_cache_preserves_layouts_and_random_stream():
    game = next(random_positions(Config(5, 5, 6), [3], 5))
    cached = Analysis(game.public(), rng=np.random.default_rng(10))
    uncached = Analysis(game.public(), rng=np.random.default_rng(10))
    left = np.random.default_rng(11)
    right = np.random.default_rng(11)
    for _ in range(100):
        uncached._sample_choices.clear()
        uncached._sample_cells.clear()
        np.testing.assert_array_equal(cached.sample(left), uncached.sample(right))
    assert left.bit_generator.state == right.bit_generator.state
