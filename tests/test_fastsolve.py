"""Native solver parity, global conditioning, and batched sampling coverage."""
import numpy as np
import pytest

from msflags.fastsolve import FastAnalysis
from msflags.game import UNKNOWN, Config, Game, PublicState, neighbour_sum
from msflags.solver import Analysis


def positions(cfg, seeds, moves):
    for seed in seeds:
        rng = np.random.default_rng(seed)
        game = Game.random(cfg, rng)
        for turn in range(moves):
            if game.done:
                break
            unknown = np.flatnonzero(game.board == UNKNOWN)
            if turn % 3 == 0:
                analysis = Analysis(game.public(), rng=rng)
                action = unknown[np.argmax(analysis.probs[unknown])]
            else:
                action = rng.choice(unknown)
            game.step(int(action))
            if turn >= 2:
                yield game.public()


def assert_consistent(pub, samples):
    assert samples.dtype == np.bool_
    assert samples.shape[1:] == (pub.cfg.n,)
    np.testing.assert_array_equal(samples.sum(axis=1), np.full(len(samples), pub.cfg.mines))
    assert np.all(samples[:, pub.board >= 9])
    assert not np.any(samples[:, (pub.board >= 0) & (pub.board <= 8)])
    counts = neighbour_sum(samples.reshape(-1, pub.cfg.h, pub.cfg.w).astype(np.int8)).reshape(-1, pub.cfg.n)
    opened = (pub.board >= 0) & (pub.board <= 8)
    np.testing.assert_array_equal(counts[:, opened], np.broadcast_to(pub.board[opened], counts[:, opened].shape))


@pytest.mark.parametrize("cfg", [Config(5, 5, 6), Config(8, 8, 13), Config(16, 16, 51)])
def test_random_midgame_parity_and_samples(cfg):
    compared = 0
    for pub in positions(cfg, range(8), 10):
        reference = Analysis(pub, rng=np.random.default_rng(200))
        native = FastAnalysis(pub, rng=np.random.default_rng(200))
        if reference.exact and native.exact:
            np.testing.assert_allclose(native.probs, reference.probs, atol=1e-9, rtol=0)
            compared += 1
        assert np.all(native.probs >= -1e-12)
        assert np.all(native.probs <= 1 + 1e-12)
        assert_consistent(pub, native.sample_many(12))
        assert_consistent(pub, native.sample()[None, :])
    assert compared >= 10


@pytest.mark.parametrize("seed", [2, 4, 8])
def test_empirical_marginals(seed):
    pub = next(positions(Config(8, 8, 13), [seed], 8))
    analysis = FastAnalysis(pub, rng=np.random.default_rng(seed))
    assert analysis.exact
    samples = analysis.sample_many(4000)
    assert_consistent(pub, samples)
    unknown = pub.board == UNKNOWN
    np.testing.assert_allclose(samples.mean(axis=0)[unknown], analysis.probs[unknown], atol=0.04, rtol=0)


def test_no_frontier_and_all_claimed():
    game = Game.random(Config(4, 5, 5), np.random.default_rng(1))
    pub = game.public()
    analysis = FastAnalysis(pub)
    assert analysis.exact
    np.testing.assert_allclose(analysis.probs, 0.25)
    assert_consistent(pub, analysis.sample_many(4000))
    all_claimed = PublicState(Config(1, 3, 3), np.array([9, 10, 9], dtype=np.int8), [2, 1], 0)
    analysis = FastAnalysis(all_claimed)
    np.testing.assert_array_equal(analysis.probs, np.zeros(3))
    assert_consistent(all_claimed, analysis.sample_many(10))
    all_safe = PublicState(Config(2, 2, 0), np.zeros(4, dtype=np.int8), [0, 0], 0)
    assert_consistent(all_safe, FastAnalysis(all_safe).sample_many(10))


def test_disconnected_components_with_interior():
    board = np.full(20, -1, dtype=np.int8)
    board[[1, 18]] = 1
    pub = PublicState(Config(1, 20, 6), board, [0, 0], 0)
    native = FastAnalysis(pub, rng=np.random.default_rng(17))
    reference = Analysis(pub)
    np.testing.assert_allclose(native.probs, reference.probs, atol=1e-9, rtol=0)
    samples = native.sample_many(4000)
    assert_consistent(pub, samples)
    np.testing.assert_allclose(samples.mean(axis=0), native.probs, atol=0.04, rtol=0)


def test_fallback_remains_consistent():
    approximated = False
    for pub in positions(Config(8, 8, 13), range(3), 7):
        analysis = FastAnalysis(pub, node_limit=1, rng=np.random.default_rng(8))
        approximated |= not analysis.exact
        assert_consistent(pub, analysis.sample_many(30))
        assert np.all((analysis.probs >= -1e-12) & (analysis.probs <= 1 + 1e-12))
    assert approximated


@pytest.mark.parametrize("cfg, board, scores", [
    (Config(1, 2, 0), [1, 0], [0, 0]),  # impossible number without unknowns
    (Config(1, 2, 1), [0, 9], [1, 0]),  # adjacent claimed mine contradicts zero
    (Config(1, 3, 1), [0, -1, 1], [0, 0]),  # contradictory constraints
    (Config(1, 3, 0), [1, -1, -1], [0, 0]),  # local/global conflict
    (Config(1, 3, 2), [-1, 0, 0], [0, 0]),  # too few unknown cells
    (Config(1, 3, 1), [9, -1, -1], [0, 0]),  # score/claimed mismatch
    (Config(1, 2, 0), [11, -1], [0, 0]),
])
def test_inconsistent_boards_raise(cfg, board, scores):
    with pytest.raises(ValueError):
        FastAnalysis(PublicState(cfg, np.array(board, dtype=np.int8), scores, 0))


def test_reproducibility_and_empty_batch():
    pub = next(positions(Config(5, 5, 6), [3], 5))
    first = FastAnalysis(pub, rng=np.random.default_rng(8))
    second = FastAnalysis(pub, rng=np.random.default_rng(8))
    np.testing.assert_array_equal(first.sample_many(100), second.sample_many(100))
    assert first.sample_many(0).shape == (0, pub.cfg.n)
    with pytest.raises(ValueError):
        first.sample_many(-1)
    with pytest.raises(ValueError):
        FastAnalysis(pub, node_limit=0)
