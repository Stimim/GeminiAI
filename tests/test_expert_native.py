"""Native decision scoring matches Python, including outcome/RNG ordering."""
import copy

import numpy as np
import pytest

from msflags.agents import SolverAgent
from msflags.expert import ExpertAgent
from msflags.fastsolve import FastAnalysis
from msflags.game import Config, Game, UNKNOWN


@pytest.mark.parametrize("cfg", [Config(4, 5, 5), Config(6, 6, 8),
                                 Config(8, 8, 13), Config(10, 12, 24),
                                 Config(16, 16, 51)])
@pytest.mark.parametrize("seed", range(10))
def test_midgame_scores_and_rng_match(cfg, seed):
    rng = np.random.default_rng(seed + 30)
    game = Game.random(cfg, rng)
    for turn in range(3 + seed % 8):
        if game.done:
            break
        if turn == 3 and seed % 2:
            legal = np.flatnonzero(game.legal_mask()[cfg.n:])
            action = cfg.n + int(rng.choice(legal)) if len(legal) else int(rng.choice(np.flatnonzero(game.board == UNKNOWN)))
        else:
            action = int(rng.choice(np.flatnonzero(game.board == UNKNOWN)))
        game.step(action)
    # Scoring is well-defined even if the random prefix ended the match.
    options = dict(samples=24, alpha=0.7 + seed % 3, cap=0.8 + seed % 2 * 0.1,
                   bomb_reserve=1.2, interior_candidates=5 + seed % 4,
                   bomb_candidates=seed % 5)
    python = ExpertAgent(np.random.default_rng(1100 + seed), native=False, **options)
    native = ExpertAgent(np.random.default_rng(1100 + seed), native=True, **options)
    a, v, p = python.scores(game.public())
    b, w, q = native.scores(game.public())
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(p, q)
    np.testing.assert_allclose(v, w, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(python.rng.integers(2**63, size=20),
                                  native.rng.integers(2**63, size=20))


def test_seeded_layouts_and_initial_zero_floods():
    game = Game.random(Config(8, 8, 10), np.random.default_rng(5))
    pub = game.public()
    rng = np.random.default_rng(70)
    first = FastAnalysis(pub, rng=rng)
    second = FastAnalysis(pub, rng=copy.deepcopy(np.random.default_rng(70)))
    np.testing.assert_array_equal(first.sample_many(48), second.sample_many(48))
    python = ExpertAgent(np.random.default_rng(71), native=False)
    native = ExpertAgent(np.random.default_rng(71))
    a, v, _ = python.scores(pub)
    b, w, _ = native.scores(pub)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(v, w, atol=1e-9, rtol=0)


def test_full_expert_vs_solver_game():
    game = Game.random(Config(8, 8, 13), np.random.default_rng(87))
    players = [ExpertAgent(np.random.default_rng(88)), SolverAgent(np.random.default_rng(89))]
    for _ in range(game.cfg.n):
        if game.done:
            break
        action = players[game.to_move].act(game.public())
        assert game.legal_mask()[action]
        game.step(action)
    assert game.done
    assert game.winner in (0, 1)


def test_custom_analysis_remains_on_python_path():
    class CustomAnalysis(FastAnalysis):
        calls = 0

        def __init__(self, *args, **kwargs):
            type(self).calls += 1
            super().__init__(*args, **kwargs)

    pub = Game.random(Config(4, 4, 4), np.random.default_rng(1)).public()
    ExpertAgent(np.random.default_rng(2), analysis=CustomAnalysis).scores(pub)
    assert CustomAnalysis.calls > 1


def test_native_scores_on_global_count_regression_board():
    rng = np.random.default_rng(1376)
    game = Game.random(Config(8, 8, 13), rng)
    opened = rng.choice(np.flatnonzero(~game.mines), 12, replace=False)
    game.board[opened] = game.counts[opened]
    python = ExpertAgent(np.random.default_rng(11376), native=False)
    native = ExpertAgent(np.random.default_rng(11376), native=True)
    actions, values, probs = python.scores(game.public())
    actual_actions, actual_values, actual_probs = native.scores(game.public())
    np.testing.assert_array_equal(actions, actual_actions)
    np.testing.assert_array_equal(probs, actual_probs)
    np.testing.assert_allclose(values, actual_values, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(python.rng.integers(2**63, size=20),
                                  native.rng.integers(2**63, size=20))
