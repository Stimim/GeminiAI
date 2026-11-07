import numpy as np
import pytest

from msflags.game import Config, Game
from msflags.mcts import Search
from msflags.agents import MCTSAgent
from msflags.train import run_searches


def evaluator(states, cfg):
    policies = []
    for board, scores, player, bombs in states:
        legal = np.concatenate((board == -1, (board == -1) & bool(bombs[player])))
        policies.append(legal / legal.sum())
    return np.asarray(policies), np.full(len(states), .125), np.zeros((len(states), cfg.n))


def game():
    cfg = Config(1, 9, 3)
    return Game(cfg, np.array([True, False, False, False, True, False, False, False, True]))


def run(batch_size, mode='puct', sims=32):
    g = game()
    s = Search(g.public(), sims, search=mode, rng=np.random.default_rng(17))
    s.analysis.sample = lambda rng: g.mines
    run_searches([s], evaluator, g.cfg, batch_size)
    return s


def test_batch_one_matches_sequential_exactly():
    expected = run(1)
    g = game()
    sequential = Search(g.public(), 32, rng=np.random.default_rng(17))
    sequential.analysis.sample = lambda rng: g.mines
    while (leaf := sequential.next_leaf()) is not None:
        p, v, _ = evaluator([leaf], g.cfg)
        sequential.provide(p[0], v[0])
    np.testing.assert_array_equal(expected.root.N, sequential.root.N)
    np.testing.assert_array_equal(expected.root.W, sequential.root.W)
    assert expected.best_action() == sequential.best_action()


@pytest.mark.parametrize('mode', ['puct', 'gumbel'])
@pytest.mark.parametrize('batch_size', [1, 4, 8])
@pytest.mark.parametrize('sims', [1, 2, 16, 32, 100])
def test_budgets_reservations_and_masks(mode, batch_size, sims):
    s = run(batch_size, mode, sims)
    assert s.finished and s.done_sims == sims
    assert s.root.N.sum() == sims - 1
    stack = [s.root]
    while stack:
        node = stack.pop()
        if node.actions is not None:
            np.testing.assert_array_equal(node.R, np.zeros_like(node.R))
        stack.extend(node.children.values())
    pi = s.improved_policy()
    assert pi.sum() == pytest.approx(1.)
    assert not pi[~s.pub.legal_mask()].any()
    assert s.pub.legal_mask()[s.best_action()]


def test_virtual_loss_does_not_mutate_real_statistics():
    g = game()
    s = Search(g.public(), 32, rng=np.random.default_rng(3))
    s.analysis.sample = lambda rng: g.mines
    leaves = s.next_leaves(8)
    p, v, _ = evaluator(leaves, g.cfg)
    s.provide_batch(p, v)
    n, w = s.root.N.copy(), s.root.W.copy()
    leaves = s.next_leaves(4)
    assert leaves and s.root.R.sum() == len(leaves)
    np.testing.assert_array_equal(n, s.root.N)
    np.testing.assert_array_equal(w, s.root.W)
    p, v, _ = evaluator(leaves, g.cfg)
    s.provide_batch(p, v)
    assert s.root.R.sum() == 0
    assert s.root.N.sum() == len(leaves)


def test_reuse_matches_public_inventory_not_only_board():
    s = run(1, sims=100)
    g = game()
    action = s.best_action()
    g.step(action)
    expected = s.child_root(action, g.board)
    assert s.matching_root(g.public()) is expected
    public = g.public()
    public.bombs = [0, 0]
    assert s.matching_root(public) is None
    assert np.all(s.pub.board == -1)


def test_agent_reuses_observed_child_and_falls_back():
    g = game()
    agent = MCTSAgent(evaluator, sims=32, rng=np.random.default_rng(3))
    first = agent.search(g.public())
    action = first.best_action()
    g.step(action)
    child = first.child_root(action, g.board)
    second = agent.search(g.public())
    if child is not None and child.actions is not None:
        assert second.root is child
    other = Game.random(g.cfg, np.random.default_rng(77))
    assert agent.search(other.public()).root is not second.root


@pytest.mark.parametrize('sims', [16, 32, 100])
def test_gumbel_halves_candidates_only_after_equal_phase_visits(sims):
    g = game()
    s = Search(g.public(), sims, search='gumbel', gumbel_k=16,
               rng=np.random.default_rng(8))
    s.analysis.sample = lambda rng: g.mines
    previous_candidates = None
    phase_counts = {}
    select = s._select_gumbel_root

    def traced_select():
        nonlocal previous_candidates, phase_counts
        i = select()
        if i is not None:
            candidates = tuple(s._g_candidates)
            if candidates != previous_candidates:
                if previous_candidates is not None and len(previous_candidates) > 1:
                    assert len(set(phase_counts.values())) == 1
                    assert len(candidates) == max(1, len(previous_candidates) // 2)
                previous_candidates = candidates
                phase_counts = {int(action): 0 for action in candidates}
            phase_counts[i] += 1
        return i

    s._select_gumbel_root = traced_select
    run_searches([s], evaluator, g.cfg)
    assert s.done_sims == sims
    assert s.root.N.sum() == sims - 1


def test_completed_q_masks_revealed_cells_and_consumed_bomb_inventory():
    g = game()
    g.step(1)
    g.bombs[g.to_move] = 0
    s = Search(g.public(), 16, search='gumbel', rng=np.random.default_rng(27))
    s.analysis.sample = lambda rng: g.mines
    run_searches([s], evaluator, g.cfg, 4)
    pi = s.improved_policy()
    legal = g.public().legal_mask()
    assert not legal.all() and not legal[g.cfg.n:].any()
    assert pi.sum() == pytest.approx(1.)
    assert np.all(pi[~legal] == 0)
