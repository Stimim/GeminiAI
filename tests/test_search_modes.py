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


def test_reuse_and_search_are_invariant_to_unobserved_real_mines():
    first = game()
    second = Game(first.cfg, np.array([True, False, False, False, False, True, False, False, True]))
    for actual in (first, second):
        actual.step(0)
        actual.step(1)
    np.testing.assert_array_equal(first.board, second.board)
    agents = [MCTSAgent(evaluator, sims=32, rng=np.random.default_rng(37)) for _ in range(2)]
    searches = [agent.search(actual.public()) for agent, actual in zip(agents, (first, second))]
    np.testing.assert_array_equal(searches[0].root.N, searches[1].root.N)
    np.testing.assert_array_equal(searches[0].root.W, searches[1].root.W)
    assert searches[0].best_action() == searches[1].best_action()
    # Opposite real layouts do not change the public-only descendant lookup.
    observed = first.public()
    assert searches[0].matching_root(observed) is searches[0].root
    assert searches[1].matching_root(observed) is searches[1].root


def test_noise_free_single_candidate_single_simulation_is_exactly_greedy():
    g = game()
    policy = np.arange(1, 2 * g.cfg.n + 1, dtype=float)
    policy /= policy.sum()
    s = Search(g.public(), 1, search='gumbel', gumbel_k=1, gumbel_noise=False,
               rng=np.random.default_rng(99))
    assert s.next_leaf() is not None
    s.provide(policy, .3)
    assert s.best_action() == int(policy.argmax())
    np.testing.assert_allclose(s.improved_policy(), policy)
    np.testing.assert_array_equal(s._g_noise, np.zeros_like(s._g_noise))


@pytest.mark.parametrize('player', [0, 1])
@pytest.mark.parametrize('action', [0, 1, 9])
def test_gumbel_completed_q_is_in_root_mover_perspective(player, action):
    g = game()
    g.to_move = player
    s = Search(g.public(), 2, search='gumbel', gumbel_k=1, gumbel_noise=False)
    s.analysis.sample = lambda rng: g.mines
    policy = np.zeros(2 * g.cfg.n)
    policy[action] = 1.
    s.next_leaf()
    s.provide(policy, 0.)
    leaf = s.next_leaf()
    assert leaf is not None
    s.provide(np.ones(2 * g.cfg.n), .6)
    i = int(np.flatnonzero(s.root.actions == action)[0])
    assert s.root.N[i] == 1
    assert s.root.W[i] == pytest.approx(.6 if action == 0 else -.6)


def test_completed_q_mixture_normalization_and_sigma_constants():
    g = game()
    s = Search(g.public(), 4, search='gumbel', gumbel_noise=False)
    s.next_leaf()
    s.provide(np.ones(2 * g.cfg.n), .2)
    s.root.N[:2] = [2, 1]
    s.root.W[:2] = [1., -.5]
    # Visited Q=.5,-.5; prior-weighted mean=0; mixed value=(.2+3*0)/4=.05.
    # Range [-.5,.5] maps to [0,1]; unvisited normalized Q=.55.
    sigma = s._sigma_q()
    assert sigma[0] == pytest.approx(5.2)  # (50 + max_visits=2) * .1
    assert sigma[1] == 0
    np.testing.assert_allclose(sigma[2:], 5.2 * .55)
