"""Search values are player-relative, not alternating-depth-relative.

These tiny fixed layouts isolate backup and public-outcome identity without a
network, stochastic solver coverage, or a training process.
"""

import numpy as np
import pytest

from msflags.game import CLAIMED, UNKNOWN, Config, Game
from msflags.mcts import Node, Search


def make_game(player, cells=(0, 3, 6), width=7):
    cfg = Config(1, width, len(cells))
    mines = np.zeros(cfg.n, dtype=bool)
    mines[list(cells)] = True
    game = Game(cfg, mines)
    game.to_move = player
    return game


def policy_for(cfg, action):
    policy = np.zeros(2 * cfg.n)
    policy[action] = 1.0
    return policy


def edge_index(node, action):
    return int(np.flatnonzero(node.actions == action)[0])


def start_search(game, sims, action, **kwargs):
    search = Search(game.public(), sims, rng=np.random.default_rng(0), **kwargs)
    search.analysis.sample = lambda rng: game.mines
    assert search.next_leaf() is not None
    search.provide(policy_for(game.cfg, action), 0.0)
    return search


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("captures", [False, True])
def test_nonterminal_bomb_passes_even_when_it_claims_a_mine(player, captures):
    game = make_game(player)
    action = game.cfg.n + (0 if captures else 6)
    if not captures:
        # Keep all three mines outside the clipped bomb window.
        game = make_game(player, cells=(0, 1, 2))
    search = start_search(game, 2, action)
    leaf = search.next_leaf()
    assert leaf is not None
    board, scores, to_move, bombs = leaf
    assert to_move == 1 - player
    assert scores[player] == int(captures)
    assert scores[1 - player] == 0
    assert bombs[player] == 0 and bombs[1 - player] == 1
    assert np.count_nonzero(board == CLAIMED[player]) == int(captures)
    search.provide(np.ones(2 * game.cfg.n), 0.625)
    i = edge_index(search.root, action)
    assert search.root.N[i] == 1
    assert search.root.W[i] == pytest.approx(-0.625)
    assert search.next_leaf() is None
    # Simulation state must not consume a real bomb or reveal the public board.
    assert game.bombs == search.pub.bombs == [1, 1]
    assert np.all(search.pub.board == UNKNOWN)


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("bomb", [False, True])
def test_winning_action_backs_up_exact_value_without_evaluating_terminal_leaf(player, bomb):
    game = make_game(player, cells=(0, 1, 6))
    if bomb:
        action = game.cfg.n
    else:
        game.step(0)
        action = 1
    search = start_search(game, 2, action)
    # The network's root value is deliberately unrelated to the winning result.
    search.root.value = -0.95
    assert search.next_leaf() is None
    assert search.finished and search.done_sims == 2
    i = edge_index(search.root, action)
    assert search.root.N[i] == 1
    assert search.root.W[i] == 1.0
    assert search._pending is None
    terminal = next(iter(search.root.children.values()))
    assert terminal.to_move == player
    assert terminal.actions is None

    observed = Game.determinize(game.public(), game.mines)
    observed.step(action)
    assert observed.winner == observed.to_move == player
    assert not observed.legal_mask().any()
    assert search.child_root(action, observed.board) is terminal


@pytest.mark.parametrize("player", [0, 1])
def test_terminal_backup_changes_sign_at_safe_turn_but_not_extra_turn(player):
    game = make_game(player)
    search = start_search(game, 4, 1, fpu_reduction=2.0)
    leaf = search.next_leaf()  # Safe 1: opponent moves next.
    assert leaf[2] == 1 - player
    after_safe = search.child_root(1, leaf[0])
    search.provide(policy_for(game.cfg, 0), 0.4)
    leaf = search.next_leaf()  # Mine 0: opponent keeps the move.
    assert leaf[2] == 1 - player
    after_mine = next(iter(after_safe.children.values()))
    search.provide(policy_for(game.cfg, 3), -0.25)
    assert search.next_leaf() is None  # Mine 3 wins for the opponent.
    root_i = edge_index(search.root, 1)
    safe_i = edge_index(after_safe, 0)
    mine_i = edge_index(after_mine, 3)
    assert search.root.N[root_i] == 3
    assert search.root.W[root_i] == pytest.approx(-0.4 + 0.25 - 1.0)
    assert after_safe.N[safe_i] == 2
    assert after_safe.W[safe_i] == pytest.approx(-0.25 + 1.0)
    assert after_mine.N[mine_i] == 1
    assert after_mine.W[mine_i] == 1.0


@pytest.mark.parametrize("player", [0, 1])
def test_chance_outcomes_share_action_statistics_not_child_values(player):
    game = make_game(player)
    layouts = [make_game(player, cells=cells).mines
               for cells in [(0, 3, 6), (1, 3, 6), (2, 3, 6)]]
    search = start_search(game, 4, 0, fpu_reduction=2.0)
    samples = iter(layouts)
    sampled = []

    def sample(rng):
        result = next(samples)
        sampled.append(result)
        return result

    search.analysis.sample = sample
    boards = []
    for expected_player, value in [(player, 0.6), (1 - player, 0.2), (1 - player, -0.4)]:
        leaf = search.next_leaf()
        assert leaf is not None and leaf[2] == expected_player
        boards.append(leaf[0].copy())
        search.provide(np.ones(2 * game.cfg.n), value)
    assert search.next_leaf() is None
    assert len(sampled) == 3  # Exactly one root determinization per rollout.
    assert boards[0][0] == CLAIMED[player]
    assert boards[1][0] == 1
    assert boards[2][0] == 0 and boards[2][1] == 1  # Distinct flood outcome.
    assert len(search.root.children) == 3
    children = [search.child_root(0, board) for board in boards]
    assert len({id(child) for child in children}) == 3
    assert [child.value for child in children] == [0.6, 0.2, -0.4]
    i = edge_index(search.root, 0)
    assert search.root.N[i] == 3
    assert search.root.W[i] == pytest.approx(0.6 - 0.2 + 0.4)
    assert np.all(search.pub.board == UNKNOWN)
    assert search.pub.scores == [0, 0]


@pytest.mark.parametrize("player", [0, 1])
def test_bomb_chance_branches_mix_terminal_win_and_opponent_leaf_values(player):
    game = make_game(player)
    action = game.cfg.n
    search = start_search(game, 4, action, fpu_reduction=2.0)
    layouts = iter([make_game(player, cells=cells).mines
                    for cells in [(0, 1, 6), (0, 3, 6), (3, 4, 6)]])
    search.analysis.sample = lambda rng: next(layouts)
    leaf = search.next_leaf()  # Winning two-mine branch backs up without a leaf.
    assert search.done_sims == 2
    assert leaf[2] == 1 - player and leaf[1][player] == 1
    assert leaf[3][player] == 0
    search.provide(np.ones(2 * game.cfg.n), 0.6)
    leaf = search.next_leaf()
    assert leaf[2] == 1 - player and leaf[1] == [0, 0]
    search.provide(np.ones(2 * game.cfg.n), -0.2)
    assert search.next_leaf() is None
    assert len(search.root.children) == 3
    terminal = [child for child in search.root.children.values() if child.actions is None]
    assert len(terminal) == 1 and terminal[0].to_move == player
    i = edge_index(search.root, action)
    assert search.root.N[i] == 3
    assert search.root.W[i] == pytest.approx(1.0 - 0.6 + 0.2)


@pytest.mark.parametrize("player", [0, 1])
def test_same_public_outcome_reuses_child_across_hidden_layouts(player):
    game = make_game(player, cells=(0, 4, 8), width=9)
    search = start_search(game, 3, 0, fpu_reduction=2.0)
    layouts = iter([game.mines, make_game(player, cells=(0, 5, 8), width=9).mines])
    search.analysis.sample = lambda rng: next(layouts)
    first = search.next_leaf()
    assert first[2] == player and first[0][0] == CLAIMED[player]
    child = search.child_root(0, first[0])
    search.provide(policy_for(game.cfg, 1), 0.3)
    second = search.next_leaf()
    assert second[2] == 1 - player and second[0][1] == 1
    search.provide(np.ones(2 * game.cfg.n), -0.7)
    assert search.next_leaf() is None
    assert len(search.root.children) == 1
    assert next(iter(search.root.children.values())) is child
    assert len(child.children) == 1
    assert search.root.W[edge_index(search.root, 0)] == pytest.approx(1.0)
    assert child.W[edge_index(child, 1)] == pytest.approx(0.7)


@pytest.mark.parametrize("player", [0, 1])
def test_determinize_preserves_public_perspective_and_owns_mutable_state(player):
    game = make_game(player)
    game.step(0)  # Claim one mine without passing.
    game.step(1)  # Then pass on a numbered safe cell.
    pub = game.public()
    sampled = Game.determinize(pub, game.mines)
    assert sampled.to_move == 1 - player
    assert sampled.board[0] == CLAIMED[player]
    assert sampled.mines[0]
    assert sampled.scores == pub.scores
    assert sampled.bombs == pub.bombs
    np.testing.assert_array_equal(sampled.counts, game.counts)
    sampled.step(game.cfg.n + 6)  # Opponent claims one mine and passes back.
    assert sampled.to_move == player
    assert sampled.scores == [1, 1]
    assert sampled.bombs[1 - player] == 0
    assert pub.bombs == game.bombs == [1, 1]
    assert pub.board[6] == game.board[6] == UNKNOWN
    assert pub.scores[player] == 1 and pub.scores[1 - player] == 0


@pytest.mark.parametrize("player", [0, 1])
def test_node_initializes_legal_priors_and_player_relative_fpu(player):
    node = Node(player)
    legal = np.array([False, True, False, True])
    node.expand(legal, np.array([100.0, 2.0, 100.0, 6.0]), -0.6)
    np.testing.assert_array_equal(node.actions, [1, 3])
    np.testing.assert_allclose(node.prior, [0.25, 0.75])
    np.testing.assert_array_equal(node.N, [0, 0])
    np.testing.assert_array_equal(node.W, [0, 0])
    assert node.value == -0.6

    game = make_game(player)
    search = Search(game.public(), 1, c_puct=0.0, fpu_reduction=0.2)
    node.N[:] = [1, 0]
    node.W[:] = [-0.85, 0]
    assert search._select(node) == 1  # Unvisited FPU = -0.8, not zero or +0.4.
    node.W[0] = -0.75
    assert search._select(node) == 0


@pytest.mark.parametrize("player", [0, 1])
def test_selection_uses_mean_q_and_exact_puct_visit_denominator(player):
    game = make_game(player)
    search = Search(game.public(), 1, c_puct=0.0)
    node = Node(player)
    node.expand(np.ones(2, dtype=bool), np.array([0.25, 0.75]), 0.0)
    node.N[:] = [4, 1]
    node.W[:] = [2.0, 0.75]
    assert search._select(node) == 1  # Q=.75 beats .5 despite smaller W.
    node.W[:] = [2.0, 0.5]  # Equal Q, exploration must settle the choice.
    search.c_puct = 1.5
    assert search._select(node) == 1
    node.prior[:] = [0.9, 0.1]
    assert search._select(node) == 0


@pytest.mark.parametrize("player", [0, 1])
def test_reused_root_retains_q_prior_and_visits_but_gets_new_budget(player):
    game = make_game(player)
    search = start_search(game, 3, 0, fpu_reduction=2.0)
    leaf = search.next_leaf()
    child = search.child_root(0, leaf[0])
    search.provide(policy_for(game.cfg, 3), 0.4)
    assert search.next_leaf() is None  # Second mine wins, seeding child stats.
    game.step(0)
    i = edge_index(child, 3)
    old_n, old_w, old_prior = child.N.copy(), child.W.copy(), child.prior.copy()
    reused = Search(game.public(), 2, root=child, fpu_reduction=2.0,
                    rng=np.random.default_rng(1))
    reused.analysis.sample = lambda rng: game.mines
    assert reused.root is child
    assert reused.done_sims == 0 and not reused.finished
    assert reused.root.to_move == player and reused.root.value == 0.4
    np.testing.assert_array_equal(reused.root.N, old_n)
    np.testing.assert_array_equal(reused.root.W, old_w)
    np.testing.assert_array_equal(reused.root.prior, old_prior)
    assert reused.next_leaf() is None  # Reused root needs no network expansion.
    assert reused.done_sims == 2
    assert reused.root.N[i] == old_n[i] + 2
    assert reused.root.W[i] == old_w[i] + 2
    assert reused.best_action() == 3
    assert reused.visit_policy()[3] == 1.0


@pytest.mark.parametrize("player", [0, 1])
def test_unexpanded_child_starts_fresh_and_single_simulation_returns_prior(player):
    game = make_game(player)
    unexpanded = Node(player)
    search = Search(game.public(), 1, root=unexpanded)
    assert search.root is not unexpanded
    assert search.next_leaf() is not None
    policy = policy_for(game.cfg, 3)
    search.provide(policy, -0.9)
    assert search.finished and search.root.value == -0.9
    assert search.root.N.sum() == 0
    np.testing.assert_array_equal(search.visit_policy(), policy)
    assert search.best_action() == 3
    assert search.next_leaf() is None


@pytest.mark.parametrize("player", [0, 1])
def test_root_sampler_keeps_claimed_mines_and_revealed_number_constraints(player):
    game = make_game(player)
    game.step(0)
    game.step(1)
    pub = game.public()
    search = Search(pub, 2, rng=np.random.default_rng(8))
    revealed = (pub.board >= 0) & (pub.board <= 8)
    for _ in range(8):
        layout = search.analysis.sample(search.rng)
        assert layout.dtype == np.bool_
        assert layout.sum() == game.cfg.mines
        assert layout[0] and not layout[1] and not layout[2]
        sampled = Game.determinize(pub, layout)
        np.testing.assert_array_equal(sampled.counts[revealed], pub.board[revealed])
        assert sampled.to_move == 1 - player
        assert sampled.scores[player] == 1 and sampled.scores[1 - player] == 0
    np.testing.assert_array_equal(pub.board, game.board)
