import numpy as np
import pytest

from msflags.compare import paired_games
from msflags.game import Config


def first_open(states, cfg):
    policy = np.zeros((len(states), 2 * cfg.n), np.float32)
    for row, (board, _, _, _) in enumerate(states):
        policy[row, np.flatnonzero(board == -1)[0]] = 1.
    return policy, np.zeros(len(states)), np.zeros((len(states), cfg.n))


@pytest.mark.parametrize('mode', ['greedy', 'mcts'])
def test_paired_comparison_repeats_layouts_with_both_seats_and_independent_batches(mode):
    kwargs = dict(games=6, seed=718, mode=mode, sims=2)
    reference = paired_games([first_open, first_open], Config(2, 2, 1),
                             concurrent_games=1, **kwargs)
    batched = paired_games([first_open, first_open], Config(2, 2, 1),
                           concurrent_games=4, **kwargs)
    assert reference['wins'] == batched['wins'] == 3
    assert reference['layouts'] == 3
    assert reference['pair_splits'] == batched['pair_splits'] == 3
    assert reference['paired_bootstrap_95'] == batched['paired_bootstrap_95'] == [.5, .5]


def test_paired_comparison_rejects_incomplete_seat_pairs():
    with pytest.raises(ValueError, match='even'):
        paired_games([first_open, first_open], Config(2, 2, 1), games=3, seed=1)
