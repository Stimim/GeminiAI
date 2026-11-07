from argparse import Namespace

import numpy as np

from msflags import train
from msflags.game import PublicState


def tiny_args(tmp_path):
    return Namespace(size=2, mines=1, games=2, sims=2, seed=123,
                     c_puct=1.5, fpu_reduction=0.2, dirichlet_alpha=None,
                     dirichlet_frac=0.25, temperature=0., temperature_moves=0,
                     concurrent_games=2, output=str(tmp_path),
                     value_validation_games=2)


class BombEvaluator:
    def __call__(self, states, cfg):
        policies = []
        for board, scores, player, bombs in states:
            legal = PublicState(cfg, board, list(scores), int(player), bombs=list(bombs)).legal_mask()
            policy = np.zeros(2 * cfg.n, np.float32)
            # Every bomb covers the entire tiny board and wins immediately.
            policy[np.flatnonzero(legal[cfg.n:])[0] + cfg.n] = 1.
            policies.append(policy)
        return np.stack(policies), np.ones(len(states), np.float32), np.zeros((len(states), cfg.n))


def test_action_metrics_include_terminal_bombs_and_every_game(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    records = []
    monkeypatch.setattr(train, 'log', lambda args, event, **fields: records.append((event, fields)))
    data = train.selfplay_batch(args, BombEvaluator(), 123)
    event, metrics = records[-1]
    assert event == 'selfplay_games'
    assert metrics['split'] == 'replay'
    assert metrics['positions'] == len(data['board']) == 2
    assert metrics['action_counts'] == dict(bomb=2, safe=0, mine=0)
    assert metrics['game_statistics'] == [dict(moves=1, bomb=1, safe=0, mine=0)] * 2
    np.testing.assert_array_equal(data['value'], [1., 1.])


def test_value_validation_uses_separate_whole_games_without_mutating_training_args(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    args.games = 96
    records = []
    monkeypatch.setattr(train, 'log', lambda args, event, **fields: records.append((event, fields)))
    train.validate_value(args, BombEvaluator(), 1_000_000_123)
    assert args.games == 96
    assert not hasattr(args, 'generation_split')
    generation, validation = records
    assert generation[1]['split'] == 'validation'
    assert generation[1]['games'] == validation[1]['games'] == 2
    assert validation[0] == 'value_validation'
    metrics = validation[1]
    assert metrics['seed'] == 1_000_000_123
    assert metrics['replay_inserted'] is False
    assert metrics['mse'] == 0.
    assert metrics['accuracy'] == 1.
    assert metrics['zero_mse'] == 1.
    assert metrics['score_bands'][0]['positions'] == 2
    assert metrics['score_bands'][0]['mse'] == 0.


def test_action_metrics_distinguish_safe_and_mine_openings(tmp_path, monkeypatch):
    from msflags.game import Game
    args = tiny_args(tmp_path)
    args.size, args.mines, args.games = 3, 3, 1
    layout = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    monkeypatch.setattr(train.Game, 'random', lambda cfg, rng: Game(cfg, layout))
    records = []
    monkeypatch.setattr(train, 'log', lambda args, event, **fields: records.append(fields))

    def evaluator(states, cfg):
        policies = np.zeros((len(states), 2 * cfg.n), np.float32)
        for row, (board, _, _, _) in enumerate(states):
            action = 3 if (board == -1).all() else np.flatnonzero(board == -1)[0]
            policies[row, action] = 1.
        return policies, np.zeros(len(states)), np.zeros((len(states), cfg.n))

    data = train.selfplay_batch(args, evaluator, 123)
    assert len(data['board']) == 3
    assert records[-1]['action_counts'] == dict(bomb=0, safe=1, mine=2)
    assert records[-1]['game_statistics'] == [dict(moves=3, bomb=0, safe=1, mine=2)]


def test_value_validation_defaults_and_disable_switch():
    args = train.parser().parse_args(['selfplay'])
    assert args.value_validation_every == 5
    assert args.value_validation_games == 16
    disabled = train.parser().parse_args(['selfplay', '--value-validation-every', '0'])
    assert disabled.value_validation_every == 0
