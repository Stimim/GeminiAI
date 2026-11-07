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
    assert metrics['policy_target_entropy'] == 0.
    assert metrics['policy_target_effective_actions'] == 1.
    assert metrics['played_not_prior_argmax'] == 0
    assert metrics['played_not_prior_argmax_fraction'] == 0.
    assert metrics['mean_game_length'] == 1.
    assert metrics['bombs_per_game'] == 1.
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


def test_clean_prior_argmax_survives_root_noise():
    from msflags.game import Config, Game
    from msflags.mcts import Node, Search

    cfg = Config(2, 2, 1)
    game = Game.random(cfg, np.random.default_rng(123))
    root = Node(game.to_move)
    policy = np.zeros(2 * cfg.n)
    policy[0], policy[1] = .6, .4
    root.expand(game.public().legal_mask(), policy, 0.)

    class Noise:
        def dirichlet(self, alpha):
            noise = np.zeros(len(alpha))
            noise[-1] = 1.
            return noise

    search = Search(game.public(), 2, root=root, rng=Noise(),
                    dirichlet_alpha=.03, dirichlet_frac=1.)
    assert root.prior_argmax == 0
    assert root.actions[root.prior.argmax()] != root.prior_argmax
    # Reusing an expanded root must not overwrite the clean network choice.
    Search(game.public(), 2, root=search.root, rng=Noise(),
           dirichlet_alpha=.03, dirichlet_frac=1.)
    assert root.prior_argmax == 0


def test_exploration_metrics_measure_target_entropy_and_sampled_actions(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    args.sims, args.games = 1, 32
    args.temperature, args.temperature_moves = 1., 60
    records = []
    monkeypatch.setattr(train, 'log', lambda args, event, **fields: records.append(fields))

    class SplitBombEvaluator(BombEvaluator):
        def __call__(self, states, cfg):
            policy, values, mines = super().__call__(states, cfg)
            policy[:, cfg.n:cfg.n + 2] = .5
            return policy, values, mines

    train.selfplay_batch(args, SplitBombEvaluator(), 123)
    metrics = records[-1]
    np.testing.assert_allclose(metrics['policy_target_entropy'], np.log(2))
    np.testing.assert_allclose(metrics['policy_target_effective_actions'], 2.)
    assert 0 < metrics['played_not_prior_argmax'] < args.games
    assert metrics['played_not_prior_argmax_fraction'] == metrics['played_not_prior_argmax'] / args.games
