import numpy as np
import pytest

from msflags.game import Config, Game, UNKNOWN, encode
from msflags.mcts import Search
from msflags.train import augment, dihedral, wilson


@pytest.mark.parametrize('symmetry', range(8))
def test_augmentation_transforms_board_policy_and_mines_together(symmetry):
    cfg = Config(3, 3, 3)
    board = np.array([-1, 1, 2, 9, 3, -1, 10, 2, 1], dtype=np.int8)
    x = encode(board, [1, 1], 1, cfg)[None]
    policy = np.arange(18, dtype=np.float32)[None]
    mine = (np.arange(9, dtype=np.float32) / 10)[None]
    actual_x, actual_pi, actual_mine = augment(x, policy, mine, symmetry)
    transformed = dihedral(board.reshape(3, 3), symmetry).reshape(-1)
    np.testing.assert_allclose(actual_x[0], encode(transformed, [1, 1], 1, cfg))
    indices = dihedral(np.arange(9).reshape(3, 3), symmetry).reshape(-1)
    np.testing.assert_array_equal(actual_pi[0, :cfg.n], policy[0, indices])
    np.testing.assert_array_equal(actual_pi[0, cfg.n:], policy[0, cfg.n + indices])
    np.testing.assert_array_equal(actual_mine[0], mine[0, indices])


@pytest.mark.parametrize('player', [0, 1])
@pytest.mark.parametrize('hit', [False, True])
def test_search_backup_respects_extra_turn_not_depth_parity(player, hit):
    cfg = Config(3, 3, 3)
    layout = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    game = Game(cfg, layout)
    game.to_move = player
    action = 0 if hit else 3
    search = Search(game.public(), 2, fpu_reduction=0)
    search.analysis.sample = lambda rng: layout
    policy = np.zeros(2 * cfg.n)
    policy[action] = 1
    search.next_leaf()
    search.provide(policy, 0.)
    leaf = search.next_leaf()
    assert leaf is not None
    assert leaf[2] == (player if hit else 1-player)
    search.provide(np.ones(2 * cfg.n), 0.75)
    index = int(np.flatnonzero(search.root.actions == action)[0])
    assert search.root.W[index] == pytest.approx(0.75 if hit else -0.75)
    assert search.root.N[index] == 1
    assert search.finished


@pytest.mark.parametrize('player', [0, 1])
def test_terminal_backup_uses_winner_perspective(player):
    cfg = Config(2, 2, 1)
    game = Game(cfg, np.array([1, 0, 0, 0], dtype=bool))
    game.to_move = player
    search = Search(game.public(), 2)
    search.analysis.sample = lambda rng: game.mines
    search.next_leaf()
    search.provide(np.array([1., 0., 0., 0., 0., 0., 0., 0.]), -0.8)
    assert search.next_leaf() is None
    assert search.root.W[0] == 1.


def test_confidence_interval_remains_finite_at_extreme_results():
    for wins in [0, 100]:
        low, high = wilson(wins, 100)
        assert 0 <= low <= high <= 1
        assert high - low > 0


def test_bootstrap_skips_approximate_labels_without_changing_play(monkeypatch):
    from msflags import train
    real_analysis = train.Analysis
    job = (Config(4, 4, 3), 12, 0.1)
    expected = train.bootstrap_game(job)
    calls = 0

    def with_first_position_inexact(*args, **kwargs):
        nonlocal calls
        result = real_analysis(*args, **kwargs)
        if calls == 0:
            result.exact = False
        calls += 1
        return result

    monkeypatch.setattr(train, 'Analysis', with_first_position_inexact)
    actual = train.bootstrap_game(job)
    assert len(actual) == len(expected) - 1
    for observed, reference in zip(actual, expected[1:]):
        np.testing.assert_array_equal(observed.board, reference.board)
        np.testing.assert_array_equal(observed.policy, reference.policy)
        np.testing.assert_array_equal(observed.mine, reference.mine)
        assert observed.value == reference.value


@pytest.mark.parametrize('symmetry', range(8))
def test_bomb_augmentation_transforms_both_action_halves(symmetry):
    cfg = Config(3, 3, 3)
    board = np.full(cfg.n, UNKNOWN, np.int8)
    x = encode(board, [0, 0], 0, cfg, [0, 1])[None]
    policy = np.arange(2 * cfg.n, dtype=np.float32)[None]
    mine = np.arange(cfg.n, dtype=np.float32)[None]
    actual_x, actual_policy, _ = augment(x, policy, mine, symmetry)
    indices = dihedral(np.arange(cfg.n).reshape(3, 3), symmetry).reshape(-1)
    np.testing.assert_array_equal(actual_policy[0, :cfg.n], policy[0, indices])
    np.testing.assert_array_equal(actual_policy[0, cfg.n:], policy[0, cfg.n + indices])
    assert not actual_x[0, 18].any()
    assert actual_x[0, 19].all()


def test_checkpoint_round_trip_preserves_historical_metadata(tmp_path):
    import torch
    from msflags.model import Net, NetConfig, load, save
    net = Net(NetConfig(channels=8, blocks=1)).eval()
    path = tmp_path / 'bomb.pt'
    save(net, path, transferred_from='historical/source.pt', bootstrap_steps=42)
    checkpoint = torch.load(path, weights_only=False)
    assert 'rules' not in checkpoint['net_cfg']
    # Earlier bomb checkpoints recorded this now-obsolete metadata field.
    checkpoint['net_cfg']['rules'] = 'bomb'
    torch.save(checkpoint, path)
    restored, metadata = load(path)
    assert restored.stem[0].in_channels == 20
    assert metadata['transferred_from'] == 'historical/source.pt'
    assert metadata['bootstrap_steps'] == 42
    for key, value in net.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_checkpoint_with_18_input_planes_is_rejected(tmp_path):
    import torch
    from msflags.model import Net, NetConfig, load, save
    path = tmp_path / 'unsupported.pt'
    save(Net(NetConfig(channels=8, blocks=1)), path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint['state']['stem.0.weight'] = checkpoint['state']['stem.0.weight'][:, :18]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='20 input planes'):
        load(path)


def test_network_has_joint_policy_and_masks_spent_bomb():
    import torch
    from msflags.model import Evaluator, Net, NetConfig
    cfg = Config(5, 5, 5)
    game = Game.random(cfg, np.random.default_rng(2))
    torch.manual_seed(0)
    net = Net(NetConfig(channels=8, blocks=1)).eval()
    evaluator = Evaluator(net, 'cpu')
    policy, value, mine = evaluator([(game.board, game.scores, 0, [0, 1])], cfg)
    assert policy.shape == (1, 2 * cfg.n)
    assert value.shape == (1,) and mine.shape == (1, cfg.n)
    assert not policy[0, cfg.n:].any()
    np.testing.assert_allclose(policy.sum(1), 1., rtol=1e-5)
    x = torch.from_numpy(encode(game.board, game.scores, 0, cfg, [1, 0])[None])
    with torch.inference_mode():
        logits, _, _ = net(x)
    assert (logits[:, cfg.n:] > -1e9).all()


def test_evaluator_requires_bomb_availability():
    from msflags.model import Evaluator, Net, NetConfig
    game = Game.random(Config(3, 3, 3), np.random.default_rng(1))
    evaluator = Evaluator(Net(NetConfig(channels=8, blocks=1)), 'cpu')
    with pytest.raises(ValueError):
        evaluator([(game.board, game.scores, game.to_move)], game.cfg)


def test_search_bomb_passes_turn_and_propagates_availability():
    cfg = Config(7, 7, 7)
    mines = np.zeros(cfg.n, bool)
    mines[:7] = True
    game = Game(cfg, mines)
    search = Search(game.public(), 2, fpu_reduction=0)
    search.analysis.sample = lambda rng: mines
    policy = np.zeros(2 * cfg.n)
    policy[cfg.n + 48] = 1
    search.next_leaf()
    search.provide(policy, 0.)
    leaf = search.next_leaf()
    assert leaf[2] == 1 and leaf[3] == [0, 1]
    search.provide(np.ones(2 * cfg.n), 0.75)
    index = int(np.flatnonzero(search.root.actions == cfg.n + 48)[0])
    assert search.root.W[index] == pytest.approx(-0.75)


def test_encoder_always_has_twenty_planes_and_player_relative_bombs():
    from msflags.game import NUM_PLANES
    cfg = Config(3, 3, 3)
    board = np.full(cfg.n, UNKNOWN, np.int8)
    initial = encode(board, [0, 0], 0, cfg)
    spent = encode(board, [0, 0], 1, cfg, [0, 1])
    assert NUM_PLANES == 20 and initial.shape == (20, cfg.h, cfg.w)
    assert initial[18:].all()
    assert spent[18].all() and not spent[19].any()


def test_bootstrap_records_joint_policies_and_bomb_inventory():
    from msflags.train import bootstrap_game, pack
    cfg = Config(4, 4, 3)
    positions = bootstrap_game((cfg, 12, 0.1))
    data = pack(positions, cfg)
    assert data['policy'].shape == (len(positions), 2 * cfg.n)
    assert data['bombs'].shape == (len(positions), 2)
    assert data['bombs'][0].tolist() == [1, 1]
    np.testing.assert_allclose(data['policy'].sum(1), 1.)


def test_training_uses_bomb_positions_and_saves_training_counts(tmp_path):
    from argparse import Namespace
    from msflags.model import Net, NetConfig, load, save
    from msflags.train import bootstrap_game, checkpoint, fit, pack
    cfg = Config(4, 4, 3)
    data = pack(bootstrap_game((cfg, 12, 0.1)), cfg)
    source = tmp_path / 'source.pt'
    net = Net(NetConfig(channels=8, blocks=1))
    save(net, source)
    args = Namespace(size=4, mines=3, seed=5, batch_size=2, device='cpu',
                     lr=0.001, weight_decay=0.0001, value_weight=0.25,
                     mine_weight=2., log_every=1, output=str(tmp_path),
                     checkpoint=str(source))
    fit(net, data, args, steps=1)
    checkpoint(net, args, 'trained', stage='bootstrap', bootstrap_steps=1)
    restored, metadata = load(tmp_path / 'latest.pt')
    assert not restored.training
    assert metadata['bootstrap_steps'] == 1
    assert 'rules' not in metadata['net_cfg']


def test_network_configuration_rejects_non_twenty_plane_inputs():
    from msflags.model import NetConfig
    with pytest.raises(ValueError, match='20 input planes'):
        NetConfig(planes=18)


@pytest.mark.parametrize('preset,channels,blocks', [('small', 32, 3), ('medium', 64, 4)])
def test_network_size_presets(tmp_path, preset, channels, blocks):
    from msflags.train import new_or_load, parser
    args = parser().parse_args(['bootstrap', '--network-size', preset, '--device', 'cpu',
                                '--output', str(tmp_path)])
    net = new_or_load(args)
    assert net.cfg.channels == channels
    assert net.cfg.blocks == blocks


def test_selfplay_resumes_latest_with_matching_bounded_replay(tmp_path, monkeypatch):
    from msflags import train
    from msflags.model import Net, NetConfig, load, save
    cfg = Config(4, 4, 3)
    data = train.pack(train.bootstrap_game((cfg, 12, 0.1)), cfg)
    data = {key: np.concatenate([value] * 4) for key, value in data.items()}
    source = tmp_path / 'bootstrap.pt'
    save(Net(NetConfig(channels=8, blocks=1)), source,
         stage='bootstrap', bootstrap_steps=8000, selfplay_iterations=0)
    args = train.parser().parse_args([
        'selfplay', '--checkpoint', str(source), '--output', str(tmp_path),
        '--device', 'cpu', '--size', '4', '--mines', '3', '--iterations', '1',
        '--steps', '1', '--batch-size', '2', '--replay-size', '3', '--eval-every', '0'])
    seeds = []
    fitted = []

    def generated(args, evaluator, seed):
        seeds.append(seed)
        return data

    def fitted_replay(net, replay, args, steps, optimizer):
        fitted.append((args.iteration, len(replay['board'])))
        return optimizer

    monkeypatch.setattr(train, 'selfplay_games', generated)
    monkeypatch.setattr(train, 'fit', fitted_replay)
    train.selfplay(args)
    first = load(tmp_path / 'latest.pt')[1]
    first_replay = (tmp_path / first['replay_file']).read_bytes()
    assert first['selfplay_iterations'] == 1
    assert 'optimizer' in first
    assert first['training_settings']['value_weight'] == 0.25
    assert first['training_settings']['lr'] == args.lr
    import json
    metrics = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    iteration_metrics = next(record for record in metrics if record['event'] == 'iteration')
    for field in ('generation_seconds', 'training_seconds', 'save_seconds', 'validation_seconds'):
        assert iteration_metrics[field] >= 0
    # Reusing the exact original launch command must prefer run/latest.pt.
    args.checkpoint = str(source)
    args.iterations = 2
    train.selfplay(args)
    second = load(tmp_path / 'latest.pt')[1]
    assert second['selfplay_iterations'] == 2
    assert second['replay_file'] != first['replay_file']
    assert (tmp_path / first['replay_file']).read_bytes() == first_replay
    assert seeds == [args.seed + 1, args.seed + 2]
    assert fitted == [(1, 3), (2, 3)]
    assert len(list(tmp_path.glob('selfplay-*.pt'))) == 2


def test_failed_checkpoint_write_keeps_latest_and_timestamped_weights(tmp_path, monkeypatch):
    from argparse import Namespace
    from msflags import train
    from msflags.model import Net, NetConfig, load
    args = Namespace(output=str(tmp_path))
    net = Net(NetConfig(channels=8, blocks=1))
    first = train.checkpoint(net, args, 'selfplay-1', selfplay_iterations=1)
    first_bytes = first.read_bytes()

    def interrupted(net, path, **metadata):
        with open(path, 'wb') as stream:
            stream.write(b'incomplete')
        raise OSError('interrupted write')

    monkeypatch.setattr(train, 'save', interrupted)
    with pytest.raises(OSError, match='interrupted write'):
        train.checkpoint(net, args, 'selfplay-2', selfplay_iterations=2)
    assert first.read_bytes() == first_bytes
    assert load(tmp_path / 'latest.pt')[1]['selfplay_iterations'] == 1
