from dataclasses import asdict

import numpy as np
import torch

from msflags.game import Config, Game, encode_batch
from msflags.model import Net, NetConfig, add_auxiliary_heads, load
from msflags.registry import _neural_entry, architecture_key
from msflags.train import Position, auxiliary_losses, augment, dihedral, finish_targets, legacy_auxiliary_targets


def test_terminal_ownership_bomb_perspective_and_unclaimed():
    cfg = Config(6, 6, 5)
    mines = np.zeros(cfg.n, bool)
    mines[[0, 1, 2, 30, 35]] = True
    game = Game(cfg, mines)
    game.to_move = 1
    game.step(35)
    game.to_move = 0
    game.step(cfg.n)
    assert game.done
    positions = [Position(np.full(cfg.n, -1), [0, 0], p, np.zeros(2 * cfg.n), np.zeros(cfg.n), [1, 1]) for p in (0, 1)]
    finish_targets(positions, game)
    np.testing.assert_array_equal(positions[0].ownership[[0, 1, 2, 30, 35]], [0, 0, 0, 2, 1])
    np.testing.assert_array_equal(positions[1].ownership[[0, 1, 2, 30, 35]], [1, 1, 1, 2, 0])
    assert np.all(positions[0].ownership[~mines] == -1)
    assert positions[0].final_scores == [3, 1]


def test_ownership_augmentation_alignment():
    board = np.arange(16).reshape(1, 1, 4, 4)
    ownership = board.reshape(1, 16)
    policy = np.tile(ownership, (1, 2))
    for symmetry in range(8):
        xb, pi, mine = augment(board, policy, ownership, symmetry)
        target = dihedral(ownership.reshape(1, 4, 4), symmetry).reshape(1, 16)
        np.testing.assert_array_equal(xb.reshape(1, 16), target)
        np.testing.assert_array_equal(mine, target)
        np.testing.assert_array_equal(pi[:, :16], target)


def test_old_checkpoint_exact_outputs_and_upgrade(tmp_path):
    torch.manual_seed(14)
    net = Net(NetConfig(channels=8, blocks=1)).eval()
    cfg = asdict(net.cfg)
    cfg.pop('score_margin')
    cfg.pop('mine_ownership')
    path = tmp_path / 'old.pt'
    torch.save(dict(net_cfg=cfg, state=net.state_dict()), path)
    restored, _ = load(path)
    game = Game.random(Config(4, 4, 3), np.random.default_rng(1))
    inputs = torch.from_numpy(encode_batch([(game.board, game.scores, 0, game.bombs)], game.cfg))
    upgraded = add_auxiliary_heads(restored, score_margin=True, mine_ownership=True)
    with torch.no_grad():
        for expected, actual, extra in zip(net(inputs), restored(inputs), upgraded(inputs)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            torch.testing.assert_close(expected, extra, rtol=0, atol=0)
        _, _, _, margin, ownership = upgraded(inputs, auxiliary=True)
        assert not margin.any()
        assert not ownership.any()


def test_legacy_auxiliary_losses_zero_and_mixed_masking():
    data = legacy_auxiliary_targets(dict(board=np.zeros((2, 4), np.int8)))
    margin = torch.tensor([10., 10.], requires_grad=True)
    ownership = torch.zeros((2, 3, 4), requires_grad=True)
    scores = torch.from_numpy(data['final_scores'])
    labels = torch.from_numpy(data['ownership'])
    player = torch.tensor([0, 1])
    losses = auxiliary_losses(margin, ownership, scores, player, labels, 26)
    assert all(loss.item() == 0 for loss in losses)
    sum(losses).backward()
    assert not margin.grad.any()
    assert not ownership.grad.any()
    scores[0] = torch.tensor([26, 0])
    labels[0, 0] = 0
    loss, own_loss = auxiliary_losses(margin, ownership, scores, player, labels, 26)
    assert loss.item() == 81 / 2
    torch.testing.assert_close(own_loss, torch.log(torch.tensor(3.)) / 2)


def test_auxiliary_generation_ids_distinct():
    base = dict(channels=64, blocks=4)
    variants = [base, dict(base, score_margin=True), dict(base, mine_ownership=True), dict(base, score_margin=True, mine_ownership=True)]
    assert len({architecture_key(shape) for shape in variants}) == 4
    ids = {_neural_entry('unused.pt', None, shape=shape, training=dict(bootstrap=8000, selfplay=181))['id'] for shape in variants}
    assert len(ids) == 4
