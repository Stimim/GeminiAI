import numpy as np
import pytest

from msflags.game import Config, Game, encode, encode_batch


@pytest.mark.parametrize('shape', [(3, 5), (16, 16)])
def test_batch_encoding_matches_scalar_for_every_plane(shape):
    cfg = Config(*shape, 3 if shape == (3, 5) else 51)
    rng = np.random.default_rng(42)
    states = []
    for _ in range(12):
        game = Game.random(cfg, rng)
        for _ in range(4):
            if not game.done:
                game.step(int(rng.choice(game.legal_actions())))
        states.append((game.board, game.scores, game.to_move, game.bombs))
    expected = np.stack([encode(board, scores, player, cfg, bombs)
                         for board, scores, player, bombs in states])
    actual = encode_batch(states, cfg)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
