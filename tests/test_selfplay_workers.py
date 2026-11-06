from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from msflags.model import Evaluator, Net, NetConfig
from msflags.train import _init_selfplay_worker, parser, selfplay_games


def test_spawned_workers_merge_the_same_seeded_batches(tmp_path):
    args = parser().parse_args([
        'selfplay', '--device', 'cpu', '--size', '3', '--mines', '3',
        '--games', '2', '--concurrent-games', '1', '--sims', '2',
        '--output', str(tmp_path), '--workers', '2'])
    evaluator = Evaluator(Net(NetConfig(channels=8, blocks=1)), 'cpu')
    expected = selfplay_games(args, evaluator, 123)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_selfplay_worker,
                             initargs=(evaluator.net.cfg, 'cpu', 1)) as pool:
        evaluator.selfplay_pool = pool
        actual = selfplay_games(args, evaluator, 123)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-6)
