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


def test_cpu_share_reserves_trainer_and_splits_workers(tmp_path, monkeypatch):
    import sys
    from msflags import train
    observed = []
    monkeypatch.setattr(sys, 'argv', ['train', 'selfplay', '--workers', '6',
                                    '--cpu-share', '13', '--output', str(tmp_path)])
    monkeypatch.setattr(train, 'selfplay', lambda args: observed.append(args))
    monkeypatch.setattr(train, 'log', lambda *args, **kwargs: None)
    train.main()
    assert observed[0].threads == 1
    assert observed[0].worker_threads == 2
    assert observed[0].threads + observed[0].workers * observed[0].worker_threads == 13


def test_cpu_share_rejects_oversubscribed_workers(tmp_path, monkeypatch):
    import sys
    import pytest
    from msflags import train
    monkeypatch.setattr(sys, 'argv', ['train', 'selfplay', '--workers', '6',
                                    '--cpu-share', '6', '--output', str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
