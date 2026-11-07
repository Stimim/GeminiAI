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
    context = multiprocessing.get_context('spawn')
    requests = context.Queue()
    with ProcessPoolExecutor(max_workers=2, mp_context=context,
                             initializer=_init_selfplay_worker,
                             initargs=(requests, 1)) as pool:
        evaluator.selfplay_pool = pool
        evaluator.selfplay_requests = requests
        actual = selfplay_games(args, evaluator, 123)
    requests.close()
    requests.join_thread()
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


def test_shared_evaluator_failure_releases_every_worker(tmp_path):
    import pytest
    args = parser().parse_args([
        'selfplay', '--device', 'cpu', '--size', '3', '--mines', '3',
        '--games', '4', '--concurrent-games', '1', '--sims', '2',
        '--output', str(tmp_path), '--workers', '2'])
    context = multiprocessing.get_context('spawn')
    requests = context.Queue()

    def fail(states, cfg):
        raise RuntimeError('deliberate inference failure')

    with ProcessPoolExecutor(max_workers=2, mp_context=context,
                             initializer=_init_selfplay_worker,
                             initargs=(requests, 1)) as pool:
        fail.selfplay_pool = pool
        fail.selfplay_requests = requests
        with pytest.raises(RuntimeError, match='deliberate inference failure'):
            selfplay_games(args, fail, 123)
    requests.close()
    requests.join_thread()


def test_shared_inference_chunks_preserve_each_worker_leaf_order():
    import queue
    from msflags.game import Config
    from msflags.train import _serve_worker_requests
    requests = queue.Queue()
    waiting, calls = [], []
    offset = 0
    for count in (33, 31, 7):
        receive, send = multiprocessing.Pipe(duplex=False)
        states = list(range(offset, offset + count))
        requests.put((states, Config(3, 3, 3), send))
        waiting.append((receive, states))
        offset += count

    def evaluate(states, cfg):
        calls.append(len(states))
        ids = np.asarray(states)
        return ids[:, None], ids, ids[:, None]

    _serve_worker_requests(evaluate, requests)
    assert calls == [64, 7]
    for receive, expected in waiting:
        ok, (policy, values, mine) = receive.recv()
        receive.close()
        assert ok
        np.testing.assert_array_equal(policy[:, 0], expected)
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(mine[:, 0], expected)
