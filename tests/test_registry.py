import json

import pytest
import torch

from msflags import registry
from msflags.model import Net, NetConfig, save


def checkpoint(tmp_path, stage, name='model.pt', channels=64, blocks=4, **metadata):
    path = tmp_path / name
    torch.save(dict(net_cfg=dict(channels=channels, blocks=blocks), state={}, stage=stage, **metadata), path)
    return path


@pytest.fixture
def pins(tmp_path, monkeypatch):
    data = registry._generations()
    for spec in data.values():
        spec['generations'] = []
    path = tmp_path / 'generations.json'
    monkeypatch.setattr(registry, 'GENERATIONS_PATH', path)

    def install(architecture, checkpoint_path, bootstrap=8000, selfplay=0):
        data[architecture]['generations'].append(dict(
            checkpoint=str(checkpoint_path), training=dict(bootstrap=bootstrap, selfplay=selfplay)))
        path.write_text(json.dumps(data))

    path.write_text(json.dumps(data))
    return install


def test_identity_from_metadata_bootstrap_and_selfplay(tmp_path):
    path = checkpoint(tmp_path, 'bootstrap', bootstrap_steps=8000, selfplay_iterations=0)
    entry = registry._neural_entry(path, None)
    assert entry['id'] == 'r64x4-bs8k.greedy'
    assert entry['generation'] == 0
    assert entry['category_id'] == 'r64x4.greedy'
    path = checkpoint(tmp_path, 'selfplay', name='another.pt', bootstrap_steps=8000, selfplay_iterations=2)
    entry = registry._neural_entry(path, dict(sims=100, c_puct=1.5))
    assert entry['id'] == 'r64x4-bs8k-sp2.mcts100-c1.5'
    assert entry['generation'] == 2
    assert entry['training'] == dict(bootstrap=8000, selfplay=2)
    assert entry['name'] == 'ResNet 64×4 · 8k bootstrap + 2 self-play · MCTS-100 (c 1.5)'
    assert registry._neural_entry(path, dict(sims=100, c_puct=2.5, fpu_reduction=0.4))['id'].endswith('mcts100-c2.5-f0.4')


def test_identity_from_completed_metrics_not_requested_iterations(tmp_path):
    events = [
        dict(event='start', command='bootstrap', arguments=dict(steps=8000, iterations=1)),
        dict(event='train', step=8000),
        dict(event='start', command='selfplay', arguments=dict(steps=300, iterations=10)),
        dict(event='train', step=300),
        dict(event='train', step=300),
    ]
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    path = checkpoint(tmp_path, 'selfplay', iteration=2)
    assert registry._neural_entry(path, None)['id'] == 'r64x4-bs8k-sp2.greedy'
    (tmp_path / 'metrics.jsonl').unlink()
    with pytest.raises(ValueError, match='Cannot derive training identity'):
        registry.training_counts(path, dict(stage='bootstrap'))


def test_metrics_after_timestamp_do_not_change_identity(tmp_path):
    events = [
        dict(event='start', time=1, command='bootstrap', arguments=dict(steps=8000)),
        dict(event='train', time=2, step=8000),
        dict(event='train', time=4, step=16000),
    ]
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    path = checkpoint(tmp_path, 'bootstrap', name='bootstrap-3000000000.pt')
    assert registry.training_counts(path, dict(stage='bootstrap')) == (8000, 0)


def test_six_categories_share_pins_and_keep_missing_identity(tmp_path, pins):
    bootstrap = tmp_path / 'arbitrary-name.pt'
    bootstrap.touch()
    missing = tmp_path / 'not-installed.pt'
    pins('r64x4', bootstrap)
    pins('r64x4', missing, selfplay=10)
    categories = registry.catalog()
    assert [e['id'] for e in categories] == [
        'r64x4.greedy', 'r64x4.mcts100-c1.5', 'r32x3.greedy',
        'r32x3.mcts100-c1.5', 'random', 'solver',
    ]
    for category in categories[:2]:
        assert [e['generation'] for e in category['generations']] == [0, 10]
        assert category['default_generation'] == category['generations'][0]['id']
        assert [e['available'] for e in category['generations']] == [True, False]
    assert registry.resolve('r64x4.greedy') == 'r64x4-bs8k.greedy'
    assert registry.resolve('r64x4-bs8k-sp10.greedy') == 'r64x4-bs8k-sp10.greedy'
    assert not registry.available('r64x4-bs8k-sp10.greedy')
    detail = registry.detail('r64x4-bs8k-sp10.greedy')
    assert detail['shape']['channels'] == 64
    assert detail['training'] == dict(bootstrap=8000, selfplay=10)
    assert registry.pinned_checkpoints() == {str(bootstrap), str(missing)}
    with pytest.raises(ValueError, match='unavailable agent'):
        registry.create('r64x4-bs8k-sp10.greedy')
    assert not categories[2]['available']
    assert categories[2]['default_generation'] is None
    with pytest.raises(ValueError, match='unavailable agent'):
        registry.resolve(categories[2]['id'])


def test_latest_available_is_generation_order_not_append_order(tmp_path, pins):
    for generation in (20, 0, 10):
        path = tmp_path / f'{generation}.pt'
        path.touch()
        pins('r32x3', path, selfplay=generation)
    assert registry.resolve('r32x3.greedy') == 'r32x3-bs8k-sp20.greedy'
    assert len(registry.entries()) == 8  # Three pins, two search categories, two baselines.


def test_duplicate_paths_and_derived_ids_rejected(tmp_path, pins):
    path = tmp_path / 'first.pt'
    pins('r64x4', path)
    pins('r64x4', tmp_path / 'second.pt')
    with pytest.raises(ValueError, match='Duplicate derived agent id'):
        registry.entries()


def test_duplicate_paths_across_architectures_rejected(tmp_path, pins):
    path = tmp_path / 'missing.pt'
    pins('r64x4', path)
    pins('r32x3', path)
    with pytest.raises(ValueError, match='Duplicate checkpoint path'):
        registry.entries()


def test_category_detail_uses_latest_available_and_create_resolves(tmp_path, pins, monkeypatch):
    path = tmp_path / 'trained.pt'
    save(Net(NetConfig(channels=32, blocks=3)), path,
         stage='selfplay', bootstrap_steps=8000, selfplay_iterations=10)
    pins('r32x3', path, selfplay=10)
    pins('r32x3', tmp_path / 'missing.pt', selfplay=20)
    detail = registry.detail('r32x3.greedy')
    assert detail['id'] == 'r32x3.greedy'
    assert detail['default_generation'] == 'r32x3-bs8k-sp10.greedy'
    assert detail['input_planes'] == 20
    assert detail['board']['bombs'] == 1
    assert 'rules' not in detail['shape']
    evaluator = object()
    monkeypatch.setattr(registry, 'Evaluator', lambda *args: evaluator)
    monkeypatch.setattr(registry, 'load', lambda *args: (None, {}))
    monkeypatch.setattr(registry, 'NetAgent', lambda value: value)
    assert registry.create('r32x3.greedy') is evaluator
    assert registry.create('r32x3-bs8k-sp10.greedy') is evaluator


def test_baseline_detail_combines_category_and_generation(pins):
    detail = registry.detail('solver')
    assert detail['default_generation'] == 'solver'
    assert detail['generations'][0]['id'] == 'solver'
    assert detail['shape'] is None
    assert registry.resolve('random') == 'random'


def test_promote_appends_shared_immutable_pin_and_rejects_duplicates(tmp_path, pins, monkeypatch):
    path = tmp_path / 'selfplay-123.pt'
    path.touch()
    shape = registry._generations()['r64x4']['shape']
    calls = []

    def load(checkpoint_path, device):
        calls.append((checkpoint_path, device))
        return None, dict(net_cfg=shape, stage='selfplay', bootstrap_steps=8000, selfplay_iterations=10)

    monkeypatch.setattr(registry, 'load', load)
    promoted = registry.promote(path)
    assert calls == [(str(path), 'cpu')]
    assert [e['id'] for e in promoted] == ['r64x4-bs8k-sp10.greedy', 'r64x4-bs8k-sp10.mcts100-c1.5']
    assert registry.pinned_checkpoints() == {str(path)}
    assert len([line for line in registry.GENERATIONS_PATH.read_text().splitlines() if '"checkpoint"' in line]) == 1
    before = registry.GENERATIONS_PATH.read_bytes()
    with pytest.raises(ValueError, match='Duplicate checkpoint path'):
        registry.promote(path.parent / '.' / path.name)
    other = tmp_path / 'different-name.pt'
    other.touch()
    with pytest.raises(ValueError, match='Duplicate derived agent id'):
        registry.promote(other)
    assert registry.GENERATIONS_PATH.read_bytes() == before
    path.unlink()
    assert registry.resolve(promoted[0]['id']) == promoted[0]['id']
    assert not registry.available(promoted[0]['id'])


def test_promote_rejects_latest_and_incompatible_weights(tmp_path, pins):
    before = registry.GENERATIONS_PATH.read_bytes()
    with pytest.raises(ValueError, match='latest.pt is mutable'):
        registry.promote(tmp_path / 'latest.pt')
    path = checkpoint(tmp_path, 'bootstrap', bootstrap_steps=8000, selfplay_iterations=0)
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        registry.promote(path)
    assert registry.GENERATIONS_PATH.read_bytes() == before


def test_cli_list_and_promote_error(pins, capsys):
    registry.main(['list'])
    assert len(json.loads(capsys.readouterr().out)) == 6
    with pytest.raises(SystemExit) as error:
        registry.main(['promote', 'latest.pt'])
    assert error.value.code == 2


def test_removed_agent_ids_are_unknown():
    for agent_id in ('bomb.r64x4-tr-bs8k.greedy', 'r64x4-tr-bs8k.greedy'):
        with pytest.raises(ValueError, match='unknown agent'):
            registry.create(agent_id)


def test_mcts_request_override_keeps_generation_settings_fixed(tmp_path, pins, monkeypatch):
    path = tmp_path / 'fixed.pt'
    path.touch()
    pins('r64x4', path, selfplay=10)
    monkeypatch.setattr(registry, 'load', lambda *args: (None, {}))
    monkeypatch.setattr(registry, 'Evaluator', lambda *args: object())
    monkeypatch.setattr(registry, 'MCTSAgent', lambda evaluator, **kwargs: kwargs)
    category = 'r64x4.mcts100-c1.5'
    generation = 'r64x4-bs8k-sp10.mcts100-c1.5'
    assert registry.create(category)['sims'] == 100
    assert registry.create(generation)['sims'] == 100
    assert registry.create(category, sims=17)['sims'] == 17
    assert registry.create(generation, sims=23)['sims'] == 23
    assert registry._entry(generation)['settings']['sims'] == 100


def test_promote_rejects_load_compatible_unsupported_architecture(tmp_path, pins):
    path = tmp_path / 'small.pt'
    save(Net(NetConfig(channels=8, blocks=1)), path,
         stage='bootstrap', bootstrap_steps=8000, selfplay_iterations=0)
    before = registry.GENERATIONS_PATH.read_bytes()
    with pytest.raises(ValueError, match='Unsupported architecture'):
        registry.promote(path)
    assert registry.GENERATIONS_PATH.read_bytes() == before
