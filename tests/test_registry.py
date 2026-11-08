import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from msflags import registry
from msflags.agents import NetAgent
from msflags.deep import DeepConfig, DeepEvaluator, DeepNet, save_net
from msflags.game import Config, Game
from msflags.rollout import RolloutAgent
from msflags.expert import ExpertAgent, Expert2Agent
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
    deep_path = tmp_path / 'deep_generations.json'
    deep_path.write_text('[]')
    monkeypatch.setattr(registry, 'DEEP_GENERATIONS_PATH', deep_path)

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


def test_ten_categories_share_pins_and_keep_missing_identity(tmp_path, pins):
    bootstrap = tmp_path / 'arbitrary-name.pt'
    bootstrap.touch()
    missing = tmp_path / 'not-installed.pt'
    pins('r64x4', bootstrap)
    pins('r64x4', missing, selfplay=10)
    categories = registry.catalog()
    assert [e['id'] for e in categories] == [
        'r64x4.greedy', 'r64x4.mcts100-c1.5', 'r32x3.greedy',
        'r32x3.mcts100-c1.5', 'p96x8.greedy', 'p96x8.rollout2x128',
        'random', 'solver', 'expert', 'expert2',
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
    assert len(registry.entries()) == 10  # Three pins, two search categories, four baselines.


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


@pytest.mark.parametrize('agent_id', ['random', 'solver', 'expert', 'expert2'])
def test_baseline_detail_combines_category_and_generation(pins, agent_id):
    detail = registry.detail(agent_id)
    assert detail['default_generation'] == agent_id
    assert len(detail['generations']) == 1
    assert detail['generations'][0]['id'] == agent_id
    assert detail['generations'][0]['generation'] == 0
    assert detail['available']
    assert detail['shape'] is None
    assert 'checkpoint' not in detail
    assert not detail['request_sims']
    assert registry.resolve(agent_id) == agent_id


@pytest.mark.parametrize('agent_id,agent_type', [('expert', ExpertAgent), ('expert2', Expert2Agent)])
def test_create_expert_baselines_uses_rng_and_reserve(pins, agent_id, agent_type):
    rng = np.random.default_rng(42)
    agent = registry.create(agent_id, rng=rng)
    assert type(agent) is agent_type
    assert agent.rng is rng
    assert agent.bomb_reserve == 3


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
    assert len(json.loads(capsys.readouterr().out)) == 10
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


@pytest.fixture
def deep_pin(tmp_path, pins, monkeypatch):
    monkeypatch.setattr(registry, '_device', 'cpu')
    monkeypatch.setattr(registry, '_evaluators', {})
    path = tmp_path / 'deep.pt'
    save_net(DeepNet(DeepConfig(channels=4, blocks=1)), path)
    records = [dict(id='p96x8-n1', checkpoint=str(path), description='Tiny test policy'),
               dict(id='p96x8-n2', checkpoint=str(tmp_path / 'missing.pt'), description='Missing policy')]
    registry.DEEP_GENERATIONS_PATH.write_text(json.dumps(records))
    return path


def test_deep_categories_preserve_missing_pins_and_detail(deep_pin):
    for suffix in ('greedy', 'rollout2x128'):
        category = f'p96x8.{suffix}'
        assert registry.resolve(category) == f'p96x8-n1.{suffix}'
        assert registry.resolve(f'p96x8-n2.{suffix}') == f'p96x8-n2.{suffix}'
        assert not registry.available(f'p96x8-n2.{suffix}')
        assert registry.detail(f'p96x8-n2.{suffix}')['shape'] == dict(channels=96, blocks=8)
        with pytest.raises(ValueError, match='unavailable agent'):
            registry.create(f'p96x8-n2.{suffix}')
    assert str(deep_pin) in registry.pinned_checkpoints()
    assert str(deep_pin.parent / 'missing.pt') in registry.pinned_checkpoints()
    detail = registry.detail('p96x8.greedy')
    assert detail['shape'] == dict(channels=4, blocks=1)
    assert detail['input_planes'] > 20
    assert detail['parameters'] > 0
    assert detail['provenance']['description'] == 'Tiny test policy'
    assert 'probabilities' in detail['explanation']


def test_create_deep_agents_share_cpu_evaluator_and_fixed_settings(deep_pin):
    rng = np.random.default_rng(42)
    greedy = registry.create('p96x8.greedy', sims=7, rng=rng)
    search = registry.create('p96x8.rollout2x128', sims=7, rng=rng)
    assert type(greedy) is NetAgent
    assert type(search) is RolloutAgent
    assert isinstance(greedy.ev, DeepEvaluator)
    assert search.ev is greedy.ev
    assert greedy.ev.device == torch.device('cpu')
    assert search.rng is rng
    assert (search.turns, search.rollouts) == (2, 128)
    # The Play path uses act(), while offline arena batches the same evaluator.
    game = Game.random(Config(2, 3, 3), rng)
    assert game.legal_mask()[greedy.act(game.public())]
    assert game.legal_mask()[search.act(game.public())]


def test_deep_greedy_two_game_arena_match(tmp_path, deep_pin):
    from msflags.arena import ArenaManager

    manager = ArenaManager(tmp_path / 'games.jsonl', cfg=Config(2, 3, 3))
    manager.batch_size = 2
    match = manager.start('p96x8.greedy', 'solver', 2, seed=7)
    manager._thread.join(timeout=15)
    assert not manager._thread.is_alive()
    result = manager.status(match['id'])
    assert result['status'] == 'completed', result['error']
    assert result['completed'] == 2
    records = [json.loads(line) for line in manager.path.read_text().splitlines()]
    assert records == result['results']
    assert [record['seats'] for record in records] == [
        ['p96x8-n1.greedy', 'solver'], ['solver', 'p96x8-n1.greedy']]
    rows = {row['id']: row for row in manager.ratings()['ratings']}
    assert rows['p96x8-n1.greedy']['games'] == 2


def test_promote_deep_cli_appends_shared_pin(tmp_path, pins, monkeypatch, capsys):
    path = tmp_path / 'net-2.pt'
    path.touch()
    calls = []

    def load(path, device):
        calls.append((path, device))
        return SimpleNamespace(cfg=DeepConfig())

    monkeypatch.setattr(registry, 'load_net', load)
    registry.main(['promote-deep', str(path), '--id', 'n2', '--description', 'New expert policy'])
    promoted = json.loads(capsys.readouterr().out)
    assert calls == [(str(path), 'cpu')]
    assert [entry['id'] for entry in promoted] == ['p96x8-n2.greedy', 'p96x8-n2.rollout2x128']
    assert registry._deep_generations() == [
        dict(id='p96x8-n2', checkpoint=str(path), description='New expert policy')]
    assert registry.resolve('p96x8.greedy') == 'p96x8-n2.greedy'
    before = registry.DEEP_GENERATIONS_PATH.read_bytes()
    with pytest.raises(ValueError, match='Duplicate deep generation id'):
        registry.promote_deep(tmp_path / 'other.pt', 'n2', 'Duplicate')
    with pytest.raises(ValueError, match='Duplicate checkpoint path'):
        registry.promote_deep(path, 'n3', 'Duplicate')
    assert registry.DEEP_GENERATIONS_PATH.read_bytes() == before


def test_promote_deep_rejects_invalid_pins(tmp_path, pins):
    before = registry.DEEP_GENERATIONS_PATH.read_bytes()
    with pytest.raises(ValueError, match='latest.pt is mutable'):
        registry.promote_deep(tmp_path / 'latest.pt', 'n2', 'Mutable')
    with pytest.raises(ValueError, match='Deep id'):
        registry.promote_deep(tmp_path / 'net.pt', 'bad', 'Invalid')
    with pytest.raises(ValueError, match='description'):
        registry.promote_deep(tmp_path / 'net.pt', 'n2', ' ')
    path = tmp_path / 'tiny.pt'
    save_net(DeepNet(DeepConfig(channels=4, blocks=1)), path)
    with pytest.raises(ValueError, match='Unsupported deep architecture'):
        registry.promote_deep(path, 'n2', 'Wrong shape')
    torch.save(dict(deep_cfg=dict(channels=96, blocks=8), state={}), path)
    with pytest.raises(RuntimeError):
        registry.promote_deep(path, 'n2', 'Invalid weights')
    assert registry.DEEP_GENERATIONS_PATH.read_bytes() == before
