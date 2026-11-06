import json

import pytest
import torch

from msflags import registry


def checkpoint(tmp_path, stage, **metadata):
    path = tmp_path / 'model.pt'
    torch.save(dict(net_cfg=dict(channels=64, blocks=4), state={}, stage=stage, **metadata), path)
    return path


def test_identity_from_metadata_bootstrap_and_selfplay(tmp_path):
    path = checkpoint(tmp_path, 'bootstrap', bootstrap_steps=8000, selfplay_iterations=0)
    assert registry._neural_entry(path, None)['id'] == 'r64x4-bs8k.greedy'
    path = checkpoint(tmp_path, 'selfplay', bootstrap_steps=8000, selfplay_iterations=2)
    entry = registry._neural_entry(path, dict(sims=100, c_puct=1.5))
    assert entry['id'] == 'r64x4-bs8k-sp2.mcts100-c1.5'
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


def test_duplicate_derived_ids_rejected(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, 'bootstrap', bootstrap_steps=5000, selfplay_iterations=0)
    monkeypatch.setattr(registry, 'VARIANTS', [(path, None), (path, None)])
    with pytest.raises(ValueError, match='Duplicate derived agent id'):
        registry.catalog()


def test_catalog_contains_only_current_agents():
    entries = registry.catalog()
    assert [entry['id'] for entry in entries] == [
        'random', 'solver', 'r64x4-bs8k.greedy', 'r32x3-bs8k.greedy', 'r64x4-bs8k.mcts100-c1.5',
    ]
    assert all('rules' not in entry for entry in entries)
    assert all(entry['available'] for entry in entries)




def test_detail_exposes_twenty_plane_bomb_board():
    entry = registry.detail('r64x4-bs8k.greedy')
    assert entry['input_planes'] == 20
    assert entry['board']['bombs'] == 1
    assert 'rules' not in entry['board']
    assert 'rules' not in entry['shape']


def test_removed_agent_ids_are_unknown():
    for agent_id in ('bomb.r64x4-tr-bs8k.greedy', 'r64x4-tr-bs8k.greedy'):
        with pytest.raises(ValueError, match='unknown agent'):
            registry.create(agent_id)
