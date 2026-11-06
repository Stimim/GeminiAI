import pytest
import torch

from msflags import registry, train


def checkpoint_set(root):
    for iteration in range(1, 14):
        torch.save({'selfplay_iterations': iteration}, root / f'selfplay-{iteration}.pt')
    (root / 'bootstrap-1.pt').write_bytes(b'bootstrap')
    (root / 'latest.pt').write_bytes(b'latest')
    (root / 'replay-0.npz').write_bytes(b'replay')


def test_retention_keeps_pins_milestones_and_newest_three(tmp_path, monkeypatch):
    checkpoint_set(tmp_path)
    pinned = tmp_path / 'selfplay-2.pt'
    monkeypatch.setattr(registry, 'pinned_checkpoints', lambda: {str(pinned)})
    original = set(tmp_path.iterdir())
    preview = train.prune_checkpoints(tmp_path, dry_run=True)
    assert len(preview['deleted']) == 8
    assert set(tmp_path.iterdir()) == original
    actual = train.prune_checkpoints(tmp_path)
    assert actual['deleted'] == preview['deleted']
    assert {p.name for p in tmp_path.iterdir()} == {
        'selfplay-2.pt', 'selfplay-10.pt', 'selfplay-11.pt',
        'selfplay-12.pt', 'selfplay-13.pt', 'bootstrap-1.pt', 'latest.pt', 'replay-0.npz'}


def test_retention_missing_registry_hook_keeps_everything(tmp_path, monkeypatch):
    checkpoint_set(tmp_path)
    monkeypatch.delattr(registry, 'pinned_checkpoints')
    with pytest.warns(RuntimeWarning, match='retaining every checkpoint'):
        result = train.prune_checkpoints(tmp_path)
    assert not result['deleted']
    assert len(list(tmp_path.iterdir())) == 16


def test_budget_failure_never_deletes_protected_files(tmp_path, monkeypatch):
    checkpoint_set(tmp_path)
    monkeypatch.setattr(registry, 'pinned_checkpoints', lambda: set())
    with pytest.warns(RuntimeWarning, match='DISK BUDGET EXCEEDED'):
        result = train.prune_checkpoints(tmp_path, limit=1)
    assert result['over_budget']
    assert (tmp_path / 'selfplay-10.pt').exists()
    assert (tmp_path / 'bootstrap-1.pt').exists()


def test_prune_cli_parser_is_independent_of_board_arguments():
    args = train.parser().parse_args(['prune', '--dry-run'])
    assert args.dry_run and args.root == 'checkpoints'
