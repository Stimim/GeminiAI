"""Stable arena identities and lazy shared checkpoint evaluators.

Transferred checkpoints retain ``tr-`` in their training identity and expose
their historical ``transferred_from`` provenance.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading

import numpy as np
import torch

from .agents import MCTSAgent, NetAgent, RandomAgent, SolverAgent
from .game import Config, NUM_PLANES
from .model import Evaluator, load

_device = 'cpu'
_lock = threading.RLock()
_evaluators: dict[tuple[str, str], Evaluator] = {}
_metadata: dict[tuple[str, int], dict] = {}


def configure(device='cpu'):
    global _device
    with _lock:
        _device = device


# Never repoint an existing variant: add a new immutable checkpoint/search pair.
VARIANTS = [
    ('checkpoints/bomb/bootstrap-1793979495800377617.pt', None),
    ('checkpoints/bomb/bootstrap-1793979495800377617.pt', dict(sims=100, c_puct=1.5, fpu_reduction=0.2)),
]
_identities: dict[tuple[str, int], tuple[dict, dict]] = {}


def training_counts(path, checkpoint):
    provenance = _provenance(path, checkpoint)
    bootstrap = checkpoint.get('bootstrap_steps')
    selfplay = checkpoint.get('selfplay_iterations')
    if selfplay is None and checkpoint.get('stage') not in ('bootstrap', 'selfplay'):
        raise ValueError(f'Cannot derive training stage: {path}')
    sessions = provenance['training']
    if bootstrap is None:
        bootstrap = sum(s['completed_steps'] for s in sessions if s['stage'] == 'bootstrap')
    if selfplay is None:
        selfplay = 0
        for session in sessions:
            if session['stage'] != 'selfplay':
                continue
            steps = session['steps_per_iteration']
            completed = session['completed_steps']
            if not steps or completed % steps:
                raise ValueError(f'Cannot derive completed self-play iterations: {path}')
            selfplay += completed // steps
        if checkpoint.get('stage') == 'selfplay' and not selfplay:
            selfplay = checkpoint.get('iteration')
    if not isinstance(bootstrap, int) or bootstrap <= 0 or not isinstance(selfplay, int) or selfplay < 0:
        raise ValueError(f'Cannot derive training identity from checkpoint/metrics: {path}')
    if checkpoint.get('stage') == 'selfplay' and selfplay == 0:
        raise ValueError(f'Missing self-play iterations: {path}')
    return bootstrap, selfplay


def _neural_entry(checkpoint, search):
    path = Path(checkpoint)
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    with _lock:
        if key not in _identities:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            training = dict(zip(('bootstrap', 'selfplay'), training_counts(path, ck)))
            training['transfer'] = ck.get('transferred_from')
            _identities[key] = (ck['net_cfg'], training)
        shape, training = _identities[key]
    bootstrap = format(training['bootstrap'] / 1000, 'g') + 'k'
    arch = f"r{shape['channels']}x{shape['blocks']}"
    trained = f'bs{bootstrap}'
    if training['transfer']:
        trained = 'tr-' + trained
    description = f'{bootstrap} bootstrap'
    if training['transfer']:
        description += ' (transferred)'
    if training['selfplay']:
        trained += f"-sp{training['selfplay']}"
        description += f" + {training['selfplay']} self-play"
    settings = dict(search or {})
    request_sims = settings.pop('request_sims', False)
    if search is None:
        suffix, label, kind = 'greedy', 'Greedy policy', 'net'
        summary = 'Solver-distilled policy, without tree search.'
    else:
        settings.setdefault('fpu_reduction', 0.2)
        sims, c, fpu = settings['sims'], settings['c_puct'], settings['fpu_reduction']
        suffix = f'mcts{sims}-c{c:g}'
        label = f'MCTS-{sims} (c {c:g}'
        if fpu != 0.2:
            suffix += f'-f{fpu:g}'
            label += f', FPU {fpu:g}'
        label += ')'
        kind = 'mcts'
        summary = f'Information-set search with {sims} simulations and exploration c {c:g}.'
        if request_sims:
            summary += " Play’s sims box overrides the simulation count."
    return dict(id=f'{arch}-{trained}.{suffix}',
                name=f"ResNet {shape['channels']}×{shape['blocks']} · {description} · {label}",
                kind=kind, checkpoint=str(path), settings=settings,
                request_sims=request_sims, summary=summary)


def _entries():
    entries = [
        dict(id='random', name='Random', kind='random', summary='Uniformly chooses a legal action.', explanation='Chooses uniformly among legal open and bomb actions. It never reads clues or scores. This is a deliberately weak baseline, not the rating anchor.'),
        dict(id='solver', name='Greedy exact solver', kind='solver', summary='Claims certain mines, otherwise maximizes immediate expected gain.', explanation='Builds constraints from visible clues and the remaining mine count. Opens certain mines first, otherwise uses the highest expected-mine 5×5 window when worth at least three mines; ties are random. Tractable frontiers are enumerated exactly; difficult components use bounded sampling. Its Elo is fixed at 1000.'),
    ]
    entries.extend(_neural_entry(checkpoint, search) for checkpoint, search in VARIANTS)
    ids = [entry['id'] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate derived agent id in registry')
    return entries


def _entry(agent_id):
    for entry in _entries():
        if entry['id'] == agent_id:
            return entry
    raise ValueError(f'unknown agent: {agent_id}')


def available(agent_id):
    e = _entry(agent_id)
    return 'checkpoint' not in e or Path(e['checkpoint']).is_file()


def catalog():
    return [{**e, 'available': 'checkpoint' not in e or Path(e['checkpoint']).is_file()}
            for e in _entries()]


def _provenance(path, checkpoint):
    records = []
    metrics = path.parent / 'metrics.jsonl'
    cutoff = path.stat().st_mtime + 1
    if path.stem.startswith(('bootstrap-', 'selfplay-')):
        cutoff = int(path.stem.split('-')[-1]) / 1e9 + 0.001
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            event = json.loads(line)
            if event.get('time', 0) <= cutoff:
                records.append(event)
    sessions = []
    for event in records:
        if event['event'] == 'start' and event.get('command') in ('bootstrap', 'selfplay'):
            args = event['arguments']
            sessions.append(dict(stage=event['command'], requested_games=args.get('games'), steps_per_iteration=args.get('steps'), requested_iterations=args.get('iterations'), seed=args.get('seed'), data=args.get('data'), positions=None, completed_steps=0, generated_selfplay_games=0))
        elif sessions:
            if event['event'] == 'loaded_data': sessions[-1]['positions'] = event['positions']
            if event['event'] == 'train':
                previous = sessions[-1].get('_last_step', 0)
                sessions[-1]['completed_steps'] += event['step'] - previous if event['step'] > previous else event['step']
                sessions[-1]['_last_step'] = event['step']
            if event['event'] == 'selfplay_games': sessions[-1]['generated_selfplay_games'] += event['games']
    for session in sessions:
        session.pop('_last_step', None)
    return dict(stage=checkpoint.get('stage'), iteration=checkpoint.get('iteration'),
                transferred_from=checkpoint.get('transferred_from'), training=sessions,
                metrics=str(metrics) if metrics.exists() else None)


def detail(agent_id):
    entry = _entry(agent_id)
    result = {**entry, 'available': available(agent_id), 'settings': entry.get('settings', {})}
    if 'checkpoint' not in entry:
        return {**result, 'shape': None, 'provenance': {'training': []}}
    path = Path(entry['checkpoint'])
    if not path.is_file():
        return {**result, 'shape': None, 'provenance': {}, 'explanation': 'This checkpoint is not installed.'}
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    with _lock:
        if key not in _metadata:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            # BatchNorm statistics are state_dict buffers, not learned parameters.
            parameters = {
                name: tensor.numel() for name, tensor in ck['state'].items()
                if not name.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))
            }
            parts = {}
            for name, count in parameters.items():
                part = name.split('.')[0]
                parts[part] = parts.get(part, 0) + count
            board = Config()
            _metadata[key] = dict(
                shape={k: v for k, v in ck['net_cfg'].items() if k != 'rules'},
                provenance=_provenance(path, ck),
                parameters=sum(parameters.values()), parameter_parts=parts,
                input_planes=NUM_PLANES,
                board=dict(h=board.h, w=board.w, mines=board.mines, bombs=board.bombs),
                policy_beta=float(ck['state']['beta'].item()),
            )
        result.update(_metadata[key])
    result['explanation'] = ('The fully convolutional residual network reads only the public board, claimed mines, scores and side to move. Its mine head predicts mine likelihood, its policy combines a scaled mine logit with a learned residual, and its value head predicts the current player’s outcome. Bootstrap distills exact solver targets; self-play updates policy and value from information-set search and completed games. ' +
        ('This agent greedily takes the highest-policy unopened cell. It is fast, but cannot explicitly compare future turn sequences.' if entry['kind'] == 'net' else
         'Information-set MCTS samples mine layouts consistent with public clues, uses the network for policy priors and leaf values, and selects the most-visited move. Mine hits keep the same player, while safe moves pass the turn. It never sees the real hidden layout. Search costs more time; sampled worlds and a finite simulation budget can miss important outcomes. Higher c_puct favors exploration. Arena uses the listed fixed settings; entries marked request_sims accept Play’s requested simulation budget.'))
    return result


def create(agent_id, sims=100, rng=None):
    entry = _entry(agent_id)
    rng = rng if rng is not None else np.random.default_rng()
    if entry['kind'] == 'random': return RandomAgent(rng)
    if entry['kind'] == 'solver': return SolverAgent(rng)
    path = Path(entry['checkpoint'])
    if not path.is_file(): raise ValueError(f'unavailable agent: {agent_id}')
    key = (str(path.resolve()), _device)
    with _lock:
        if key not in _evaluators:
            _evaluators[key] = Evaluator(load(str(path), _device)[0], _device)
        evaluator = _evaluators[key]
    if entry['kind'] == 'net': return NetAgent(evaluator)
    settings = dict(entry['settings'])
    if entry.get('request_sims'): settings['sims'] = sims
    return MCTSAgent(evaluator, rng=rng, **settings)
