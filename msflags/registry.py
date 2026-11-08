"""Immutable agent catalog backed by shared ResNet and deep checkpoint pins.

Missing weights retain their identities. ``python -m msflags.registry list``
lists the catalog; ``promote`` and ``promote-deep`` append validated pins.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading

import numpy as np
import torch

from .agents import MCTSAgent, NetAgent, RandomAgent, SolverAgent
from .expert import ExpertAgent, Expert2Agent
from .game import Config, NUM_PLANES
from .model import Evaluator, load
from .deep import DeepEvaluator, PLANES as DEEP_PLANES, load_net
from .rollout import RolloutAgent

_device = 'cpu'
_lock = threading.RLock()
_evaluators: dict[tuple[str, str], Evaluator] = {}
_metadata: dict[tuple[str, int], dict] = {}


def configure(device='cpu'):
    global _device
    with _lock:
        _device = device


GENERATIONS_PATH = Path(__file__).with_name('generations.json')
DEEP_GENERATIONS_PATH = Path(__file__).with_name('deep_generations.json')
_SEARCH = dict(sims=100, c_puct=1.5, fpu_reduction=0.2)
CATEGORIES = [
    dict(id='r64x4.greedy', name='ResNet 64×4 · Greedy policy', kind='net', architecture='r64x4', settings={},
         summary='Solver-distilled policy, without tree search.'),
    dict(id='r64x4.mcts100-c1.5', name='ResNet 64×4 · MCTS-100 (c 1.5)', kind='mcts', architecture='r64x4', settings=_SEARCH,
         summary='Information-set search with 100 simulations and exploration c 1.5.'),
    dict(id='r32x3.greedy', name='ResNet 32×3 · Greedy policy', kind='net', architecture='r32x3', settings={},
         summary='Solver-distilled policy, without tree search.'),
    dict(id='r32x3.mcts100-c1.5', name='ResNet 32×3 · MCTS-100 (c 1.5)', kind='mcts', architecture='r32x3', settings=_SEARCH,
         summary='Information-set search with 100 simulations and exploration c 1.5.'),
    dict(id='p96x8.greedy', name='Probability ResNet 96×8 · Greedy policy', kind='deep', architecture='p96x8', settings={},
         summary='Probability-aware expert-distilled policy, without search.'),
    dict(id='p96x8.rollout2x128', name='Probability ResNet 96×8 · Rollout 2×128', kind='rollout', architecture='p96x8',
         settings=dict(turns=2, rollouts=128),
         summary='Probability-aware policy with 128 sampled rollouts truncated after two changes of turn.'),
    dict(id='random', name='Random', kind='random', settings={},
         summary='Uniformly chooses a legal action.',
         explanation='Chooses uniformly among legal open and bomb actions. It never reads clues or scores. This is a deliberately weak baseline, not the rating anchor.'),
    dict(id='solver', name='Greedy exact solver', kind='solver', settings={},
         summary='Claims certain mines, otherwise maximizes immediate expected gain.',
         explanation='Builds constraints from visible clues and the remaining mine count. Opens certain mines first, otherwise uses the highest expected-mine 5×5 window when worth at least three mines; ties are random. Tractable frontiers are enumerated exactly; difficult components use bounded sampling. Its Elo is fixed at 1000.'),
    dict(id='expert', name='Expert (gift-aware, 1-ply)', kind='expert', settings=dict(bomb_reserve=3),
         summary='Claims certain mines, then balances expected gain against gifts to the opponent.',
         explanation='Claims certain mines first. Scores guesses as p/(1-p) minus the expected opponent gift from clues sampled from layouts consistent with the public board. Bombs trade expected mine gain against the opponent gift and a reserve of three. It never reads hidden mines.'),
    dict(id='expert2', name='Expert (gift-aware, 2-ply)', kind='expert2', settings=dict(bomb_reserve=3),
         summary='Gift-aware expert that re-scores top moves against the opponent’s best one-ply reply.',
         explanation='Claims certain mines first. Starts with guess scores p/(1-p) minus the expected opponent gift from clues sampled from layouts consistent with the public board, then re-scores top moves by the opponent’s best one-ply reply. Bombs use a reserve of three. It never reads hidden mines.'),
]
_identities: dict[tuple[str, int], tuple[dict, dict]] = {}


def _generations():
    return json.loads(GENERATIONS_PATH.read_text())


def _deep_generations():
    return json.loads(DEEP_GENERATIONS_PATH.read_text())


def _deep_entry(record, category, generation):
    path = Path(record['checkpoint'])
    return dict(id=f"{record['id']}.{category['id'].split('.', 1)[1]}",
                category_id=category['id'], generation=generation,
                label=record['id'], description=record['description'],
                name=f"{category['name']} · {record['description']}",
                kind=category['kind'], architecture='p96x8', checkpoint=str(path),
                settings=dict(category['settings']), available=path.is_file(),
                request_sims=False, summary=category['summary'],
                shape=dict(channels=96, blocks=8))


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


def architecture_key(shape):
    base = f"r{shape['channels']}x{shape['blocks']}"
    suffix = ('s' if shape.get('score_margin') else '') + ('o' if shape.get('mine_ownership') else '')
    return base + ('a' + suffix if suffix else '')


def _neural_entry(checkpoint, search, *, shape=None, training=None):
    path = Path(checkpoint)
    if shape is None or training is None:
        key = (str(path.resolve()), path.stat().st_mtime_ns)
        with _lock:
            if key not in _identities:
                ck = torch.load(path, map_location='cpu', weights_only=False)
                counts = dict(zip(('bootstrap', 'selfplay'), training_counts(path, ck)))
                _identities[key] = (ck['net_cfg'], counts)
            shape, training = _identities[key]
    bootstrap = format(training['bootstrap'] / 1000, 'g') + 'k'
    arch = architecture_key(shape)
    trained = f'bs{bootstrap}'
    description = f'{bootstrap} bootstrap'
    if training['selfplay']:
        trained += f"-sp{training['selfplay']}"
        description += f" + {training['selfplay']} self-play"
    settings = dict(search or {})
    request_sims = settings.pop('request_sims', search is not None)
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
    return dict(id=f'{arch}-{trained}.{suffix}', category_id=f'{arch}.{suffix}',
                generation=training['selfplay'], label=f"Generation {training['selfplay']}" + (' · Bootstrap' if not training['selfplay'] else ''),
                training=dict(training), shape={k: v for k, v in shape.items() if k != 'rules'},
                name=f"ResNet {shape['channels']}×{shape['blocks']} · {description} · {label}",
                kind=kind, checkpoint=str(path), settings=settings,
                available=path.is_file(), request_sims=request_sims, summary=summary)


def entries():
    """Flat concrete identities; missing files never erase pinned generations."""
    data = _generations()
    deep_records = _deep_generations()
    result = []
    paths = set()
    for architecture, spec in data.items():
        shape = spec['shape']
        if architecture != architecture_key(shape):
            raise ValueError(f'Architecture mismatch: {architecture}')
        for record in spec['generations']:
            path = str(Path(record['checkpoint']).resolve())
            if path in paths:
                raise ValueError('Duplicate checkpoint path in registry')
            paths.add(path)
    deep_ids = set()
    for record in deep_records:
        if not re.fullmatch(r'p96x8-n[1-9][0-9]*', record['id']):
            raise ValueError(f"Invalid deep generation id: {record['id']}")
        if record['id'] in deep_ids:
            raise ValueError('Duplicate deep generation id in registry')
        deep_ids.add(record['id'])
        path = str(Path(record['checkpoint']).resolve())
        if path in paths:
            raise ValueError('Duplicate checkpoint path in registry')
        paths.add(path)
    for category in CATEGORIES:
        architecture = category.get('architecture')
        if architecture == 'p96x8':
            result.extend(_deep_entry(record, category, generation)
                          for generation, record in enumerate(deep_records, 1))
            continue
        if architecture is None:
            result.append({**category, 'category_id': category['id'], 'generation': 0,
                           'label': category['name'], 'available': True,
                           'training': dict(bootstrap=0, selfplay=0)})
            continue
        spec = data[architecture]
        for record in sorted(spec['generations'], key=lambda r: (r['training']['selfplay'], r['training']['bootstrap'])):
            result.append(_neural_entry(record['checkpoint'], category['settings'] if category['kind'] == 'mcts' else None,
                                        shape=record.get('shape', spec['shape']), training=record['training']))
    ids = [entry['id'] for entry in result]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate derived agent id in registry')
    return result


def catalog():
    concrete = entries()
    result = []
    for category in CATEGORIES:
        generations = [e for e in concrete if e['category_id'] == category['id']]
        installed = [e for e in generations if e['available']]
        default = installed[-1] if installed else None
        summary = category.get('summary') or ('Solver-distilled policy, without tree search.' if category['kind'] == 'net'
                  else 'Information-set search with 100 simulations and exploration c 1.5.')
        result.append({**category, 'summary': summary, 'available': bool(installed),
                       'request_sims': category['kind'] == 'mcts',
                       'default_generation': default['id'] if default else None,
                       'generations': generations})
    return result


def resolve(agent_id):
    for category in catalog():
        if category['id'] == agent_id:
            if category['default_generation'] is None:
                raise ValueError(f'unavailable agent: {agent_id}')
            return category['default_generation']
        for generation in category['generations']:
            if generation['id'] == agent_id:
                return agent_id
    raise ValueError(f'unknown agent: {agent_id}')


def _entry(agent_id):
    resolved = resolve(agent_id)
    return next(entry for entry in entries() if entry['id'] == resolved)


def available(agent_id):
    for category in catalog():
        if category['id'] == agent_id:
            return category['available']
        for generation in category['generations']:
            if generation['id'] == agent_id:
                return generation['available']
    raise ValueError(f'unknown agent: {agent_id}')


def pinned_checkpoints() -> set[str]:
    """Paths protected from retention, including pins not installed locally."""
    return ({record['checkpoint'] for spec in _generations().values() for record in spec['generations']}
            | {record['checkpoint'] for record in _deep_generations()})


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
                training=sessions,
                metrics=str(metrics) if metrics.exists() else None)


def detail(agent_id):
    category = next((entry for entry in catalog() if entry['id'] == agent_id), None)
    if category is not None:
        selected = category['default_generation']
        if selected is None and category['generations']:
            selected = category['generations'][-1]['id']
        if selected is not None and selected != agent_id:
            return {**_generation_detail(selected), **category}
        if selected is None:
            return {**category, 'shape': None, 'provenance': {'training': []},
                    'explanation': 'No checkpoint is installed for this category.'}
        return {**_generation_detail(agent_id), **category}
    return _generation_detail(agent_id)


def _generation_detail(agent_id):
    entry = _entry(agent_id)
    result = {**entry, 'available': available(agent_id), 'settings': entry.get('settings', {})}
    if 'checkpoint' not in entry:
        return {**result, 'shape': None, 'provenance': {'training': []}}
    path = Path(entry['checkpoint'])
    if not path.is_file():
        return {**result, 'provenance': {}, 'explanation': 'This checkpoint is not installed.'}
    if entry.get('architecture') == 'p96x8':
        ck = torch.load(path, map_location='cpu', weights_only=False)
        parameters = {name: tensor.numel() for name, tensor in ck['state'].items()
                      if not name.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))}
        return {**result, 'shape': ck['deep_cfg'], 'input_planes': DEEP_PLANES,
                'parameters': sum(parameters.values()),
                'provenance': {'description': entry['description']},
                'explanation': 'The probability-aware residual network reads the public board and solver-derived mine probabilities. It predicts a joint open/bomb policy and a value, without reading hidden mines. ' +
                ('Greedy play chooses the highest-policy legal action.' if entry['kind'] == 'deep' else
                 'Search samples consistent mine layouts and compares candidate actions with 128 rollouts, truncated after two changes of turn.')}
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


def create(agent_id, sims=None, rng=None):
    entry = _entry(agent_id)
    rng = rng if rng is not None else np.random.default_rng()
    baselines = {'random': RandomAgent, 'solver': SolverAgent,
                 'expert': ExpertAgent, 'expert2': Expert2Agent}
    if entry['kind'] in baselines:
        return baselines[entry['kind']](rng, **entry['settings'])
    path = Path(entry['checkpoint'])
    if not path.is_file(): raise ValueError(f'unavailable agent: {agent_id}')
    key = (str(path.resolve()), _device)
    with _lock:
        if entry.get('architecture') == 'p96x8':
            if key not in _evaluators:
                _evaluators[key] = DeepEvaluator(load_net(str(path), _device), _device)
            evaluator = _evaluators[key]
            if entry['kind'] == 'deep':
                return NetAgent(evaluator)
            return RolloutAgent(evaluator, rng, **entry['settings'])
        if key not in _evaluators:
            _evaluators[key] = Evaluator(load(str(path), _device)[0], _device)
        evaluator = _evaluators[key]
    if entry['kind'] == 'net': return NetAgent(evaluator)
    settings = dict(entry['settings'])
    if sims is not None: settings['sims'] = sims
    return MCTSAgent(evaluator, rng=rng, **settings)


def promote(checkpoint):
    """Pin one load-compatible checkpoint; existing identities are immutable."""
    path = Path(checkpoint)
    if path.name == 'latest.pt' or path.resolve().name == 'latest.pt':
        raise ValueError('latest.pt is mutable; promote a timestamped checkpoint')
    data = _generations()
    if str(path.resolve()) in {str(Path(p).resolve()) for p in pinned_checkpoints()}:
        raise ValueError('Duplicate checkpoint path in registry')
    # Use the real loader: inspecting net_cfg alone misses incompatible weights.
    _, ck = load(str(path), 'cpu')
    shape = {key: value for key, value in ck['net_cfg'].items() if key != 'rules'}
    architecture = architecture_key(shape)
    if architecture not in data:
        raise ValueError(f'Unsupported architecture: {architecture}')
    training = dict(zip(('bootstrap', 'selfplay'), training_counts(path, ck)))
    if any(record['training'] == training for record in data[architecture]['generations']):
        raise ValueError('Duplicate derived agent id in registry')
    record = dict(checkpoint=str(path), training=training)
    if shape != data[architecture]['shape']:
        record['shape'] = shape
    data[architecture]['generations'].append(record)
    # Keep every pin on one line: promotion is a single append-only data record.
    lines = ['{']
    for index, (arch, spec) in enumerate(data.items()):
        lines.extend([f'  {json.dumps(arch)}: {{',
                      f'    "shape": {json.dumps(spec["shape"])},',
                      '    "generations": ['])
        records = spec['generations']
        lines.extend('      ' + json.dumps(item) + (',' if i < len(records) - 1 else '')
                     for i, item in enumerate(records))
        lines.extend(['    ]', '  }' + (',' if index < len(data) - 1 else '')])
    lines.append('}')
    GENERATIONS_PATH.write_text('\n'.join(lines) + '\n')
    return [_neural_entry(path, search, shape=shape, training=training)
            for search in (None, _SEARCH)]


def promote_deep(checkpoint, generation_id, description):
    """Append one immutable probability-aware 96×8 checkpoint pin."""
    path = Path(checkpoint)
    if path.name == 'latest.pt' or path.resolve().name == 'latest.pt':
        raise ValueError('latest.pt is mutable; promote an immutable checkpoint')
    if not re.fullmatch(r'n[1-9][0-9]*', generation_id):
        raise ValueError('Deep id must be n followed by a positive integer')
    if not description.strip():
        raise ValueError('Deep description must not be empty')
    records = _deep_generations()
    identity = f'p96x8-{generation_id}'
    if any(record['id'] == identity for record in records):
        raise ValueError('Duplicate deep generation id in registry')
    if str(path.resolve()) in {str(Path(p).resolve()) for p in pinned_checkpoints()}:
        raise ValueError('Duplicate checkpoint path in registry')
    net = load_net(str(path), 'cpu')
    if (net.cfg.channels, net.cfg.blocks) != (96, 8):
        raise ValueError('Unsupported deep architecture: expected 96×8')
    record = dict(id=identity, checkpoint=str(path), description=description.strip())
    records.append(record)
    DEEP_GENERATIONS_PATH.write_text('[\n' + ',\n'.join('  ' + json.dumps(item) for item in records) + '\n]\n')
    return [_deep_entry(record, category, len(records)) for category in CATEGORIES
            if category.get('architecture') == 'p96x8']


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description='List immutable agent generations or promote a checkpoint.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='List categories and every pinned generation.')
    promotion = commands.add_parser('promote', help='Append an immutable checkpoint to its architecture.')
    promotion.add_argument('checkpoint')
    deep_promotion = commands.add_parser('promote-deep', help='Append an immutable probability-aware checkpoint.')
    deep_promotion.add_argument('checkpoint')
    deep_promotion.add_argument('--id', required=True)
    deep_promotion.add_argument('--description', required=True)
    args = parser.parse_args(argv)
    if args.command == 'list':
        print(json.dumps(catalog(), indent=2))
    else:
        try:
            promoted = (promote_deep(args.checkpoint, args.id, args.description)
                        if args.command == 'promote-deep' else promote(args.checkpoint))
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        print(json.dumps(promoted, indent=2))


if __name__ == '__main__':
    main()
