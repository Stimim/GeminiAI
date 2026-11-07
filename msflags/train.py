"""Solver distillation, batched information-set self-play, and alternating-seat arenas."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import itertools
import math
import os
from pathlib import Path
import multiprocessing
import signal
import queue
import warnings
import time

import numpy as np
import torch
import torch.nn.functional as F

from .agents import RandomAgent, SolverAgent
from .game import Config, Game, UNKNOWN, encode_batch
from .mcts import Search
from .model import Evaluator, Net, NetConfig, load, save
from .solver import Analysis


@dataclass
class Position:
    board: np.ndarray
    scores: list[int]
    player: int
    policy: np.ndarray
    mine: np.ndarray
    bombs: list[int]
    value: float = 0.0


def dihedral(x: np.ndarray, symmetry: int) -> np.ndarray:
    """Apply one of eight square symmetries on the final two axes."""
    y = np.rot90(x, symmetry % 4, axes=(-2, -1))
    return np.flip(y, axis=-1) if symmetry >= 4 else y


def augment(x, policy, mine, symmetry):
    h, w = x.shape[-2:]
    return (dihedral(x, symmetry).copy(),
            dihedral(policy.reshape(-1, h, w), symmetry).reshape(policy.shape).copy(),
            dihedral(mine.reshape(-1, h, w), symmetry).reshape(mine.shape).copy())


def bootstrap_game(job):
    cfg, seed, epsilon = job
    rng = np.random.default_rng(seed)
    g = Game.random(cfg, rng)
    positions = []
    while not g.done:
        pub = g.public()
        a = Analysis(pub, rng=rng)
        best = SolverAgent(rng).best_actions(pub, a.probs)
        # The solver can fall back to sampling on hard components. Those
        # positions may drive play, but must not masquerade as exact labels.
        if a.exact:
            target = np.zeros(2 * cfg.n, np.float32)
            target[best] = 1 / len(best)
            positions.append(Position(g.board.copy(), list(g.scores), g.to_move, target,
                                      a.probs.astype(np.float32), bombs=list(g.bombs)))
        legal = g.legal_actions()
        action = rng.choice(legal if rng.random() < epsilon else best)
        g.step(int(action))
    for p in positions:
        p.value = 1. if p.player == g.winner else -1.
    return positions


def pack(positions, cfg):
    return dict(board=np.stack([p.board for p in positions]),
                scores=np.array([p.scores for p in positions], np.int16),
                bombs=np.array([p.bombs for p in positions], np.int8),
                player=np.array([p.player for p in positions], np.int8),
                policy=np.stack([p.policy for p in positions]).astype(np.float32),
                mine=np.stack([p.mine for p in positions]).astype(np.float32),
                value=np.array([p.value for p in positions], np.float32))


def log(args, event, **fields):
    record = dict(event=event, time=time.time(), **fields)
    if hasattr(args, 'iteration'):
        record.setdefault('iteration', args.iteration)
    print(json.dumps(record), flush=True)
    with (Path(args.output) / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')


def checkpoint(net, args, tag, **metadata):
    """Publish complete timestamped weights before atomically updating latest."""
    path = Path(args.output)
    temporary = path / '.checkpoint.tmp'
    target = path / f'{tag}.pt'
    save(net, str(temporary), **metadata)
    os.replace(temporary, target)
    latest = path / '.latest.tmp'
    # Checkpoints are immutable; a hard link avoids copying optimizer tensors.
    if latest.exists():
        latest.unlink()
    os.link(target, latest)
    os.replace(latest, path / 'latest.pt')
    return target


def fit(net, data, args, steps, optimizer=None):
    """Random replay minibatches; augmentation is shared by inputs and both cell heads."""
    optimizer = optimizer or torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(data['board'])
    rng = np.random.default_rng(args.seed + getattr(args, 'iteration', 0))
    cfg = Config(args.size, args.size, args.mines)
    start = time.monotonic()
    # Compact preencoded CPU storage avoids repeating neighbour computations per epoch.
    availability = data['bombs']
    x = np.empty((n, 20, cfg.h, cfg.w), dtype=np.float16)
    for offset in range(0, n, 1024):
        part = slice(offset, offset + 1024)
        states = list(zip(data['board'][part], data['scores'][part], data['player'][part], availability[part]))
        x[part] = encode_batch(states, cfg)
    net.train()
    total = np.zeros(3)
    for step in range(1, steps + 1):
        ids = rng.integers(n, size=args.batch_size)
        xb, pi, mine = augment(x[ids], data['policy'][ids], data['mine'][ids], int(rng.integers(8)))
        xb = torch.as_tensor(xb, device=args.device, dtype=torch.float32)
        pi = torch.as_tensor(pi, device=args.device)
        mine = torch.as_tensor(mine, device=args.device)
        value = torch.as_tensor(data['value'][ids], device=args.device)
        with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                            enabled=torch.device(args.device).type == 'cuda'):
            logits, pred_value, pred_mine = net(xb)
            policy_loss = -(pi * F.log_softmax(logits.float(), 1)).sum(1).mean()
            value_loss = F.mse_loss(pred_value.float(), value)
            mask = xb[:, 0].flatten(1)
            mine_loss = (F.binary_cross_entropy_with_logits(pred_mine.float(), mine, reduction='none') * mask).sum() / mask.sum()
            loss = policy_loss + args.value_weight * value_loss + args.mine_weight * mine_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += [policy_loss.item(), value_loss.item(), mine_loss.item()]
        if step % args.log_every == 0 or step == steps:
            denom = args.log_every if step % args.log_every == 0 else step % args.log_every
            log(args, 'train', step=step, policy_loss=total[0]/denom, value_loss=total[1]/denom,
                mine_loss=total[2]/denom, steps_per_second=step/(time.monotonic()-start))
            total[:] = 0
    net.eval()
    return optimizer


def new_or_load(args, required=False):
    path = Path(args.checkpoint or str(Path(args.output) / 'latest.pt'))
    if path.exists():
        net = load(str(path), args.device)[0]
        return net
    if required or args.checkpoint:
        raise FileNotFoundError(path)
    preset = getattr(args, 'network_size', None)
    channels, blocks = {'small': (32, 3), 'medium': (64, 4)}[preset] if preset else (args.channels, args.blocks)
    return Net(NetConfig(channels, blocks, args.policy_beta)).to(args.device)


def bootstrap(args):
    cfg = Config(args.size, args.size, args.mines)
    data_path = Path(args.data or str(Path(args.output) / 'bootstrap-exact.npz'))
    if data_path.exists():
        with np.load(data_path) as stored:
            data = dict(stored)
        if data['board'].shape[1] != cfg.n or data['policy'].shape[1] != 2 * cfg.n:
            raise ValueError('dataset must match the board size and contain 2N actions')
        log(args, 'loaded_data', positions=len(data['board']))
    else:
        positions = []
        start = time.monotonic()
        jobs = [(cfg, args.seed + i, args.epsilon) for i in range(args.games)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, samples in enumerate(pool.map(bootstrap_game, jobs, chunksize=1), 1):
                positions.extend(samples)
                if i % args.game_log_every == 0 or i == args.games:
                    log(args, 'bootstrap_games', games=i, positions=len(positions), games_per_second=i/(time.monotonic()-start))
        data = pack(positions, cfg)
        del positions
        np.savez(data_path, **data)
    net = new_or_load(args)
    for epoch in range(1, args.iterations + 1):
        fit(net, data, args, args.steps)
        checkpoint(net, args, f'bootstrap-{time.time_ns()}', stage='bootstrap', iteration=epoch,
                   bootstrap_steps=args.steps * epoch, selfplay_iterations=0)


def run_searches(searches, evaluator, cfg, leaf_batch_size=1):
    """Batch multiple virtual-loss-diversified leaves from every live search."""
    active = list(searches)
    while active:
        pending, leaves = [], []
        for s in active:
            batch = s.next_leaves(leaf_batch_size)
            if batch:
                pending.append((s, len(batch)))
                leaves.extend(batch)
        if not pending:
            break
        policies, values, _ = evaluator(leaves, cfg)
        offset = 0
        for s, count in pending:
            s.provide_batch(policies[offset:offset + count], values[offset:offset + count])
            offset += count
        active = [s for s, _ in pending]


def selfplay_batch(args, evaluator, seed):
    cfg = Config(args.size, args.size, args.mines)
    rng = np.random.default_rng(seed)
    games = [Game.random(cfg, rng) for _ in range(args.games)]
    histories = [[] for _ in games]
    roots = [None] * len(games)
    moves = [0] * len(games)
    action_counts = [dict(bomb=0, safe=0, mine=0) for _ in games]
    active = list(range(len(games)))
    start = time.monotonic()
    while active:
        searches = [Search(games[i].public(), args.sims, rng=rng, root=roots[i],
                           c_puct=args.c_puct, fpu_reduction=args.fpu_reduction,
                           dirichlet_alpha=args.dirichlet_alpha,
                           dirichlet_frac=args.dirichlet_frac,
                           search=getattr(args, 'search', 'puct'),
                           gumbel_k=getattr(args, 'gumbel_k', 16),
                           gumbel_noise=getattr(args, 'gumbel_noise', True)) for i in active]
        run_searches(searches, evaluator, cfg, getattr(args, 'leaf_batch_size', 1))
        next_active = []
        for i, s in zip(active, searches):
            g = games[i]
            pi = s.improved_policy().astype(np.float32)
            histories[i].append(Position(g.board.copy(), list(g.scores), g.to_move, pi,
                                         s.analysis.probs.astype(np.float32), bombs=list(g.bombs)))
            if moves[i] < args.temperature_moves and args.temperature > 0:
                weights = pi.astype(np.float64) ** (1 / args.temperature)
                action = int(rng.choice(len(pi), p=weights / weights.sum()))
            else:
                action = s.best_action()
            kind = 'bomb' if action >= cfg.n else 'mine' if g.mines[action] else 'safe'
            action_counts[i][kind] += 1
            g.step(action)
            roots[i] = s.child_root(action, g.board)
            moves[i] += 1
            if not g.done:
                next_active.append(i)
        active = next_active
    positions = []
    for g, history in zip(games, histories):
        for p in history:
            p.value = 1. if p.player == g.winner else -1.
        positions.extend(history)
    log(args, 'selfplay_games', games=len(games), positions=len(positions),
        games_per_second=len(games)/(time.monotonic()-start),
        split=getattr(args, 'generation_split', 'replay'),
        game_statistics=[dict(moves=sum(counts.values()), **counts) for counts in action_counts],
        action_counts={kind: sum(counts[kind] for counts in action_counts)
                       for kind in ('bomb', 'safe', 'mine')})
    return pack(positions, cfg)


_worker_evaluator = None


class _WorkerEvaluator:
    """CPU search workers share the trainer's single GPU model through a queue."""

    def __init__(self, requests):
        self.requests = requests

    def __call__(self, states, cfg):
        receive, send = multiprocessing.Pipe(duplex=False)
        try:
            self.requests.put((states, cfg, send))
            ok, result = receive.recv()
            if not ok:
                raise RuntimeError(result)
            return result
        finally:
            receive.close()
            send.close()


def _init_selfplay_worker(requests, threads):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # A terminated trainer must not strand large, signal-ignoring worker processes.
    if os.name == 'posix' and Path('/proc/self').exists():
        import ctypes
        parent_process = multiprocessing.parent_process()
        parent = parent_process.pid if parent_process is not None else os.getppid()
        if os.getppid() != parent:
            os._exit(1)
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot set worker parent-death signal')
        if os.getppid() != parent:
            os._exit(1)
    global _worker_evaluator
    torch.set_num_threads(threads)
    _worker_evaluator = _WorkerEvaluator(requests)


def _selfplay_worker(job):
    args, seed = job
    return selfplay_batch(args, _worker_evaluator, seed)


def _serve_worker_requests(evaluator, requests):
    states, cfg, connection = requests.get(timeout=.05)
    pending = [(states, cfg, connection)]
    # Drain already waiting workers without delaying the first requester.
    while True:
        try:
            pending.append(requests.get_nowait())
        except queue.Empty:
            break
    leaves = [state for states, _, _ in pending for state in states]
    try:
        # Bound inference activation memory independently of worker count and K.
        evaluated = [evaluator(leaves[i:i + 64], cfg) for i in range(0, len(leaves), 64)]
        outputs = [np.concatenate([part[k] for part in evaluated]) for k in range(3)]
        offset = 0
        for states, _, connection in pending:
            count = len(states)
            connection.send((True, tuple(output[offset:offset + count] for output in outputs)))
            offset += count
    except BaseException as error:
        for _, _, connection in pending:
            try:
                connection.send((False, str(error)))
            except (BrokenPipeError, EOFError, OSError):
                pass
        raise
    finally:
        for _, _, connection in pending:
            connection.close()


def selfplay_games(args, evaluator, seed):
    jobs = []
    for offset in range(0, args.games, args.concurrent_games):
        batch_args = argparse.Namespace(**vars(args))
        batch_args.games = min(args.concurrent_games, args.games - offset)
        jobs.append((batch_args, seed + offset))
    pool = getattr(evaluator, 'selfplay_pool', None)
    if pool is None:
        batches = [selfplay_batch(batch_args, evaluator, batch_seed) for batch_args, batch_seed in jobs]
    else:
        requests = evaluator.selfplay_requests
        futures = [pool.submit(_selfplay_worker, job) for job in jobs]
        try:
            while not all(future.done() for future in futures):
                for future in futures:
                    if future.done() and not future.cancelled() and future.exception() is not None:
                        raise future.exception()
                try:
                    _serve_worker_requests(evaluator, requests)
                except queue.Empty:
                    pass
            batches = [future.result() for future in futures]
        except BaseException as error:
            # Release every waiting worker before the executor's shutdown joins it.
            for future in futures:
                future.cancel()
            while not all(future.done() for future in futures):
                try:
                    _, _, connection = requests.get(timeout=.05)
                except queue.Empty:
                    continue
                try:
                    connection.send((False, str(error)))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                finally:
                    connection.close()
            raise
    return {k: np.concatenate([batch[k] for batch in batches]) for k in batches[0]}


def validate_value(args, evaluator, seed):
    """Fresh complete self-play games, scored but never inserted into replay."""
    validation_args = argparse.Namespace(**vars(args))
    validation_args.games = args.value_validation_games
    validation_args.generation_split = 'validation'
    started = time.monotonic()
    data = selfplay_games(validation_args, evaluator, seed)
    generation_seconds = time.monotonic() - started
    predictions = []
    cfg = Config(args.size, args.size, args.mines)
    for offset in range(0, len(data['board']), 128):
        part = slice(offset, offset + 128)
        states = list(zip(data['board'][part], data['scores'][part],
                          data['player'][part], data['bombs'][part]))
        _, values, _ = evaluator(states, cfg)
        predictions.append(values)
    values = np.concatenate(predictions)
    target = data['value']
    rows = np.arange(len(target))
    margin = (data['scores'][rows, data['player']] -
              data['scores'][rows, 1 - data['player']])
    scores = data['scores'].sum(1)
    bands = []
    for low, high in ((0, 10), (10, 25), (25, 40), (40, 2 * cfg.win_score + 1)):
        mask = (scores >= low) & (scores < high)
        if mask.any():
            bands.append(dict(score_min=low, score_max_exclusive=high,
                              positions=int(mask.sum()),
                              mse=float(np.mean((values[mask] - target[mask]) ** 2)),
                              accuracy=float(np.mean((values[mask] > 0) == (target[mask] > 0)))))
    log(args, 'value_validation', games=validation_args.games, positions=len(target),
        seed=seed, policy='selfplay', replay_inserted=False,
        mse=float(np.mean((values - target) ** 2)),
        zero_mse=float(np.mean(target ** 2)),
        score_leader_accuracy=float(np.mean((margin > 0) == (target > 0))),
        accuracy=float(np.mean((values > 0) == (target > 0))),
        prediction_mean=float(values.mean()), target_mean=float(target.mean()),
        score_bands=bands, generation_seconds=generation_seconds,
        seconds=time.monotonic() - started)


DISK_LIMIT_BYTES = 100_000_000_000


def disk_bytes(root):
    """Count allocated project file sizes once per inode (latest.pt is a hard link)."""
    seen = set()
    total = 0
    for directory, _, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                seen.add(key)
                total += stat.st_size
    return total


def prune_checkpoints(root='checkpoints', dry_run=False, limit=DISK_LIMIT_BYTES, *, output=None):
    """Measure root globally, but training may delete only inside its own output."""
    root = Path(root)
    deletion_root = Path(output).resolve() if output is not None else root.resolve()
    before = disk_bytes(root)
    project_root = Path.cwd() if root.resolve().is_relative_to(Path.cwd()) else root
    project_usage = disk_bytes(project_root)
    try:
        from .registry import pinned_checkpoints
    except ImportError:
        warnings.warn('registry.pinned_checkpoints unavailable: retaining every checkpoint', RuntimeWarning)
        return dict(before_bytes=before, after_bytes=before, project_bytes=project_usage,
                    deleted=[], over_budget=max(before, project_usage) > limit)
    pinned = {Path(path).resolve() for path in pinned_checkpoints()}
    deleted = []
    for directory, _, files in os.walk(deletion_root):
        directory = Path(directory)
        candidates = []
        for name in files:
            if not (name.startswith('selfplay-') and name.endswith('.pt')):
                continue
            path = directory / name
            if not path.resolve().is_relative_to(deletion_root):
                continue
            try:
                metadata = torch.load(path, map_location='cpu', weights_only=False)
                iteration = metadata.get('selfplay_iterations', metadata.get('iteration'))
                if iteration is not None:
                    candidates.append((int(iteration), path))
            except (OSError, RuntimeError, EOFError, ValueError):
                warnings.warn(f'Cannot inspect {path}; retaining it', RuntimeWarning)
        newest = set(sorted({iteration for iteration, _ in candidates})[-3:])
        for iteration, path in candidates:
            if path.resolve() in pinned or iteration % 10 == 0 or iteration in newest:
                continue
            deleted.append(str(path))
            if not dry_run:
                path.unlink(missing_ok=True)
    after = disk_bytes(root)
    project_usage = disk_bytes(project_root)
    over_budget = max(after, project_usage) > limit
    if over_budget:
        warnings.warn(f'DISK BUDGET EXCEEDED: project={project_usage}, checkpoints={after} bytes > {limit}; training must stop. Milestones and pinned weights retained.', RuntimeWarning)
    return dict(before_bytes=before, after_bytes=after, project_bytes=project_usage,
                deleted=deleted, over_budget=over_budget)


def prune(args):
    print(json.dumps(prune_checkpoints(args.root, args.dry_run)), flush=True)


def selfplay(args):
    # A run directory always wins over its original bootstrap --checkpoint.
    latest = Path(args.output) / 'latest.pt'
    source = latest if latest.exists() else Path(args.checkpoint or latest)
    args.checkpoint = str(source)
    net, prior = load(source, args.device)
    from .registry import training_counts
    bootstrap_steps, previous_iterations = training_counts(source, prior)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if 'optimizer' in prior:
        optimizer.load_state_dict(prior['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr
    replay = None
    replay_path = Path(args.output) / prior.get('replay_file', 'replay.npz')
    if replay_path.exists():
        with np.load(replay_path) as stored:
            replay = {k: stored[k][-args.replay_size:].copy() for k in stored.files}
    elif prior.get('replay_file'):
        raise FileNotFoundError(f'checkpoint replay buffer missing: {replay_path}')
    log(args, 'resume', checkpoint=str(source), completed_iterations=previous_iterations,
        replay_positions=0 if replay is None else len(replay['board']))
    initial_retention = prune_checkpoints(args.disk_root, output=args.output)
    log(args, 'retention', **initial_retention)
    if initial_retention['over_budget']:
        return
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    pool = None
    requests = None
    if args.workers > 1:
        context = multiprocessing.get_context('spawn')
        requests = context.Queue()
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                   initializer=_init_selfplay_worker,
                                   initargs=(requests, args.worker_threads))
    evaluator = Evaluator(net, args.device)
    evaluator.selfplay_pool = pool
    evaluator.selfplay_requests = requests
    iterations = range(previous_iterations + 1, args.iterations + 1) if args.iterations else itertools.count(previous_iterations + 1)
    for iteration in iterations:
        args.iteration = iteration
        started = time.monotonic()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(args.device)
        generation_start = time.monotonic()
        data = selfplay_games(args, evaluator, args.seed + iteration)
        generation_seconds = time.monotonic() - generation_start
        new_positions = len(data['board'])
        replay = {k: data[k][-args.replay_size:].copy() if replay is None else
                  np.concatenate((replay[k], data[k]))[-args.replay_size:].copy() for k in data}
        del data
        training_start = time.monotonic()
        fit(net, replay, args, args.steps, optimizer)
        training_seconds = time.monotonic() - training_start
        save_start = time.monotonic()
        # Alternate slots keep the last published checkpoint's replay intact
        # if interrupted while writing the next iteration.
        replay_file = f'replay-{iteration % 2}.npz'
        temporary = Path(args.output) / '.replay.tmp'
        with temporary.open('wb') as stream:
            np.savez(stream, **replay)
        os.replace(temporary, Path(args.output) / replay_file)
        saved = checkpoint(net, args, f'selfplay-{time.time_ns()}', stage='selfplay', iteration=iteration,
                           bootstrap_steps=bootstrap_steps, selfplay_iterations=iteration,
                           replay_file=replay_file, optimizer=optimizer.state_dict(),
                           training_settings=vars(args).copy())
        save_seconds = time.monotonic() - save_start
        validation_seconds = 0.0
        if args.value_validation_every and iteration % args.value_validation_every == 0:
            validation_start = time.monotonic()
            validate_value(args, evaluator, args.seed + 1_000_000_000 + iteration * args.games)
            validation_seconds = time.monotonic() - validation_start
        if args.eval_every and iteration % args.eval_every == 0:
            result = arena_games(args, 'greedy', 'solver', args.eval_games, net=net)
            log(args, 'evaluation', iteration=iteration, **result)
        log(args, 'iteration', iteration=iteration, seconds=time.monotonic() - started,
            generation_seconds=generation_seconds, training_seconds=training_seconds,
            save_seconds=save_seconds, validation_seconds=validation_seconds, games=args.games,
            games_per_second=args.games / generation_seconds, new_positions=new_positions,
            replay_positions=len(replay['board']), sample_reuse=args.steps * args.batch_size / new_positions,
            checkpoint=str(saved),
            peak_vram_mb=(torch.cuda.max_memory_allocated(args.device) / 2**20
                          if torch.device(args.device).type == 'cuda' else 0))
        retention = prune_checkpoints(args.disk_root, output=args.output)
        log(args, 'retention', **retention)
        if retention['over_budget'] or stop_requested:
            break
    if pool is not None:
        pool.shutdown()
    if requests is not None:
        requests.close()
        requests.join_thread()
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)


def wilson(wins, games):
    z = 1.959963984540054
    p = wins / games
    d = 1 + z*z/games
    middle = (p + z*z/(2*games)) / d
    half = z * math.sqrt(p*(1-p)/games + z*z/(4*games*games)) / d
    return [middle-half, middle+half]


def arena_games(args, first, second, count, net=None):
    cfg = Config(args.size, args.size, args.mines)
    rng = np.random.default_rng(args.seed)
    from . import registry
    entries = [
        dict(id=agent_id, kind='net', settings={}) if net is not None and agent_id == 'greedy'
        else registry._entry(agent_id)
        for agent_id in (first, second)
    ]
    kinds = [entry['kind'] for entry in entries]
    evaluators = {}
    for side, entry in enumerate(entries):
        if entry['kind'] in ('net', 'mcts'):
            override = args.checkpoint if side == 0 else getattr(args, 'checkpoint_b', None)
            path = override or entry.get('checkpoint')
            model = net if net is not None else load(path, args.device)[0]
            evaluators[side] = Evaluator(model, args.device)
    basic = {'solver': SolverAgent(rng), 'random': RandomAgent(rng)}
    wins = 0
    start = time.monotonic()
    completed = 0
    for offset in range(0, count, args.concurrent_games):
        ids = list(range(offset, min(count, offset+args.concurrent_games)))
        games = [Game.random(cfg, rng) for _ in ids]
        previous = [[None] * len(games) for _ in range(2)]
        active = list(range(len(games)))
        while active:
            actions = {}
            for side, kind in enumerate(kinds):
                selected = [i for i in active if (games[i].to_move ^ (ids[i] % 2)) == side]
                if not selected:
                    continue
                if kind in basic:
                    actions.update({i: basic[kind].act(games[i].public()) for i in selected})
                elif kind == 'net':
                    policies, _, _ = evaluators[side]([(games[i].board, games[i].scores, games[i].to_move, games[i].bombs) for i in selected], cfg)
                    actions.update({i: int(np.argmax(p)) for i, p in zip(selected, policies)})
                else:
                    settings = entries[side]['settings']
                    override = args.checkpoint if side == 0 else getattr(args, 'checkpoint_b', None)
                    if override:
                        settings = dict(sims=args.sims, c_puct=args.c_puct, fpu_reduction=args.fpu_reduction,
                                        search=getattr(args, 'search', 'puct'),
                                        gumbel_k=getattr(args, 'gumbel_k', 16),
                                        gumbel_noise=getattr(args, 'gumbel_noise', False))
                    searches = []
                    for i in selected:
                        pub = games[i].public()
                        old = previous[side][i]
                        root = old.matching_root(pub) if old is not None else None
                        search = Search(pub, rng=rng, root=root, **settings)
                        searches.append(search)
                        previous[side][i] = search
                    run_searches(searches, evaluators[side], cfg, getattr(args, 'leaf_batch_size', 1))
                    actions.update({i: s.best_action() for i, s in zip(selected, searches)})
            remaining = []
            for i in active:
                games[i].step(actions[i])
                if games[i].done:
                    wins += int((games[i].winner ^ (ids[i] % 2)) == 0)
                    completed += 1
                else:
                    remaining.append(i)
            active = remaining
        log(args, 'arena_progress', games=completed, wins=wins, games_per_second=completed/(time.monotonic()-start))
    return dict(agent_a=first, agent_b=second, games=count, wins=wins, win_rate=wins/count,
                confidence_95=wilson(wins,count), seconds=time.monotonic()-start)


def arena(args):
    log(args, 'arena', **arena_games(args, args.agent_a, args.agent_b, args.games))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    pruning = subs.add_parser('prune', help='Retain pinned, milestone, and newest-three self-play checkpoints.')
    pruning.add_argument('--root', default='checkpoints')
    pruning.add_argument('--dry-run', action='store_true')
    for name in ('bootstrap', 'selfplay', 'arena'):
        s = subs.add_parser(name)
        s.add_argument('--output', default='checkpoints')
        s.add_argument('--checkpoint')
        s.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        s.add_argument('--size', type=int, default=16)
        s.add_argument('--mines', type=int, default=51)
        s.add_argument('--seed', type=int, default=42)
        s.add_argument('--games', type=int, default=1000 if name == 'bootstrap' else 32 if name == 'selfplay' else 200)
        s.add_argument('--threads', type=int, default=4)
        s.add_argument('--sims', type=int, default=100)
        s.add_argument('--c-puct', type=float, default=1.5)
        s.add_argument('--fpu-reduction', type=float, default=0.2)
        s.add_argument('--search', choices=('puct', 'gumbel'), default='puct')
        s.add_argument('--gumbel-k', type=int, default=16)
        s.add_argument('--gumbel-noise', action=argparse.BooleanOptionalAction,
                       default=name == 'selfplay', help='Gumbel exploration (off by default for arena).')
        s.add_argument('--leaf-batch-size', type=int, default=1)
        s.add_argument('--concurrent-games', type=int, default=32)
        if name != 'arena':
            s.add_argument('--iterations', type=int, default=1 if name == 'bootstrap' else 10)
            s.add_argument('--steps', type=int, default=5000 if name == 'bootstrap' else 500)
            s.add_argument('--batch-size', type=int, default=128)
            s.add_argument('--lr', type=float, default=0.001)
            s.add_argument('--weight-decay', type=float, default=0.0001)
            s.add_argument('--mine-weight', type=float, default=2.)
            s.add_argument('--value-weight', type=float, default=0.25)
            s.add_argument('--log-every', type=int, default=100)
        if name == 'bootstrap':
            s.add_argument('--workers', type=int, default=min(12, os.cpu_count() or 1))
            s.add_argument('--epsilon', type=float, default=0.08)
            s.add_argument('--channels', type=int, default=64)
            s.add_argument('--blocks', type=int, default=4)
            s.add_argument('--network-size', choices=('small', 'medium'),
                           help='Architecture preset: small=32x3, medium=64x4; overrides channels/blocks.')
            s.add_argument('--policy-beta', type=float, default=3.)
            s.add_argument('--data')
            s.add_argument('--game-log-every', type=int, default=100)
        if name == 'selfplay':
            s.add_argument('--workers', type=int, default=1, help='Independent spawned self-play processes.')
            s.add_argument('--worker-threads', type=int, default=1)
            s.add_argument('--cpu-share', type=int,
                           help='Total CPU thread budget for trainer plus spawned workers.')
            s.add_argument('--disk-root', default='checkpoints')
            s.add_argument('--replay-size', type=int, default=100000)
            s.add_argument('--dirichlet-alpha', type=float, default=0.15)
            s.add_argument('--dirichlet-frac', type=float, default=0.25)
            s.add_argument('--temperature', type=float, default=1.)
            s.add_argument('--temperature-moves', type=int, default=30)
            s.add_argument('--eval-every', type=int, default=1)
            s.add_argument('--eval-games', type=int, default=40)
            s.add_argument('--value-validation-every', type=int, default=5,
                           help='Fresh held-out self-play validation cadence; zero disables.')
            s.add_argument('--value-validation-games', type=int, default=16)
        if name == 'arena':
            from . import registry
            entries = registry.entries()
            ids = [entry['id'] for entry in entries] + [entry['id'] for entry in registry.catalog()]
            default = next((entry['id'] for entry in entries if entry.get('request_sims')), 'solver')
            s.add_argument('--agent-a', choices=ids, default=default,
                           help='Catalog identity; --checkpoint overrides its weights for an unrecorded evaluation.')
            s.add_argument('--agent-b', choices=ids, default='solver')
            s.add_argument('--checkpoint-b')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.command == 'prune':
        prune(args)
        return
    if args.size < 2 or args.mines < 1 or args.mines >= args.size**2 or args.mines % 2 == 0:
        p.error('use a square board with a positive odd mine count smaller than its area (no draws)')
    if args.games < 1 or args.sims < 1 or args.concurrent_games < 1:
        p.error('games, sims, and concurrent-games must be positive')
    if args.leaf_batch_size < 1 or args.gumbel_k < 1:
        p.error('leaf-batch-size and gumbel-k must be positive')
    if args.command == 'selfplay' and args.cpu_share is not None:
        minimum = args.workers + 1 if args.workers > 1 else 1
        if args.cpu_share < minimum:
            p.error('cpu-share must cover one thread per worker plus the trainer')
        if args.workers > 1:
            args.threads = 1
            args.worker_threads = (args.cpu_share - 1) // args.workers
        else:
            args.threads = args.cpu_share
    if args.command != 'arena' and (args.steps < 1 or args.batch_size < 1 or args.iterations < 0 or args.log_every < 1):
        p.error('steps, batch-size, and log-every must be positive; iterations must be nonnegative')
    if args.command == 'bootstrap' and args.iterations == 0:
        p.error('bootstrap iterations must be positive')
    if args.command == 'selfplay' and (args.replay_size < 1 or args.eval_every < 0 or args.eval_games < 1):
        p.error('replay-size and eval-games must be positive; eval-every must be nonnegative')
    if args.command == 'selfplay' and (args.workers < 1 or args.worker_threads < 1):
        p.error('workers and worker-threads must be positive')
    if args.command == 'selfplay' and (args.value_validation_every < 0 or args.value_validation_games < 1):
        p.error('value-validation-every must be nonnegative; value-validation-games must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    log(args, 'start', command=args.command, arguments=vars(args))
    globals()[args.command](args)


if __name__ == '__main__':
    main()
