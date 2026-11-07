"""Unrecorded paired-layout, alternating-seat search comparisons."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .game import Config, Game
from .mcts import Search
from .model import Evaluator, load
from .train import run_searches


def paired_games(evaluators, cfg, *, games, seed, mode='mcts', sims=100,
                 c_puct=1.5, fpu_reduction=0.2, concurrent_games=16,
                 mode_b=None, sims_b=None, gumbel_k=16, leaf_batch_size=1,
                 tree_reuse=True, tree_reuse_b=None, leaf_batch_size_b=None):
    if games < 2 or games % 2:
        raise ValueError('games must be positive and even (two seats per layout)')
    modes = [mode, mode if mode_b is None else mode_b]
    budgets = [sims, sims if sims_b is None else sims_b]
    reuse = [tree_reuse, tree_reuse if tree_reuse_b is None else tree_reuse_b]
    batches = [leaf_batch_size, leaf_batch_size if leaf_batch_size_b is None else leaf_batch_size_b]
    if min(*budgets, *batches, concurrent_games, gumbel_k) < 1:
        raise ValueError('sims, batch sizes, gumbel-k and concurrent-games must be positive')
    if any(m not in ('greedy', 'mcts', 'puct', 'gumbel') for m in modes):
        raise ValueError('mode must be greedy, mcts, puct or gumbel')
    started = time.monotonic()
    outcomes, move_counts, side_seconds = [], [0, 0], [0., 0.]
    for offset in range(0, games, concurrent_games):
        ids = list(range(offset, min(games, offset + concurrent_games)))
        batch = [Game.random(cfg, np.random.default_rng(np.random.SeedSequence([seed, 0, i // 2])))
                 for i in ids]
        rngs = [[np.random.default_rng(np.random.SeedSequence([seed, 1, i // 2, i % 2, side]))
                 for i in ids] for side in range(2)]
        previous = [[None] * len(batch) for _ in range(2)]
        active = list(range(len(batch)))
        while active:
            actions = {}
            for side, evaluator in enumerate(evaluators):
                selected = [i for i in active if (batch[i].to_move ^ (ids[i] % 2)) == side]
                if not selected:
                    continue
                tick = time.monotonic()
                if modes[side] == 'greedy':
                    policies, _, _ = evaluator([(batch[i].board, batch[i].scores,
                                                batch[i].to_move, batch[i].bombs)
                                               for i in selected], cfg)
                    actions.update((i, int(policy.argmax())) for i, policy in zip(selected, policies))
                else:
                    searches = []
                    for i in selected:
                        pub = batch[i].public()
                        old = previous[side][i]
                        root = old.matching_root(pub) if reuse[side] and old is not None else None
                        search = Search(pub, budgets[side], rng=rngs[side][i], root=root,
                                        c_puct=c_puct, fpu_reduction=fpu_reduction,
                                        search='gumbel' if modes[side] == 'gumbel' else 'puct',
                                        gumbel_k=gumbel_k)
                        searches.append(search)
                        previous[side][i] = search
                    run_searches(searches, evaluator, cfg, batches[side])
                    actions.update((i, search.best_action()) for i, search in zip(selected, searches))
                side_seconds[side] += time.monotonic() - tick
                move_counts[side] += len(selected)
            remaining = []
            for i in active:
                batch[i].step(actions[i])
                if not batch[i].done:
                    remaining.append(i)
            active = remaining
        outcomes.extend(int((game.winner ^ (i % 2)) == 0) for i, game in zip(ids, batch))
    pairs = np.asarray(outcomes).reshape(-1, 2).mean(1)
    bootstrap_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    means = pairs[bootstrap_rng.integers(len(pairs), size=(10000, len(pairs)))].mean(1)
    return dict(games=games, layouts=len(pairs), wins=sum(outcomes), win_rate=float(pairs.mean()),
                paired_bootstrap_95=np.quantile(means, [.025, .975]).tolist(),
                pair_sweeps=int((pairs == 1).sum()), pair_splits=int((pairs == .5).sum()),
                pair_losses=int((pairs == 0).sum()), seed=seed, mode=mode, mode_b=modes[1],
                sims=sims if mode != 'greedy' else 0, sims_b=budgets[1], c_puct=c_puct,
                fpu_reduction=fpu_reduction, leaf_batch_size=batches, tree_reuse=reuse,
                moves=move_counts, side_seconds=side_seconds,
                mean_seconds_per_move=[s / max(1, n) for s, n in zip(side_seconds, move_counts)],
                seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-a', required=True)
    parser.add_argument('--checkpoint-b', required=True)
    parser.add_argument('--mode', choices=('greedy', 'mcts', 'puct', 'gumbel'), default='mcts')
    parser.add_argument('--mode-b', choices=('greedy', 'mcts', 'puct', 'gumbel'))
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=910000)
    parser.add_argument('--sims', type=int, default=100)
    parser.add_argument('--sims-b', type=int)
    parser.add_argument('--gumbel-k', type=int, default=16)
    parser.add_argument('--leaf-batch-size', type=int, default=1)
    parser.add_argument('--leaf-batch-size-b', type=int)
    parser.add_argument('--tree-reuse', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tree-reuse-b', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--c-puct', type=float, default=1.5)
    parser.add_argument('--fpu-reduction', type=float, default=.2)
    parser.add_argument('--concurrent-games', type=int, default=16)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        torch.backends.cudnn.benchmark = True
    evaluators = [Evaluator(load(path, args.device)[0], args.device)
                  for path in (args.checkpoint_a, args.checkpoint_b)]
    options = vars(args).copy()
    for key in ('checkpoint_a', 'checkpoint_b', 'device', 'threads'):
        options.pop(key)
    try:
        result = paired_games(evaluators, Config(16, 16, 51), **options)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(dict(checkpoint_a=args.checkpoint_a, checkpoint_b=args.checkpoint_b, **result)), flush=True)


if __name__ == '__main__':
    main()
