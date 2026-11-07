"""Unrecorded paired, alternating-seat checkpoint comparisons.

Each layout is generated independently of play and played with both seats.
Search uses fresh public-information trees with no exploration noise.
"""
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
                 c_puct=1.5, fpu_reduction=0.2, concurrent_games=16):
    if games < 2 or games % 2:
        raise ValueError('games must be positive and even (two seats per layout)')
    if sims < 1 or concurrent_games < 1:
        raise ValueError('sims and concurrent-games must be positive')
    if mode not in ('greedy', 'mcts'):
        raise ValueError('mode must be greedy or mcts')
    started = time.monotonic()
    outcomes = []
    for offset in range(0, games, concurrent_games):
        ids = list(range(offset, min(games, offset + concurrent_games)))
        batch = [Game.random(cfg, np.random.default_rng(np.random.SeedSequence([seed, 0, i // 2])))
                 for i in ids]
        rngs = [np.random.default_rng(np.random.SeedSequence([seed, 1, i // 2, i % 2])) for i in ids]
        active = list(range(len(batch)))
        while active:
            actions = {}
            for side, evaluator in enumerate(evaluators):
                selected = [i for i in active if (batch[i].to_move ^ (ids[i] % 2)) == side]
                if not selected:
                    continue
                if mode == 'greedy':
                    policies, _, _ = evaluator([(batch[i].board, batch[i].scores,
                                                batch[i].to_move, batch[i].bombs)
                                               for i in selected], cfg)
                    actions.update((i, int(policy.argmax())) for i, policy in zip(selected, policies))
                else:
                    searches = [Search(batch[i].public(), sims, rng=rngs[i], c_puct=c_puct,
                                       fpu_reduction=fpu_reduction) for i in selected]
                    run_searches(searches, evaluator, cfg)
                    actions.update((i, search.best_action()) for i, search in zip(selected, searches))
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
                pair_losses=int((pairs == 0).sum()), seed=seed, mode=mode,
                sims=sims if mode == 'mcts' else 0, c_puct=c_puct,
                fpu_reduction=fpu_reduction, seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-a', required=True)
    parser.add_argument('--checkpoint-b', required=True)
    parser.add_argument('--mode', choices=('greedy', 'mcts'), default='mcts')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=910000)
    parser.add_argument('--sims', type=int, default=100)
    parser.add_argument('--c-puct', type=float, default=1.5)
    parser.add_argument('--fpu-reduction', type=float, default=.2)
    parser.add_argument('--concurrent-games', type=int, default=16)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.sims < 1 or args.concurrent_games < 1 or args.threads < 1:
        parser.error('games must be even and >=2; sims, concurrent-games and threads must be positive')
    torch.set_num_threads(args.threads)
    evaluators = [Evaluator(load(path, args.device)[0], args.device)
                  for path in (args.checkpoint_a, args.checkpoint_b)]
    result = paired_games(evaluators, Config(16, 16, 51), games=args.games, seed=args.seed,
                          mode=args.mode, sims=args.sims, c_puct=args.c_puct,
                          fpu_reduction=args.fpu_reduction, concurrent_games=args.concurrent_games)
    print(json.dumps(dict(checkpoint_a=args.checkpoint_a, checkpoint_b=args.checkpoint_b, **result)), flush=True)


if __name__ == '__main__':
    main()
