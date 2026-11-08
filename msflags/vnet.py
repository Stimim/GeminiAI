"""Distill the two-ply expert's reply advantage into a scalar value network.

Generate targets with ``python -m msflags.vnet gen`` and fit with
``python -m msflags.vnet train --data checkpoints/deep/vtarget-*.npz``.
Values and regression errors are in mines, not win probabilities.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .deep import DeepConfig, PLANES, encode_deep
from .expert import EPS, ExpertAgent, reply_advantage
from .fastsolve import batch_probs
from .game import UNKNOWN, Config, PublicState
from .model import ResBlock


class ValueNet(nn.Module):
    """DeepNet's residual trunk with an unbounded scalar regression head."""
    def __init__(self, cfg: DeepConfig = DeepConfig()):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(PLANES, c, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(cfg.blocks)])
        self.value = nn.Sequential(nn.Linear(2 * c, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        f = self.body(self.stem(x))
        return self.value(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)).squeeze(1)


def load_net(path, device="cpu") -> ValueNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ValueNet(DeepConfig(**ck["value_cfg"])).to(device)
    net.load_state_dict(ck["state"])
    return net.eval()


class ValueEvaluator:
    def __init__(self, net: ValueNet, device="cpu", rng=None):
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.rng = rng if rng is not None else np.random.default_rng()

    @torch.inference_mode()
    def __call__(self, states, cfg: Config, beta=1.0):
        if not states:
            return np.empty(0, dtype=np.float32)
        boards = np.asarray([s[0] for s in states], dtype=np.int8)
        scores = np.asarray([s[1] for s in states])
        to_move = np.asarray([s[2] for s in states])
        bombs = np.asarray([s[3] for s in states])
        probs = batch_probs(boards, scores, cfg, seed=int(self.rng.integers(1 << 62)))
        x = torch.from_numpy(encode_deep(boards, scores, to_move, bombs, probs, cfg)).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            values = self.net(x)
        values = values.float().cpu().numpy()
        sure = ((boards == UNKNOWN) & (probs >= 1 - EPS)).sum(1)
        values = sure + beta * (values - sure)
        wins = scores[np.arange(len(states)), to_move] + sure >= cfg.win_score
        return np.where(wins, float(cfg.mines), values)


_KEYS = ("board", "scores", "to_move", "bombs", "probs")
_TARGET_DATA = None


def _target_chunk(job):
    start, stop, seed = job
    inner = ExpertAgent(np.random.default_rng(seed), bomb_reserve=3)
    cfg = Config()
    values = np.empty(stop - start, dtype=np.float32)
    for j, i in enumerate(range(start, stop)):
        pub = PublicState(cfg, _TARGET_DATA["board"][i], _TARGET_DATA["scores"][i].tolist(),
                          int(_TARGET_DATA["to_move"][i]), None, _TARGET_DATA["bombs"][i].tolist())
        values[j] = reply_advantage(pub, inner)
    return start, values


def _paths(patterns):
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise ValueError("No data shards match the supplied paths")
    return paths


def distribution(values):
    return dict(mean=float(values.mean()), std=float(values.std()), min=float(values.min()),
                max=float(values.max()), quantiles=np.quantile(values, [0, .01, .1, .5, .9, .99, 1]).tolist())


def generate(data_paths, out, *, positions=300000, workers=6, shard=10000, seed=1):
    """Uniformly select source positions; retain source-game IDs across output shards."""
    global _TARGET_DATA
    if positions <= 0 or shard <= 0 or not 1 <= workers <= 6:
        raise ValueError("positions/shard must be positive and workers must be between 1 and 6")
    paths = _paths(data_paths)
    rng = np.random.default_rng(seed)
    lengths = []
    for path in paths:
        with np.load(path) as part:
            lengths.append(len(part["board"]))
    total = sum(lengths)
    if positions > total:
        raise ValueError(f"Requested {positions} unique positions but only {total} are available")
    selected = rng.choice(total, positions, replace=False)
    data = {}
    cursor = 0
    game_offset = 0
    for path, length in zip(paths, lengths):
        slots = np.flatnonzero((selected >= cursor) & (selected < cursor + length))
        local = selected[slots] - cursor
        with np.load(path) as part:
            for key in _KEYS:
                rows = part[key][local]
                if key not in data:
                    data[key] = np.empty((positions, *rows.shape[1:]), dtype=rows.dtype)
                data[key][slots] = rows
            # Source game IDs are scoped to each expert shard.
            games = part["game"].astype(np.int64)
            if "game" not in data:
                data["game"] = np.empty(positions, dtype=np.int64)
            data["game"][slots] = games[local] + game_offset
            game_offset += int(games.max()) + 1
        cursor += length
    _TARGET_DATA = data
    targets = np.empty(positions, dtype=np.float32)
    ready = np.zeros(positions, dtype=bool)
    next_shard = 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    jobs = [(i, min(i + 128, positions), int(rng.integers(1 << 62)))
            for i in range(0, positions, 128)]
    started = time.monotonic()
    done = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for start, values in pool.imap_unordered(_target_chunk, jobs):
            targets[start:start + len(values)] = values
            done += len(values)
            ready[start:start + len(values)] = True
            while next_shard * shard < positions:
                lo, hi = next_shard * shard, min((next_shard + 1) * shard, positions)
                if not ready[lo:hi].all():
                    break
                np.savez_compressed(f"{out}-{next_shard:03d}.npz",
                                    **{k: v[lo:hi] for k, v in data.items()},
                                    target=targets[lo:hi], source_paths=np.asarray(paths), target_seed=seed)
                next_shard += 1
            if done % 4096 < len(values) or done == positions:
                print(json.dumps(dict(positions=done, total=positions,
                                      seconds=round(time.monotonic() - started, 1))), flush=True)
    print(json.dumps(dict(positions=positions, distribution=distribution(targets),
                          seconds=round(time.monotonic() - started, 1))), flush=True)
    _TARGET_DATA = None


def train(data_paths, out, *, steps=10000, batch=256, lr=.001, device="cuda",
          channels=96, blocks=8, seed=1, held_fraction=.1, log_every=500):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    paths = _paths(data_paths)
    parts = []
    source_paths = None
    for path in paths:
        with np.load(path) as part:
            sources = part["source_paths"].tolist()
            if source_paths is not None and sources != source_paths:
                raise ValueError("Target shards must share a source_paths manifest for stable game IDs")
            source_paths = sources
            parts.append({k: part[k] for k in (*_KEYS, "target", "game")})
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    games = np.unique(data["game"])
    if len(games) < 2 or not 0 < held_fraction < 1:
        raise ValueError("Need at least two source games and 0 < held_fraction < 1")
    rng.shuffle(games)
    held_games = games[:min(len(games) - 1, max(1, round(len(games) * held_fraction)))]
    held_mask = np.isin(data["game"], held_games)
    held, fit = np.flatnonzero(held_mask), np.flatnonzero(~held_mask)
    cfg = Config()
    net = ValueNet(DeepConfig(channels, blocks)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    dtype = torch.device(device).type

    def tensors(idx, augment=False):
        x = encode_deep(*(data[k][idx] for k in _KEYS), cfg)
        if augment:
            x = np.rot90(x, int(rng.integers(4)), (2, 3))
            if rng.integers(2):
                x = x[..., ::-1]
        return (torch.from_numpy(np.ascontiguousarray(x)).to(device),
                torch.from_numpy(data["target"][idx]).to(device))

    def predict(x):
        with torch.autocast(device_type=dtype, dtype=torch.bfloat16, enabled=dtype == "cuda"):
            return net(x).float()

    started = time.monotonic()
    stats = dict(train_positions=len(fit), held_positions=len(held), held_games=len(held_games),
                 target_std=float(data["target"][held].std()))
    print(json.dumps(stats), flush=True)
    for step in range(1, steps + 1):
        net.train()
        x, y = tensors(rng.choice(fit, batch), augment=True)
        loss = F.mse_loss(predict(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()
        if step % log_every == 0 or step == steps:
            net.eval()
            squared = 0.
            with torch.inference_mode():
                for start in range(0, len(held), batch):
                    xh, yh = tensors(held[start:start + batch])
                    squared += float(F.mse_loss(predict(xh), yh, reduction="sum"))
            stats.update(step=step, mse=loss.detach().item(), held_rmse=float(np.sqrt(squared / len(held))),
                         seconds=round(time.monotonic() - started, 1))
            print(json.dumps(stats), flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(value_cfg=asdict(net.cfg), state=net.state_dict(), data=paths,
                    held_games=held_games, metrics=stats, seed=seed), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--data", nargs="+", default=["checkpoints/deep/expert-a-*.npz", "checkpoints/deep/expert-b-*.npz"])
    g.add_argument("--out", default="checkpoints/deep/vtarget")
    g.add_argument("--positions", type=int, default=300000)
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--shard", type=int, default=10000)
    g.add_argument("--seed", type=int, default=1)
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", default=["checkpoints/deep/vtarget-*.npz"])
    t.add_argument("--out", default="checkpoints/deep/value.pt")
    t.add_argument("--steps", type=int, default=10000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=.001)
    t.add_argument("--channels", type=int, default=96)
    t.add_argument("--blocks", type=int, default=8)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--held-fraction", type=float, default=.1)
    t.add_argument("--log-every", type=int, default=500)
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["data_paths"] = args.pop("data")
    (generate if command == "gen" else train)(**args)


from .netexpert import NetExpert2Agent


if __name__ == "__main__":
    main()
