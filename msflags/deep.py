"""Probability-aware network ("deep expert") and its training data.

Inputs are the classic 20 public planes plus three planes from the exact
solver: mine probability, certain mines and certain safe cells. Every input is
derived from the public board, so the network never sees hidden information.

    uv run python -m msflags.deep gen --games 20000 --out checkpoints/deep/expert-0.npz
    uv run python -m msflags.deep train --data checkpoints/deep/expert-0.npz --out checkpoints/deep/net-0.pt
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert import EPS, ExpertAgent
from .fastsolve import FastAnalysis
from .game import NUM_PLANES, UNKNOWN, Config, Game, PublicState, encode_batch
from .model import ResBlock

PLANES = NUM_PLANES + 3


def encode_deep(boards: np.ndarray, scores: np.ndarray, to_move: np.ndarray, bombs: np.ndarray,
                probs: np.ndarray, cfg: Config) -> np.ndarray:
    """[B, PLANES, h, w] float32 from public states and their exact mine probabilities."""
    states = list(zip(boards, scores.tolist(), to_move.tolist(), bombs.tolist()))
    base = encode_batch(states, cfg)
    unknown = boards == UNKNOWN
    p = np.where(unknown, probs, 0.0).astype(np.float32)
    extra = np.stack([p, (unknown & (p >= 1 - EPS)), (unknown & (p <= EPS))], 1).astype(np.float32)
    return np.concatenate([base, extra.reshape(len(boards), 3, cfg.h, cfg.w)], 1)


@dataclass
class DeepConfig:
    channels: int = 96
    blocks: int = 8


class DeepNet(nn.Module):
    def __init__(self, cfg: DeepConfig = DeepConfig()):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(cfg.blocks)])
        self.policy = nn.Sequential(nn.Conv2d(c, 32, 1), nn.ReLU(), nn.Conv2d(32, 2, 1))
        self.value = nn.Sequential(nn.Linear(2 * c, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor):
        f = self.body(self.stem(x))
        logits = self.policy(f).flatten(2)  # [B, 2, N]: open, bomb centre
        unknown = x[:, 0].flatten(1) > 0.5
        available = x[:, 18].flatten(1) > 0.5
        centres = F.max_pool2d(x[:, :1], 5, stride=1, padding=2).flatten(1) > 0.5
        open_logits = logits[:, 0].masked_fill(~unknown, -1e9)
        bomb_logits = logits[:, 1].masked_fill(~(centres & available), -1e9)
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        return torch.cat([open_logits, bomb_logits], 1), torch.tanh(self.value(pooled)).squeeze(1)


def save_net(net: DeepNet, path: str | Path, **extra) -> None:
    torch.save({"deep_cfg": asdict(net.cfg), "state": net.state_dict(), **extra}, path)


def load_net(path: str | Path, device="cpu") -> DeepNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = DeepNet(DeepConfig(**ck["deep_cfg"])).to(device)
    net.load_state_dict(ck["state"])
    return net.eval()


class DeepEvaluator:
    """Search evaluator: (board, scores, to_move, bombs) states -> (policy, value, probs)."""

    def __init__(self, net: DeepNet, device="cpu"):
        self.net = net.to(device).eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def __call__(self, states, cfg: Config, probs: np.ndarray | None = None):
        boards = np.array([s[0] for s in states], dtype=np.int8)
        scores = np.array([s[1] for s in states])
        to_move = np.array([s[2] for s in states])
        bombs = np.array([s[3] for s in states])
        if probs is None:
            probs = np.array([FastAnalysis(PublicState(cfg, b, list(sc), int(t), None, list(bo))).probs
                              for b, sc, t, bo in zip(boards, scores, to_move, bombs)])
        x = torch.from_numpy(encode_deep(boards, scores, to_move, bombs, probs, cfg)).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            logits, value = self.net(x)
        policy = torch.softmax(logits.float(), 1)
        return policy.cpu().numpy(), value.float().cpu().numpy(), probs


# ------------------------------------------------------------------ data
def expert_game(job):
    """One expert-vs-expert game with softmax exploration; returns position arrays.

    ``teacher="expert"`` labels positions with the 1-ply softmax of randomly
    parameterized experts. ``teacher="expert2"`` uses fixed best parameters and
    labels each position with the two-ply expert's chosen action (one-hot); the
    first ``explore_moves`` moves are still sampled from the 1-ply softmax.
    """
    seed, temperature, explore_moves, teacher = job
    cfg = Config()
    rng = np.random.default_rng(seed)
    game = Game.random(cfg, rng)
    if teacher == "expert2":
        from .expert import Expert2Agent
        agents = [Expert2Agent(np.random.default_rng(rng.integers(1 << 62)), bomb_reserve=3.0) for _ in range(2)]
    else:
        params = [dict(alpha=float(rng.uniform(0.6, 1.6)), bomb_reserve=float(rng.uniform(0.5, 2.5)))
                  for _ in range(2)]
        agents = [ExpertAgent(np.random.default_rng(rng.integers(1 << 62)), **p) for p in params]
    rows = dict(board=[], scores=[], to_move=[], bombs=[], probs=[], policy=[])
    moves = 0
    while not game.done:
        pub = game.public()
        agent = agents[pub.to_move]
        probs = FastAnalysis(pub, rng=agent.rng).probs
        pi = np.zeros(2 * cfg.n, dtype=np.float32)
        sure = np.flatnonzero((pub.board == UNKNOWN) & (probs >= 1 - EPS))
        if len(sure):
            pi[sure] = 1.0 / len(sure)
            action = int(sure[0])
        else:
            actions, values, _ = agent.scores(pub)
            w = np.exp((values - values.max()) / temperature)
            w /= w.sum()
            if teacher == "expert2":
                label = agent.act(pub)
                pi[label] = 1.0
            else:
                label = int(actions[np.argmax(values)])
                pi[actions] = w
            if moves < explore_moves:
                action = int(actions[rng.choice(len(actions), p=w)])
            else:
                action = label
        rows["board"].append(pub.board.copy())
        rows["scores"].append(list(pub.scores))
        rows["to_move"].append(pub.to_move)
        rows["bombs"].append(list(pub.bombs))
        rows["probs"].append(probs.astype(np.float32))
        rows["policy"].append(pi)
        game.step(action)
        moves += 1
    out = {k: np.array(v) for k, v in rows.items()}
    out["z"] = np.where(out["to_move"] == game.winner, 1.0, -1.0).astype(np.float32)
    return out


_NETS: dict[tuple[str, str], DeepEvaluator] = {}


def deep_agent(rng=None, *, net: str, sims: int = 200, device: str = "cpu", c_puct: float = 1.5,
               fpu_reduction: float = 0.2, batch: int = 8):
    """Greedy network policy when sims == 0, otherwise public-state MCTS with the network."""
    from .agents import MCTSAgent, NetAgent
    key = (net, device)
    if key not in _NETS:
        torch.set_num_threads(1)
        _NETS[key] = DeepEvaluator(load_net(net, device), device)
    evaluator = _NETS[key]
    if sims == 0:
        return NetAgent(evaluator)
    return MCTSAgent(evaluator, sims=sims, rng=rng, leaf_batch_size=batch,
                     c_puct=c_puct, fpu_reduction=fpu_reduction, analysis=FastAnalysis)


def _safe_expert_game(job):
    try:
        return expert_game(job)
    except ValueError as error:  # reported and counted; one bad game must not discard a shard
        return f"{type(error).__name__}: {error}"


def _save(parts: list[dict], path: Path) -> int:
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["game"] = np.concatenate([np.full(len(p["z"]), i) for i, p in enumerate(parts)])
    data["policy"] = data["policy"].astype(np.float16)
    data["probs"] = data["probs"].astype(np.float16)
    np.savez_compressed(path, **data)
    return len(data["z"])


def generate(games: int, out: str, seed: int, workers: int, temperature: float, explore_moves: int,
             shard: int = 500, teacher: str = "expert") -> None:
    """Write expert games to ``{out}-NNN.npz`` shards of ``shard`` games each."""
    started = time.monotonic()
    jobs = [(int(s), temperature, explore_moves, teacher)
            for s in np.random.SeedSequence(seed).generate_state(games, np.uint64)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    parts, shards, positions, failures = [], 0, 0, []
    with mp.get_context("fork").Pool(workers) as pool:
        for result in pool.imap_unordered(_safe_expert_game, jobs, chunksize=2):
            if isinstance(result, str):
                failures.append(result)
            else:
                parts.append(result)
            if len(parts) == shard:
                positions += _save(parts, Path(f"{out}-{shards:03d}.npz"))
                shards += 1
                parts = []
                print(json.dumps(dict(shards=shards, positions=positions, failures=len(failures),
                                      seconds=round(time.monotonic() - started))), flush=True)
    if parts:
        positions += _save(parts, Path(f"{out}-{shards:03d}.npz"))
        shards += 1
    print(json.dumps(dict(games=games, shards=shards, positions=positions, failures=failures[:5],
                          failure_count=len(failures), seconds=round(time.monotonic() - started, 1))), flush=True)


def selfplay(net_path: str, games: int, out: str, *, seed: int, sims: int, concurrency: int,
             device: str, gumbel_k: int = 16, batch: int = 8) -> None:
    """MCTS self-play in one process; leaves of all live games share GPU batches."""
    from .mcts import Search
    from .train import run_searches
    started = time.monotonic()
    cfg = Config()
    evaluator = DeepEvaluator(load_net(net_path, device), device)
    rng = np.random.default_rng(seed)
    rows = dict(board=[], scores=[], to_move=[], bombs=[], probs=[], policy=[], z=[], game=[])
    pending = list(range(games))
    live: dict[int, dict] = {}
    finished = 0
    while pending or live:
        while pending and len(live) < concurrency:
            gid = pending.pop()
            live[gid] = dict(game=Game.random(cfg, rng), hist=[], last=None)
        searches = []
        for gid, item in live.items():
            pub = item["game"].public()
            root = item["last"].matching_root(pub) if item["last"] is not None else None
            s = Search(pub, sims, rng=rng, root=root, search="gumbel", gumbel_k=gumbel_k,
                       gumbel_noise=True, analysis=FastAnalysis)
            searches.append((gid, s))
        run_searches([s for _, s in searches], evaluator, cfg, batch)
        for gid, s in searches:
            item = live[gid]
            game = item["game"]
            pub = s.pub
            item["hist"].append((pub.board.copy(), list(pub.scores), pub.to_move, list(pub.bombs),
                                 s.analysis.probs.astype(np.float16), s.improved_policy().astype(np.float16)))
            item["last"] = s
            game.step(s.best_action())
            if game.done:
                for board, sc, tm, bo, pr, pi in item["hist"]:
                    for key, value in zip(("board", "scores", "to_move", "bombs", "probs", "policy"),
                                          (board, sc, tm, bo, pr, pi)):
                        rows[key].append(value)
                    rows["z"].append(1.0 if tm == game.winner else -1.0)
                    rows["game"].append(gid)
                del live[gid]
                finished += 1
                if finished % 50 == 0:
                    print(json.dumps(dict(finished=finished, seconds=round(time.monotonic() - started))), flush=True)
    data = {k: np.array(v) for k, v in rows.items()}
    data["z"] = data["z"].astype(np.float32)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(json.dumps(dict(games=games, positions=len(data["z"]), out=out,
                          seconds=round(time.monotonic() - started, 1))), flush=True)


# -------------------------------------------------------------- training
def train(data_paths: list[str], out: str, *, steps: int, batch: int, lr: float, device: str,
          channels: int, blocks: int, init: str | None, value_weight: float, seed: int) -> None:
    torch.manual_seed(seed)
    cfg = Config()
    parts = [np.load(p) for p in data_paths]
    keys = ("board", "scores", "to_move", "bombs", "probs", "policy", "z")
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    n = len(data["z"])
    rng = np.random.default_rng(seed)
    # Hold out whole games (positions within a game are strongly correlated).
    game = np.cumsum((data["board"] == UNKNOWN).all(1)) - 1
    held_mask = game % 50 == 0
    held = np.flatnonzero(held_mask)
    train_idx = np.flatnonzero(~held_mask)
    net = load_net(init, device) if init else DeepNet(DeepConfig(channels, blocks)).to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    def batch_tensors(idx):
        x = encode_deep(data["board"][idx], data["scores"][idx], data["to_move"][idx], data["bombs"][idx],
                        data["probs"][idx].astype(np.float32), cfg)
        # Random dihedral symmetry: square boards keep all 8.
        k, flip = int(rng.integers(4)), bool(rng.integers(2))
        pi = data["policy"][idx].astype(np.float32).reshape(len(idx), 2, cfg.h, cfg.w)
        x = np.rot90(x, k, (2, 3))
        pi = np.rot90(pi, k, (2, 3))
        if flip:
            x, pi = x[..., ::-1], pi[..., ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        pi = torch.from_numpy(np.ascontiguousarray(pi).reshape(len(idx), -1)).to(device)
        z = torch.from_numpy(data["z"][idx]).to(device)
        return x, pi, z

    def losses(x, pi, z):
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            logits, v = net(x)
        logp = torch.log_softmax(logits.float(), 1)
        pl = -(pi * logp.clamp_min(-1e4)).sum(1).mean()
        vl = F.mse_loss(v.float(), z)
        return pl, vl

    started = time.monotonic()
    for step in range(1, steps + 1):
        idx = train_idx[rng.integers(len(train_idx), size=batch)]
        pl, vl = losses(*batch_tensors(idx))
        loss = pl + value_weight * vl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                hp, hv = [], []
                for i in range(0, min(len(held), 8192), 1024):
                    a, b = losses(*batch_tensors(held[i:i + 1024]))
                    hp.append(a.item()); hv.append(b.item())
            net.train()
            print(json.dumps(dict(step=step, policy=round(pl.item(), 4), value=round(vl.item(), 4),
                                  held_policy=round(float(np.mean(hp)), 4), held_value=round(float(np.mean(hv)), 4),
                                  seconds=round(time.monotonic() - started))), flush=True)
    net.eval()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_net(net, out, data=data_paths, steps=steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--games", type=int, default=2000)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--workers", type=int, default=18)
    g.add_argument("--temperature", type=float, default=0.3)
    g.add_argument("--explore-moves", type=int, default=20)
    g.add_argument("--teacher", choices=("expert", "expert2"), default="expert")
    g.add_argument("--shard", type=int, default=500)
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--channels", type=int, default=96)
    t.add_argument("--blocks", type=int, default=8)
    t.add_argument("--init")
    t.add_argument("--value-weight", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s = sub.add_parser("selfplay")
    s.add_argument("--net", required=True)
    s.add_argument("--games", type=int, default=200)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--sims", type=int, default=64)
    s.add_argument("--gumbel-k", type=int, default=16)
    s.add_argument("--concurrency", type=int, default=64)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.cmd == "gen":
        generate(args.games, args.out, args.seed, args.workers, args.temperature, args.explore_moves,
                 shard=args.shard, teacher=args.teacher)
    elif args.cmd == "selfplay":
        torch.set_num_threads(1)
        selfplay(args.net, args.games, args.out, seed=args.seed, sims=args.sims,
                 concurrency=args.concurrency, device=args.device, gumbel_k=args.gumbel_k)
    else:
        train(args.data, args.out, steps=args.steps, batch=args.batch, lr=args.lr, device=args.device,
              channels=args.channels, blocks=args.blocks, init=args.init,
              value_weight=args.value_weight, seed=args.seed)


if __name__ == "__main__":
    main()
