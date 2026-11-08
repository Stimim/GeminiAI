"""Self-play improvement loop for the probability-aware network.

Each iteration runs ``--procs`` self-play processes (Gumbel MCTS, GPU leaves)
with the current network, then fine-tunes it on a sliding window of the most
recent self-play files plus optional fixed anchor data. Every iteration is
appended to ``{dir}/loop.jsonl``; rerunning resumes after the last finished
iteration.

    uv run python -m msflags.deeploop --dir checkpoints/deep/loop-a --start checkpoints/deep/net-0.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], log) -> None:
    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--start", required=True, help="initial network checkpoint")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games", type=int, default=600, help="self-play games per iteration")
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--gumbel-k", type=int, default=16)
    p.add_argument("--window", type=int, default=8, help="recent iterations of self-play to train on")
    p.add_argument("--anchor", nargs="*", default=[], help="fixed data always included (e.g. expert shards)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    record = root / "loop.jsonl"
    done = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
    net = done[-1]["net"] if done else args.start
    log = open(root / "loop.log", "a")
    py = sys.executable
    for it in range(len(done), args.iterations):
        started = time.monotonic()
        per = -(-args.games // args.procs)
        outs = [root / f"sp-{it:03d}-{k}.npz" for k in range(args.procs)]
        procs = [subprocess.Popen([py, "-m", "msflags.deep", "selfplay", "--net", net, "--games", str(per),
                                   "--out", str(out), "--seed", str(args.seed + 1000 * it + k),
                                   "--sims", str(args.sims), "--gumbel-k", str(args.gumbel_k),
                                   "--concurrency", "48"], stdout=log, stderr=subprocess.STDOUT)
                 for k, out in enumerate(outs)]
        if any(proc.wait() for proc in procs):
            raise SystemExit(f"self-play failed in iteration {it}")
        played = time.monotonic() - started
        window = sorted(root.glob("sp-*.npz"))
        window = [f for f in window if int(f.name.split("-")[1]) > it - args.window]
        new = root / f"net-{it:03d}.pt"
        run([py, "-m", "msflags.deep", "train", "--init", net, "--out", str(new),
             "--data", *map(str, window), *args.anchor, "--steps", str(args.steps),
             "--lr", str(args.lr), "--value-weight", str(args.value_weight), "--seed", str(it)], log)
        net = str(new)
        entry = dict(iteration=it, net=net, games=per * args.procs, selfplay_seconds=round(played),
                     seconds=round(time.monotonic() - started), window=len(window), time=time.time())
        with record.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
