"""Expert iteration for the probability-aware network.

Each iteration: (1) truncated-rollout search with the current net plays
``--games`` games and labels every position with its chosen move; (2) the net
is fine-tuned on the newest ``--window`` iterations of that data plus fixed
``--anchor`` shards; (3) the new net plays the old one greedily on paired
layouts. Results are appended to ``{dir}/loop.jsonl``; a rerun resumes after
the last finished iteration. The next teacher is always the newest net.

    uv run python -m msflags.deeploop --dir checkpoints/deep/iter --start checkpoints/deep/net-2.pt \
        --anchor checkpoints/deep/expert2-a-00[0-4].npz
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], log) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
    log.write(proc.stdout)
    log.flush()
    if proc.returncode:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True)
    p.add_argument("--start", required=True, help="initial network checkpoint")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--games", type=int, default=600, help="teacher games per iteration")
    p.add_argument("--workers", type=int, default=5, help="CUDA teacher processes (~1.4 GB RAM each)")
    p.add_argument("--window", type=int, default=2, help="recent iterations of teacher data to train on")
    p.add_argument("--anchor", nargs="*", default=[], help="fixed shards always included")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gate-layouts", type=int, default=300)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    record = root / "loop.jsonl"
    done = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
    net = done[-1]["next_teacher"] if done else args.start
    py = sys.executable
    with open(root / "loop.log", "a") as log:
        for it in range(len(done), args.iterations):
            started = time.monotonic()
            prefix = root / f"teacher-{it:03d}"
            run([py, "-m", "msflags.deep", "gen", "--games", str(args.games), "--workers", str(args.workers),
                 "--teacher", f"rollout:{net}", "--seed", str(args.seed + it), "--shard", "100",
                 "--out", str(prefix)], log)
            generated = time.monotonic() - started
            data = [str(f) for f in sorted(root.glob("teacher-*.npz"))
                    if int(f.name.split("-")[1]) > it - args.window]
            new = root / f"net-{it:03d}.pt"
            run([py, "-m", "msflags.deep", "train", "--init", net, "--out", str(new), "--data", *data,
                 *args.anchor, "--steps", str(args.steps), "--lr", str(args.lr), "--value-weight", "0.1",
                 "--seed", str(it)], log)
            gate = json.loads(run([py, "-m", "msflags.bench", f"deep:net={new},sims=0,device=cuda",
                                   f"deep:net={net},sims=0,device=cuda", "--layouts", str(args.gate_layouts),
                                   "--workers", "4", "--seed", str(9000 + it)], log).strip().splitlines()[-1])
            # Promote only improvements; otherwise the old net remains the teacher.
            promoted = gate["win_rate"] > 0.5
            entry = dict(iteration=it, teacher=net, net=str(new), games=args.games, data_files=len(data),
                         gate_win_rate=gate["win_rate"], gate_ci95=gate["ci95"], promoted=promoted,
                         next_teacher=str(new) if promoted else net,
                         generate_seconds=round(generated), seconds=round(time.monotonic() - started),
                         time=time.time())
            with record.open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            net = entry["next_teacher"]


if __name__ == "__main__":
    main()
