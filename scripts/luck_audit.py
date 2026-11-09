import json, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def run(chunk):
    from msflags.game import Config, Game, UNKNOWN, window_sum
    from msflags.agents import SolverAgent
    from msflags.deep import deep_agent
    from msflags.fastsolve import FastAnalysis
    cfg = Config(); out = []
    net = deep_agent(np.random.default_rng(0), net="checkpoints/deep/iter/net-003.pt", sims=0, device="cuda")
    for layout in chunk:
        for swap in (0, 1):
            g = Game.random(cfg, np.random.default_rng(np.random.SeedSequence([77, layout])))
            solver = SolverAgent(np.random.default_rng(np.random.SeedSequence([78, layout, swap])))
            agents = [net, solver] if swap == 0 else [solver, net]
            me = swap  # seat of n3
            luck = [0.0, 0.0]; certain = [0, 0]
            while not g.done:
                pub = g.public(); p = g.to_move; side = 0 if p == me else 1
                a = agents[p].act(pub)
                probs = FastAnalysis(pub).probs
                before = g.scores[p]
                if a < cfg.n:
                    if probs[a] >= 1 - 1e-9:
                        certain[side] += 1; g.step(a)
                    else:
                        g.step(a); luck[side] += (g.scores[p] - before) - probs[a]
                else:
                    exp = window_sum(np.where(pub.board == UNKNOWN, probs, 0.0).reshape(cfg.h, cfg.w)).reshape(-1)[a - cfg.n]
                    g.step(a); luck[side] += (g.scores[p] - before) - exp
            margin = g.scores[me] - g.scores[1 - me]
            out.append(dict(won=int(g.winner == me), margin=margin, luck=luck[0] - luck[1], certain=certain))
    return out

if __name__ == "__main__":
    layouts = list(range(int(sys.argv[1])))
    chunks = [layouts[i::2] for i in range(2)]
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as pool:
        res = [r for part in pool.map(run, chunks) for r in part]
    won = np.array([r["won"] for r in res]); margin = np.array([r["margin"] for r in res], float)
    luck = np.array([r["luck"] for r in res])
    print(json.dumps(dict(games=len(res), win_rate=won.mean(),
        luck_in_wins=luck[won == 1].mean(), luck_in_losses=luck[won == 0].mean(),
        margin_in_losses=margin[won == 0].mean(),
        losses=int((won == 0).sum()),
        losses_with_luck_deficit_exceeding_margin=int(((won == 0) & (-luck > -margin)).sum()),
        luck_sd=luck.std())))
