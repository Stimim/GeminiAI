# Minesweeper Flags

A two-player Minesweeper game and AI playground. Players alternate opening cells: claiming a mine scores a point and grants another move; a safe cell reveals its adjacent-mine count (zeros flood-fill) and passes the turn. Each player also has one 5×5 bomb: it claims unopened mines, reveals safe cells, and always ends the turn. First to a majority of the mines wins. The default board is 16×16 with 51 mines, first to 26.

## Architecture

- **Exact constraint solver:** derives mine probabilities from the visible clues and enumerates consistent assignments.
- **ResNet evaluator:** fully convolutional mine, open-policy, bomb-policy, and value heads, using 20 public-information input planes. Open policy is `beta * mine_logit + residual(features)`; the bomb head starts from clipped 5×5 expected mine counts. Open and bomb actions share a masked softmax over `2 × board cells`.
- **Information-set MCTS:** searches from public information, sampling consistent determinizations with the constraint solver.

## Setup

```sh
uv sync
```

## Training

```sh
uv run python -m msflags.train bootstrap --output checkpoints/bootstrap-medium --data checkpoints/bootstrap-bomb-2000.npz --network-size medium --games 2000 --steps 8000 --workers 12
uv run python -m msflags.train bootstrap --output checkpoints/bootstrap-small --data checkpoints/bootstrap-bomb-2000.npz --network-size small --steps 8000
uv run python -m msflags.train arena --agent-a r64x4-bs8k.mcts100-c1.5 --agent-b solver --games 200
```

Check subcommand options with `uv run python -m msflags.train bootstrap --help`, `uv run python -m msflags.train selfplay --help`, or `uv run python -m msflags.train arena --help`.

Train from scratch or continue a supported 20-plane checkpoint:

```sh
uv run python -m msflags.train bootstrap --checkpoint checkpoints/bootstrap-medium/latest.pt --output checkpoints/continued --data checkpoints/bootstrap-bomb-2000.npz --steps 8000 --device cuda
```

Bomb solver baseline: open any certain mine first; otherwise detonate the highest expected-mine window when its expectation is at least three (and exceeds the best single opening). A bomb gives up the usual extra turn, so the threshold trades immediate multi-mine gain against saving the one-use resource. Bootstrap explores all legal actions and stores a joint open/bomb policy; uncertain solver fallback positions are excluded from exact labels.

A quick alternating-seat sweep (40 games each, seeds 81000–81039) gave thresholds two and four each 20/40 wins against threshold three (6.3 seconds total). This small experiment does not establish an optimum; three remains the conservative midpoint baseline.

### Fresh bomb bootstrap

The registered medium (64 channels × 4 blocks) and small (32 × 3) networks start from random weights and share 2,000 explored bomb-solver games. `--network-size small|medium` selects architecture; numeric `--size` remains the square board width. No transferred variant is registered. Its checkpoint remains on disk; retired-agent arena results are preserved in `checkpoints/arena/archive-transferred.jsonl`, with the pre-reset ledger backed up alongside it. Solver/random-only results remain active.

The shared corpus contains **112,302 exact-labelled positions** (epsilon 0.08, seed 20261107), generated in 11.9 seconds. Both sizes trained for 8,000 steps at lr=0.001, batch size 128; medium took 126.5 seconds end to end (including corpus generation), small 98.0 seconds using the same dataset. On 200 alternating-seat greedy games against solver (seed 310000), medium won **136/200 (68.0%, 95% CI 61.2–74.1%)** and small **105/200 (52.5%, 45.6–59.3%)**.

Medium MCTS-100 won **131/200 (65.5%, 95% CI 58.7–71.7%)** against solver with the same arena seed, in 412.3 seconds under concurrent load; greedy evaluations each took 9.7 seconds. These evaluations are unrecorded head-to-head tests. The fresh 20-games-per-pair round robin added 200 ledger games in 195.0 seconds. With 21 retained solver/random games, the 221-game leaderboard is medium greedy **1145.4 ±113.2**, small greedy **1054.8 ±109.4**, medium MCTS **1019.3 ±109.3**, solver **1000**, random **6.9 ±607.0** Elo. Small pair counts do not establish a reliable ordering between close agents.

The reset archived **100** retired-agent games and retained **21** solver/random games. Backup: `checkpoints/arena/games-before-reset-1793983677327548715.jsonl`. Bootstrap and promoted self-play checkpoints are pinned in `msflags/generations.json`.

## Web UI

Run the server with `uv run python -m msflags.server` and open http://127.0.0.1:8000. Neural catalog entries use pinned timestamped checkpoints; random and solver need no weights. Use `--host`, `--port`, and `--device cpu|cuda` to configure it. The interface supports human-vs-AI, AI-vs-AI, and hot-seat games, with hints and move-by-move animation.

### AI catalog and arena

The catalog has six categories: Random, Greedy exact solver, and ResNet 64×4 / 32×3 with either Greedy policy or MCTS-100 (c 1.5). Each neural category has immutable checkpoint generations; both search modes of an architecture share its pinned generation list. Generation 0 is bootstrap; generation k is completed self-play iteration k. Category IDs (`r64x4.greedy`, `r64x4.mcts100-c1.5`, `r32x3.greedy`, `r32x3.mcts100-c1.5`) select the latest available generation. Baselines have one generation each. Category detail shows architecture, search settings, training summaries and generation ratings; generation detail retains full checkpoint provenance.

Generation IDs remain `{arch}-{training}.{search}`: for example `r64x4-bs8k-sp12.greedy`. They derive from checkpoint metadata, not filenames or list position. Bootstrap uses `bs{steps in k}`, followed by `-sp{completed iterations}` for self-play. Search is `greedy` or `mcts{sims}-c{c_puct}` (plus `-f{fpu}` for nondefault FPU reduction; default 0.2). Missing provenance and duplicate identities are rejected. Arena records and Elo always belong to concrete generations, never moving categories. Play’s simulation budget applies only to MCTS; arena always uses the category’s fixed settings.

Only 20-plane checkpoints with joint open/bomb policies are supported.


Arena games alternate the starting seat and append to `checkpoints/arena/games.jsonl`. One match runs at a time per server process; live boards expose only public information. Match progress is in memory; completed games survive restarts. Omit the API seed for fresh random games, or supply a nonnegative seed to reproduce a match. The offline round-robin batches up to 32 games for efficient neural inference; the web arena runs one game at a time for a coherent live view. Ratings reread the ledger, so newly completed CLI games appear without restarting the server. Run one server process when using the arena.

#### Round-robin CLI

`--games` is the number of games **per pair**. By default the latest available generation of each category participates; six categories produce fifteen pairs, so `--games 20` records 300 games. Use `--agents r64x4.greedy solver r32x3.mcts100-c1.5` to select categories, or pass immutable generation IDs. `--seed 200000` makes the tournament reproducible. The `ratings` command refits all persisted results without additional games. The training CLI's `arena` command accepts generation IDs; `--checkpoint` / `--checkpoint-b` override the corresponding neural weights for unrecorded evaluations.

```sh
uv run python -m msflags.arena roundrobin --games 20 --device cuda
uv run python -m msflags.arena ratings
```

#### Ratings method

Ratings fit all recorded games jointly with a Bradley–Terry logistic model, fixing the greedy solver at **1000 Elo**. The win probability is `1 / (1 + 10 ** ((rating_b - rating_a) / 400))`. The fit maximizes the likelihood with a Gaussian regularizer centered at 1000, standard deviation **800 Elo**, for every non-anchor agent, keeping perfect win/loss records finite. The leaderboard reports games played and approximate 95% uncertainty half-widths (`1.96 × standard error`) from the inverse fitted Hessian, conditional on the fixed anchor; these are not guarantees of future performance. Random is rated, not used as the anchor.


API: `GET /api/agents` returns `{agents: [category, …]}`. Each category has `id`, `name`, `kind`, `summary`, `available`, `default_generation` (latest available ID), and `generations`. Generation rows include `id`, `category_id`, `generation`, `label`, `available`, `games`, and, when rated, `elo` and `uncertainty` (95% half-width). `GET /api/agents/{id}` accepts category or generation IDs. Game seats and hints also accept either; game creation resolves categories once and stores concrete generation IDs. `POST /api/arena/matches` (`{a,b,games,seed}`) resolves contenders before recording results. `GET /api/arena/matches/{id}` includes the live public board; `GET /api/ratings` retains per-generation ratings. Unknown/unavailable agents and invalid counts are rejected; a second active match returns HTTP 409.

#### Adding a checkpoint or agent

1. Train to a timestamped checkpoint, keeping its run's `metrics.jsonl` alongside it. Only completed immutable checkpoints are eligible; never promote `latest.pt`.
2. Run `uv run python -m msflags.registry promote checkpoints/my-run/selfplay-<timestamp>.pt`. This validates the checkpoint and architecture, rejects duplicate paths/identities, and appends it to that architecture's tracked generation list in `msflags/generations.json`. Both search categories acquire the generation together.
3. Inspect `uv run python -m msflags.registry list`, open `/agents.html`, and rate the generation with `uv run python -m msflags.arena roundrobin --agents <generation-id> solver --games 20 --device cuda`.

Never overwrite, rename or repoint pinned weights: historical results are keyed by their derived generation IDs. A missing pinned file remains visible as unavailable, and category defaults skip it. `registry.pinned_checkpoints()` exposes all referenced paths so checkpoint pruning preserves promoted weights. New architecture/search categories are explicit definitions in `registry.py`; promotion does not invent categories.

## Training runs

`checkpoints/run-medium/` (64×4) and `checkpoints/run-small/` (32×3) resume existing optimizer, replay, and iteration counters. The output directory's `latest.pt` always takes precedence over `--checkpoint`. `--iterations 0` is open-ended; positive values are total target iteration numbers, not additional iterations. Interrupted unpublished work is regenerated; the command's learning rate overrides the restored optimizer learning rate.

### Round 2 launch and operation

Both runs use **96 games / iteration, MCTS-100, c_puct=1.5, unchanged Dirichlet noise, 300 updates × 128 samples**, and 50,000-position replay. Each has two persistent spawned self-play processes, each batching 16 games with one Torch thread; the trainer uses two Torch threads. Equal CPU budgets avoid assigning all 20 hardware threads to either run. Each worker owns its CUDA evaluator; completed games merge in seed/batch order.

Medium (detached process group **1734838**, also in `train.pid`):

```sh
setsid nohup uv run python -m msflags.train selfplay --output checkpoints/run-medium --checkpoint checkpoints/bootstrap-medium/latest.pt --iterations 0 --games 96 --concurrent-games 16 --workers 2 --worker-threads 1 --sims 100 --steps 300 --batch-size 128 --lr 0.0002 --threads 2 --replay-size 50000 --eval-every 5 --eval-games 100 --seed 610000 --device cuda >> checkpoints/run-medium/train.log 2>&1 &
echo $! > checkpoints/run-medium/train.pid
```

Small (detached process group **1734839**):

```sh
setsid nohup uv run python -m msflags.train selfplay --output checkpoints/run-small --checkpoint checkpoints/bootstrap-small/latest.pt --iterations 0 --games 96 --concurrent-games 16 --workers 2 --worker-threads 1 --sims 100 --steps 300 --batch-size 128 --lr 0.001 --threads 2 --replay-size 50000 --eval-every 5 --eval-games 100 --seed 620000 --device cuda >> checkpoints/run-small/train.log 2>&1 &
echo $! > checkpoints/run-small/train.pid
```

Monitor:

```sh
tail -f checkpoints/run-medium/train.log checkpoints/run-small/train.log
nvidia-smi
```

Stop gracefully at the next complete iteration; workers finish games and publish checkpoint/replay before exit:

```sh
kill -TERM -- -"$(cat checkpoints/run-medium/train.pid)"
kill -TERM -- -"$(cat checkpoints/run-small/train.pid)"
```

Wait for process groups to exit before repeating launch commands. Never launch two trainers into one run directory. `latest.pt` is atomically published after immutable weights, optimizer state, and replay are complete. Two replay slots preserve the previous published checkpoint during writes; only the latest checkpoint is guaranteed a matching replay for continuation. Metrics include self-play generation, losses, iteration throughput/reuse, evaluations every five iterations, and retention decisions.

### Throughput measurements

Fixed seed **730000**, bootstrap weights, **32 games / 100 simulations / 100 updates**, batch size 128, CUDA, outputs under `/tmp`. Both original live process groups were SIGSTOP-paused during each benchmark pair and SIGCONT-resumed afterward; the concurrent arena was paused for optimized measurements. GPU percentages are sampled whole-GPU means; CPU is approximate steady-state search-process utilization (100% = one logical CPU). Iteration time includes startup/pool costs.

| Network | Pipeline | Games/s | Seconds/iteration | GPU utilization | Search CPU |
|---|---|---:|---:|---:|---:|
| medium | baseline, 1×32 | 0.332 | 99.03 | 34% | ~102% |
| medium | optimized, 2×16 | 0.581 | 58.30 | 37% | ~206% |
| small | baseline, 1×32 | 0.350 | 93.78 | 29% | ~103% |
| small | optimized, 2×16 | 0.576 | 58.38 | 38% | ~203% |

Generation improved **1.75× medium / 1.65× small**. The earlier profile attributed 64% to forward+encoding and 15% to solver sampling. Encoding now batches neighbour sums instead of looping over states: a 32-state CPU microbenchmark improved **4.01 ms → 0.282 ms (14.2×)**. Neighbour sums accumulate in place; solver sampling caches conditional distributions and decoded reservoir cells without changing RNG draws. Persistent workers overlap CPU tree traversal and independent batched CUDA evaluation. Existing inference mode and CUDA BF16 remain enabled; search budgets, selection/noise, and sampling distributions are unchanged. Replay encoding uses bounded 1,024-position chunks.

First completed iterations of the final tested-code launches, with both runs active:

| Run | Iteration | Seconds | Games/s | New/replay positions | Sample reuse |
|---|---:|---:|---:|---:|---:|
| medium | 6 | 286.03 | 0.345 | 6,655 / 44,995 | 5.77× |
| small | 5 | 250.98 | 0.404 | 6,190 / 35,041 | 6.20× |

These contention-inclusive results are not the isolated benchmark. Both detached launches remain running and resumed previously published self-play weights, not bootstrap weights. The concurrent GPU snapshot reached 85% utilization at 2,024 / 8,188 MiB (including the arena).

### Disk budget

The budget is **100 GB decimal for the whole project**, including checkpoints, datasets, replay, logs, and the environment. After these iterations: **6.92 GB total project; 1.31 GB checkpoints** (logical file sizes, hard-linked `latest.pt` counted once). Self-play weights with AdamW occupy **3,934,581 bytes medium / 883,833 bytes small**. Each replay position is **3,339 bytes**: one saturated 50,000-position slot is ~167 MB; both slots for both runs plateau at ~668 MB. Until saturation, ~6,200–6,700 new positions add ~21–22 MB per run/iteration; replay is subsequently replaced, not accumulated.

Pruning runs before generation and after each published iteration. It keeps **all bootstrap checkpoints; every tenth self-play iteration; the newest three iterations per run; `latest.pt`; both replay slots; datasets; and every path returned by lazily imported `registry.pinned_checkpoints()`**. Only other immutable `selfplay-*.pt` files are removed. Iteration numbers come from checkpoint metadata, never timestamps. An unavailable registry hook keeps everything and emits a warning. If protected data still puts the project/checkpoints over 100 GB, training warns loudly and stops; milestones and pinned generations are never sacrificed.

```sh
uv run python -m msflags.train prune --dry-run
uv run python -m msflags.train prune
```

Without retention, a completed iteration pair adds ~4.82 MB of weights; every-tenth retention averages ~0.482 MB per pair, plus rolling newest-three files and additional pins. At observed concurrent iteration times, milestone growth is **~0.15 GB/day**, versus ~1.5 GB/day unpruned. Replay growth stops at its bound; logs and extra registered generations add to the estimate. The hard cap eventually stops training rather than deleting permanent generations.

Promote immutable timestamped weights using the registry CLI described above (`uv run python -m msflags.registry --help`), never `latest.pt`. Never repoint a registered generation to different weights.

## Tests

```sh
uv run pytest -q
```

## Deployment considerations (future: public hosting)

- Games and active arena matches live in process-memory dictionaries without expiry or eviction; memory grows, and multiple processes/replicas do not share them. Use a shared store (e.g. SQLite/Postgres or Redis) and expire sessions.
- Arena results use a local JSONL ledger, while web matches run in a server thread. For hosting, move persistence to a database and execution to a background job worker; restrict arena-match creation to authenticated admins because it starts long CPU/GPU jobs.
- MCTS is CPU-bound Python search with per-move network evaluations; one process may serve few concurrent AI moves. Cap simulations server-side, add rate limits and request timeouts, and consider separate batched inference; GPU versus CPU-only hosting is a cost choice (CPU PyTorch images are much smaller than CUDA images).
- The board's mine layout is withheld until game end, but client-supplied seeds make layouts reproducible and hints expose AI help. For ranked/online play, use server-side secret seeds and disable hints.
- Anyone with a game ID can act for either human seat (hot-seat design). Online human-vs-human needs per-seat tokens/accounts and push updates (WebSocket/SSE) instead of polling.
- Checkpoints are loaded with `torch.load(weights_only=False)`; never load untrusted files. Prefer `weights_only=True` or `safetensors` for deployment.
- Put HTTPS behind a reverse proxy; static files can be served by a proxy/CDN. Add a health-check endpoint and structured logs.

