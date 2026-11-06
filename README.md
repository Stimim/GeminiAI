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

The reset archived **100** retired-agent games and retained **21** solver/random games. Backup: `checkpoints/arena/games-before-reset-1793983677327548715.jsonl`. Bootstrap initialization checkpoints are pinned in `registry.VARIANTS`; self-play checkpoints are not registered.

## Web UI

Run the server with `uv run python -m msflags.server` and open http://127.0.0.1:8000. Neural catalog entries use pinned timestamped checkpoints; random and solver need no weights. Use `--host`, `--port`, and `--device cpu|cuda` to configure it. The interface supports human-vs-AI, AI-vs-AI, and hot-seat games, with hints and move-by-move animation.

### AI catalog and arena

Use **Play** (`/`), **AIs** (`/agents.html`), and **Arena** (`/arena.html`). The catalog explains each strategy and reads network shape and training provenance from real checkpoints and their `metrics.jsonl`. Evaluators are loaded lazily and shared by checkpoint. Neural IDs are derived as `{arch}-{training}.{search}`: `r{channels}x{blocks}`, `bs{bootstrap steps in k}` plus `-sp{completed self-play iterations}` when applicable, and `greedy` or `mcts{sims}-c{c_puct}` (with `-f{fpu}` only for nondefault FPU reduction, default 0.2). Missing training provenance and duplicate derived IDs raise errors rather than guessing. The registered neural entries are `r64x4-bs8k.greedy`, `r32x3-bs8k.greedy`, and `r64x4-bs8k.mcts100-c1.5`. Arena always uses the listed fixed settings.

Only 20-plane checkpoints with joint open/bomb policies are supported.


Arena games alternate the starting seat and append to `checkpoints/arena/games.jsonl`. One match runs at a time per server process; live boards expose only public information. Match progress is in memory; completed games survive restarts. Omit the API seed for fresh random games, or supply a nonnegative seed to reproduce a match. The offline round-robin batches up to 32 games for efficient neural inference; the web arena runs one game at a time for a coherent live view. Ratings reread the ledger, so newly completed CLI games appear without restarting the server. Run one server process when using the arena.

#### Round-robin CLI

`--games` is the number of games **per pair**. By default, every available registry entry participates; five agents produce ten pairs, so `--games 20` records 200 games. Use `--agents r64x4-bs8k.greedy solver r64x4-bs8k.mcts100-c1.5` to select contenders, and `--seed 200000` for a reproducible tournament. The `ratings` command refits all persisted results without playing additional games. The training CLI's `arena` command also accepts catalog IDs; `--checkpoint` / `--checkpoint-b` still override the corresponding neural weights for unrecorded evaluations, with `--sims`, `--c-puct`, and `--fpu-reduction` controlling overridden search.

```sh
uv run python -m msflags.arena roundrobin --games 20 --device cuda
uv run python -m msflags.arena ratings
```

#### Ratings method

Ratings fit all recorded games jointly with a Bradley–Terry logistic model, fixing the greedy solver at **1000 Elo**. The win probability is `1 / (1 + 10 ** ((rating_b - rating_a) / 400))`. The fit maximizes the likelihood with a Gaussian regularizer centered at 1000, standard deviation **800 Elo**, for every non-anchor agent, keeping perfect win/loss records finite. The leaderboard reports games played and approximate 95% uncertainty half-widths (`1.96 × standard error`) from the inverse fitted Hessian, conditional on the fixed anchor; these are not guarantees of future performance. Random is rated, not used as the anchor.


API: `GET /api/agents`, `GET /api/agents/{id}`, `POST /api/arena/matches` (`{a,b,games,seed}`), `GET /api/arena/matches/{id}` (including `current` live board), and `GET /api/ratings`. Unknown/unavailable agents and invalid counts are rejected; a second active match returns HTTP 409. Existing game, move, AI and hint routes accept catalog IDs (plus `human` for game seats).

#### Adding a checkpoint or agent

1. Train to a new timestamped checkpoint, such as `checkpoints/my-run/bootstrap-<timestamp>.pt`; keep the run's `metrics.jsonl` alongside it. Shape comes from `net_cfg`; completed training comes from checkpoint metadata (`bootstrap_steps`, `selfplay_iterations`) or completed metrics, not requested iteration counts.
2. Add a `(checkpoint, search)` line to `VARIANTS` in `msflags/registry.py`. Use `None` for greedy policy, or search settings for MCTS:

   ```python
   ('checkpoints/my-run/bootstrap-<timestamp>.pt', None),
   ('checkpoints/my-run/bootstrap-<timestamp>.pt',
    dict(sims=100, c_puct=1.5, fpu_reduction=0.2)),
   ```

3. IDs, display names, summaries, and explanations are generated automatically. Restart the server, open `/agents.html`, and run `uv run python -m msflags.arena roundrobin --agents <derived-id> solver --games 20 --device cuda`.
4. Never repoint an entry to different weights or use `latest.pt` / newest-file globs: persisted results are keyed by derived ID. If a new checkpoint derives an existing ID, do not register it alongside the existing variant or overwrite its weights; use a genuinely distinct architecture/training/search configuration.

## Training runs

`checkpoints/run-medium/` and `checkpoints/run-small/` are separate, unregistered self-play runs initialized from their corresponding fresh bootstrap checkpoints. Each iteration uses MCTS-100 (c=1.5), bounded replay (50,000 positions), and saves immutable `selfplay-<timestamp>.pt` weights, optimizer state, and a matching replay snapshot. `latest.pt` is atomically published only after the checkpoint and replay are complete. Two replay slots bound buffer disk use; old timestamped checkpoints remain usable for inference/promotion, but only the latest run checkpoint is guaranteed a matching replay for continuation.

An existing run's `latest.pt` takes precedence over `--checkpoint`, so repeat the original command to resume safely. `--iterations` is the total target iteration number, not an additional count. Iteration seeds and counters continue after the latest completed checkpoint. Interrupted work since that checkpoint is regenerated. Learning rate is taken from the resume command; optimizer moments are restored.

Medium launch (PID **1729825**, also recorded in `checkpoints/run-medium/train.pid`):

```sh
setsid nohup uv run python -m msflags.train selfplay --output checkpoints/run-medium --checkpoint checkpoints/bootstrap-medium/bootstrap-1793983777620140569.pt --iterations 1000 --games 128 --concurrent-games 32 --sims 100 --steps 400 --batch-size 128 --lr 0.0002 --threads 2 --replay-size 50000 --eval-every 5 --eval-games 100 --seed 610000 --device cuda > checkpoints/run-medium/train.log 2>&1 &
```

Small launch (PID **1729929**, also recorded in `checkpoints/run-small/train.pid`):

```sh
setsid nohup uv run python -m msflags.train selfplay --output checkpoints/run-small --checkpoint checkpoints/bootstrap-small/bootstrap-1793983891981972304.pt --iterations 1000 --games 128 --concurrent-games 32 --sims 100 --steps 400 --batch-size 128 --lr 0.001 --threads 2 --replay-size 50000 --eval-every 5 --eval-games 100 --seed 620000 --device cuda > checkpoints/run-small/train.log 2>&1 &
```

Sizing trials (32 games, 100 updates) completed in **100.3 s medium** and **129.8 s small**: 0.330 and 0.253 games/s, with peak allocated VRAM **132.4 MiB** and **71.4 MiB**, respectively. Trials overlapped other arena/training jobs, so these are contention-inclusive measurements, not isolated benchmarks. Their sample reuse was 7.18× / 6.09×; the long runs scale games and updates together by four (128 / 400). Each process uses two Torch threads and 32 concurrent games; both fit comfortably in 8 GB VRAM.

Both detached runs completed iteration 1 and continued training:

| Run | Iteration time | Games/s | New/replay positions | Sample reuse | Policy/value/mine loss (last 100 updates) |
|---|---:|---:|---:|---:|---|
| medium | 379.0 s | 0.343 | 7,186 | 7.12× | 1.9529 / 0.5473 / 0.3844 |
| small | 384.6 s | 0.338 | 7,737 | 6.62× | 2.3325 / 0.6193 / 0.3704 |

Their first immutable checkpoints are `checkpoints/run-medium/selfplay-1793984389238123137.pt` and `checkpoints/run-small/selfplay-1793984438726959293.pt`. The two-run steady-state GPU snapshot used **1,359 / 8,188 MiB**, at 35% utilization; peak allocated tensor memory remained 132.4 / 71.4 MiB. WSL's per-process `nvidia-smi` memory was unavailable, so allocated peaks come from PyTorch metrics. Neither run is GPU-memory constrained; concurrent throughput supports keeping both running.

Monitor with `tail -f checkpoints/run-medium/train.log checkpoints/run-small/train.log`. Machine-readable progress is in each run's `metrics.jsonl`: `selfplay_games` records batch generation, `train` records losses, `iteration` records wall time, throughput, replay size, sample reuse, peak allocated VRAM, and the timestamped checkpoint. Greedy solver evaluations are recorded as `evaluation` every five iterations (100 games). Replay and encoded minibatch storage are bounded; timestamped checkpoints and logs deliberately accumulate on disk.

Stop medium with `kill -- -1729825`, or small with `kill -- -1729929`. These send SIGTERM to the entire launch process group, including the `uv` child. Resume by repeating that run's exact launch command above, changing `> .../train.log` to `>> .../train.log` to preserve console history; keep the same output directory and bootstrap argument (`latest.pt` wins automatically). The new launch has a new PID; record it with `echo $!`. Do not launch two processes into the same run directory.

To promote a result, stop or inspect the run, select an actual immutable `selfplay-<timestamp>.pt`, and add its path plus search settings to `registry.VARIANTS` as described above. Never register `latest.pt`, never repoint an existing entry, and do not register the in-progress runs automatically.

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

